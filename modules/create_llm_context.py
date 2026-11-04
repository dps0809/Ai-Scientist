import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def create_llm_context(
    df: pd.DataFrame,
    target: str,
    patterns: List[Dict[str, Any]],
    stage1_data: Optional[Dict[str, Any]] = None,
    stage2_data: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs structured scientific context for the LLM containing:
    1. Stage 1: Ranked individual feature dependencies (Pearson, Spearman, Mutual Information).
    2. Stage 2: Feature combination discovery (Single AUCs, Combined AUC, Synergy gain, Joint MI).
    3. Stage 2: Feature interaction analysis (Additive AUC vs Interaction AUC, interaction term significance).
    4. Inter-variable correlations (between predictors) to identify confounders/mediators.
    5. Summary statistics (mean, std, min, max) for quantitative context.
    """
    evidence = []
    for pattern in patterns:
        p_type = pattern.get("type")
        if p_type == "individual_dependency":
            evidence.append({
                "type": "individual_dependency",
                "feature": pattern.get("feature"),
                "target": pattern.get("target"),
                "pearson_r": pattern.get("pearson_r"),
                "spearman_rho": pattern.get("spearman_rho"),
                "mutual_information": pattern.get("mutual_information"),
                "p_value": pattern.get("pearson_p_value"),
                "direction": pattern.get("direction"),
                "rank": pattern.get("rank")
            })
        elif p_type == "feature_combination_synergy":
            evidence.append({
                "type": "feature_combination_synergy",
                "combination": pattern.get("combination"),
                "score_a": pattern.get("score_a"),
                "score_b": pattern.get("score_b"),
                "combined_score": pattern.get("combined_score"),
                "synergy_gain": pattern.get("synergy_gain"),
                "joint_mi": pattern.get("joint_mi"),
                "interaction_delta": pattern.get("interaction_delta"),
                "conclusion": pattern.get("conclusion")
            })
        elif p_type == "correlation":
            evidence.append({
                "type": "statistical_relationship",
                "feature": pattern.get("feature"),
                "target": pattern.get("target"),
                "correlation": pattern.get("correlation"),
                "p_value": pattern.get("p_value"),
                "direction": pattern.get("direction")
            })
        elif p_type == "ml_feature_importance":
            evidence.append({
                "type": "feature_importance",
                "feature": pattern.get("feature"),
                "importance": pattern.get("importance")
            })
        elif p_type in ["group_difference", "categorical_association"]:
            evidence.append(pattern)

    # ------------------------------------------------------------------
    # Inter-variable correlations: find notable correlations between
    # feature pairs (excluding target) so the LLM can reason about
    # multi-variable relationships.
    # ------------------------------------------------------------------
    inter_variable_correlations = []
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    features_only = [c for c in numerical_cols if c != target]

    if len(features_only) >= 2:
        corr_matrix = df[features_only].corr()
        seen = set()
        for i, col_a in enumerate(features_only):
            for j, col_b in enumerate(features_only):
                if i >= j:
                    continue
                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                r = float(corr_matrix.loc[col_a, col_b])
                if abs(r) >= 0.15:
                    inter_variable_correlations.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(r, 4),
                        "strength": "strong" if abs(r) >= 0.4 else
                                    "moderate" if abs(r) >= 0.25 else "weak"
                    })
        # Sort by absolute correlation descending, keep top 10
        inter_variable_correlations.sort(
            key=lambda x: abs(x["correlation"]), reverse=True
        )
        inter_variable_correlations = inter_variable_correlations[:10]

    # ------------------------------------------------------------------
    # Summary statistics per feature (mean, std, min, max)
    # ------------------------------------------------------------------
    summary_stats = {}
    for col in features_only:
        desc = df[col].describe()
        summary_stats[col] = {
            "mean": round(float(desc["mean"]), 2),
            "std": round(float(desc["std"]), 2),
            "min": round(float(desc["min"]), 2),
            "max": round(float(desc["max"]), 2),
        }

    # Extract Stage 1 structured tables
    stage1_table = []
    if stage1_data and "table" in stage1_data:
        stage1_table = stage1_data["table"]

    # Extract Stage 2 structured tables
    stage2_combinations = []
    stage2_interactions = []
    if stage2_data and "combinations" in stage2_data:
        for item in stage2_data["combinations"]:
            stage2_combinations.append({
                "combination": item["combination"],
                "score_a": item["score_a"],
                "score_b": item["score_b"],
                "combined_score": item["combined_score"],
                "synergy_gain": item["synergy_delta"],
                "joint_mi": item["joint_mi"],
                "mi_gain": item["mi_gain"],
                "status": item["synergy_status"],
                "metric": item["metric"]
            })
            if item["interaction_delta"] != 0 or item["interaction_p_value"] != "N/A":
                stage2_interactions.append({
                    "combination": item["combination"],
                    "additive_score": item["combined_score"],
                    "interaction_score": item["interaction_score"],
                    "interaction_delta": item["interaction_delta"],
                    "interaction_p_value": item["interaction_p_value"],
                    "status": item["interaction_status"],
                    "metric": item["metric"]
                })

    context = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target": str(target),
        "all_features": features_only,
        "stage_1_individual_dependencies": stage1_table,
        "stage_2_feature_combinations": stage2_combinations[:8],
        "stage_2_feature_interactions": stage2_interactions[:8],
        "inter_variable_correlations": inter_variable_correlations,
        "summary_statistics": summary_stats,
        "discovered_evidence": evidence,
    }

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "llm_context.json"), "w") as f:
        json.dump(context, f, indent=4)

    return context
