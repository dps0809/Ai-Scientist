import os
import json
import numpy as np
import pandas as pd
from typing import cast
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def individual_dependency_analysis(df: pd.DataFrame, target: str, output_dir=None, force_recompute: bool = False):
    """
    Stage 1: Individual Feature Dependency Analysis
    Calculates the independent dependency of each feature on the target:
      - Pearson correlation (linear association & p-value)
      - Spearman correlation (monotonic association & p-value)
      - Mutual Information (linear + non-linear dependency)
      - Composite ranking of individual features
    """
    out_dir = output_dir or OUTPUT_DIR
    stage1_json_path = os.path.join(out_dir, "stage1_data.json")

    # Seek cached processed data for this dataset
    if os.path.exists(stage1_json_path) and not force_recompute:
        try:
            with open(stage1_json_path, "r") as f:
                cached = json.load(f)
            cached_feats = set(cached.get("ranked_features", []))
            df_numeric_feats = set(df.select_dtypes(include=np.number).columns) - {target}
            if cached_feats and cached_feats == df_numeric_feats:
                print(f"[CACHE HIT] Loaded Stage 1 dependency analysis from: {stage1_json_path}")
                return cached
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("STAGE 1: INDIVIDUAL FEATURE DEPENDENCY ANALYSIS")
    print("=" * 70)

    results = []
    patterns = []
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    is_classification = df[target].nunique() <= 10 or df[target].dtype == "object"

    # Prepare target variable
    y = df[target].copy()
    if is_classification and (y.dtype == "object" or str(y.dtype) == "category"):
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=df.index)
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0)

    # Compute individual dependencies for each feature
    candidate_features = [col for col in df.columns if col != target]

    for col in candidate_features:
        x = df[col].copy()
        if x.nunique() < 2:
            continue

        # Handle numeric vs categorical feature encoding
        if col in numerical_cols:
            x_clean = pd.to_numeric(x, errors="coerce").fillna(x.median() if pd.notnull(x.median()) else 0)
        else:
            x_clean = pd.Series(LabelEncoder().fit_transform(x.astype(str)), index=df.index)

        # 1. Pearson Correlation
        try:
            p_res = cast(tuple[float, float], pearsonr(x_clean, y))
            r_val = float(p_res[0])
            p_val = float(p_res[1])
        except Exception:
            r_val, p_val = 0.0, 1.0

        # 2. Spearman Correlation
        try:
            s_res = cast(tuple[float, float], spearmanr(x_clean, y))
            rho_val = float(s_res[0])
            s_pval = float(s_res[1])
        except Exception:
            rho_val, s_pval = 0.0, 1.0

        # 3. Mutual Information
        try:
            X_2d = x_clean.values.reshape(-1, 1)
            if is_classification:
                mi_array = mutual_info_classif(X_2d, y, random_state=42)
            else:
                mi_array = mutual_info_regression(X_2d, y, random_state=42)
            mi_val = float(mi_array[0])
        except Exception:
            mi_val = 0.0

        direction = "positive" if r_val >= 0 else "negative"
        significance = "significant (p < 0.05)" if (p_val < 0.05 or s_pval < 0.05) else "not significant"

        p_val_str = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        s_pval_str = "< 0.0001" if s_pval < 0.0001 else f"{s_pval:.4f}"

        item = {
            "feature": col,
            "pearson_r": round(r_val, 4),
            "pearson_p_value": p_val_str,
            "spearman_rho": round(rho_val, 4),
            "spearman_p_value": s_pval_str,
            "mutual_information": round(mi_val, 4),
            "direction": direction,
            "significance": significance
        }
        results.append(item)

    if not results:
        print("No valid features analyzed in Stage 1.")
        return {"ranked_features": [], "table": [], "patterns": []}

    # Rank features: prioritize Mutual Information and absolute correlations
    # Composite score = 0.5 * MI_normalized + 0.25 * |Pearson| + 0.25 * |Spearman|
    max_mi = max([r["mutual_information"] for r in results]) or 1.0
    for r in results:
        norm_mi = r["mutual_information"] / max_mi
        r["composite_score"] = round(0.5 * norm_mi + 0.25 * abs(r["pearson_r"]) + 0.25 * abs(r["spearman_rho"]), 4)

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Assign ranks
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    # Display ranked table
    print(f"\n{'Rank':<5} {'Feature':<25} {'Pearson r':<12} {'Spearman rho':<14} {'MI':<10} {'Significance':<15}")
    print("-" * 85)
    for r in results:
        print(f"{r['rank']:<5} {r['feature']:<25} {r['pearson_r']:<12} {r['spearman_rho']:<14} {r['mutual_information']:<10.4f} {r['significance']:<15}")

    # Export to CSV
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "individual_dependencies.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nStage 1 results saved to: {csv_path}")

    # Format patterns for general pipeline compatibility
    for r in results:
        patterns.append({
            "type": "individual_dependency",
            "feature": r["feature"],
            "target": target,
            "pearson_r": r["pearson_r"],
            "pearson_p_value": r["pearson_p_value"],
            "spearman_rho": r["spearman_rho"],
            "mutual_information": r["mutual_information"],
            "direction": r["direction"],
            "rank": r["rank"],
            "composite_score": r["composite_score"]
        })

    ranked_feature_names = [r["feature"] for r in results]

    output_dict = {
        "ranked_features": ranked_feature_names,
        "table": results,
        "patterns": patterns
    }

    try:
        with open(os.path.join(out_dir, "stage1_data.json"), "w") as f:
            json.dump(output_dict, f, indent=4)
    except Exception:
        pass

    return output_dict
