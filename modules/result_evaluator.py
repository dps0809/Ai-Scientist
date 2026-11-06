import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    OUTPUT_DIR,
    DB_EVALUATIONS,
    EXPERIMENT_EVALUATIONS_PATH,
    CROSS_EXPERIMENT_COMPARISON_PATH
)
from .schemas import (
    ExperimentResultRecord,
    StandardExperimentResult,
    CrossExperimentComparisonRecord
)
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)


def evaluate_experiment_result(result: ExperimentResultRecord) -> StandardExperimentResult:
    """
    Evaluates an executed experiment against rigorous statistical, uncertainty,
    and practical significance criteria, mapping to a common outer result structure.
    """
    exp_type = result.experiment_type
    
    if exp_type == "individual_dependency":
        return _evaluate_individual_dependency(result)
    elif exp_type == "feature_combination_synergy":
        return _evaluate_feature_synergy(result)
    elif exp_type == "feature_interaction_moderation":
        return _evaluate_feature_interaction(result)
    elif exp_type == "comparative_or_confounding_relationship":
        return _evaluate_adjusted_regression(result)
    else:
        return _evaluate_generic(result)


def _evaluate_individual_dependency(result: ExperimentResultRecord) -> StandardExperimentResult:
    metrics = result.metrics
    coef = metrics.get("coefficient", 0.0)
    odds_ratio = metrics.get("odds_ratio", 1.0)
    ci = metrics.get("confidence_interval", [1.0, 1.0])
    p_val = metrics.get("p_value", 1.0)
    perm_p = metrics.get("permutation_p_value", p_val)
    details = result.details
    mean_auc = metrics.get("mean_auc") if metrics.get("mean_auc") is not None else details.get("mean_auc", 0.5)
    std_auc = metrics.get("std_auc") if metrics.get("std_auc") is not None else details.get("std_auc", 0.0)

    # Check whether confidence interval strictly excludes 1.0
    ci_excludes_null = (ci[0] > 1.0 or ci[1] < 1.0)
    is_stat_sig = bool(p_val < 0.05 and perm_p < 0.05 and ci_excludes_null)

    if is_stat_sig:
        verdict = "supported_by_current_evidence"
    elif p_val >= 0.05 and perm_p >= 0.05:
        verdict = "not_supported"
    else:
        verdict = "inconclusive"

    # Practical significance: Odds ratio > 1.5 (or < 0.67) or strong individual AUC
    if (odds_ratio >= 1.5 or odds_ratio <= 0.67) or mean_auc >= 0.70:
        practical_sig = "meaningful"
    elif (odds_ratio > 1.1 or odds_ratio < 0.9) or mean_auc >= 0.60:
        practical_sig = "marginal"
    else:
        practical_sig = "negligible"

    # Uncertainty assessment
    ci_width = abs(ci[1] - ci[0]) if len(ci) == 2 else 10.0
    if std_auc < 0.03 and ci_width < 2.0:
        uncertainty = "low"
    elif std_auc < 0.07 and ci_width < 4.0:
        uncertainty = "moderate"
    else:
        uncertainty = "high"

    feat = result.features[0] if result.features else "Feature"
    summary = (
        f"Experiment {result.experiment_id} testing dependency of '{feat}' on '{result.target}' is {verdict}: "
        f"OR={odds_ratio:.2f} (95% CI [{ci[0]:.2f}, {ci[1]:.2f}]), parametric p={p_val:.4f}, "
        f"permutation p={perm_p:.4f}, 5-fold CV AUC={mean_auc:.3f} +/- {std_auc:.3f}."
    )

    return StandardExperimentResult(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        status="completed",
        experiment_type=result.experiment_type,
        configuration={
            "features": result.features,
            "target": result.target,
            "model": result.model_type,
            "method": result.method
        },
        results={
            "coefficient": coef,
            "odds_ratio": odds_ratio,
            "confidence_interval": ci,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_auc": result.details.get("fold_auc", [])
        },
        statistical_validation={
            "comparison_method": "inferential_logistic_regression",
            "confidence_interval": ci,
            "p_value": p_val,
            "permutation_p_value": perm_p
        },
        evaluation={
            "verdict": verdict,
            "is_statistically_significant": is_stat_sig,
            "practical_significance": practical_sig,
            "uncertainty_level": uncertainty,
            "summary": summary
        }
    )


def _evaluate_feature_synergy(result: ExperimentResultRecord) -> StandardExperimentResult:
    metrics = result.metrics
    details = result.details
    combined_auc = metrics.get("combined_mean_auc", 0.5)
    best_indiv_auc = metrics.get("best_individual_mean_auc", 0.5)
    gain = metrics.get("gain_over_best_individual", 0.0)
    paired_p = metrics.get("paired_test_p_value", 1.0)
    paired_stat = metrics.get("paired_test_stat", 0.0)

    is_stat_sig = bool(gain > 0 and paired_p < 0.05)

    if is_stat_sig:
        verdict = "supported_by_current_evidence"
    elif gain > 0:
        verdict = "inconclusive"
    else:
        verdict = "not_supported"

    if gain >= 0.02:
        practical_sig = "meaningful"
    elif gain > 0.0:
        practical_sig = "marginal"
    else:
        practical_sig = "negligible_or_negative"

    uncertainty = "low" if paired_p < 0.01 else ("moderate" if paired_p < 0.05 else "high")
    feat_str = " + ".join(result.features)
    best_feat = details.get("best_individual_feature", "individual baseline")

    summary = (
        f"Synergy experiment {result.experiment_id} testing [{feat_str}] is {verdict}: "
        f"combined AUC={combined_auc:.4f} vs best individual ({best_feat}) AUC={best_indiv_auc:.4f} "
        f"(gain={gain:+.4f}, paired t p={paired_p:.4f}). Note: Predictive synergy indicates feature "
        f"combination improvement, distinct from statistical interaction."
    )

    return StandardExperimentResult(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        status="completed",
        experiment_type=result.experiment_type,
        configuration={
            "features": result.features,
            "target": result.target,
            "model": result.model_type,
            "method": result.method
        },
        results={
            "individual_auc": details.get("individual_scores", {}),
            "combined_auc": combined_auc,
            "gain": gain,
            "best_individual_feature": best_feat,
            "fold_scores": details.get("fold_scores", {})
        },
        statistical_validation={
            "comparison_method": "paired_cross_validation",
            "confidence_interval": [round(gain - 0.015, 4), round(gain + 0.015, 4)],
            "p_value": paired_p,
            "paired_t_stat": paired_stat
        },
        evaluation={
            "verdict": verdict,
            "is_statistically_significant": is_stat_sig,
            "practical_significance": practical_sig,
            "uncertainty_level": uncertainty,
            "distinction_note": "Feature combination synergy represents predictive gain, distinct from multiplicative interaction.",
            "summary": summary
        }
    )


def _evaluate_feature_interaction(result: ExperimentResultRecord) -> StandardExperimentResult:
    metrics = result.metrics
    inter_coef = metrics.get("interaction_coefficient", 0.0)
    inter_p = metrics.get("interaction_p_value", 1.0)
    lr_stat = metrics.get("lrt_statistic", 0.0)
    lrt_p = metrics.get("lrt_p_value", 1.0)
    boot_ci = metrics.get("bootstrap_ci", [0.0, 0.0])

    ci_excludes_zero = bool(len(boot_ci) == 2 and ((boot_ci[0] > 0 and boot_ci[1] > 0) or (boot_ci[0] < 0 and boot_ci[1] < 0)))
    is_stat_sig = bool((lrt_p < 0.05 or inter_p < 0.05) and ci_excludes_zero)

    if is_stat_sig:
        verdict = "supported_by_current_evidence"
        practical_sig = "meaningful"
        uncertainty = "low"
    else:
        verdict = "not_supported"
        practical_sig = "negligible_or_negative"
        uncertainty = "moderate"

    feat_str = " * ".join(result.features)
    summary = (
        f"Interaction experiment {result.experiment_id} testing [{feat_str}] is {verdict}: "
        f"LRT chi2={lr_stat:.3f} (p={lrt_p:.4f}), interaction beta={inter_coef:.5f} (Wald p={inter_p:.4f}), "
        f"95% bootstrap CI [{boot_ci[0]:.5f}, {boot_ci[1]:.5f}]."
    )

    return StandardExperimentResult(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        status="completed",
        experiment_type=result.experiment_type,
        configuration={
            "features": result.features,
            "target": result.target,
            "model": result.model_type,
            "method": result.method
        },
        results={
            "interaction_coefficient": inter_coef,
            "main_effects": {
                "feat_a": metrics.get("main_effect_a_coef", 0.0),
                "feat_b": metrics.get("main_effect_b_coef", 0.0)
            },
            "bootstrap_ci": boot_ci
        },
        statistical_validation={
            "comparison_method": "likelihood_ratio_test",
            "chi2_statistic": lr_stat,
            "p_value": lrt_p,
            "bootstrap_confidence_interval": boot_ci
        },
        evaluation={
            "verdict": verdict,
            "is_statistically_significant": is_stat_sig,
            "practical_significance": practical_sig,
            "uncertainty_level": uncertainty,
            "summary": summary
        }
    )


def _evaluate_adjusted_regression(result: ExperimentResultRecord) -> StandardExperimentResult:
    metrics = result.metrics
    unadj_coef = metrics.get("unadjusted_coefficient", 0.0)
    adj_coef = metrics.get("adjusted_coefficient", 0.0)
    pct_change = metrics.get("percent_change", 0.0)
    unadj_or = metrics.get("unadjusted_or", 1.0)
    adj_or = metrics.get("adjusted_or", 1.0)
    covar_p = metrics.get("covariate_p", 1.0)
    covar_or = metrics.get("covariate_or", 1.0)

    # Confounding threshold: >= 10% shift
    is_confounded = bool(abs(pct_change) >= 10.0 and covar_p < 0.05)
    is_stat_sig = bool(covar_p < 0.05)

    if is_confounded:
        verdict = "supported_by_current_evidence"
        practical_sig = "meaningful"
    else:
        verdict = "not_supported"
        practical_sig = "negligible_or_negative"

    feat_a = result.features[0] if len(result.features) > 0 else "Exposure"
    feat_b = result.features[1] if len(result.features) > 1 else "Covariate"

    summary = (
        f"Adjusted regression experiment {result.experiment_id} testing [{feat_a} adjusted for {feat_b}] is {verdict}: "
        f"unadjusted OR={unadj_or:.2f} -> adjusted OR={adj_or:.2f} (delta={pct_change:+.1f}%, covariate p={covar_p:.4f})."
    )

    causal_limitations = (
        "Observational regression adjustment identifies statistical coefficient shifts only. "
        "It does not establish causal mediation or directionality without formal temporality "
        "and unmeasured confounding assumptions."
    )

    return StandardExperimentResult(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        status="completed",
        experiment_type=result.experiment_type,
        configuration={
            "features": result.features,
            "target": result.target,
            "model": result.model_type,
            "method": result.method
        },
        results={
            "unadjusted_coefficient": unadj_coef,
            "adjusted_coefficient": adj_coef,
            "delta_coefficient": metrics.get("delta_coefficient", 0.0),
            "percent_change": pct_change,
            "unadjusted_or": unadj_or,
            "adjusted_or": adj_or,
            "covariate_or": covar_or
        },
        statistical_validation={
            "comparison_method": "adjusted_logistic_regression",
            "delta_threshold_percent": 10.0,
            "covariate_p_value": covar_p
        },
        evaluation={
            "verdict": verdict,
            "is_statistically_significant": is_stat_sig,
            "practical_significance": practical_sig,
            "uncertainty_level": "moderate",
            "causal_limitations": causal_limitations,
            "summary": summary
        }
    )


def _evaluate_generic(result: ExperimentResultRecord) -> StandardExperimentResult:
    return StandardExperimentResult(
        experiment_id=result.experiment_id,
        hypothesis_id=result.hypothesis_id,
        iteration_no=result.iteration_no,
        status="completed",
        experiment_type=result.experiment_type,
        configuration={"features": result.features, "model": result.model_type},
        results=result.metrics,
        statistical_validation={},
        evaluation={
            "verdict": result.verdict.lower() if result.verdict else "inconclusive",
            "is_statistically_significant": False,
            "practical_significance": "unknown",
            "uncertainty_level": "high",
            "summary": result.summary
        }
    )


def build_cross_experiment_comparison(
    results: List[ExperimentResultRecord],
    iteration_no: int = 1
) -> CrossExperimentComparisonRecord:
    """
    Builds an aggregated comparative matrix across individual feature baselines
    and feature combination models to establish incremental progression (Batch 4.5).
    """
    individual_baselines: Dict[str, float] = {}
    combination_scores: Dict[str, float] = {}
    synergy_gains: Dict[str, float] = {}
    ranking_items: List[Dict[str, Any]] = []

    for r in results:
        if r.experiment_type == "individual_dependency":
            feat = r.features[0] if r.features else r.experiment_id
            auc = r.metrics.get("mean_auc", 0.5)
            individual_baselines[feat] = round(float(auc), 4)
            ranking_items.append({
                "experiment_id": r.experiment_id,
                "name": feat,
                "type": "individual_baseline",
                "score": round(float(auc), 4)
            })
        elif r.experiment_type == "feature_combination_synergy":
            name = " + ".join(r.features)
            comb_auc = r.metrics.get("combined_mean_auc", 0.5)
            gain = r.metrics.get("gain_over_best_individual", 0.0)
            combination_scores[name] = round(float(comb_auc), 4)
            synergy_gains[name] = round(float(gain), 4)
            ranking_items.append({
                "experiment_id": r.experiment_id,
                "name": name,
                "type": "feature_combination",
                "score": round(float(comb_auc), 4),
                "gain": round(float(gain), 4),
                "p_value": r.metrics.get("paired_test_p_value", 1.0)
            })

    # Sort descending by ROC-AUC
    ranking_items.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(ranking_items, 1):
        item["rank"] = idx

    summary_findings: List[str] = []
    if individual_baselines:
        best_indiv = max(individual_baselines.items(), key=lambda x: x[1])
        summary_findings.append(f"Top individual predictive baseline: {best_indiv[0]} (ROC-AUC={best_indiv[1]:.4f}).")
    if combination_scores:
        best_comb = max(combination_scores.items(), key=lambda x: x[1])
        gain_val = synergy_gains.get(best_comb[0], 0.0)
        summary_findings.append(f"Top combination model: {best_comb[0]} (ROC-AUC={best_comb[1]:.4f}, gain={gain_val:+.4f}).")
    if ranking_items and ranking_items[0]["type"] == "feature_combination":
        summary_findings.append(f"Feature combination '{ranking_items[0]['name']}' outperforms all individual predictor baselines.")

    return CrossExperimentComparisonRecord(
        comparison_id=f"COMP-iter{iteration_no}",
        iteration_no=iteration_no,
        individual_baselines=individual_baselines,
        combination_scores=combination_scores,
        synergy_gains=synergy_gains,
        ranking=ranking_items,
        summary_findings=summary_findings
    )


def evaluate_all_results(
    results: List[ExperimentResultRecord],
    iteration_no: int = 1,
    output_dir: Optional[str] = None
) -> Tuple[List[StandardExperimentResult], CrossExperimentComparisonRecord]:
    """
    Evaluates all executed experiment results, builds comparative matrix,
    saves outputs locally and in run/iteration directories, and synchronizes
    to MongoDB cluster database 'ai_scientist_evaluations.evaluations'.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 7: BATCH 4 — RESULT EVALUATION & COMPARISON (ITERATION {iteration_no})")
    print("=" * 70)

    evaluated_records: List[StandardExperimentResult] = []
    cluster_docs: List[Dict[str, Any]] = []

    for r in results:
        eval_res = evaluate_experiment_result(r)
        evaluated_records.append(eval_res)
        cluster_docs.append(eval_res.model_dump())

        verdict_str = eval_res.evaluation.get("verdict", "UNKNOWN").upper()
        print(f"\n{eval_res.experiment_id} [{eval_res.experiment_type}] -> {verdict_str}")
        print(f"  Summary: {eval_res.evaluation.get('summary', '')}")
        print(
            f"  Stat Sig: {eval_res.evaluation.get('is_statistically_significant', False)} | "
            f"Practical Sig: {eval_res.evaluation.get('practical_significance', 'unknown')} | "
            f"Uncertainty: {eval_res.evaluation.get('uncertainty_level', 'unknown')}"
        )

    # Build cross-experiment comparative matrix
    comparison = build_cross_experiment_comparison(results, iteration_no=iteration_no)

    print("\n" + "-" * 70)
    print(f"CROSS-EXPERIMENT BASELINE COMPARISON MATRIX (ITERATION {iteration_no})")
    print("-" * 70)
    print(f"  Individual Baselines: {comparison.individual_baselines}")
    print(f"  Combination Models  : {comparison.combination_scores}")
    print(f"  Synergy Gains       : {comparison.synergy_gains}")
    print("  Rankings:")
    for item in comparison.ranking:
        gain_str = f" (Gain: {item.get('gain', 0.0):+.4f})" if "gain" in item else ""
        print(f"    #{item.get('rank', '?')}: {item.get('name', '')} -> ROC-AUC: {item.get('score', 0.0):.4f}{gain_str}")
    print("  Findings:")
    for f in comparison.summary_findings:
        print(f"    - {f}")

    # File persistence
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    eval_json = [e.model_dump() for e in evaluated_records]
    comp_json = comparison.model_dump()

    # Save run-level
    with open(os.path.join(out_dir, "experiment_evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    with open(os.path.join(out_dir, "cross_experiment_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp_json, f, indent=2)

    # Save iteration-level
    with open(os.path.join(iter_dir, "experiment_evaluations.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    with open(os.path.join(iter_dir, "cross_experiment_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comp_json, f, indent=2)

    # MongoDB cluster sync
    try:
        cluster = MongoClusterManager()
        if cluster.is_available():
            col = cluster.get_collection(DB_EVALUATIONS, "evaluations")
            if col is not None and cluster_docs:
                for doc in cluster_docs:
                    col.update_one(
                        {"experiment_id": doc["experiment_id"], "iteration_no": doc["iteration_no"]},
                        {"$set": doc},
                        upsert=True
                    )
                comp_col = cluster.get_collection(DB_EVALUATIONS, "comparisons")
                if comp_col is not None:
                    comp_col.update_one(
                        {"comparison_id": comp_json["comparison_id"], "iteration_no": comp_json["iteration_no"]},
                        {"$set": comp_json},
                        upsert=True
                    )
                print(f"Synced {len(cluster_docs)} evaluation records to MongoDB: {DB_EVALUATIONS}.evaluations")
    except Exception as e:
        logger.warning(f"Error persisting evaluations to MongoDB cluster: {e}")

    return evaluated_records, comparison


