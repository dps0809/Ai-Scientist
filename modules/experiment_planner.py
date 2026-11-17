import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from .config import OUTPUT_DIR, EXPERIMENT_SPECS_PATH, DB_EXPERIMENTS
from .schemas import ExperimentSpecRecord, ValidationReportRecord, ValidationStatus
from .mongo_cluster import MongoClusterManager

logger = logging.getLogger(__name__)

# Predefined Registry mapping hypothesis types to validated statistical execution targets
EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "individual_dependency": {
        "method": "logistic_regression",
        "model_type": "statsmodels.Logit",
        "parameters": {"fit_intercept": True, "maxiter": 100},
        "evaluation_template": {
            "odds_ratio": True,
            "confidence_interval": True,
            "confidence_level": 0.95,
            "p_value": True,
            "permutation_test": True,
            "permutation_iterations": 1000
        }
    },
    "feature_combination_synergy": {
        "method": "paired_model_comparison",
        "model_type": "sklearn.linear_model.LogisticRegression",
        "parameters": {"solver": "lbfgs", "max_iter": 500},
        "evaluation_template": {
            "metric": "roc_auc",
            "cross_validation": "StratifiedKFold",
            "folds": 5
        }
    },
    "feature_interaction_moderation": {
        "method": "interaction_regression",
        "model_type": "statsmodels.Logit",
        "parameters": {"include_interaction_term": True, "maxiter": 100},
        "evaluation_template": {
            "interaction_term": True,
            "interaction_p_value": True,
            "likelihood_ratio_test": True,
            "bootstrap_samples": 500
        }
    },
    "comparative_or_confounding_relationship": {
        "method": "adjusted_regression",
        "model_type": "statsmodels.Logit",
        "parameters": {"adjust_for_covariates": True, "maxiter": 100},
        "evaluation_template": {
            "adjusted_odds_ratio": True,
            "unadjusted_odds_ratio": True,
            "delta_coefficient": True,
            "confidence_interval": True,
            "p_value": True
        }
    }
}

def plan_experiments(
    validation_report: ValidationReportRecord,
    target: str = "Outcome",
    df: Optional[pd.DataFrame] = None,
    iteration_no: int = 1,
    output_dir: Optional[str] = None
) -> List[ExperimentSpecRecord]:
    """
    Converts all validated and reviewed hypotheses into formal, executable ExperimentSpecRecord objects.
    Enforces registry compliance and routes H5 initially to adjusted regression.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 5: BATCH 2 — EXPERIMENT REGISTRY & PLANNING (ITER {iteration_no})")
    print("=" * 70)

    specs: List[ExperimentSpecRecord] = []
    cluster_docs: List[Dict[str, Any]] = []

    for idx, val_res in enumerate(validation_report.results, 1):
        h_id = val_res.hypothesis_id
        features = list(val_res.features_involved)
        exp_id = f"iter{iteration_no}_E{idx}" if "iter" in h_id else f"E{idx}"

        # Determine hypothesis type / category
        h_type = getattr(val_res, "hypothesis_type", None)
        if h_type in EXPERIMENT_REGISTRY:
            category = h_type
            if category == "comparative_or_confounding_relationship":
                routing_note = "Routed H5 initially to adjusted regression baseline to determine direct vs indirect association before attempting formal causal mediation."
            elif category == "feature_interaction_moderation":
                routing_note = "Interaction logistic regression with product term and bootstrap."
            elif category == "feature_combination_synergy":
                routing_note = "Multi-feature paired comparative predictive cross-validation."
            else:
                routing_note = "Standard individual dependency statistical testing."
        elif h_id.endswith("_H1") or h_id.endswith("_H2") or h_id in ("H1", "H2") or (len(features) == 1 and idx <= 2):
            category = "individual_dependency"
            routing_note = "Standard individual dependency statistical testing."
        elif h_id.endswith("_H3") or h_id == "H3" or (len(features) == 2 and idx == 3):
            category = "feature_combination_synergy"
            routing_note = "Multi-feature paired comparative predictive cross-validation."
        elif h_id.endswith("_H4") or h_id == "H4" or idx == 4:
            category = "feature_interaction_moderation"
            routing_note = "Interaction logistic regression with product term and bootstrap."
        elif h_id.endswith("_H5") or h_id == "H5" or idx >= 5:
            # Explicit user rule: H5 routed to adjusted regression rather than raw mediation
            category = "comparative_or_confounding_relationship"
            routing_note = "Routed H5 initially to adjusted regression baseline to determine direct vs indirect association before attempting formal causal mediation."
        else:
            category = "individual_dependency"
            routing_note = "Defaulted to individual regression."

        reg_info = EXPERIMENT_REGISTRY.get(category, EXPERIMENT_REGISTRY["individual_dependency"])

        comparison = None
        if category == "feature_combination_synergy" and len(features) >= 2:
            f1, f2 = features[0], features[1]
            comparison = [[f1], [f2], [f1, f2]]

        spec = ExperimentSpecRecord(
            experiment_id=exp_id,
            hypothesis_id=h_id,
            iteration_no=iteration_no,
            experiment_type=category,
            method=reg_info["method"],
            model_type=reg_info["model_type"],
            features=features,
            target=target,
            comparison=comparison,
            parameters=dict(reg_info["parameters"]),
            evaluation=dict(reg_info["evaluation_template"]),
            routing_notes=routing_note
        )

        specs.append(spec)
        cluster_docs.append(spec.model_dump())

        print(f"\n{exp_id} -> Linked to {h_id} [{category}]")
        print(f"  Method     : {spec.method} ({spec.model_type})")
        print(f"  Features   : {', '.join(spec.features)} | Target: {spec.target}")
        if spec.comparison:
            print(f"  Comparison : {spec.comparison}")
        if spec.routing_notes:
            print(f"  Notes      : {spec.routing_notes}")

    # Save to JSON outputs
    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    iter_dir = os.path.join(out_dir, "iterations", f"iter_{iteration_no}")
    os.makedirs(iter_dir, exist_ok=True)

    specs_dump = [s.model_dump() for s in specs]
    with open(os.path.join(out_dir, "experiment_specifications.json"), "w") as f:
        json.dump(specs_dump, f, indent=4)
    with open(os.path.join(iter_dir, "experiment_specifications.json"), "w") as f:
        json.dump(specs_dump, f, indent=4)

    # Persist to MongoDB cluster scoped by run_id
    run_id = os.path.basename(out_dir.rstrip("\\/"))
    cluster = MongoClusterManager()
    cluster.save_documents(
        db_name=DB_EXPERIMENTS,
        collection_name="specifications",
        docs=cluster_docs,
        id_key="experiment_id",
        run_id=run_id
    )

    print("\n" + "-" * 70)
    print(f"PLANNED EXPERIMENTS SUMMARY: Total Specifications = {len(specs)}")
    print(f"Saved specifications to: {EXPERIMENT_SPECS_PATH}")
    print(f"Saved to MongoDB: {DB_EXPERIMENTS}.specifications")
    print("-" * 70)

    return specs
