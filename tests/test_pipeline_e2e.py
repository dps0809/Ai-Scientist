import os
import json
import pytest
import pandas as pd
from modules.config import (
    OUTPUT_DIR,
    VALIDATED_HYPOTHESES_PATH,
    EXPERIMENT_SPECS_PATH,
    DB_VALIDATION,
    DB_EXPERIMENTS
)
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments
from modules.mongo_cluster import MongoClusterManager
from modules.schemas import ValidationStatus

def test_pipeline_e2e_integration(tmp_path):
    hypotheses_path = os.path.join(OUTPUT_DIR, "generated_hypotheses.json")
    if os.path.exists(hypotheses_path):
        with open(hypotheses_path, "r") as f:
            hypotheses = json.load(f)
    else:
        # Fallback to MongoDB if cluster has saved hypotheses, or standard fixture
        cluster = MongoClusterManager()
        db_hyps = cluster.query_documents("ai_scientist_hypotheses", "hypotheses") if cluster.is_available() else []
        if len(db_hyps) >= 5:
            hypotheses = db_hyps
        else:
            hypotheses = [
                {
                    "hypothesis_id": "iter1_H1",
                    "hypothesis_type": "individual_dependency",
                    "features_involved": ["Glucose"],
                    "statement": "Higher glucose leads to higher outcome.",
                    "rationale": "High Pearson correlation with Outcome.",
                    "evidence": {"correlation": 0.4665},
                    "proposed_experiment": "Logistic regression on Glucose."
                },
                {
                    "hypothesis_id": "iter1_H2",
                    "hypothesis_type": "individual_dependency",
                    "features_involved": ["BMI"],
                    "statement": "BMI positively associates with diabetes risk.",
                    "rationale": "Moderate dependency.",
                    "evidence": {"correlation": 0.2926},
                    "proposed_experiment": "Logistic regression on BMI."
                },
                {
                    "hypothesis_id": "iter1_H3",
                    "hypothesis_type": "feature_combination_synergy",
                    "features_involved": ["Glucose", "BMI"],
                    "statement": "Glucose and BMI jointly predict diabetes.",
                    "rationale": "Synergy gain observed.",
                    "evidence": {"synergy_gain": 0.0223},
                    "proposed_experiment": "Paired model comparison."
                },
                {
                    "hypothesis_id": "iter1_H4",
                    "hypothesis_type": "feature_interaction_moderation",
                    "features_involved": ["Glucose", "Age"],
                    "statement": "Age moderates the effect of Glucose on Outcome.",
                    "rationale": "Significant interaction.",
                    "evidence": {"interaction_delta": 0.0002, "p_value": 0.0026},
                    "proposed_experiment": "Interaction regression with bootstrap."
                },
                {
                    "hypothesis_id": "iter1_H5",
                    "hypothesis_type": "comparative_or_confounding_relationship",
                    "features_involved": ["Pregnancies", "Age"],
                    "statement": "Pregnancies and Age are correlated, suggesting Age may confound Outcome.",
                    "rationale": "Strong collinearity.",
                    "evidence": {"correlation": 0.5443},
                    "proposed_experiment": "Adjusted regression analysis."
                }
            ]
        
    assert len(hypotheses) >= 5, "Should have at least 5 hypotheses"
    
    df = pd.read_csv("diabetes.csv")
    out_dir = str(tmp_path)
    
    # Run Batch 1 Validation
    report = validate_hypotheses(
        hypotheses=hypotheses,
        df=df,
        target="Outcome",
        iteration_no=1,
        output_dir=out_dir
    )
    
    assert report.total == len(hypotheses)
    assert os.path.exists(os.path.join(out_dir, "validated_hypotheses.json"))
    
    # Verify comparative / confounding hypothesis is flagged for review
    confounding_h = next((r for r in report.results if r.reasons and ("confounding" in r.reasons[0].lower() or "adjusted regression" in r.reasons[0].lower())), None)
    assert confounding_h is not None, "Confounding/comparative relationship must be flagged for review"
    assert confounding_h.status == ValidationStatus.NEEDS_REVIEW
    
    # Run Batch 2 Experiment Planning
    specs = plan_experiments(
        validation_report=report,
        target="Outcome",
        df=df,
        iteration_no=1,
        output_dir=out_dir
    )
    
    assert len(specs) == len(hypotheses)
    assert os.path.exists(os.path.join(out_dir, "experiment_specifications.json"))
    
    # Verify at least one adjusted_regression spec exists (routed from confounding/mediation)
    adj_spec = next((s for s in specs if s.method == "adjusted_regression"), None)
    assert adj_spec is not None, "Should have an adjusted_regression experiment specification"
    assert adj_spec.model_type == "statsmodels.Logit"
        
    # Verify MongoDB cluster persistence if server is available
    cluster = MongoClusterManager()
    if cluster.is_available():
        val_docs = cluster.query_documents(DB_VALIDATION, "validations")
        spec_docs = cluster.query_documents(DB_EXPERIMENTS, "specifications")
        assert len(val_docs) >= len(hypotheses)
        assert len(spec_docs) >= len(hypotheses)

