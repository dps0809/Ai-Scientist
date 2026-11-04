import pytest
import pandas as pd
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments, EXPERIMENT_REGISTRY
from modules.schemas import ExperimentSpecRecord

def test_registry_contains_supported_types():
    assert "individual_dependency" in EXPERIMENT_REGISTRY
    assert "feature_combination_synergy" in EXPERIMENT_REGISTRY
    assert "feature_interaction_moderation" in EXPERIMENT_REGISTRY
    assert "comparative_or_confounding_relationship" in EXPERIMENT_REGISTRY

def test_plan_experiments_produces_specs_for_all_five_hypotheses(tmp_path):
    raw = [
        {"hypothesis_id": "H1", "hypothesis_type": "individual_dependency", "hypothesis": "Glucose association", "evidence_from_data": "r=0.46", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H2", "hypothesis_type": "individual_dependency", "hypothesis": "BMI association", "evidence_from_data": "r=0.29", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H3", "hypothesis_type": "feature_combination_synergy", "hypothesis": "Glucose + BMI synergy", "evidence_from_data": "AUC=0.8097", "proposed_experiment": "Model comparison"},
        {"hypothesis_id": "H4", "hypothesis_type": "feature_interaction_moderation", "hypothesis": "Glucose + Age interaction", "evidence_from_data": "delta=0.0002", "proposed_experiment": "Interaction test"},
        {"hypothesis_id": "H5", "hypothesis_type": "comparative_or_confounding_relationship", "hypothesis": "Glucose and BMI correlation r=0.22", "evidence_from_data": "r=0.2211", "proposed_experiment": "Mediation analysis"}
    ]
    df = pd.DataFrame({"Glucose": [100, 120], "BMI": [25, 30], "Age": [30, 40], "Outcome": [0, 1]})
    report = validate_hypotheses(raw, df, target="Outcome", iteration_no=1, output_dir=str(tmp_path))
    
    specs = plan_experiments(report, target="Outcome", iteration_no=1, output_dir=str(tmp_path))
    
    # Must generate specs for ALL 5 hypotheses without dropping any
    assert len(specs) == 5
    
    # Check H1 spec
    e1 = next(s for s in specs if s.hypothesis_id == "H1")
    assert e1.experiment_id == "E1"
    assert e1.method == "logistic_regression"
    assert e1.model_type == "statsmodels.Logit"
    assert e1.evaluation["odds_ratio"] is True
    
    # Check H3 spec
    e3 = next(s for s in specs if s.hypothesis_id == "H3")
    assert e3.experiment_id == "E3"
    assert e3.method == "paired_model_comparison"
    assert e3.comparison == [["Glucose"], ["BMI"], ["Glucose", "BMI"]]
    assert e3.evaluation["metric"] == "roc_auc"
    
    # Check H4 spec
    e4 = next(s for s in specs if s.hypothesis_id == "H4")
    assert e4.experiment_id == "E4"
    assert e4.method == "interaction_regression"
    assert e4.evaluation["interaction_term"] is True
    
    # Check H5 routing to adjusted regression
    e5 = next(s for s in specs if s.hypothesis_id == "H5")
    assert e5.experiment_id == "E5"
    assert e5.method == "adjusted_regression"
    assert e5.model_type == "statsmodels.Logit"
    assert "adjusted regression" in e5.routing_notes.lower()
