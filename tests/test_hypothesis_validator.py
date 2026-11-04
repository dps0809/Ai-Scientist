import pytest
import pandas as pd
from modules.hypothesis_validator import validate_hypotheses
from modules.schemas import ValidationStatus

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Glucose": [120, 140, 90, 180, 110],
        "BMI": [28.5, 32.1, 22.0, 35.4, 26.0],
        "Age": [33, 45, 25, 52, 29],
        "Outcome": [0, 1, 0, 1, 0]
    })

def test_validate_all_hypotheses_without_omission(sample_df, tmp_path):
    raw = [
        {"hypothesis_id": "H1", "hypothesis_type": "individual_dependency", "hypothesis": "Glucose association", "evidence_from_data": "Glucose: pearson_r=0.4666", "proposed_experiment": "Fit logistic regression"},
        {"hypothesis_id": "H2", "hypothesis_type": "individual_dependency", "hypothesis": "BMI association", "evidence_from_data": "BMI: pearson_r=0.2927", "proposed_experiment": "Fit logistic regression"},
        {"hypothesis_id": "H3", "hypothesis_type": "feature_combination_synergy", "hypothesis": "Glucose + BMI synergy", "evidence_from_data": "combined_score=0.8097", "proposed_experiment": "Compare models"},
        {"hypothesis_id": "H4", "hypothesis_type": "feature_interaction_moderation", "hypothesis": "Glucose and Age interaction delta = -0.0007, p-value = 0.0364", "evidence_from_data": "interaction_delta=0.0002, interaction_p_value=0.0026", "proposed_experiment": "Fit interaction model"},
        {"hypothesis_id": "H5", "hypothesis_type": "comparative_or_confounding_relationship", "hypothesis": "Glucose and BMI correlation r=0.22", "evidence_from_data": "correlation=0.2211", "proposed_experiment": "Mediation analysis"}
    ]
    stage2_mock = {
        "combinations": [{"combination": ["Glucose", "Age"], "interaction_delta": 0.0002, "interaction_p_value": 0.0026}]
    }
    
    report = validate_hypotheses(
        hypotheses=raw,
        df=sample_df,
        target="Outcome",
        stage2_data=stage2_mock,
        iteration_no=1,
        output_dir=str(tmp_path)
    )
    
    # Must process ALL 5 hypotheses without dropping any
    assert report.total == 5
    assert len(report.results) == 5
    
    # H1, H2, H3 should be VALIDATED
    h1 = next(r for r in report.results if r.hypothesis_id == "H1")
    assert h1.status == ValidationStatus.VALIDATED
    
    # H4 must be flagged as NEEDS_REVIEW with reconciled ground-truth evidence attached
    h4 = next(r for r in report.results if r.hypothesis_id == "H4")
    assert h4.status == ValidationStatus.NEEDS_REVIEW
    assert "Discrepancy detected" in h4.reasons[0]
    assert h4.reconciled_evidence["interaction_delta"] == 0.0002
    assert h4.reconciled_evidence["interaction_p_value"] == 0.0026

def test_validate_rejects_missing_column(sample_df, tmp_path):
    raw = [
        {"hypothesis_id": "H_Err", "hypothesis_type": "individual_dependency", "hypothesis": "UnknownFeature association", "evidence_from_data": "r=0.5", "proposed_experiment": "Fit regression"}
    ]
    report = validate_hypotheses(raw, sample_df, target="Outcome", iteration_no=1, output_dir=str(tmp_path))
    assert report.rejected_count == 1
    assert report.results[0].status == ValidationStatus.REJECTED
    assert any("not found" in r.lower() for r in report.results[0].reasons)
