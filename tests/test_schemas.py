import pytest
from datetime import datetime
from modules.schemas import (
    InputRecord,
    PatternRecord,
    HypothesisRecord,
    ValidationResultRecord,
    ValidationReportRecord,
    ExperimentSpecRecord,
    ValidationStatus
)

def test_input_record_schema():
    inp = InputRecord(
        input_id="INP-20261104-001",
        source="diabetes.csv",
        row_count=768,
        column_count=9,
        features=["Glucose", "BMI", "Age"],
        target="Outcome"
    )
    dump = inp.model_dump()
    assert dump["input_id"] == "INP-20261104-001"
    assert dump["row_count"] == 768
    assert "timestamp" in dump

def test_pattern_record_schema():
    pat = PatternRecord(
        pattern_id="PAT-001",
        input_id="INP-20261104-001",
        pattern_type="individual_dependency",
        features=["Glucose"],
        metrics={"pearson_r": 0.4666, "p_value": 0.0001},
        description="Strong positive dependency"
    )
    dump = pat.model_dump()
    assert dump["pattern_id"] == "PAT-001"
    assert dump["metrics"]["pearson_r"] == 0.4666

def test_hypothesis_record_schema():
    hypo = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose concentration has a strong positive association with diabetes outcome.",
        features_involved=["Glucose"],
        evidence_from_data="pearson_r=0.4666",
        proposed_experiment="Fit logistic regression using Glucose."
    )
    dump = hypo.model_dump()
    assert dump["hypothesis_id"] == "H1"
    assert dump["iteration_no"] == 1
    assert dump["fingerprint"] == "('Glucose',)-individual_dependency"

def test_validation_records_schema():
    val = ValidationResultRecord(
        validation_id="VAL-H1-1",
        hypothesis_id="H1",
        iteration_no=1,
        status=ValidationStatus.VALIDATED,
        features_involved=["Glucose"],
        reconciled_evidence={"pearson_r": 0.4666},
        reasons=[]
    )
    report = ValidationReportRecord(
        report_id="REP-1",
        iteration_no=1,
        total=1,
        validated_count=1,
        needs_review_count=0,
        rejected_count=0,
        results=[val]
    )
    dump = report.model_dump()
    assert dump["validated_count"] == 1
    assert len(dump["results"]) == 1

def test_experiment_spec_schema():
    spec = ExperimentSpecRecord(
        experiment_id="E1",
        hypothesis_id="H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        evaluation={"odds_ratio": True, "p_value": True}
    )
    dump = spec.model_dump()
    assert dump["experiment_id"] == "E1"
    assert dump["model_type"] == "statsmodels.Logit"
