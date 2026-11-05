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

def test_experiment_result_schema():
    from modules.schemas import ExperimentResultRecord
    record = ExperimentResultRecord(
        result_id="RES-iter1_E1",
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        metrics={"odds_ratio": 1.78, "p_value": 0.0001, "permutation_p_value": 0.001},
        details={"fold_auc": [0.75, 0.78, 0.74, 0.81, 0.76], "mean_auc": 0.768},
        verdict="SUPPORTED",
        summary="Glucose is significantly associated with diabetes odds."
    )
    dump = record.model_dump()
    assert dump["result_id"] == "RES-iter1_E1"
    assert dump["metrics"]["odds_ratio"] == 1.78
    assert dump["verdict"] == "SUPPORTED"
    assert "timestamp" in dump


def test_evaluation_and_refinement_schemas():
    from modules.schemas import (
        StandardExperimentResult,
        CrossExperimentComparisonRecord,
        LLMInterpretationRecord,
        RefinedHypothesisRecord
    )
    eval_rec = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={
            "features": ["Glucose", "BMI"],
            "model": "LogisticRegression"
        },
        results={
            "individual_auc": {"Glucose": 0.79, "BMI": 0.686},
            "combined_auc": 0.817,
            "gain": 0.0274
        },
        statistical_validation={
            "comparison_method": "paired_cross_validation",
            "confidence_interval": [0.012, 0.043],
            "p_value": 0.0026
        },
        evaluation={
            "verdict": "supported_by_current_evidence",
            "is_statistically_significant": True,
            "practical_significance": "meaningful",
            "uncertainty_level": "low",
            "summary": "Combining Glucose and BMI significantly improved ROC-AUC by 0.0274."
        }
    )
    assert eval_rec.experiment_id == "iter1_E3"
    assert eval_rec.results["gain"] == 0.0274
    assert eval_rec.evaluation["is_statistically_significant"] is True

    comp_rec = CrossExperimentComparisonRecord(
        comparison_id="COMP-iter1",
        iteration_no=1,
        individual_baselines={"Glucose": 0.79, "BMI": 0.686},
        combination_scores={"Glucose + BMI": 0.817},
        synergy_gains={"Glucose + BMI": 0.0274},
        ranking=[{"rank": 1, "name": "Glucose + BMI", "score": 0.817}],
        summary_findings=["Glucose + BMI achieves top score"]
    )
    assert comp_rec.comparison_id == "COMP-iter1"
    assert comp_rec.ranking[0]["score"] == 0.817

    interp_rec = LLMInterpretationRecord(
        interpretation_id="INT-iter1_E3",
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_status="supported_by_current_evidence",
        interpretation="The combined feature model outperformed individual feature models.",
        observations=["Glucose was the strongest predictor.", "BMI added complementary information."],
        limitations=["Causal mediation between Glucose and BMI has not been established."],
        next_hypothesis="Adding Age to Glucose and BMI will further improve diabetes prediction.",
        proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}
    )
    assert interp_rec.hypothesis_status == "supported_by_current_evidence"
    assert "Glucose" in interp_rec.proposed_experiment["features"]

    refined_rec = RefinedHypothesisRecord(
        hypothesis_id="iter2_H1",
        parent_hypothesis_id="iter1_H3",
        iteration_no=2,
        hypothesis="Adding Age to Glucose and BMI will further improve diabetes prediction.",
        rationale="Synergy observed in iter1_E3 warrants testing third predictor.",
        proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"},
        features_involved=["Glucose", "BMI", "Age"]
    )
    assert refined_rec.hypothesis_id == "iter2_H1"
    assert refined_rec.parent_hypothesis_id == "iter1_H3"

