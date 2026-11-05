import pytest
from modules.schemas import StandardExperimentResult, HypothesisRecord


def test_build_interpretation_prompt():
    from modules.llm_interpreter import build_interpretation_prompt

    std_res = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
        statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
    )
    hypo = HypothesisRecord(
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_type="feature_combination_synergy",
        hypothesis="Combining Glucose and BMI provides greater predictive power for diabetes than either feature alone.",
        features_involved=["Glucose", "BMI"],
        evidence_from_data="Joint MI = 0.18, Synergy Gain = +0.027",
        proposed_experiment="Run 5-fold Stratified CV comparing individual vs combined models."
    )
    prompt = build_interpretation_prompt(std_res, hypo)
    assert "Role" in prompt
    assert "Original Hypothesis" in prompt
    assert "Experiment Configuration" in prompt
    assert "Experimental Results" in prompt
    assert "Instructions" in prompt
    assert "hypothesis_status" in prompt
    assert "next_hypothesis" in prompt
    assert "Glucose" in prompt
    assert "BMI" in prompt


def test_interpret_experiment_with_mocked_ollama(monkeypatch):
    import requests
    from modules.llm_interpreter import interpret_single_experiment

    std_res = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
        statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
    )
    hypo = HypothesisRecord(
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_type="feature_combination_synergy",
        hypothesis="Combining Glucose and BMI provides greater predictive power for diabetes.",
        features_involved=["Glucose", "BMI"],
        evidence_from_data="Synergy Gain = +0.027",
        proposed_experiment="5-fold CV"
    )

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": '{"hypothesis_status": "supported_by_current_evidence", "interpretation": "Combined model outperforms baselines.", "observations": ["Glucose is strongest baseline", "BMI adds synergy"], "limitations": ["Requires validation on external cohorts"], "next_hypothesis": "Adding Age to Glucose and BMI will further improve diabetes prediction.", "proposed_experiment": {"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}}'
            }

    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockResponse())

    interp = interpret_single_experiment(std_res, hypo)
    assert interp.hypothesis_status == "supported_by_current_evidence"
    assert interp.experiment_id == "iter1_E3"
    assert "Age" in interp.proposed_experiment["features"]
    assert len(interp.observations) == 2


def test_interpret_experiment_raises_on_unreachable_ollama(monkeypatch):
    import requests
    from modules.llm_interpreter import interpret_single_experiment

    std_res = StandardExperimentResult(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        status="completed",
        experiment_type="individual_dependency",
        configuration={"features": ["Glucose"], "model": "statsmodels.Logit"},
        results={"odds_ratio": 3.46},
        statistical_validation={"p_value": 0.0001},
        evaluation={"verdict": "supported_by_current_evidence"}
    )

    def mock_fail(url, json, **kwargs):
        raise requests.exceptions.ConnectionError("Could not connect to Ollama")

    monkeypatch.setattr(requests, "post", mock_fail)

    with pytest.raises(ConnectionError, match="Could not connect to Ollama"):
        interpret_single_experiment(std_res, None)


def test_format_and_register_refined_hypotheses(tmp_path):
    from modules.schemas import LLMInterpretationRecord
    from modules.hypothesis_registry import HypothesisRegistry
    from modules.llm_interpreter import register_refined_hypotheses

    interps = [
        LLMInterpretationRecord(
            interpretation_id="INT-iter1_E3",
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            hypothesis_status="supported_by_current_evidence",
            interpretation="Combined model improved performance.",
            observations=["Glucose and BMI synergistic"],
            limitations=["Need Age evaluation"],
            next_hypothesis="Adding Age to Glucose and BMI will further improve diabetes prediction beyond the 0.817 ROC-AUC achieved by Glucose and BMI.",
            proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}
        )
    ]
    reg_file = str(tmp_path / "hypo_reg.json")
    registry = HypothesisRegistry(registry_path=reg_file)
    refined_records = register_refined_hypotheses(interps, next_iteration_no=2, registry=registry)
    assert len(refined_records) == 1
    assert refined_records[0].hypothesis_id == "iter2_H1"
    assert refined_records[0].parent_hypothesis_id == "iter1_H3"
    assert "Age" in refined_records[0].features_involved
    # Verify registered in registry as duplicate on re-registration
    from modules.schemas import HypothesisRecord
    dup_check = HypothesisRecord(
        hypothesis_id="temp_check",
        iteration_no=2,
        hypothesis_type="refined_follow_up",
        hypothesis=refined_records[0].hypothesis,
        features_involved=refined_records[0].features_involved
    )
    assert registry.is_duplicate(dup_check)
    assert not registry.register_hypothesis(dup_check)


def test_interpret_and_refine_all_experiments(monkeypatch, tmp_path):
    import requests
    from modules.schemas import StandardExperimentResult, HypothesisRecord
    from modules.llm_interpreter import interpret_and_refine_all

    eval_results = [
        StandardExperimentResult(
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            status="completed",
            experiment_type="feature_combination_synergy",
            configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
            results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
            statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
            evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
        )
    ]
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="iter1_H3",
            iteration_no=1,
            hypothesis_type="feature_combination_synergy",
            hypothesis="Glucose and BMI synergy.",
            features_involved=["Glucose", "BMI"],
            evidence_from_data="gain=0.027",
            proposed_experiment="5-fold CV"
        )
    ]

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": '{"hypothesis_status": "supported_by_current_evidence", "interpretation": "Combined model outperforms baselines.", "observations": ["Glucose is strongest baseline"], "limitations": ["Requires external cohort"], "next_hypothesis": "Add Age to Glucose + BMI.", "proposed_experiment": {"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}}'
            }

    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockResponse())

    interps, refined = interpret_and_refine_all(
        eval_results=eval_results,
        hypotheses=hypotheses,
        iteration_no=1,
        output_dir=str(tmp_path)
    )
    assert len(interps) == 1
    assert len(refined) == 1
    assert interps[0].experiment_id == "iter1_E3"
    assert refined[0].hypothesis_id == "iter2_H1"
    assert (tmp_path / "llm_interpretations.json").exists()
    assert (tmp_path / "refined_next_hypotheses.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "llm_interpretations.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "refined_next_hypotheses.json").exists()




