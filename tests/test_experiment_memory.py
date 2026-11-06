import os
import pytest
from modules.schemas import (
    ExperimentSpecRecord,
    ExperimentResultRecord,
    HypothesisRecord,
    StandardExperimentResult
)
from modules.experiment_memory import ExperimentMemory


def test_experiment_memory_duplicate_detection(tmp_path):
    mem_file = os.path.join(tmp_path, "test_memory.json")
    memory = ExperimentMemory(storage_path=mem_file, sync_mongo=False)

    spec1 = ExperimentSpecRecord(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    assert not memory.is_duplicate_experiment(spec1)
    memory.register_experiment_spec(spec1)
    assert memory.is_duplicate_experiment(spec1)

    # Different feature order must still be detected as duplicate!
    spec_reversed = ExperimentSpecRecord(
        experiment_id="iter2_E1",
        hypothesis_id="iter2_H1",
        iteration_no=2,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["BMI", "Glucose"],
        target="Outcome"
    )
    assert memory.is_duplicate_experiment(spec_reversed)

    # Different model is distinct and allowed
    spec_diff_model = ExperimentSpecRecord(
        experiment_id="iter2_E2",
        hypothesis_id="iter2_H2",
        iteration_no=2,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="statsmodels.Logit",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    assert not memory.is_duplicate_experiment(spec_diff_model)


def test_experiment_memory_history_and_querying(tmp_path):
    mem_file = os.path.join(tmp_path, "test_memory.json")
    memory = ExperimentMemory(storage_path=mem_file, sync_mongo=False)

    h1 = HypothesisRecord(
        hypothesis_id="iter1_H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose associates with Outcome.",
        features_involved=["Glucose"]
    )
    memory.save_hypothesis(h1)

    res1 = StandardExperimentResult(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        status="completed",
        experiment_type="individual_dependency",
        configuration={"features": ["Glucose"], "model": "statsmodels.Logit"},
        results={"odds_ratio": 3.46, "roc_auc": 0.79},
        statistical_validation={"p_value": 0.0001},
        evaluation={"verdict": "supported_by_current_evidence"}
    )
    memory.save_experiment_result(res1)

    res2 = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"combined_auc": 0.8178, "gain": 0.0274},
        statistical_validation={"p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence"}
    )
    memory.save_experiment_result(res2)

    history = memory.get_all_experiment_results()
    assert len(history) == 2

    best = memory.get_best_result()
    assert best is not None
    assert best["experiment_id"] == "iter1_E3"
    assert best["score"] == 0.8178

    stats = memory.get_summary_stats()
    assert stats["total_experiments"] == 2
    assert stats["supported_count"] == 2
    assert stats["refuted_count"] == 0
