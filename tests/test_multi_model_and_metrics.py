import pytest
import numpy as np
import pandas as pd
from modules.schemas import (
    ExperimentSpecRecord,
    ExperimentResultRecord,
    StandardExperimentResult,
    CrossExperimentComparisonRecord,
    AutonomousStateRecord
)
from modules.experiment_executor import (
    get_candidate_models,
    run_individual_dependency_experiment,
    run_feature_combination_synergy_experiment,
    prepare_dataset_for_experiments
)
from modules.result_evaluator import (
    evaluate_experiment_result,
    build_cross_experiment_comparison
)
from modules.experiment_controller import ExperimentController
from modules.experiment_memory import ExperimentMemory


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 120
    glucose = np.random.normal(120, 30, n)
    bmi = np.random.normal(30, 6, n)
    age = np.random.normal(35, 10, n)
    logits = -3.5 + 0.03 * glucose + 0.04 * bmi + 0.01 * age
    probs = 1 / (1 + np.exp(-logits))
    outcome = (np.random.rand(n) < probs).astype(int)
    glucose[0] = 0
    bmi[1] = 0
    return pd.DataFrame({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Outcome": outcome
    })


def test_candidate_models_available():
    models = get_candidate_models()
    assert "LogisticRegression" in models
    assert "RandomForest" in models
    assert "XGBoost" in models


def test_individual_dependency_multi_model_and_metrics(sample_df):
    prep_df = prepare_dataset_for_experiments(sample_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="LogisticRegression",
        features=["Glucose"],
        target="Outcome"
    )
    result = run_individual_dependency_experiment(prep_df, spec, n_permutations=20)
    
    # Primary & Secondary Metrics
    assert "mean_auc" in result.metrics
    assert "mean_accuracy" in result.metrics
    assert result.metrics["primary_metric"] == "ROC_AUC"
    assert result.metrics["secondary_metric"] == "Accuracy"
    
    # Model Tournament
    assert "best_model" in result.metrics
    assert result.metrics["best_model"] in ("LogisticRegression", "RandomForest", "XGBoost")
    assert "model_comparison" in result.metrics
    comp = result.metrics["model_comparison"]
    assert "LogisticRegression" in comp
    assert "RandomForest" in comp
    assert "XGBoost" in comp
    for m in ("LogisticRegression", "RandomForest", "XGBoost"):
        assert "mean_auc" in comp[m]
        assert "mean_accuracy" in comp[m]


def test_feature_combination_synergy_multi_model_and_metrics(sample_df):
    prep_df = prepare_dataset_for_experiments(sample_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    result = run_feature_combination_synergy_experiment(prep_df, spec)

    # Primary & Secondary Metrics
    assert "combined_mean_auc" in result.metrics
    assert "gain_over_best_individual" in result.metrics
    assert "combined_mean_accuracy" in result.metrics
    assert "accuracy_gain" in result.metrics
    assert result.metrics["primary_metric"] == "ROC_AUC"
    assert result.metrics["secondary_metric"] == "Accuracy"

    # Multi-model Comparison
    assert "best_model" in result.metrics
    assert result.metrics["best_model"] in ("LogisticRegression", "RandomForest", "XGBoost")
    assert "model_comparison" in result.metrics
    comp = result.metrics["model_comparison"]
    assert "LogisticRegression" in comp
    assert "RandomForest" in comp
    assert "XGBoost" in comp
    for m in ("LogisticRegression", "RandomForest", "XGBoost"):
        assert "combined_mean_auc" in comp[m]
        assert "combined_mean_accuracy" in comp[m]
        assert "auc_gain" in comp[m]
        assert "accuracy_gain" in comp[m]


def test_result_evaluator_tracks_multi_model_and_accuracy(sample_df):
    prep_df = prepare_dataset_for_experiments(sample_df)
    spec1 = ExperimentSpecRecord(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="LogisticRegression",
        features=["Glucose"],
        target="Outcome"
    )
    spec2 = ExperimentSpecRecord(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    r1 = run_individual_dependency_experiment(prep_df, spec1, n_permutations=20)
    r2 = run_feature_combination_synergy_experiment(prep_df, spec2)

    eval1 = evaluate_experiment_result(r1)
    eval2 = evaluate_experiment_result(r2)

    assert "mean_accuracy" in eval1.results
    assert "best_model" in eval1.results
    assert "combined_accuracy" in eval2.results
    assert "accuracy_gain" in eval2.results
    assert "best_model" in eval2.results

    comparison = build_cross_experiment_comparison([r1, r2], iteration_no=1)
    assert "Glucose" in comparison.individual_accuracies
    assert "Glucose + BMI" in comparison.combination_accuracies
    assert "Glucose + BMI" in comparison.accuracy_gains
    assert len(comparison.ranking) == 2
    assert "accuracy" in comparison.ranking[0]
    assert "best_model" in comparison.ranking[0]


def test_experiment_controller_multi_model_report(sample_df, tmp_path):
    run_dir = tmp_path / "20261107_123456"
    run_dir.mkdir()
    memory = ExperimentMemory(storage_path=str(run_dir / "experiment_memory.json"), run_id="20261107_123456")
    controller = ExperimentController(output_dir=str(run_dir), memory=memory, df=sample_df)

    assert controller.state.run_id == "20261107_123456"

    # Mock standard experiment result
    eval_res = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "target": "Outcome", "model": "LogisticRegression"},
        results={
            "combined_auc": 0.8123,
            "gain": 0.0210,
            "combined_accuracy": 0.7650,
            "accuracy_gain": 0.0150,
            "best_model": "LogisticRegression",
            "model_comparison": {
                "LogisticRegression": {"combined_mean_auc": 0.8123, "combined_mean_accuracy": 0.7650},
                "RandomForest": {"combined_mean_auc": 0.7950, "combined_mean_accuracy": 0.7420},
                "XGBoost": {"combined_mean_auc": 0.7810, "combined_mean_accuracy": 0.7310}
            }
        },
        statistical_validation={},
        evaluation={"verdict": "supported_by_current_evidence", "summary": "Strong synergy observed"}
    )
    memory.save_experiment_result(eval_res)
    controller.update_progress(eval_res)

    assert controller.state.best_score == 0.8123
    assert controller.state.best_accuracy == 0.7650
    assert controller.state.best_model == "LogisticRegression"

    md, report_data = controller.generate_final_report()
    assert report_data["primary_metric"] == "ROC_AUC"
    assert report_data["secondary_metric"] == "Accuracy"
    assert report_data["best_accuracy"] == 0.7650
    assert "Multi-Model Architecture Tournament" in md
    assert "LogisticRegression" in md
    assert "RandomForest" in md
    assert "XGBoost" in md

