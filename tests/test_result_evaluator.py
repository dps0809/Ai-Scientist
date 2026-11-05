import pytest
import numpy as np
import pandas as pd
from modules.schemas import ExperimentResultRecord


def test_evaluate_individual_dependency_result():
    from modules.result_evaluator import evaluate_experiment_result

    raw_res = ExperimentResultRecord(
        result_id="RES-iter1_E1",
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        metrics={
            "coefficient": 1.242,
            "odds_ratio": 3.46,
            "confidence_interval": [2.82, 4.23],
            "p_value": 0.00001,
            "permutation_p_value": 0.000,
            "mean_auc": 0.790,
            "std_auc": 0.021
        },
        details={"fold_auc": [0.78, 0.81, 0.77, 0.80, 0.79]},
        verdict="SUPPORTED",
        summary="Glucose strongly supported"
    )
    standard_res = evaluate_experiment_result(raw_res)
    assert standard_res.experiment_id == "iter1_E1"
    assert standard_res.hypothesis_id == "iter1_H1"
    assert standard_res.status == "completed"
    assert standard_res.results["odds_ratio"] == 3.46
    assert standard_res.evaluation["verdict"] == "supported_by_current_evidence"
    assert standard_res.evaluation["is_statistically_significant"] is True
    assert standard_res.evaluation["practical_significance"] == "meaningful"
    assert standard_res.evaluation["uncertainty_level"] == "low"


def test_evaluate_feature_synergy_result():
    from modules.result_evaluator import evaluate_experiment_result

    raw_res = ExperimentResultRecord(
        result_id="RES-iter1_E3",
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome",
        metrics={
            "combined_mean_auc": 0.817,
            "best_individual_mean_auc": 0.790,
            "gain_over_best_individual": 0.0274,
            "paired_test_stat": 4.12,
            "paired_test_p_value": 0.0026
        },
        details={
            "individual_scores": {"Glucose": 0.790, "BMI": 0.686},
            "best_individual_feature": "Glucose",
            "fold_scores": {
                "Glucose": [0.78, 0.81, 0.77, 0.80, 0.79],
                "BMI": [0.67, 0.70, 0.68, 0.69, 0.69],
                "Glucose_BMI": [0.81, 0.84, 0.80, 0.82, 0.815]
            }
        },
        verdict="SUPPORTED",
        summary="Synergy strongly supported"
    )
    standard_res = evaluate_experiment_result(raw_res)
    assert standard_res.experiment_id == "iter1_E3"
    assert standard_res.results["gain"] == 0.0274
    assert standard_res.evaluation["verdict"] == "supported_by_current_evidence"
    assert standard_res.evaluation["is_statistically_significant"] is True
    assert standard_res.evaluation["practical_significance"] == "meaningful"
    assert "Glucose" in standard_res.results["individual_auc"]


def test_evaluate_feature_interaction_result():
    from modules.result_evaluator import evaluate_experiment_result

    raw_res = ExperimentResultRecord(
        result_id="RES-iter1_E5",
        experiment_id="iter1_E5",
        hypothesis_id="iter1_H5",
        iteration_no=1,
        experiment_type="feature_interaction_moderation",
        method="interaction_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "Age"],
        target="Outcome",
        metrics={
            "interaction_coefficient": -0.00075,
            "interaction_p_value": 0.0044,
            "lrt_statistic": 7.632,
            "lrt_p_value": 0.0057,
            "bootstrap_ci": [-0.00129, -0.00019],
            "main_effect_a_coef": 0.035,
            "main_effect_b_coef": 0.042
        },
        details={"log_likelihood_additive": -320.1, "log_likelihood_interaction": -316.28},
        verdict="SUPPORTED",
        summary="Significant moderation detected"
    )
    standard_res = evaluate_experiment_result(raw_res)
    assert standard_res.experiment_id == "iter1_E5"
    assert standard_res.evaluation["verdict"] == "supported_by_current_evidence"
    assert standard_res.evaluation["is_statistically_significant"] is True
    assert standard_res.statistical_validation["comparison_method"] == "likelihood_ratio_test"


def test_evaluate_adjusted_regression_result():
    from modules.result_evaluator import evaluate_experiment_result

    raw_res = ExperimentResultRecord(
        result_id="RES-iter1_E6",
        experiment_id="iter1_E6",
        hypothesis_id="iter1_H6",
        iteration_no=1,
        experiment_type="comparative_or_confounding_relationship",
        method="adjusted_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "BMI"],
        target="Outcome",
        metrics={
            "unadjusted_coefficient": 1.242,
            "adjusted_coefficient": 1.168,
            "delta_coefficient": -0.074,
            "percent_change": -5.93,
            "unadjusted_or": 3.46,
            "adjusted_or": 3.22,
            "covariate_coefficient": 0.45,
            "covariate_p": 0.0001,
            "covariate_or": 1.57
        },
        details={"sample_size": 750},
        verdict="REFUTED",
        summary="Confounding refuted"
    )
    standard_res = evaluate_experiment_result(raw_res)
    assert standard_res.experiment_id == "iter1_E6"
    assert standard_res.results["percent_change"] == -5.93
    assert "causal_limitations" in standard_res.evaluation


def test_build_cross_experiment_comparison():
    from modules.result_evaluator import build_cross_experiment_comparison

    raw_results = [
        ExperimentResultRecord(
            result_id="RES-iter1_E1",
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome",
            metrics={"mean_auc": 0.790, "odds_ratio": 3.46, "p_value": 0.0001},
            verdict="SUPPORTED",
            summary="Glucose"
        ),
        ExperimentResultRecord(
            result_id="RES-iter1_E2",
            experiment_id="iter1_E2",
            hypothesis_id="iter1_H2",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["BMI"],
            target="Outcome",
            metrics={"mean_auc": 0.686, "odds_ratio": 2.03, "p_value": 0.0001},
            verdict="SUPPORTED",
            summary="BMI"
        ),
        ExperimentResultRecord(
            result_id="RES-iter1_E3",
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            experiment_type="feature_combination_synergy",
            method="paired_model_comparison",
            model_type="sklearn.linear_model.LogisticRegression",
            features=["Glucose", "BMI"],
            target="Outcome",
            metrics={"combined_mean_auc": 0.817, "gain_over_best_individual": 0.0274, "paired_test_p_value": 0.0026},
            verdict="SUPPORTED",
            summary="Glucose + BMI"
        )
    ]
    comparison = build_cross_experiment_comparison(raw_results, iteration_no=1)
    assert comparison.iteration_no == 1
    assert comparison.individual_baselines["Glucose"] == 0.790
    assert comparison.combination_scores["Glucose + BMI"] == 0.817
    assert comparison.synergy_gains["Glucose + BMI"] == 0.0274
    assert len(comparison.ranking) == 3
    assert comparison.ranking[0]["name"] == "Glucose + BMI"
    assert len(comparison.summary_findings) >= 1


def test_evaluate_all_results_and_save(tmp_path):
    from modules.result_evaluator import evaluate_all_results

    sample_results = [
        ExperimentResultRecord(
            result_id="RES-iter1_E1",
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome",
            metrics={"odds_ratio": 3.46, "p_value": 0.0001, "permutation_p_value": 0.000, "mean_auc": 0.790, "confidence_interval": [2.8, 4.2]},
            verdict="SUPPORTED",
            summary="Glucose"
        )
    ]
    eval_list, comparison = evaluate_all_results(
        results=sample_results,
        iteration_no=1,
        output_dir=str(tmp_path)
    )
    assert len(eval_list) == 1
    assert eval_list[0].experiment_id == "iter1_E1"
    assert comparison.individual_baselines["Glucose"] == 0.790
    assert (tmp_path / "experiment_evaluations.json").exists()
    assert (tmp_path / "cross_experiment_comparison.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "experiment_evaluations.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "cross_experiment_comparison.json").exists()


