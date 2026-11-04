import pytest
import numpy as np
import pandas as pd
from modules.schemas import ExperimentSpecRecord

def test_prepare_dataset_replaces_invalid_zeros():
    from modules.experiment_executor import prepare_dataset_for_experiments
    raw_df = pd.DataFrame({
        "Glucose": [0, 120, 140],
        "BMI": [25.0, 0, 30.0],
        "BloodPressure": [70, 80, 0],
        "Insulin": [0, 100, 150],
        "SkinThickness": [20, 0, 30],
        "Pregnancies": [0, 2, 4], # 0 pregnancies is a valid physiological count!
        "Outcome": [0, 1, 1]
    })
    prep_df = prepare_dataset_for_experiments(raw_df)
    assert pd.isna(prep_df.loc[0, "Glucose"])
    assert pd.isna(prep_df.loc[1, "BMI"])
    assert pd.isna(prep_df.loc[2, "BloodPressure"])
    assert pd.isna(prep_df.loc[0, "Insulin"])
    assert pd.isna(prep_df.loc[1, "SkinThickness"])
    assert prep_df.loc[0, "Pregnancies"] == 0 # Preserved!
    assert prep_df.loc[0, "Outcome"] == 0 # Preserved!

@pytest.fixture
def sample_diabetes_df():
    np.random.seed(42)
    n = 100
    glucose = np.random.normal(120, 30, n)
    bmi = np.random.normal(30, 6, n)
    age = np.random.normal(35, 10, n)
    pregnancies = np.random.poisson(3, n)
    # Binary outcome dependent on glucose and bmi
    logits = -4.0 + 0.03 * glucose + 0.05 * bmi + 0.01 * age
    probs = 1 / (1 + np.exp(-logits))
    outcome = (np.random.rand(n) < probs).astype(int)
    # inject a few invalid zeros to test robustness
    glucose[0] = 0
    bmi[1] = 0
    return pd.DataFrame({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age,
        "Pregnancies": pregnancies,
        "Outcome": outcome
    })

def test_run_individual_dependency_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_individual_dependency_experiment, prepare_dataset_for_experiments
    
    prep_df = prepare_dataset_for_experiments(sample_diabetes_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        parameters={"fit_intercept": True, "maxiter": 100},
        evaluation={"odds_ratio": True, "confidence_interval": True, "permutation_iterations": 50}
    )
    result = run_individual_dependency_experiment(prep_df, spec, n_permutations=50)
    assert result.experiment_id == "iter1_E1"
    assert result.hypothesis_id == "iter1_H1"
    assert "coefficient" in result.metrics
    assert "odds_ratio" in result.metrics
    assert "confidence_interval" in result.metrics
    assert len(result.metrics["confidence_interval"]) == 2
    assert "p_value" in result.metrics
    assert "permutation_p_value" in result.metrics
    assert "fold_auc" in result.details
    assert len(result.details["fold_auc"]) == 5
    assert "mean_auc" in result.details
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
    assert result.summary != ""

def test_run_feature_combination_synergy_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_feature_combination_synergy_experiment, prepare_dataset_for_experiments
    
    prep_df = prepare_dataset_for_experiments(sample_diabetes_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome",
        comparison=[["Glucose"], ["BMI"], ["Glucose", "BMI"]],
        parameters={"solver": "lbfgs", "max_iter": 500},
        evaluation={"metric": "roc_auc", "folds": 5}
    )
    result = run_feature_combination_synergy_experiment(prep_df, spec)
    assert result.experiment_id == "iter1_E3"
    assert result.hypothesis_id == "iter1_H3"
    assert "combined_mean_auc" in result.metrics
    assert "gain_over_best_individual" in result.metrics
    assert "individual_scores" in result.details
    assert "Glucose" in result.details["individual_scores"]
    assert "BMI" in result.details["individual_scores"]
    assert "fold_scores" in result.details
    assert len(result.details["fold_scores"]["Glucose_BMI"]) == 5
    assert "paired_test_p_value" in result.metrics
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
    assert result.summary != ""

def test_run_feature_interaction_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_feature_interaction_experiment, prepare_dataset_for_experiments
    
    prep_df = prepare_dataset_for_experiments(sample_diabetes_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E4",
        hypothesis_id="iter1_H4",
        iteration_no=1,
        experiment_type="feature_interaction_moderation",
        method="interaction_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "Age"],
        target="Outcome",
        parameters={"include_interaction_term": True, "maxiter": 100},
        evaluation={"interaction_term": True, "likelihood_ratio_test": True, "bootstrap_samples": 30}
    )
    result = run_feature_interaction_experiment(prep_df, spec, n_bootstrap=30)
    assert result.experiment_id == "iter1_E4"
    assert result.hypothesis_id == "iter1_H4"
    assert "interaction_coefficient" in result.metrics
    assert "interaction_p_value" in result.metrics
    assert "lrt_statistic" in result.metrics
    assert "lrt_p_value" in result.metrics
    assert "bootstrap_ci" in result.metrics
    assert len(result.metrics["bootstrap_ci"]) == 2
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
    assert result.summary != ""

def test_run_adjusted_regression_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_adjusted_regression_experiment, prepare_dataset_for_experiments
    
    prep_df = prepare_dataset_for_experiments(sample_diabetes_df)
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E5",
        hypothesis_id="iter1_H5",
        iteration_no=1,
        experiment_type="comparative_or_confounding_relationship",
        method="adjusted_regression",
        model_type="statsmodels.Logit",
        features=["Pregnancies", "Age"],
        target="Outcome",
        parameters={"adjust_for_covariates": True, "maxiter": 100},
        evaluation={"adjusted_odds_ratio": True, "unadjusted_odds_ratio": True, "delta_coefficient": True}
    )
    result = run_adjusted_regression_experiment(prep_df, spec)
    assert result.experiment_id == "iter1_E5"
    assert result.hypothesis_id == "iter1_H5"
    assert "unadjusted_coefficient" in result.metrics
    assert "adjusted_coefficient" in result.metrics
    assert "delta_coefficient" in result.metrics
    assert "percent_change" in result.metrics
    assert "unadjusted_or" in result.metrics
    assert "adjusted_or" in result.metrics
    assert "unadjusted_ci" in result.metrics
    assert "adjusted_ci" in result.metrics
    assert "unadjusted_p_value" in result.metrics
    assert "adjusted_p_value" in result.metrics
    assert "covariate_coefficient" in result.metrics
    assert "covariate_p_value" in result.metrics
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
    assert result.summary != ""

def test_execute_all_experiments_and_persistence(sample_diabetes_df, tmp_path):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import execute_all_experiments
    
    specs = [
        ExperimentSpecRecord(
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome"
        ),
        ExperimentSpecRecord(
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            experiment_type="feature_combination_synergy",
            method="paired_model_comparison",
            model_type="sklearn.linear_model.LogisticRegression",
            features=["Glucose", "BMI"],
            target="Outcome",
            comparison=[["Glucose"], ["BMI"], ["Glucose", "BMI"]]
        )
    ]
    results = execute_all_experiments(
        specs=specs,
        df=sample_diabetes_df,
        iteration_no=1,
        output_dir=str(tmp_path),
        n_permutations=20,
        n_bootstrap=20
    )
    assert len(results) == 2
    assert results[0].experiment_id == "iter1_E1"
    assert results[1].experiment_id == "iter1_E3"
    
    # Check disk persistence in output_dir and iterations/iter_1/
    assert (tmp_path / "experiment_results.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "experiment_results.json").exists()


def test_experiment_executor_statsmodels_fallback(sample_diabetes_df, monkeypatch):
    import modules.experiment_executor as executor
    monkeypatch.setattr(executor, "STATSMODELS_AVAILABLE", False)
    monkeypatch.setattr(executor, "sm", None)

    spec_indiv = ExperimentSpecRecord(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome"
    )
    res_indiv = executor.run_individual_dependency_experiment(sample_diabetes_df, spec_indiv, n_permutations=10)
    assert res_indiv.experiment_id == "iter1_E1"
    assert "odds_ratio" in res_indiv.metrics
    assert res_indiv.verdict in ("SUPPORTED", "REFUTED")

    spec_inter = ExperimentSpecRecord(
        experiment_id="iter1_E5",
        hypothesis_id="iter1_H5",
        iteration_no=1,
        experiment_type="feature_interaction_moderation",
        method="interaction_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "Age"],
        target="Outcome"
    )
    res_inter = executor.run_feature_interaction_experiment(sample_diabetes_df, spec_inter, n_bootstrap=10)
    assert res_inter.experiment_id == "iter1_E5"
    assert "interaction_coefficient" in res_inter.metrics

    spec_adj = ExperimentSpecRecord(
        experiment_id="iter1_E6",
        hypothesis_id="iter1_H6",
        iteration_no=1,
        experiment_type="comparative_or_confounding_relationship",
        method="adjusted_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    res_adj = executor.run_adjusted_regression_experiment(sample_diabetes_df, spec_adj)
    assert res_adj.experiment_id == "iter1_E6"
    assert "delta_coefficient" in res_adj.metrics






