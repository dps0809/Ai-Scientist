import os
import pytest
import pandas as pd
from modules.schemas import (
    StoppingConditionConfig,
    AutonomousStateRecord,
    ExperimentSpecRecord,
    StandardExperimentResult
)
from modules.experiment_controller import ExperimentController
from modules.experiment_memory import ExperimentMemory


def test_controller_stopping_conditions(tmp_path):
    out_dir = str(tmp_path)
    cfg = StoppingConditionConfig(
        max_experiments=3,
        max_iterations=2,
        no_improvement_tolerance=2,
        target_roc_auc=0.85
    )
    mem = ExperimentMemory(storage_path=os.path.join(out_dir, "mem.json"), sync_mongo=False)
    controller = ExperimentController(
        config=cfg,
        memory=mem,
        output_dir=out_dir
    )

    # Initial state: should not stop
    should_stop, reason = controller.should_stop()
    assert not should_stop

    # 1. Test max experiments condition
    controller.state.experiments_completed = 3
    should_stop, reason = controller.should_stop()
    assert should_stop
    assert "Maximum experiments limit" in reason

    # Reset
    controller.state.experiments_completed = 1

    # 2. Test target ROC-AUC achieved
    controller.state.best_score = 0.86
    should_stop, reason = controller.should_stop()
    assert should_stop
    assert "Target ROC-AUC achieved" in reason

    # Reset
    controller.state.best_score = 0.80

    # 3. Test consecutive no improvement tolerance
    controller.state.consecutive_no_improvement = 2
    should_stop, reason = controller.should_stop()
    assert should_stop
    assert "No improvement tolerance" in reason


def test_controller_validates_and_rejects_hallucinated_features(tmp_path):
    df = pd.DataFrame({"Glucose": [100, 120], "BMI": [25, 30], "Outcome": [0, 1]})
    controller = ExperimentController(
        config=StoppingConditionConfig(max_experiments=5),
        df=df,
        target="Outcome",
        output_dir=str(tmp_path)
    )

    valid_spec = ExperimentSpecRecord(
        experiment_id="test_E1",
        hypothesis_id="test_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome"
    )
    is_valid, msg = controller.validate_experiment_spec(valid_spec)
    assert is_valid

    invalid_spec = ExperimentSpecRecord(
        experiment_id="test_E2",
        hypothesis_id="test_H2",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["NonExistentMetabolite"],
        target="Outcome"
    )
    is_valid, msg = controller.validate_experiment_spec(invalid_spec)
    assert not is_valid
    assert "not found in dataset" in msg


def test_controller_updates_progress_and_detects_gain(tmp_path):
    out_dir = str(tmp_path)
    cfg = StoppingConditionConfig(
        max_experiments=5,
        min_meaningful_gain=0.01
    )
    mem = ExperimentMemory(storage_path=os.path.join(out_dir, "mem.json"), sync_mongo=False)
    controller = ExperimentController(
        config=cfg,
        memory=mem,
        output_dir=out_dir
    )

    # 1. First result: 0.79 -> sets baseline
    res1 = StandardExperimentResult(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        status="completed",
        experiment_type="individual_dependency",
        configuration={"features": ["Glucose"], "model": "statsmodels.Logit"},
        results={"roc_auc": 0.79},
        statistical_validation={"p_value": 0.0001},
        evaluation={"verdict": "supported_by_current_evidence"}
    )
    controller.update_progress(res1)
    assert controller.state.experiments_completed == 1
    assert controller.state.best_score == 0.79
    assert controller.state.consecutive_no_improvement == 0

    # 2. Second result: 0.8178 -> gain = +0.0278 >= 0.01 -> resets consecutive_no_improvement
    res2 = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"combined_auc": 0.8178, "gain": 0.0278},
        statistical_validation={"p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence"}
    )
    controller.update_progress(res2)
    assert controller.state.experiments_completed == 2
    assert controller.state.best_score == 0.8178
    assert controller.state.consecutive_no_improvement == 0

    # 3. Third result: 0.8180 -> gain = +0.0002 < 0.01 -> increments consecutive_no_improvement
    res3 = StandardExperimentResult(
        experiment_id="iter2_E1",
        hypothesis_id="iter2_H1",
        iteration_no=2,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "Pregnancies"], "model": "LogisticRegression"},
        results={"combined_auc": 0.8180, "gain": 0.0002},
        statistical_validation={"p_value": 0.12},
        evaluation={"verdict": "not_supported"}
    )
    controller.update_progress(res3)
    assert controller.state.experiments_completed == 3
    assert controller.state.consecutive_no_improvement == 1


def test_autonomous_loop_duplicate_recovery_temp_spec(tmp_path):
    from modules.schemas import RefinedHypothesisRecord
    from modules.experiment_controller import run_autonomous_scientist

    df = pd.DataFrame({
        "Glucose": [100, 120, 140, 110, 130, 90, 115, 125],
        "BMI": [25, 30, 35, 28, 32, 24, 27, 31],
        "Outcome": [0, 1, 1, 0, 1, 0, 0, 1]
    })
    
    # Create an initial hypothesis
    initial = [{
        "hypothesis_id": "iter1_H1",
        "hypothesis_type": "feature_combination_synergy",
        "hypothesis": "Glucose + BMI synergy",
        "features_involved": ["Glucose", "BMI"],
        "proposed_experiment": "Train model on Glucose and BMI"
    }]

    cfg = StoppingConditionConfig(
        max_experiments=2,
        max_iterations=3,
        no_improvement_tolerance=2
    )

    # Pre-register the spec so it is detected as duplicate immediately
    mem_path = str(tmp_path / "experiment_memory.json")
    mem = ExperimentMemory(storage_path=mem_path, sync_mongo=False)
    dup_spec = ExperimentSpecRecord(
        experiment_id="dup_E1",
        hypothesis_id="dup_H1",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    mem.register_experiment_spec(dup_spec)

    # run_autonomous_scientist should handle duplicate detection and fallback without raising ValidationError
    md, report = run_autonomous_scientist(
        initial_hypotheses=initial,
        df=df,
        target="Outcome",
        config=cfg,
        output_dir=str(tmp_path),
        n_permutations=5,
        n_bootstrap=5,
        sync_mongo=False
    )

    assert report is not None
    assert "stopping_reason" in report

