import os
import json
import pytest
import pandas as pd
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments
from modules.schemas import ValidationStatus

def test_iteration_prefixed_hypothesis_ids_and_run_output_directory(tmp_path):
    run_output_dir = tmp_path / "20261104_015000"
    run_output_dir.mkdir()
    
    # Simulate LLM raw outputs across Iteration 1 and Iteration 2
    raw_iter1 = [
        {"hypothesis_id": "iter1_H1", "hypothesis_type": "individual_dependency", "hypothesis": "Glucose association", "evidence_from_data": "r=0.46", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "iter1_H2", "hypothesis_type": "individual_dependency", "hypothesis": "BMI association", "evidence_from_data": "r=0.29", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "iter1_H3", "hypothesis_type": "feature_combination_synergy", "hypothesis": "Glucose and BMI synergy", "evidence_from_data": "AUC=0.80", "proposed_experiment": "Paired comparison"},
    ]
    raw_iter2 = [
        {"hypothesis_id": "iter2_H1", "hypothesis_type": "individual_dependency", "hypothesis": "Age association", "evidence_from_data": "r=0.23", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "iter2_H2", "hypothesis_type": "feature_interaction_moderation", "hypothesis": "Glucose and Age interaction", "evidence_from_data": "delta=0.002", "proposed_experiment": "Interaction test"},
    ]
    
    df = pd.DataFrame({"Glucose": [100, 120], "BMI": [25, 30], "Age": [30, 40], "Outcome": [0, 1]})
    
    # 1. Test Iteration 1 validation and output storage
    report1 = validate_hypotheses(
        hypotheses=raw_iter1,
        df=df,
        target="Outcome",
        iteration_no=1,
        output_dir=str(run_output_dir)
    )
    
    assert [r.hypothesis_id for r in report1.results] == ["iter1_H1", "iter1_H2", "iter1_H3"]
    assert [r.validation_id for r in report1.results] == ["VAL-iter1_H1", "VAL-iter1_H2", "VAL-iter1_H3"]
    
    # Check that report is stored inside the run_output_dir
    val_path = run_output_dir / "validated_hypotheses.json"
    assert val_path.exists()
    iter1_dir = run_output_dir / "iterations" / "iter_1"
    assert (iter1_dir / "validated_hypotheses.json").exists()
    
    # 2. Test Iteration 1 experiment planning
    specs1 = plan_experiments(
        validation_report=report1,
        target="Outcome",
        df=df,
        iteration_no=1,
        output_dir=str(run_output_dir)
    )
    assert [s.hypothesis_id for s in specs1] == ["iter1_H1", "iter1_H2", "iter1_H3"]
    assert [s.experiment_id for s in specs1] == ["iter1_E1", "iter1_E2", "iter1_E3"]
    assert (run_output_dir / "experiment_specifications.json").exists()
    
    # 3. Test Iteration 2 validation and experiment planning
    report2 = validate_hypotheses(
        hypotheses=raw_iter2,
        df=df,
        target="Outcome",
        iteration_no=2,
        output_dir=str(run_output_dir)
    )
    assert [r.hypothesis_id for r in report2.results] == ["iter2_H1", "iter2_H2"]
    assert [r.validation_id for r in report2.results] == ["VAL-iter2_H1", "VAL-iter2_H2"]
    
    specs2 = plan_experiments(
        validation_report=report2,
        target="Outcome",
        df=df,
        iteration_no=2,
        output_dir=str(run_output_dir)
    )
    assert [s.hypothesis_id for s in specs2] == ["iter2_H1", "iter2_H2"]
    assert [s.experiment_id for s in specs2] == ["iter2_E1", "iter2_E2"]
