import os
import json
import pytest
import pandas as pd
from modules.schemas import StoppingConditionConfig
from modules.experiment_controller import run_autonomous_scientist


def test_autonomous_scientist_loop_executes_and_generates_final_report(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    df = pd.read_csv("diabetes.csv")

    class MockOllamaResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": json.dumps({
                    "hypothesis_status": "supported_by_current_evidence",
                    "interpretation": "Combination shows significant predictive synergy.",
                    "observations": ["ROC-AUC improved", "p-value < 0.05"],
                    "limitations": ["Requires external validation"],
                    "next_hypothesis": "Glucose, BMI, and Age together provide highest discrimination.",
                    "proposed_experiment": {
                        "features": ["Glucose", "BMI", "Age"],
                        "model": "sklearn.linear_model.LogisticRegression",
                        "metric": "ROC_AUC"
                    }
                })
            }

    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockOllamaResponse())

    config = StoppingConditionConfig(
        max_experiments=3,
        max_iterations=2,
        no_improvement_tolerance=2
    )

    initial_hypotheses = [
        {
            "hypothesis_id": "iter1_H1",
            "iteration_no": 1,
            "hypothesis_type": "individual_dependency",
            "hypothesis": "Glucose positively associates with Outcome.",
            "features_involved": ["Glucose"],
            "evidence_from_data": "Pearson r = 0.4665",
            "proposed_experiment": "Logistic regression on Glucose."
        },
        {
            "hypothesis_id": "iter1_H2",
            "iteration_no": 1,
            "hypothesis_type": "feature_combination_synergy",
            "hypothesis": "Glucose and BMI jointly predict diabetes.",
            "features_involved": ["Glucose", "BMI"],
            "evidence_from_data": "Synergy gain = 0.0223",
            "proposed_experiment": "Paired CV comparison."
        }
    ]

    report_md, report_data = run_autonomous_scientist(
        initial_hypotheses=initial_hypotheses,
        df=df,
        target="Outcome",
        config=config,
        output_dir=out_dir,
        n_permutations=20,
        n_bootstrap=20,
        sync_mongo=False
    )

    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.md"))
    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.json"))
    assert os.path.exists(os.path.join(out_dir, "autonomous_state.json"))
    assert report_data["total_experiments"] >= 2
    assert "Executive Summary" in report_md or "Confirmed Scientific Discoveries" in report_md
    assert len(report_data["confirmed_hypotheses"]) >= 1
