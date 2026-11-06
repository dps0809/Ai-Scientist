import os
import json
import pytest
import pandas as pd
from modules.config import (
    OUTPUT_DIR,
    VALIDATED_HYPOTHESES_PATH,
    EXPERIMENT_SPECS_PATH,
    DB_VALIDATION,
    DB_EXPERIMENTS
)
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments
from modules.mongo_cluster import MongoClusterManager
from modules.schemas import ValidationStatus

def test_pipeline_e2e_integration(tmp_path):
    hypotheses = [
        {
            "hypothesis_id": "iter1_H1",
            "hypothesis_type": "individual_dependency",
            "features_involved": ["Glucose"],
            "statement": "Higher glucose leads to higher outcome.",
            "rationale": "High Pearson correlation with Outcome.",
            "evidence": {"correlation": 0.4665},
            "proposed_experiment": "Logistic regression on Glucose."
        },
                {
                    "hypothesis_id": "iter1_H2",
                    "hypothesis_type": "individual_dependency",
                    "features_involved": ["BMI"],
                    "statement": "BMI positively associates with diabetes risk.",
                    "rationale": "Moderate dependency.",
                    "evidence": {"correlation": 0.2926},
                    "proposed_experiment": "Logistic regression on BMI."
                },
                {
                    "hypothesis_id": "iter1_H3",
                    "hypothesis_type": "feature_combination_synergy",
                    "features_involved": ["Glucose", "BMI"],
                    "statement": "Glucose and BMI jointly predict diabetes.",
                    "rationale": "Synergy gain observed.",
                    "evidence": {"synergy_gain": 0.0223},
                    "proposed_experiment": "Paired model comparison."
                },
                {
                    "hypothesis_id": "iter1_H4",
                    "hypothesis_type": "feature_interaction_moderation",
                    "features_involved": ["Glucose", "Age"],
                    "statement": "Age moderates the effect of Glucose on Outcome.",
                    "rationale": "Significant interaction.",
                    "evidence": {"interaction_delta": 0.0002, "p_value": 0.0026},
                    "proposed_experiment": "Interaction regression with bootstrap."
                },
                {
                    "hypothesis_id": "iter1_H5",
                    "hypothesis_type": "comparative_or_confounding_relationship",
                    "features_involved": ["Pregnancies", "Age"],
                    "statement": "Pregnancies and Age are correlated, suggesting Age may confound Outcome.",
                    "rationale": "Strong collinearity.",
                    "evidence": {"correlation": 0.5443},
                    "proposed_experiment": "Adjusted regression analysis."
                }
            ]
        
    assert len(hypotheses) >= 5, "Should have at least 5 hypotheses"
    
    df = pd.read_csv("diabetes.csv")
    out_dir = str(tmp_path)
    
    # Run Batch 1 Validation
    report = validate_hypotheses(
        hypotheses=hypotheses,
        df=df,
        target="Outcome",
        iteration_no=1,
        output_dir=out_dir
    )
    
    assert report.total == len(hypotheses)
    assert os.path.exists(os.path.join(out_dir, "validated_hypotheses.json"))
    
    # Verify comparative / confounding hypothesis is flagged for review
    confounding_h = next((r for r in report.results if r.reasons and ("confounding" in r.reasons[0].lower() or "adjusted regression" in r.reasons[0].lower())), None)
    assert confounding_h is not None, "Confounding/comparative relationship must be flagged for review"
    assert confounding_h.status == ValidationStatus.NEEDS_REVIEW
    
    # Run Batch 2 Experiment Planning
    specs = plan_experiments(
        validation_report=report,
        target="Outcome",
        df=df,
        iteration_no=1,
        output_dir=out_dir
    )
    
    assert len(specs) == len(hypotheses)
    assert os.path.exists(os.path.join(out_dir, "experiment_specifications.json"))
    
    # Verify at least one adjusted_regression spec exists (routed from confounding/mediation)
    adj_spec = next((s for s in specs if s.method == "adjusted_regression"), None)
    assert adj_spec is not None, "Should have an adjusted_regression experiment specification"
    assert adj_spec.model_type == "statsmodels.Logit"
        
    # Run Batch 3 Experiment Execution
    from modules.experiment_executor import execute_all_experiments
    from modules.config import DB_RESULTS
    
    results = execute_all_experiments(
        specs=specs,
        df=df,
        iteration_no=1,
        output_dir=out_dir,
        n_permutations=20,
        n_bootstrap=20
    )
    assert len(results) == len(specs)
    assert os.path.exists(os.path.join(out_dir, "experiment_results.json"))
    assert os.path.exists(os.path.join(out_dir, "iterations", "iter_1", "experiment_results.json"))
        
    # Verify MongoDB cluster persistence if server is available
    cluster = MongoClusterManager()
    if cluster.is_available():
        val_docs = cluster.query_documents(DB_VALIDATION, "validations")
        spec_docs = cluster.query_documents(DB_EXPERIMENTS, "specifications")
        res_docs = cluster.query_documents(DB_RESULTS, "results")
        assert len(val_docs) >= len(hypotheses)
        assert len(spec_docs) >= len(hypotheses)
        assert len(res_docs) >= len(hypotheses)

    # Run Batch 4 Result Evaluation Engine
    from modules.result_evaluator import evaluate_all_results
    from modules.config import DB_EVALUATIONS, DB_REFINEMENTS

    eval_results, comp_record = evaluate_all_results(
        results=results,
        iteration_no=1,
        output_dir=out_dir
    )
    assert len(eval_results) == len(results)
    assert comp_record is not None
    assert os.path.exists(os.path.join(out_dir, "experiment_evaluations.json"))
    assert os.path.exists(os.path.join(out_dir, "cross_experiment_comparison.json"))
    assert os.path.exists(os.path.join(out_dir, "iterations", "iter_1", "experiment_evaluations.json"))

    # Run Batch 5 LLM Interpretation & Hypothesis Refinement
    import requests
    from modules.hypothesis_registry import HypothesisRegistry
    from modules.llm_interpreter import interpret_and_refine_all
    from modules.schemas import HypothesisRecord

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": '{"hypothesis_status": "supported_by_current_evidence", "interpretation": "Strong confirmation under cross-validation.", "observations": ["Evidence holds firmly"], "limitations": ["Need more clinical samples"], "next_hypothesis": "Combining Glucose, BMI, and Insulin will further improve risk prediction.", "proposed_experiment": {"features": ["Glucose", "BMI", "Insulin"], "model": "LogisticRegression", "metric": "ROC_AUC"}}'
            }

    monkeypatch_post = lambda url, json, **kwargs: MockResponse()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(requests, "post", monkeypatch_post)

    hypo_records = [
        HypothesisRecord(
            hypothesis_id=h.get("hypothesis_id", f"iter1_H{i+1}"),
            iteration_no=1,
            hypothesis_type=h.get("hypothesis_type", "individual_dependency"),
            hypothesis=h.get("statement", h.get("hypothesis", "")),
            features_involved=h.get("features_involved", []),
            evidence_from_data=str(h.get("evidence", h.get("evidence_from_data", ""))),
            proposed_experiment=h.get("proposed_experiment", "")
        )
        for i, h in enumerate(hypotheses)
    ]

    registry = HypothesisRegistry(registry_path=os.path.join(out_dir, "hypothesis_registry.json"))
    for h in hypo_records:
        registry.register_hypothesis(h)

    try:
        interps, next_hyps = interpret_and_refine_all(
            eval_results=eval_results,
            hypotheses=hypo_records,
            iteration_no=1,
            output_dir=out_dir,
            registry=registry
        )
    finally:
        monkeypatch.undo()

    assert len(interps) == len(eval_results)
    assert len(next_hyps) > 0
    assert os.path.exists(os.path.join(out_dir, "llm_interpretations.json"))
    assert os.path.exists(os.path.join(out_dir, "refined_next_hypotheses.json"))
    assert os.path.exists(os.path.join(out_dir, "iterations", "iter_1", "llm_interpretations.json"))
    assert os.path.exists(os.path.join(out_dir, "iterations", "iter_1", "refined_next_hypotheses.json"))

    # Verify Cluster persistence for evaluations and refinements
    if cluster.is_available():
        eval_docs = cluster.query_documents(DB_EVALUATIONS, "evaluations")
        comp_docs = cluster.query_documents(DB_EVALUATIONS, "comparisons")
        interp_docs = cluster.query_documents(DB_REFINEMENTS, "interpretations")
        ref_docs = cluster.query_documents(DB_REFINEMENTS, "refined_hypotheses")
        assert len(eval_docs) >= len(eval_results)
        assert len(comp_docs) >= 1
        assert len(interp_docs) >= len(interps)
        assert len(ref_docs) >= len(next_hyps)

    # Test Batch 6 Autonomous Loop & Batch 7 Experiment Memory Orchestration
    from modules.experiment_controller import run_autonomous_scientist
    from modules.schemas import StoppingConditionConfig
    from modules.config import DB_MEMORY

    monkeypatch.setattr(requests, "post", monkeypatch_post)
    try:
        report_md, report_data = run_autonomous_scientist(
            initial_hypotheses=hypotheses[:2],
            df=df,
            target="Outcome",
            config=StoppingConditionConfig(max_experiments=2, max_iterations=1),
            output_dir=out_dir,
            n_permutations=10,
            n_bootstrap=10,
            sync_mongo=False
        )
    finally:
        monkeypatch.undo()

    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.md"))
    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.json"))
    assert os.path.exists(os.path.join(out_dir, "experiment_memory.json"))
    assert report_data["total_experiments"] >= 1
    assert "Autonomous AI Scientist" in report_md




