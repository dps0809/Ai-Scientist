import json

from ai_scientist.data.profiler import profile_dataset, synthetic_dataset
from ai_scientist.evaluation.evaluator import evaluate
from ai_scientist.experiments.planner import plan_experiment
from ai_scientist.experiments.runner import run_experiment
from ai_scientist.hypotheses.generator import generate_hypothesis
from ai_scientist.patterns.discovery import discover_patterns
from ai_scientist.pipeline import run_demo


def test_demo_supports_hypothesis(tmp_path):
    report = run_demo(tmp_path)
    assert report.evaluation.status == "supported"
    assert (tmp_path / "scientific_report.md").exists()
    payload = json.loads((tmp_path / "scientific_report.json").read_text())
    assert payload["history"][0]["experiment_id"] == "E-001"


def test_experiment_is_reproducible():
    dataset = synthetic_dataset()
    profile = profile_dataset(dataset)
    hypothesis = generate_hypothesis(discover_patterns(profile))
    plan = plan_experiment(hypothesis)
    first = run_experiment(dataset, plan)
    second = run_experiment(dataset, plan)
    assert first.control_metric == second.control_metric
    assert first.treatment_metric == second.treatment_metric
    assert evaluate(hypothesis, first).status == "supported"
