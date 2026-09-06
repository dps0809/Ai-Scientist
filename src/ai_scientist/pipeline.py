from pathlib import Path

from ai_scientist.data.profiler import profile_dataset, synthetic_dataset
from ai_scientist.evaluation.evaluator import evaluate
from ai_scientist.experiments.planner import plan_experiment
from ai_scientist.experiments.runner import run_experiment
from ai_scientist.hypotheses.generator import generate_hypothesis
from ai_scientist.models import ScientificReport
from ai_scientist.patterns.discovery import discover_patterns
from ai_scientist.reporting.writer import write_report

RESEARCH_QUESTION = "Can an AI system autonomously discover useful hypotheses and improve model performance through iterative experimentation?"


def run_demo(output_dir: Path = Path("artifacts"), seed: int = 7) -> ScientificReport:
    dataset = synthetic_dataset()
    profile = profile_dataset(dataset)
    patterns = discover_patterns(profile)
    hypothesis = generate_hypothesis(patterns)
    plan = plan_experiment(hypothesis, seed=seed)
    result = run_experiment(dataset, plan)
    evaluation = evaluate(hypothesis, result)
    next_step = "Try a nonlinear model or a held-out validation split in the next iteration."
    report = ScientificReport(
        research_question=RESEARCH_QUESTION,
        dataset_name=dataset.name,
        profile=profile,
        patterns=patterns,
        hypothesis=hypothesis,
        plan=plan,
        result=result,
        evaluation=evaluation,
        next_step=next_step,
        history=[{"experiment_id": plan.experiment_id, "status": evaluation.status}],
    )
    write_report(report, output_dir)
    return report
