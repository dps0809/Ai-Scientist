from ai_scientist.models import Evaluation, ExperimentResult, Hypothesis


def evaluate(hypothesis: Hypothesis, result: ExperimentResult, threshold: float = 0.10) -> Evaluation:
    improvement = (result.control_metric - result.treatment_metric) / result.control_metric
    if improvement >= threshold:
        status = "supported"
        explanation = f"Treatment improved {result.experiment_id} by {improvement:.1%}."
    elif improvement <= 0:
        status = "rejected"
        explanation = f"Treatment did not improve the metric ({improvement:.1%})."
    else:
        status = "inconclusive"
        explanation = f"Improvement of {improvement:.1%} did not reach the threshold."
    return Evaluation(hypothesis.hypothesis_id, status, improvement, explanation)
