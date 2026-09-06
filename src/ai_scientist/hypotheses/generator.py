from ai_scientist.models import Hypothesis, Pattern


def generate_hypothesis(patterns: tuple[Pattern, ...]) -> Hypothesis:
    pattern = patterns[0]
    return Hypothesis(
        hypothesis_id="H-001",
        statement="Using the signal feature to predict the target will reduce mean squared error versus a mean-only baseline.",
        rationale=f"Generated from: {pattern.description} ({pattern.evidence}).",
        success_criterion="Treatment MSE is at least 10% lower than control MSE.",
        expected_metric_direction="lower",
    )
