from ai_scientist.models import ExperimentPlan, Hypothesis


def plan_experiment(hypothesis: Hypothesis, seed: int = 7) -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="E-001",
        hypothesis_id=hypothesis.hypothesis_id,
        control="predict the target mean",
        treatment="fit a linear relationship using signal",
        metric="mean_squared_error",
        seed=seed,
    )
