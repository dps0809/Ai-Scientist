from time import perf_counter

from ai_scientist.models import Dataset, ExperimentPlan, ExperimentResult


def run_experiment(dataset: Dataset, plan: ExperimentPlan) -> ExperimentResult:
    started = perf_counter()
    targets = [target for _, target in dataset.rows]
    features = [feature for feature, _ in dataset.rows]
    target_mean = sum(targets) / len(targets)
    control_mse = sum((target - target_mean) ** 2 for target in targets) / len(targets)

    feature_mean = sum(features) / len(features)
    numerator = sum((x - feature_mean) * (y - target_mean) for x, y in dataset.rows)
    denominator = sum((x - feature_mean) ** 2 for x in features)
    slope = numerator / denominator
    intercept = target_mean - slope * feature_mean
    treatment_mse = sum((target - (intercept + slope * feature)) ** 2 for feature, target in dataset.rows) / len(targets)
    return ExperimentResult(
        experiment_id=plan.experiment_id,
        control_metric=control_mse,
        treatment_metric=treatment_mse,
        runtime_seconds=perf_counter() - started,
        reproducible=True,
    )
