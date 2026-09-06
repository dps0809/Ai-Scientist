import math

from ai_scientist.models import Dataset, DatasetProfile


def synthetic_dataset() -> Dataset:
    rows = tuple((index / 10, 1.0 + 2.0 * (index / 10)) for index in range(10))
    return Dataset(name="synthetic_linear", rows=rows)


def profile_dataset(dataset: Dataset) -> DatasetProfile:
    features = [row[0] for row in dataset.rows]
    targets = [row[1] for row in dataset.rows]
    feature_mean = sum(features) / len(features)
    target_mean = sum(targets) / len(targets)
    numerator = sum((x - feature_mean) * (y - target_mean) for x, y in dataset.rows)
    feature_std = math.sqrt(sum((x - feature_mean) ** 2 for x in features))
    target_std = math.sqrt(sum((y - target_mean) ** 2 for y in targets))
    correlation = numerator / (feature_std * target_std)
    return DatasetProfile(
        dataset_name=dataset.name,
        row_count=len(dataset.rows),
        feature_names=(dataset.feature_name,),
        target_mean=target_mean,
        feature_mean=feature_mean,
        feature_target_correlation=correlation,
    )
