from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Dataset:
    name: str
    rows: tuple[tuple[float, float], ...]
    feature_name: str = "signal"
    target_name: str = "target"


@dataclass(frozen=True)
class DatasetProfile:
    dataset_name: str
    row_count: int
    feature_names: tuple[str, ...]
    target_mean: float
    feature_mean: float
    feature_target_correlation: float


@dataclass(frozen=True)
class Pattern:
    description: str
    evidence: str
    confidence: float


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    rationale: str
    success_criterion: str
    expected_metric_direction: Literal["higher", "lower"]


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    hypothesis_id: str
    control: str
    treatment: str
    metric: str
    seed: int


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    control_metric: float
    treatment_metric: float
    runtime_seconds: float
    reproducible: bool


@dataclass(frozen=True)
class Evaluation:
    hypothesis_id: str
    status: Literal["supported", "rejected", "inconclusive"]
    improvement: float
    explanation: str


@dataclass(frozen=True)
class ScientificReport:
    research_question: str
    dataset_name: str
    profile: DatasetProfile
    patterns: tuple[Pattern, ...]
    hypothesis: Hypothesis
    plan: ExperimentPlan
    result: ExperimentResult
    evaluation: Evaluation
    next_step: str
    history: list[dict[str, object]] = field(default_factory=list)
