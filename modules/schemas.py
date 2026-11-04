from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"

class InputRecord(BaseModel):
    input_id: str
    source: str
    row_count: int
    column_count: int
    features: List[str]
    target: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PatternRecord(BaseModel):
    pattern_id: str
    input_id: str
    pattern_type: str
    features: List[str]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    description: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class HypothesisRecord(BaseModel):
    hypothesis_id: str
    iteration_no: int = 1
    hypothesis_type: str
    hypothesis: str
    features_involved: List[str] = Field(default_factory=list)
    evidence_from_data: str = ""
    proposed_experiment: str = ""
    fingerprint: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context: Any) -> None:
        if self.fingerprint is None:
            sorted_feats = tuple(sorted(self.features_involved))
            self.fingerprint = f"{sorted_feats}-{self.hypothesis_type}"

class ValidationResultRecord(BaseModel):
    validation_id: str
    hypothesis_id: str
    iteration_no: int = 1
    hypothesis_type: Optional[str] = None
    status: ValidationStatus
    features_involved: List[str] = Field(default_factory=list)
    reconciled_evidence: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ValidationReportRecord(BaseModel):
    report_id: str
    iteration_no: int = 1
    total: int
    validated_count: int
    needs_review_count: int
    rejected_count: int
    results: List[ValidationResultRecord] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ExperimentSpecRecord(BaseModel):
    experiment_id: str
    hypothesis_id: str
    iteration_no: int = 1
    experiment_type: str
    method: str
    model_type: str
    features: List[str]
    target: str
    comparison: Optional[List[List[str]]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    routing_notes: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
