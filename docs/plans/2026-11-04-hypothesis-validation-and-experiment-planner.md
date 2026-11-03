# Hypothesis Validation (Batch 1) and Experiment Planning (Batch 2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a robust Pydantic-powered Hypothesis Validator with evidence reconciliation and graceful MongoDB storage (Batch 1), alongside an extensible Experiment Registry & Planner that converts validated hypotheses into structured, executable experiment specifications (Batch 2).

**Architecture:**
- **Validator Layer (`modules/hypothesis_validator.py`)**: Uses Pydantic to validate hypothesis structure, Pandas for dataset schema/type/binary checks, and compares numerical claims against Stage 1/Stage 2 ground-truth metrics. Contradictory claims (e.g., H4 interaction delta) are flagged as `NEEDS_REVIEW` with reconciled ground-truth evidence attached.
- **Persistence Layer (`modules/mongo_client.py` & `modules/config.py`)**: Implements graceful dual storage—saving structured JSON to `ai_scientist_output/` while synchronizing documents to MongoDB (database `ai_scientist`) when available, failing safely if offline.
- **Experiment Registry & Planner (`modules/experiment_planner.py`)**: Houses a schema-driven registry mapping hypothesis types to statistical models (`statsmodels.Logit`, `sklearn` CV comparisons, interaction terms, adjusted regressions). Converts hypotheses into executable `ExperimentSpec` models, routing H5 initially to adjusted regression.
- **Pipeline Integration (`mains.py`)**: Chains Stage 4 (Validation) and Stage 5 (Planning) directly after LLM hypothesis generation.

**Tech Stack:** Python 3.13, Pydantic 2.x, Pandas 3.0+, statsmodels 0.15+, scikit-learn 1.9+, pymongo 4.19+, pytest 9.x.

---

### Codebase Audit: What We Keep vs What Changes

| Component | Status | Rationale |
|---|---|---|
| `modules/config.py` | **UPDATE** | Add MongoDB configuration (`MONGO_URI`, `MONGO_DB_NAME`, collection names) and output file paths. |
| `pyproject.toml` | **UPDATE** | Add `statsmodels` and `pymongo` to project dependencies. |
| `modules/mongo_client.py` | **CREATE (NEW)** | Resilient MongoDB connection manager with timeout and graceful offline fallback. |
| `modules/hypothesis_validator.py` | **CREATE (NEW - Batch 1)** | Pydantic schema validation, dataset consistency checks, evidence reconciliation, and status assignment (`VALIDATED`, `NEEDS_REVIEW`, `REJECTED`). |
| `modules/experiment_planner.py` | **CREATE (NEW - Batch 2)** | Predefined experiment registry, parameter templates, H5 adjusted regression routing, and experiment spec generation. |
| `mains.py` | **UPDATE** | Integrate Stage 4 (Validator) and Stage 5 (Planner) into pipeline execution flow. |
| `tests/test_mongo_client.py` | **CREATE (NEW)** | Test MongoDB client connection and offline fallback. |
| `tests/test_hypothesis_validator.py` | **CREATE (NEW)** | Test schema validation, column checks, binary target verification, discrepancy detection (H4), and reconciliation. |
| `tests/test_experiment_planner.py` | **CREATE (NEW)** | Test experiment registry mapping, H1-E1 through H4-E4 spec creation, and H5 mediation-to-adjusted-regression routing. |
| `tests/test_pipeline_integration.py` | **CREATE (NEW)** | Test end-to-end integration of validation and experiment planning stages. |

---

## Detailed Task Breakdown

### Task 1: Persistence Configuration & Graceful MongoDB Client
**Files:**
- Modify: `modules/config.py`
- Modify: `pyproject.toml`
- Create: `modules/mongo_client.py`
- Test: `tests/test_mongo_client.py`

**Step 1: Write the failing test**

Create `tests/test_mongo_client.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from modules.mongo_client import get_mongo_client, sync_to_mongodb

def test_get_mongo_client_offline_fallback():
    # If connection fails or times out, get_mongo_client returns None without throwing
    with patch("pymongo.MongoClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.admin.command.side_effect = Exception("Connection refused")
        mock_client.return_value = mock_instance
        
        client = get_mongo_client(uri="mongodb://invalid:27017/", timeout_ms=50)
        assert client is None

def test_sync_to_mongodb_graceful():
    # sync_to_mongodb should return False and not raise when client is None
    success = sync_to_mongodb(collection_name="test_col", docs=[{"test": 1}], client=None)
    assert success is False
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_mongo_client.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.mongo_client'`

**Step 3: Implement minimal code**

1. Modify `modules/config.py` to add:
```python
# MongoDB & Persistence Settings
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "ai_scientist"
MONGO_VALIDATION_COLLECTION = "hypotheses_validation"
MONGO_EXPERIMENTS_COLLECTION = "experiment_specifications"
MONGO_TIMEOUT_MS = 2000

VALIDATED_HYPOTHESES_PATH = os.path.join(OUTPUT_DIR, "validated_hypotheses.json")
EXPERIMENT_SPECS_PATH = os.path.join(OUTPUT_DIR, "experiment_specifications.json")
```

2. Modify `pyproject.toml` dependencies:
Add `"statsmodels>=0.14",` and `"pymongo>=4.9",` to `dependencies`.

3. Create `modules/mongo_client.py`:
```python
import logging
from typing import Optional, List, Dict, Any
import pymongo
from pymongo.errors import PyMongoError
from .config import MONGO_URI, MONGO_DB_NAME, MONGO_TIMEOUT_MS

logger = logging.getLogger(__name__)

def get_mongo_client(uri: str = MONGO_URI, timeout_ms: int = MONGO_TIMEOUT_MS) -> Optional[pymongo.MongoClient]:
    """
    Connect to MongoDB with a short timeout.
    Returns MongoClient instance or None if offline/unavailable.
    """
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        return client
    except Exception as e:
        logger.warning(f"MongoDB connection unavailable ({e}). Continuing in standalone file-based mode.")
        return None

def sync_to_mongodb(
    collection_name: str,
    docs: List[Dict[str, Any]],
    id_key: str = "hypothesis_id",
    db_name: str = MONGO_DB_NAME,
    client: Optional[pymongo.MongoClient] = None
) -> bool:
    """
    Upsert documents into MongoDB collection. Gracefully skips if MongoDB is not available.
    """
    if client is None:
        client = get_mongo_client()
    if client is None:
        return False

    try:
        db = client[db_name]
        col = db[collection_name]
        for doc in docs:
            filter_query = {id_key: doc[id_key]} if id_key in doc else doc
            col.update_one(filter_query, {"$set": doc}, upsert=True)
        return True
    except PyMongoError as e:
        logger.warning(f"Failed to sync to MongoDB collection '{collection_name}': {e}")
        return False
```

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_mongo_client.py -v
```
Expected: PASS (2 tests passed)

**Step 5: Commit**

```bash
git add modules/config.py pyproject.toml modules/mongo_client.py tests/test_mongo_client.py
git commit -m "feat: add config settings and graceful MongoDB persistence client"
```

---

### Task 2: Implement Batch 1 — Hypothesis Validator
**Files:**
- Create: `modules/hypothesis_validator.py`
- Test: `tests/test_hypothesis_validator.py`

**Step 1: Write the failing test**

Create `tests/test_hypothesis_validator.py`:
```python
import pytest
import pandas as pd
from modules.hypothesis_validator import (
    validate_hypotheses,
    ValidationStatus,
    RawHypothesis
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Glucose": [120, 140, 90, 180, 110],
        "BMI": [28.5, 32.1, 22.0, 35.4, 26.0],
        "Age": [33, 45, 25, 52, 29],
        "Outcome": [0, 1, 0, 1, 0]
    })

def test_validate_valid_hypothesis(sample_df):
    raw = [{
        "hypothesis_id": "H1",
        "hypothesis_type": "individual_dependency",
        "hypothesis": "Glucose concentration has a strong positive association with diabetes outcome.",
        "evidence_from_data": "Glucose: pearson_r=0.4666, p < 0.05",
        "proposed_experiment": "Fit logistic regression using Glucose."
    }]
    report = validate_hypotheses(raw, sample_df, target="Outcome")
    assert report.total == 1
    assert report.validated_count == 1
    assert report.results[0].status == ValidationStatus.VALIDATED

def test_validate_missing_feature(sample_df):
    raw = [{
        "hypothesis_id": "H_Err",
        "hypothesis_type": "individual_dependency",
        "hypothesis": "InsulinLevel has strong positive association.",
        "evidence_from_data": "InsulinLevel: pearson_r=0.45",
        "proposed_experiment": "Fit logistic regression."
    }]
    report = validate_hypotheses(raw, sample_df, target="Outcome")
    assert report.rejected_count == 1
    assert report.results[0].status == ValidationStatus.REJECTED
    assert any("not found" in r for r in report.results[0].reasons)

def test_validate_discrepancy_needs_review(sample_df):
    # Simulates H4 conflicting numbers
    raw = [{
        "hypothesis_id": "H4",
        "hypothesis_type": "feature_interaction_moderation",
        "hypothesis": "Glucose and Age interaction (interaction delta = -0.0007, p-value = 0.0364)",
        "evidence_from_data": "Glucose + Age: interaction_delta=0.0002, interaction_p_value=0.0026",
        "proposed_experiment": "Fit logistic regression with interaction term."
    }]
    stage2_mock = {
        "combinations": [{
            "combination": ["Glucose", "Age"],
            "interaction_delta": 0.0002,
            "interaction_p_value": 0.0026
        }]
    }
    report = validate_hypotheses(raw, sample_df, target="Outcome", stage2_data=stage2_mock)
    res = report.results[0]
    assert res.status == ValidationStatus.NEEDS_REVIEW
    assert "Discrepancy detected" in res.reasons[0]
    assert res.reconciled_evidence["interaction_delta"] == 0.0002
    assert res.reconciled_evidence["interaction_p_value"] == 0.0026
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_hypothesis_validator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.hypothesis_validator'`

**Step 3: Implement minimal code**

Create `modules/hypothesis_validator.py`:
- Use `pydantic.BaseModel` and `enum.Enum`:
  ```python
  class ValidationStatus(str, Enum):
      VALIDATED = "VALIDATED"
      NEEDS_REVIEW = "NEEDS_REVIEW"
      REJECTED = "REJECTED"
  ```
- Define `RawHypothesis`, `ValidationResult`, `ValidationReport` Pydantic models.
- Implement `extract_features_from_hypothesis(hypothesis, df_columns)`: dynamically identifies dataset features mentioned in the hypothesis fields.
- Implement `check_numerical_evidence(hypothesis, stage1_data, stage2_data)`: regex extracts metrics like `interaction_delta`, `p_value`, `r` from both hypothesis sentence and `evidence_from_data`. Flags discrepancies against calculated ground-truth values.
- Implement dataset verification:
  - `target in df.columns`
  - `set(df[target].dropna().unique()).issubset({0, 1})`
  - For each extracted feature: exists in `df.columns` and `pd.api.types.is_numeric_dtype(df[feature])`.
- Save results to `OUTPUT_DIR/validated_hypotheses.json`.
- Synchronize records to MongoDB collection `hypotheses_validation` using `sync_to_mongodb()`.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_hypothesis_validator.py -v
```
Expected: PASS (3 tests passed)

**Step 5: Commit**

```bash
git add modules/hypothesis_validator.py tests/test_hypothesis_validator.py
git commit -m "feat: implement Batch 1 Hypothesis Validator with evidence reconciliation"
```

---

### Task 3: Implement Batch 2 — Experiment Registry & Planner
**Files:**
- Create: `modules/experiment_planner.py`
- Test: `tests/test_experiment_planner.py`

**Step 1: Write the failing test**

Create `tests/test_experiment_planner.py`:
```python
import pytest
from modules.hypothesis_validator import ValidationReport, ValidationResult, ValidationStatus
from modules.experiment_planner import plan_experiments, ExperimentSpec, EXPERIMENT_REGISTRY

def test_registry_contains_supported_methods():
    assert "individual_dependency" in EXPERIMENT_REGISTRY
    assert "feature_combination_synergy" in EXPERIMENT_REGISTRY
    assert "feature_interaction_moderation" in EXPERIMENT_REGISTRY
    assert "comparative_or_confounding_relationship" in EXPERIMENT_REGISTRY

def test_plan_experiments_routing_h5():
    # H5 specifies mediation in hypothesis text, but planner routes to adjusted regression
    report = ValidationReport(
        total=2,
        validated_count=1,
        needs_review_count=1,
        rejected_count=0,
        results=[
            ValidationResult(
                hypothesis_id="H1",
                hypothesis_type="individual_dependency",
                status=ValidationStatus.VALIDATED,
                features_involved=["Glucose"],
                reconciled_evidence={"pearson_r": 0.4666},
                reasons=[]
            ),
            ValidationResult(
                hypothesis_id="H5",
                hypothesis_type="comparative_or_confounding_relationship",
                status=ValidationStatus.NEEDS_REVIEW,
                features_involved=["Glucose", "BMI"],
                reconciled_evidence={"correlation": 0.2211},
                reasons=["Needs adjusted regression routing"]
            )
        ]
    )
    
    specs = plan_experiments(report, target="Outcome")
    assert len(specs) == 2
    
    # Check H1 spec
    e1 = next(s for s in specs if s.hypothesis_id == "H1")
    assert e1.experiment_id == "E1"
    assert e1.method == "logistic_regression"
    assert e1.evaluation["odds_ratio"] is True
    
    # Check H5 routing to adjusted regression
    e5 = next(s for s in specs if s.hypothesis_id == "H5")
    assert e5.experiment_id == "E5"
    assert e5.method == "adjusted_regression"
    assert "adjusted_odds_ratio" in e5.evaluation
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_experiment_planner.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_planner'`

**Step 3: Implement minimal code**

Create `modules/experiment_planner.py`:
- Define `EXPERIMENT_REGISTRY` mapping:
  - `individual_dependency`:
    - `method`: `logistic_regression`
    - `model_class`: `statsmodels.Logit`
    - `evaluation_template`: `{"odds_ratio": True, "confidence_interval": True, "permutation_test": True, "permutation_iterations": 1000}`
  - `feature_combination_synergy`:
    - `method`: `paired_model_comparison`
    - `model_class`: `sklearn.linear_model.LogisticRegression`
    - `evaluation_template`: `{"metric": "roc_auc", "cross_validation": "StratifiedKFold", "folds": 5}`
  - `feature_interaction_moderation`:
    - `method`: `interaction_regression`
    - `model_class`: `statsmodels.Logit`
    - `evaluation_template`: `{"interaction_term": True, "likelihood_ratio_test": True, "bootstrap_samples": 500}`
  - `comparative_or_confounding_relationship`:
    - `method`: `adjusted_regression`
    - `model_class`: `statsmodels.Logit`
    - `evaluation_template`: `{"adjusted_odds_ratio": True, "unadjusted_odds_ratio": True, "delta_coefficient": True}`
- Define `ExperimentSpec` Pydantic model:
  - `experiment_id: str`
  - `hypothesis_id: str`
  - `experiment_type: str`
  - `method: str`
  - `model_type: str`
  - `features: List[str]`
  - `target: str`
  - `comparison: Optional[List[List[str]]]`
  - `evaluation: Dict[str, Any]`
  - `routing_notes: Optional[str]`
- Build `plan_experiments(validation_report, target, df)`:
  - Loops over validated and reviewed hypotheses.
  - Resolves features, comparisons, and parameters.
  - Applies explicit H5 routing rule: routes comparative/mediation hypotheses to `adjusted_regression`.
  - Saves all 5 specs to `OUTPUT_DIR/experiment_specifications.json`.
  - Synchronizes to MongoDB collection `experiment_specifications` using `sync_to_mongodb()`.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_experiment_planner.py -v
```
Expected: PASS (2 tests passed)

**Step 5: Commit**

```bash
git add modules/experiment_planner.py tests/test_experiment_planner.py
git commit -m "feat: implement Batch 2 Experiment Registry and Planner"
```

---

### Task 4: Pipeline Integration in `mains.py` & End-to-End Test
**Files:**
- Modify: `mains.py`
- Create: `tests/test_pipeline_integration.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline_integration.py`:
```python
import os
import json
import pytest
import pandas as pd
from modules.config import OUTPUT_DIR, VALIDATED_HYPOTHESES_PATH, EXPERIMENT_SPECS_PATH
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments

def test_pipeline_integration_with_generated_hypotheses():
    hypotheses_file = os.path.join(OUTPUT_DIR, "generated_hypotheses.json")
    assert os.path.exists(hypotheses_file), "generated_hypotheses.json must exist"
    
    with open(hypotheses_file, "r") as f:
        hypotheses = json.load(f)
        
    df = pd.read_csv("diabetes.csv")
    
    # 1. Validation
    report = validate_hypotheses(hypotheses, df, target="Outcome")
    assert report.total == 5
    assert os.path.exists(VALIDATED_HYPOTHESES_PATH)
    
    # Verify H4 flagged as NEEDS_REVIEW due to conflicting interaction delta
    h4_res = next(r for r in report.results if r.hypothesis_id == "H4")
    assert h4_res.status == "NEEDS_REVIEW"
    
    # 2. Experiment Planning
    specs = plan_experiments(report, target="Outcome", df=df)
    assert len(specs) == 5
    assert os.path.exists(EXPERIMENT_SPECS_PATH)
    
    # Verify H5 routed to adjusted_regression
    e5 = next(s for s in specs if s.hypothesis_id == "H5")
    assert e5.method == "adjusted_regression"
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_pipeline_integration.py -v
```
Expected: Tests pipeline integration.

**Step 3: Modify `mains.py`**

In `mains.py`:
1. Import `validate_hypotheses` from `modules.hypothesis_validator`.
2. Import `plan_experiments` from `modules.experiment_planner`.
3. Add Stage 4 and Stage 5 blocks after hypothesis generation:
```python
    # Stage 4: Hypothesis Validation (Batch 1)
    print("\n" + "=" * 70)
    print("STAGE 4: HYPOTHESIS VALIDATION & RECONCILIATION")
    print("=" * 70)
    validation_report = validate_hypotheses(
        hypotheses=hypotheses,
        df=df,
        target=target,
        stage1_data=stage1,
        stage2_data=stage2
    )

    # Stage 5: Experiment Planning & Specification (Batch 2)
    print("\n" + "=" * 70)
    print("STAGE 5: EXPERIMENT PLANNING & REGISTRY SPECIFICATION")
    print("=" * 70)
    experiment_specs = plan_experiments(
        validation_report=validation_report,
        target=target,
        df=df
    )
```
4. Update the summary report printout at the end of `mains.py`.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/ -v
```
Expected: PASS (all tests pass)

**Step 5: Commit**

```bash
git add mains.py tests/test_pipeline_integration.py
git commit -m "feat: integrate Stage 4 Hypothesis Validator and Stage 5 Experiment Planner into mains.py"
```

---

## Verification Plan

### Automated Tests
1. `python -m pytest tests/test_mongo_client.py -v`
2. `python -m pytest tests/test_hypothesis_validator.py -v`
3. `python -m pytest tests/test_experiment_planner.py -v`
4. `python -m pytest tests/test_pipeline_integration.py -v`
5. Full test suite:
   ```powershell
   python -m pytest -v
   ```

### End-to-End Pipeline Verification
Run `python mains.py` and verify:
1. Console outputs clear Stage 4 validation summary showing H1-H3 as VALIDATED, H4 as NEEDS_REVIEW (with reconciled delta = 0.0002, p = 0.0026), and H5 status.
2. Console outputs Stage 5 experiment specifications E1 to E5, with H5 routed to `adjusted_regression`.
3. Verify files generated:
   - `ai_scientist_output/validated_hypotheses.json`
   - `ai_scientist_output/experiment_specifications.json`
4. Verify MongoDB documents populated in database `ai_scientist`:
   - Collection `hypotheses_validation` contains 5 documents.
   - Collection `experiment_specifications` contains 5 documents.
