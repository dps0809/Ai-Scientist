# Iterative Hypothesis Validation & Experiment Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement an end-to-end, reproducible research architecture featuring centralized uniform schemas, unique input/pattern/hypothesis IDs, multi-database MongoDB cluster segregation on open port 27017, de-duplication across iterative LLM hypothesis generations, Batch 1 hypothesis validation with ground-truth evidence reconciliation, Batch 2 experiment planning with H5 adjusted regression routing, and automatic `README.md` updates after every task.

**Architecture:**
- **Centralized Schema Contract (`modules/schemas.py`)**: Pydantic models for `InputRecord`, `PatternRecord`, `HypothesisRecord`, `ValidationRecord`, and `ExperimentSpecRecord` ensuring data uniformity across all pipeline stages, JSON outputs, and database documents.
- **Multi-Database Cluster Manager (`modules/mongo_cluster.py`)**: Connects to the active local MongoDB port (27017) and routes distinct stage artifacts into isolated dedicated databases:
  1. `ai_scientist_inputs`: Input datasets, schemas, and `input_id` metadata.
  2. `ai_scientist_patterns`: Stage 1 individual dependencies and Stage 2 combination/interaction metrics.
  3. `ai_scientist_hypotheses`: LLM contexts, generated hypotheses with `iteration_no`, and deduplication index.
  4. `ai_scientist_validation`: Batch 1 validation reports, discrepancy checks, and reconciled evidence.
  5. `ai_scientist_experiments`: Batch 2 structured experiment specifications.
- **De-duplication & Iterative Registry (`modules/hypothesis_registry.py`)**: Indexes all past hypotheses across iterations, generates structural/semantic fingerprints (features, hypothesis type, relationship direction), prevents duplicate hypothesis generation across iterations, and formats inputs with explicit iteration numbers.
- **Hypothesis Validator (`modules/hypothesis_validator.py` - Batch 1)**: Validates ALL LLM hypotheses without omission, enforces dataset and target integrity, resolves numerical discrepancies against Stage 1/Stage 2 ground truth (e.g. H4 interaction delta), flags statuses (`VALIDATED`, `NEEDS_REVIEW`, `REJECTED`), and records audit trails.
- **Experiment Planner (`modules/experiment_planner.py` - Batch 2)**: Registry mapping hypothesis types to executable statistical models (`statsmodels.Logit`, `sklearn` CV comparisons, interaction terms, adjusted regression), routing H5 to adjusted regression.
- **Documentation & Discipline**: `README.md` is updated and committed after each task.

**Tech Stack:** Python 3.13, Pydantic 2.x, Pandas 3.0+, statsmodels 0.15+, scikit-learn 1.9+, pymongo 4.19+, pytest 9.x.

---

### Codebase Audit: What We Keep vs What Changes

| Component | Status | Rationale |
|---|---|---|
| `modules/config.py` | **UPDATE** | Add multi-database configurations, collection names, iteration settings, and output paths. |
| `pyproject.toml` | **UPDATE** | Register `statsmodels>=0.14` and `pymongo>=4.9` in dependencies. |
| `modules/schemas.py` | **CREATE (NEW)** | Single source of truth: Pydantic models for inputs, patterns, hypotheses, validations, and experiment specs. |
| `modules/mongo_cluster.py` | **CREATE (NEW)** | Multi-database client managing separate databases for inputs, patterns, hypotheses, validation, and experiments. |
| `modules/load_dataset.py` | **UPDATE** | Generates `input_id` and logs dataset ingestion to `ai_scientist_inputs`. |
| `modules/save_patterns.py` | **UPDATE** | Assigns `pattern_id` and `input_id`, logs to `ai_scientist_patterns` DB and JSON. |
| `modules/hypothesis_registry.py` | **CREATE (NEW)** | Manages iteration history, fingerprints hypotheses, and prevents duplicates across iterative loops. |
| `modules/generate_hypotheses_with_llm.py` | **UPDATE** | Supports iteration tracking, passes past hypotheses to avoid duplicate generation, produces uniform schema. |
| `modules/hypothesis_validator.py` | **CREATE (NEW - Batch 1)** | Schema checks, column/type checks, discrepancy reconciliation (H4), and persistence to `ai_scientist_validation`. |
| `modules/experiment_planner.py` | **CREATE (NEW - Batch 2)** | Predefined experiment registry, H5 adjusted regression routing, uniform `ExperimentSpecRecord` generation, and persistence to `ai_scientist_experiments`. |
| `mains.py` | **UPDATE** | Orchestrates full pipeline: Input -> Patterns -> Hypotheses -> Validation -> Planning with multi-DB sync. |
| `README.md` | **UPDATE CONTINUOUSLY** | Updated after every single task to reflect latest architecture and schemas. |

---

## Detailed Task Breakdown

### Task 1: Centralized Uniform Schemas & Multi-Database Cluster Manager
**Files:**
- Create: `modules/schemas.py`
- Create: `modules/mongo_cluster.py`
- Modify: `modules/config.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_mongo_cluster.py`
- Test: `tests/test_schemas.py`

**Step 1: Write the failing tests**

1. Create `tests/test_schemas.py`:
```python
import pytest
from modules.schemas import (
    InputRecord,
    PatternRecord,
    HypothesisRecord,
    ValidationRecord,
    ExperimentSpecRecord,
    ValidationStatus
)

def test_schemas_instantiation_and_serialization():
    hypo = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose concentration has strong association.",
        features_involved=["Glucose"],
        evidence_from_data="pearson_r=0.4666",
        proposed_experiment="Fit logistic regression"
    )
    assert hypo.hypothesis_id == "H1"
    assert hypo.iteration_no == 1
    dump = hypo.model_dump()
    assert dump["hypothesis_id"] == "H1"
    assert "timestamp" in dump
```

2. Create `tests/test_mongo_cluster.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from modules.mongo_cluster import MongoClusterManager

def test_mongo_cluster_offline_fallback():
    with patch("pymongo.MongoClient") as mock_client:
        mock_instance = MagicMock()
        mock_instance.admin.command.side_effect = Exception("Connection refused")
        mock_client.return_value = mock_instance
        
        cluster = MongoClusterManager(uri="mongodb://invalid:27017/", timeout_ms=50)
        assert cluster.is_available() is False
        success = cluster.save_document("ai_scientist_inputs", "inputs", {"input_id": "test"})
        assert success is False
```

**Step 2: Run tests to verify they fail**

Run:
```powershell
python -m pytest tests/test_schemas.py tests/test_mongo_cluster.py -v
```
Expected: FAIL with `ModuleNotFoundError` for `modules.schemas` and `modules.mongo_cluster`.

**Step 3: Implement minimal code**

1. Modify `modules/config.py` to add database names and paths:
```python
# Multi-database cluster settings (port 27017)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_TIMEOUT_MS = 2000

DB_INPUTS = "ai_scientist_inputs"
DB_PATTERNS = "ai_scientist_patterns"
DB_HYPOTHESES = "ai_scientist_hypotheses"
DB_VALIDATION = "ai_scientist_validation"
DB_EXPERIMENTS = "ai_scientist_experiments"

VALIDATED_HYPOTHESES_PATH = os.path.join(OUTPUT_DIR, "validated_hypotheses.json")
EXPERIMENT_SPECS_PATH = os.path.join(OUTPUT_DIR, "experiment_specifications.json")
HYPOTHESIS_REGISTRY_PATH = os.path.join(OUTPUT_DIR, "hypothesis_registry.json")
```

2. Update `pyproject.toml` dependencies with `statsmodels>=0.14` and `pymongo>=4.9`.

3. Create `modules/schemas.py` with standard Pydantic models for `InputRecord`, `PatternRecord`, `HypothesisRecord`, `ValidationRecord`, `ValidationReportRecord`, `ExperimentSpecRecord`, ensuring uniform IDs and datetime stamps.

4. Create `modules/mongo_cluster.py` implementing `MongoClusterManager`:
   - Connects to `MONGO_URI`.
   - Routes saves to specified DB and collection.
   - Upserts using primary key (`input_id`, `pattern_id`, `hypothesis_id`, `validation_id`, `experiment_id`).
   - Gracefully returns `False` without breaking pipeline if MongoDB is offline.

5. Update `README.md` to document the centralized schemas and multi-database cluster architecture.

**Step 4: Run tests to verify they pass**

Run:
```powershell
python -m pytest tests/test_schemas.py tests/test_mongo_cluster.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add modules/config.py pyproject.toml modules/schemas.py modules/mongo_cluster.py README.md tests/test_schemas.py tests/test_mongo_cluster.py
git commit -m "feat: implement centralized schemas and multi-database cluster manager"
```

---

### Task 2: Input & Discovered Patterns Tracking with Unique IDs
**Files:**
- Modify: `modules/load_dataset.py`
- Modify: `modules/save_patterns.py`
- Modify: `README.md`
- Test: `tests/test_input_and_pattern_tracking.py`

**Step 1: Write the failing test**

Create `tests/test_input_and_pattern_tracking.py`:
```python
import pytest
import pandas as pd
from modules.load_dataset import load_dataset_with_id
from modules.save_patterns import save_patterns_with_ids

def test_load_dataset_assigns_input_id(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,Outcome\n1,2,0\n3,4,1")
    df, input_meta = load_dataset_with_id(str(csv_file))
    assert input_meta.input_id.startswith("INP-")
    assert input_meta.row_count == 2
    assert input_meta.column_count == 3

def test_save_patterns_assigns_pattern_ids():
    raw_patterns = [{"type": "pearson", "feature": "Glucose", "r": 0.46}]
    processed = save_patterns_with_ids(raw_patterns, input_id="INP-TEST-001")
    assert len(processed) == 1
    assert processed[0]["pattern_id"].startswith("PAT-")
    assert processed[0]["input_id"] == "INP-TEST-001"
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_input_and_pattern_tracking.py -v
```
Expected: FAIL with `ImportError: cannot import name 'load_dataset_with_id'`

**Step 3: Implement minimal code**

1. In `modules/load_dataset.py`:
   - Add `load_dataset_with_id(path)` which generates a unique `input_id` (e.g. `INP-<hash/timestamp>`), records metadata in `InputRecord`, logs to MongoDB `ai_scientist_inputs.inputs`, and returns `(df, input_meta)`.
   - Maintain backward-compatible `load_dataset(path)` wrapper.

2. In `modules/save_patterns.py`:
   - Add `save_patterns_with_ids(patterns, input_id)` which injects unique `pattern_id` (`PAT-<num>`), associates `input_id`, saves to `ai_scientist_patterns.patterns` DB and JSON file.
   - Maintain backward-compatible `save_patterns(patterns)`.

3. Update `README.md` with input tracking and pattern persistence details.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_input_and_pattern_tracking.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add modules/load_dataset.py modules/save_patterns.py README.md tests/test_input_and_pattern_tracking.py
git commit -m "feat: attach unique IDs to inputs and patterns with multi-DB persistence"
```

---

### Task 3: Iteration-Aware Hypothesis Registry & De-duplication Engine
**Files:**
- Create: `modules/hypothesis_registry.py`
- Modify: `modules/generate_hypotheses_with_llm.py`
- Modify: `README.md`
- Test: `tests/test_hypothesis_registry.py`

**Step 1: Write the failing test**

Create `tests/test_hypothesis_registry.py`:
```python
import pytest
from modules.hypothesis_registry import HypothesisRegistry
from modules.schemas import HypothesisRecord

def test_registry_prevents_duplicate_hypotheses():
    registry = HypothesisRegistry()
    h1 = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose concentration has strong positive association with outcome.",
        features_involved=["Glucose"],
        evidence_from_data="r=0.4666",
        proposed_experiment="Logistic regression"
    )
    is_new = registry.register_hypothesis(h1)
    assert is_new is True
    
    # Same features and hypothesis type in iteration 2
    h1_duplicate = HypothesisRecord(
        hypothesis_id="H1",
        iteration_no=2,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose is strongly positively associated with outcome.",
        features_involved=["Glucose"],
        evidence_from_data="r=0.4666",
        proposed_experiment="Logistic regression"
    )
    assert registry.is_duplicate(h1_duplicate) is True
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_hypothesis_registry.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.hypothesis_registry'`

**Step 3: Implement minimal code**

1. Create `modules/hypothesis_registry.py`:
   - Class `HypothesisRegistry`:
     - Loads existing registry from `ai_scientist_hypotheses.hypotheses` and `HYPOTHESIS_REGISTRY_PATH`.
     - Generates canonical structural fingerprint: `(sorted(features), hypothesis_type)`.
     - Checks duplicate status.
     - Adds new hypotheses tagged with `iteration_no` and `hypothesis_no`.
     - Formats past hypotheses summary string to inject into subsequent LLM prompts so the LLM is explicitly forbidden from repeating existing hypotheses.

2. In `modules/generate_hypotheses_with_llm.py`:
   - Accept optional `iteration_no = 1` and `registry = None`.
   - Pass existing hypotheses to the prompt: *"Do NOT regenerate these existing hypotheses: [...]"*.
   - Parse and wrap hypotheses into `HypothesisRecord` models with `iteration_no` and `hypothesis_id`.
   - Save to `ai_scientist_hypotheses.hypotheses` and `OUTPUT_DIR/generated_hypotheses.json`.

3. Update `README.md` with de-duplication and iterative generation documentation.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_hypothesis_registry.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add modules/hypothesis_registry.py modules/generate_hypotheses_with_llm.py README.md tests/test_hypothesis_registry.py
git commit -m "feat: implement iteration-aware hypothesis registry and deduplication engine"
```

---

### Task 4: Batch 1 — Schema-Driven Hypothesis Validator with Discrepancy Reconciliation
**Files:**
- Create: `modules/hypothesis_validator.py`
- Modify: `README.md`
- Test: `tests/test_hypothesis_validator.py`

**Step 1: Write the failing test**

Create `tests/test_hypothesis_validator.py`:
```python
import pytest
import pandas as pd
from modules.hypothesis_validator import validate_hypotheses
from modules.schemas import ValidationStatus

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Glucose": [120, 140, 90, 180, 110],
        "BMI": [28.5, 32.1, 22.0, 35.4, 26.0],
        "Age": [33, 45, 25, 52, 29],
        "Outcome": [0, 1, 0, 1, 0]
    })

def test_validate_all_hypotheses_without_omission(sample_df):
    raw = [
        {"hypothesis_id": "H1", "hypothesis_type": "individual_dependency", "hypothesis": "Glucose association", "evidence_from_data": "Glucose r=0.46", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H2", "hypothesis_type": "individual_dependency", "hypothesis": "BMI association", "evidence_from_data": "BMI r=0.29", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H3", "hypothesis_type": "feature_combination_synergy", "hypothesis": "Glucose + BMI synergy", "evidence_from_data": "AUC=0.8097", "proposed_experiment": "Model comparison"},
        {"hypothesis_id": "H4", "hypothesis_type": "feature_interaction_moderation", "hypothesis": "Glucose + Age interaction delta = -0.0007, p = 0.0364", "evidence_from_data": "delta=0.0002, p=0.0026", "proposed_experiment": "Interaction test"},
        {"hypothesis_id": "H5", "hypothesis_type": "comparative_or_confounding_relationship", "hypothesis": "Glucose and BMI correlation r=0.22", "evidence_from_data": "r=0.2211", "proposed_experiment": "Mediation analysis"}
    ]
    stage2_mock = {
        "combinations": [{"combination": ["Glucose", "Age"], "interaction_delta": 0.0002, "interaction_p_value": 0.0026}]
    }
    report = validate_hypotheses(raw, sample_df, target="Outcome", stage2_data=stage2_mock, iteration_no=1)
    
    # Must validate ALL 5 hypotheses, none dropped
    assert report.total == 5
    assert len(report.results) == 5
    
    # H4 must be flagged NEEDS_REVIEW with reconciled evidence attached
    h4 = next(r for r in report.results if r.hypothesis_id == "H4")
    assert h4.status == ValidationStatus.NEEDS_REVIEW
    assert h4.reconciled_evidence["interaction_delta"] == 0.0002
    assert h4.reconciled_evidence["interaction_p_value"] == 0.0026
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_hypothesis_validator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.hypothesis_validator'`

**Step 3: Implement minimal code**

1. Create `modules/hypothesis_validator.py`:
   - Validates every incoming hypothesis against `HypothesisRecord`.
   - Checks:
     1. Required fields presence.
     2. Target existence in dataset and binary `{0, 1}` value check.
     3. Supported hypothesis type check.
     4. Feature extraction: dynamically identifies dataset features, verifies they exist in `df.columns` and have valid numeric types.
     5. Evidence consistency check & reconciliation: checks numbers cited in text against Stage 1/Stage 2 ground truth. Conflicting values (e.g. H4 delta -0.0007 vs 0.0002) are flagged as `NEEDS_REVIEW` with reconciled ground-truth metrics stored in `reconciled_evidence`.
   - Saves `ValidationReportRecord` to `OUTPUT_DIR/validated_hypotheses.json`.
   - Synchronizes records to MongoDB database `ai_scientist_validation`, collection `validations`.

2. Update `README.md` to document Batch 1 validator, discrepancy reconciliation, and validation statuses.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_hypothesis_validator.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add modules/hypothesis_validator.py README.md tests/test_hypothesis_validator.py
git commit -m "feat: implement Batch 1 Hypothesis Validator with discrepancy reconciliation"
```

---

### Task 5: Batch 2 — Experiment Registry & Iterative Planner (H5 Adjusted Regression Routing)
**Files:**
- Create: `modules/experiment_planner.py`
- Modify: `README.md`
- Test: `tests/test_experiment_planner.py`

**Step 1: Write the failing test**

Create `tests/test_experiment_planner.py`:
```python
import pytest
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments, EXPERIMENT_REGISTRY
import pandas as pd

def test_experiment_planner_routes_all_five_hypotheses():
    raw = [
        {"hypothesis_id": "H1", "hypothesis_type": "individual_dependency", "hypothesis": "Glucose association", "evidence_from_data": "r=0.46", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H2", "hypothesis_type": "individual_dependency", "hypothesis": "BMI association", "evidence_from_data": "r=0.29", "proposed_experiment": "Logistic regression"},
        {"hypothesis_id": "H3", "hypothesis_type": "feature_combination_synergy", "hypothesis": "Glucose + BMI synergy", "evidence_from_data": "AUC=0.8097", "proposed_experiment": "Model comparison"},
        {"hypothesis_id": "H4", "hypothesis_type": "feature_interaction_moderation", "hypothesis": "Glucose + Age interaction", "evidence_from_data": "delta=0.0002", "proposed_experiment": "Interaction test"},
        {"hypothesis_id": "H5", "hypothesis_type": "comparative_or_confounding_relationship", "hypothesis": "Glucose and BMI correlation r=0.22", "evidence_from_data": "r=0.2211", "proposed_experiment": "Mediation analysis"}
    ]
    df = pd.DataFrame({"Glucose": [100, 120], "BMI": [25, 30], "Age": [30, 40], "Outcome": [0, 1]})
    report = validate_hypotheses(raw, df, target="Outcome", iteration_no=1)
    
    specs = plan_experiments(report, target="Outcome", iteration_no=1)
    
    # Must generate specs for ALL 5 hypotheses
    assert len(specs) == 5
    
    # H5 must be routed to adjusted_regression rather than raw mediation
    e5 = next(s for s in specs if s.hypothesis_id == "H5")
    assert e5.method == "adjusted_regression"
    assert e5.model_type == "statsmodels.Logit"
    assert "routed to adjusted regression" in e5.routing_notes.lower()
```

**Step 2: Run test to verify it fails**

Run:
```powershell
python -m pytest tests/test_experiment_planner.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_planner'`

**Step 3: Implement minimal code**

1. Create `modules/experiment_planner.py`:
   - `EXPERIMENT_REGISTRY` mapping:
     - `individual_dependency` -> `statsmodels.Logit` (odds ratio, 95% CI, p-value, permutation test)
     - `feature_combination_synergy` -> `sklearn.linear_model.LogisticRegression` (5-fold Stratified CV ROC AUC, paired comparison)
     - `feature_interaction_moderation` -> `statsmodels.Logit` (interaction term, likelihood ratio test, bootstrap CI)
     - `comparative_or_confounding_relationship` -> `statsmodels.Logit` (adjusted odds ratio, confounding delta)
   - Routing logic for H5:
     Even if hypothesis proposes mediation, routes explicitly to `adjusted_regression` with clear audit note.
   - Saves all `ExperimentSpecRecord` objects to `OUTPUT_DIR/experiment_specifications.json`.
   - Synchronizes specifications to MongoDB database `ai_scientist_experiments`, collection `specifications`.

2. Update `README.md` to document the Experiment Registry and Batch 2 specifications.

**Step 4: Run test to verify it passes**

Run:
```powershell
python -m pytest tests/test_experiment_planner.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add modules/experiment_planner.py README.md tests/test_experiment_planner.py
git commit -m "feat: implement Batch 2 Experiment Registry and Planner with H5 routing"
```

---

### Task 6: Master Pipeline Orchestration & End-to-End Test
**Files:**
- Modify: `mains.py`
- Modify: `README.md`
- Create: `tests/test_pipeline_e2e.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline_e2e.py`:
```python
import os
import json
import pytest
import pandas as pd
from modules.config import (
    OUTPUT_DIR,
    VALIDATED_HYPOTHESES_PATH,
    EXPERIMENT_SPECS_PATH
)
from modules.hypothesis_validator import validate_hypotheses
from modules.experiment_planner import plan_experiments
from modules.mongo_cluster import MongoClusterManager

def test_full_pipeline_multi_db_and_spec_integrity():
    hypotheses_file = os.path.join(OUTPUT_DIR, "generated_hypotheses.json")
    with open(hypotheses_file, "r") as f:
        hypotheses = json.load(f)
    
    df = pd.read_csv("diabetes.csv")
    report = validate_hypotheses(hypotheses, df, target="Outcome", iteration_no=1)
    assert report.total == 5
    
    specs = plan_experiments(report, target="Outcome", iteration_no=1)
    assert len(specs) == 5
    
    # Check JSON files exist
    assert os.path.exists(VALIDATED_HYPOTHESES_PATH)
    assert os.path.exists(EXPERIMENT_SPECS_PATH)
    
    # If MongoDB is running on port 27017, verify documents in separate DBs
    cluster = MongoClusterManager()
    if cluster.is_available():
        val_count = cluster.get_collection("ai_scientist_validation", "validations").count_documents({})
        exp_count = cluster.get_collection("ai_scientist_experiments", "specifications").count_documents({})
        assert val_count >= 5
        assert exp_count >= 5
```

**Step 2: Run test to verify it fails/passes**

Run:
```powershell
python -m pytest tests/test_pipeline_e2e.py -v
```

**Step 3: Update `mains.py`**

In `mains.py`:
1. Use `load_dataset_with_id` to generate `input_id`.
2. Pass `input_id` into `save_patterns_with_ids`.
3. Call `validate_hypotheses` with `stage1_data`, `stage2_data`, and `iteration_no=1`.
4. Call `plan_experiments` with `validation_report` and `iteration_no=1`.
5. Output detailed summary of validation statuses, reconciled values, and planned experiment specifications.
6. Display multi-database cluster sync status.

**Step 4: Run full test suite and `mains.py`**

Run:
```powershell
python -m pytest -v
python mains.py
```
Expected: All tests pass and `mains.py` completes with full multi-DB sync.

**Step 5: Final documentation & commit**

Update `README.md` with:
- Full lifecycle documentation (Stages 1 through 5).
- Multi-database MongoDB cluster guide.
- Iterative hypothesis generation and deduplication instructions.

```bash
git add mains.py README.md tests/test_pipeline_e2e.py
git commit -m "feat: complete end-to-end integration of iterative validation and experiment engine"
```

---

## Verification Plan

### Automated Tests
1. `python -m pytest tests/test_schemas.py -v`
2. `python -m pytest tests/test_mongo_cluster.py -v`
3. `python -m pytest tests/test_input_and_pattern_tracking.py -v`
4. `python -m pytest tests/test_hypothesis_registry.py -v`
5. `python -m pytest tests/test_hypothesis_validator.py -v`
6. `python -m pytest tests/test_experiment_planner.py -v`
7. `python -m pytest tests/test_pipeline_e2e.py -v`
8. Full regression test suite:
   ```powershell
   python -m pytest -v
   ```

### End-to-End Pipeline Verification
Run `python mains.py` and verify:
1. `input_id` generated and saved to MongoDB `ai_scientist_inputs.inputs`.
2. Discovered patterns tagged with `pattern_id` and saved to `ai_scientist_patterns.patterns`.
3. All 5 hypotheses processed and registered without omission in `ai_scientist_hypotheses.hypotheses`.
4. Stage 4 validation checks all 5 hypotheses:
   - H1-H3 marked `VALIDATED`
   - H4 marked `NEEDS_REVIEW` with reconciled interaction delta (0.0002) and p-value (0.0026)
   - Recorded in `ai_scientist_validation.validations`
5. Stage 5 experiment planning generates 5 `ExperimentSpecRecord` objects:
   - H5 routed to `adjusted_regression` with audit notes
   - Recorded in `ai_scientist_experiments.specifications`
6. Verify output files in `ai_scientist_output/`:
   - `validated_hypotheses.json`
   - `experiment_specifications.json`
   - `hypothesis_registry.json`
7. Check MongoDB collections on localhost:27017 across all 5 databases.
