# Batch 6 & Batch 7 — Autonomous Experiment Controller & Experiment Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Autonomous Experiment Controller & Selector (Batch 6) and MongoDB Experiment Memory (Batch 7) to run a self-directed, closed-loop AI Scientist that iteratively selects hypotheses, prevents duplicate experiments, enforces stopping budgets, tracks scientific lineage, and synthesizes a final publication-grade scientific report.

**Architecture:** 
- `modules/experiment_memory.py` (Batch 7): Centralized experiment memory maintaining normalized configuration fingerprints (`sorted_features + model + target + metric`), cross-iteration lineage, and synchronized MongoDB storage across `ai_scientist_experiments`, `ai_scientist_results`, and `ai_scientist_memory`.
- `modules/experiment_selector.py` (Batch 6 Component A): LLM Experiment Selector that reasons over past findings, unexplored feature combinations, and prior synergy gains to prioritize the most promising follow-up hypotheses.
- `modules/experiment_controller.py` (Batch 6 Component B): The governing authority enforcing plan validation, normalized duplicate detection, retry limits for hallucinated features/models, stopping condition monitoring, autonomous loop lifecycle orchestration, and final report compilation (`final_scientific_report.md` / `.json`).
- Pipeline loop in `mains.py`: Integrates autonomous multi-iteration execution driving the complete empirical discovery cycle ($\text{Evidence} \rightarrow \text{Hypothesis} \rightarrow \text{Experiment} \rightarrow \text{Result} \rightarrow \text{Interpretation} \rightarrow \text{New Hypothesis}$).

**Tech Stack:** Python 3.13, Pandas, NumPy, Scikit-learn, SciPy, Statsmodels, Requests, Pydantic v2, PyMongo, Ollama (`qwen3:4b`), Pytest.

---

### Task 1: Schema Updates & Centralized Configuration for Autonomous Loop and Memory

**Files:**
- Modify: `modules/schemas.py`
- Modify: `modules/config.py`
- Test: `tests/test_schemas.py`

**Step 1: Write the failing test**
In `tests/test_schemas.py`, add `test_autonomous_loop_and_memory_schemas`:
```python
def test_autonomous_loop_and_memory_schemas():
    from modules.schemas import (
        StoppingConditionConfig,
        AutonomousStateRecord,
        NormalizedExperimentFingerprint,
        FinalScientificReportRecord,
        RefinedHypothesisRecord,
        HypothesisRecord
    )
    stop_cfg = StoppingConditionConfig(
        max_experiments=15,
        max_iterations=3,
        max_runtime_minutes=20,
        no_improvement_tolerance=3,
        target_roc_auc=0.85
    )
    assert stop_cfg.max_experiments == 15

    fp = NormalizedExperimentFingerprint(
        features=["Glucose", "BMI"],
        model="LogisticRegression",
        target="Outcome",
        metric="ROC_AUC"
    )
    assert fp.fingerprint == "BMI,Glucose|LogisticRegression|Outcome|ROC_AUC"

    state = AutonomousStateRecord(
        run_id="RUN-20261106-001",
        experiments_completed=4,
        iterations_completed=1,
        best_score=0.8178,
        best_features=["Glucose", "BMI"],
        best_model="LogisticRegression",
        status="RUNNING"
    )
    assert state.experiments_completed == 4

    refined = RefinedHypothesisRecord(
        hypothesis_id="iter2_H1",
        parent_hypothesis_id="iter1_H1",
        iteration_no=2,
        hypothesis="Glucose + Age interact non-linearly.",
        rationale="Strong interaction seen in iter1.",
        proposed_experiment={"features": ["Glucose", "Age"], "model": "statsmodels.Logit", "metric": "AUC"},
        features_involved=["Glucose", "Age"]
    )
    parent_adapted = refined.to_hypothesis_record()
    assert isinstance(parent_adapted, HypothesisRecord)
    assert parent_adapted.hypothesis_id == "iter2_H1"
    assert parent_adapted.features_involved == ["Glucose", "Age"]
    assert parent_adapted.iteration_no == 2
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py::test_autonomous_loop_and_memory_schemas -v`
Expected: FAIL with `ImportError: cannot import name 'StoppingConditionConfig'`

**Step 3: Write minimal implementation**
1. In `modules/schemas.py`:
   - Define `StoppingConditionConfig` (Pydantic model: `max_experiments: int = 15`, `max_iterations: int = 3`, `max_runtime_minutes: int = 20`, `no_improvement_tolerance: int = 3`, `target_roc_auc: float = 0.85`, `min_meaningful_gain: float = 0.005`).
   - Define `NormalizedExperimentFingerprint` (computes sorted feature string + model + target + metric).
   - Define `AutonomousStateRecord` (tracks progress: `run_id`, `experiments_completed`, `iterations_completed`, `consecutive_no_improvement`, `best_score`, `best_features`, `best_model`, `history`, `status`, `stopping_reason`).
   - Add `to_hypothesis_record(self) -> HypothesisRecord` method to `RefinedHypothesisRecord` ensuring child hypotheses seamlessly adapt into standard `HypothesisRecord` objects for subsequent iterations.
   - Define `FinalScientificReportRecord` (contains title, overview, dataset_profile, confirmed_hypotheses, refuted_hypotheses, baseline_rankings, top_combination, lineage_tree, limitations, conclusions).
2. In `modules/config.py`:
   - Add `DB_MEMORY = "ai_scientist_memory"`
   - Add `AUTONOMOUS_STATE_PATH = os.path.join(OUTPUT_DIR, "autonomous_state.json")`
   - Add `FINAL_REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "final_scientific_report.md")`
   - Add `FINAL_REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "final_scientific_report.json")`
   - Add `MAX_AUTONOMOUS_EXPERIMENTS = 15`, `MAX_AUTONOMOUS_ITERATIONS = 3`, `MAX_RUNTIME_MINUTES = 20`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add modules/schemas.py modules/config.py tests/test_schemas.py
```

---

### Task 2: Batch 7 — MongoDB Experiment Memory & Normalized Duplicate Prevention

**Files:**
- Create: `modules/experiment_memory.py`
- Test: `tests/test_experiment_memory.py`

**Step 1: Write the failing test**
In `tests/test_experiment_memory.py`:
```python
import os
import pytest
from modules.schemas import (
    ExperimentSpecRecord,
    ExperimentResultRecord,
    HypothesisRecord,
    StandardExperimentResult
)
from modules.experiment_memory import ExperimentMemory

def test_experiment_memory_duplicate_detection(tmp_path):
    mem_file = os.path.join(tmp_path, "test_memory.json")
    memory = ExperimentMemory(storage_path=mem_file)

    spec1 = ExperimentSpecRecord(
        spec_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    assert not memory.is_duplicate_experiment(spec1)
    memory.register_experiment_spec(spec1)
    assert memory.is_duplicate_experiment(spec1)

    # Different feature order must still be detected as duplicate!
    spec_reversed = ExperimentSpecRecord(
        spec_id="iter2_E1",
        hypothesis_id="iter2_H1",
        iteration_no=2,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["BMI", "Glucose"],
        target="Outcome"
    )
    assert memory.is_duplicate_experiment(spec_reversed)

    # Different model or target is distinct
    spec_diff_model = ExperimentSpecRecord(
        spec_id="iter2_E2",
        hypothesis_id="iter2_H2",
        iteration_no=2,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="statsmodels.Logit",
        features=["Glucose", "BMI"],
        target="Outcome"
    )
    assert not memory.is_duplicate_experiment(spec_diff_model)

def test_experiment_memory_history_and_querying(tmp_path):
    mem_file = os.path.join(tmp_path, "test_memory.json")
    memory = ExperimentMemory(storage_path=mem_file)

    h1 = HypothesisRecord(
        hypothesis_id="iter1_H1",
        iteration_no=1,
        hypothesis_type="individual_dependency",
        hypothesis="Glucose associates with Outcome.",
        features_involved=["Glucose"]
    )
    memory.save_hypothesis(h1)

    res1 = StandardExperimentResult(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        status="completed",
        experiment_type="individual_dependency",
        configuration={"features": ["Glucose"], "model": "statsmodels.Logit"},
        results={"odds_ratio": 3.46, "roc_auc": 0.79},
        statistical_validation={"p_value": 0.0001},
        evaluation={"verdict": "supported_by_current_evidence"}
    )
    memory.save_experiment_result(res1)

    history = memory.get_all_experiment_results()
    assert len(history) == 1
    assert memory.get_best_result() is not None
    assert memory.get_best_result()["results"]["roc_auc"] == 0.79
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_memory'`

**Step 3: Write minimal implementation**
In `modules/experiment_memory.py`:
- Implement `ExperimentMemory` class:
  - `__init__(storage_path: Optional[str] = None)`
  - `compute_normalized_fingerprint(features: List[str], model: str, target: str, metric: str = "ROC_AUC") -> str`:
    Normalizes feature names (strips whitespace, case-preserved, sorted alphabetically), joins with model, target, metric.
  - `is_duplicate_experiment(spec: Union[ExperimentSpecRecord, Dict[str, Any]]) -> bool`:
    Checks local set of fingerprints and queries MongoDB (`ai_scientist_experiments.specifications` and `ai_scientist_memory.experiments`) if available.
  - `register_experiment_spec(spec: ExperimentSpecRecord) -> None`:
    Stores specification, tracks fingerprint, syncs to MongoDB.
  - `save_hypothesis(hypothesis: HypothesisRecord) -> None`:
    Syncs to memory cache and MongoDB `ai_scientist_memory.hypotheses`.
  - `save_experiment_result(result: StandardExperimentResult) -> None`:
    Records result, tracks best AUC and top features, syncs to MongoDB `ai_scientist_memory.results`.
  - `get_best_result() -> Optional[Dict[str, Any]]`:
    Returns highest scoring model and feature set across all completed experiments.
  - `get_summary_stats() -> Dict[str, Any]`:
    Returns total experiments, supported count, refuted count, best AUC.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_memory.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add modules/experiment_memory.py tests/test_experiment_memory.py
```

---

### Task 3: Batch 6 Component A — LLM Experiment Selector & Candidate Prioritizer

**Files:**
- Create: `modules/experiment_selector.py`
- Test: `tests/test_experiment_selector.py`

**Step 1: Write the failing test**
In `tests/test_experiment_selector.py`:
```python
import pytest
from modules.schemas import HypothesisRecord, RefinedHypothesisRecord
from modules.experiment_selector import rank_and_select_candidates

def test_rank_and_select_candidates_prioritizes_synergies():
    pool = [
        RefinedHypothesisRecord(
            hypothesis_id="iter2_H1",
            parent_hypothesis_id="iter1_H1",
            iteration_no=2,
            hypothesis="Glucose and Insulin interact.",
            features_involved=["Glucose", "Insulin"],
            proposed_experiment={"features": ["Glucose", "Insulin"], "model": "statsmodels.Logit"}
        ),
        RefinedHypothesisRecord(
            hypothesis_id="iter2_H2",
            parent_hypothesis_id="iter1_H3",
            iteration_no=2,
            hypothesis="Glucose, BMI, and Age yield 3-feature predictive synergy.",
            features_involved=["Glucose", "BMI", "Age"],
            proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "sklearn.linear_model.LogisticRegression"}
        )
    ]
    # Given dataset columns:
    valid_cols = ["Glucose", "BMI", "Age", "Pregnancies", "Outcome"]
    ranked = rank_and_select_candidates(
        candidate_pool=pool,
        valid_columns=valid_cols,
        top_k=2
    )
    assert len(ranked) >= 1
    assert ranked[0].hypothesis_id in ("iter2_H1", "iter2_H2")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_selector'`

**Step 3: Write minimal implementation**
In `modules/experiment_selector.py`:
- `score_candidate_promise(hypothesis: Union[HypothesisRecord, RefinedHypothesisRecord], valid_columns: List[str], memory_stats: Dict[str, Any]) -> float`:
  - Penalizes missing/hallucinated features (score = 0 if any feature not in `valid_columns`).
  - Rewards features linked to previous top performers (e.g. if Glucose or BMI had highest baseline AUC).
  - Rewards multi-feature synergy exploration (2 or 3 features).
  - Rewards hypotheses with actionable model specifications.
- `rank_and_select_candidates(candidate_pool: List[Any], valid_columns: List[str], memory_stats: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Any]`:
  - Filters out unviable candidates (e.g., features not in dataset columns).
  - Sorts remaining candidate hypotheses by promise score descending.
  - Returns top K candidate hypotheses.
- `prompt_llm_for_alternative_experiment(rejected_hypothesis: Any, reason: str, valid_columns: List[str], model: str, url: str) -> Optional[RefinedHypothesisRecord]`:
  - If a plan was rejected (e.g. duplicate or hallucinated columns), prompts Ollama with valid columns to provide a corrected, novel alternative.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_selector.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add modules/experiment_selector.py tests/test_experiment_selector.py
```

---

### Task 4: Batch 6 Component B — Experiment Controller & Stopping Conditions

**Files:**
- Create: `modules/experiment_controller.py`
- Test: `tests/test_experiment_controller.py`

**Step 1: Write the failing test**
In `tests/test_experiment_controller.py`:
```python
import os
import pytest
import pandas as pd
from modules.schemas import StoppingConditionConfig, AutonomousStateRecord, ExperimentSpecRecord
from modules.experiment_controller import ExperimentController
from modules.experiment_memory import ExperimentMemory

def test_controller_stopping_conditions(tmp_path):
    out_dir = str(tmp_path)
    cfg = StoppingConditionConfig(
        max_experiments=3,
        max_iterations=2,
        no_improvement_tolerance=2
    )
    mem = ExperimentMemory(storage_path=os.path.join(out_dir, "mem.json"))
    controller = ExperimentController(
        config=cfg,
        memory=mem,
        output_dir=out_dir
    )

    should_stop, reason = controller.should_stop()
    assert not should_stop

    controller.state.experiments_completed = 3
    should_stop, reason = controller.should_stop()
    assert should_stop
    assert "Maximum experiments limit" in reason

def test_controller_validates_and_rejects_hallucinated_features(tmp_path):
    df = pd.DataFrame({"Glucose": [100, 120], "BMI": [25, 30], "Outcome": [0, 1]})
    controller = ExperimentController(
        config=StoppingConditionConfig(max_experiments=5),
        df=df,
        target="Outcome",
        output_dir=str(tmp_path)
    )
    valid_spec = ExperimentSpecRecord(
        spec_id="test_E1",
        hypothesis_id="test_H1",
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome"
    )
    is_valid, msg = controller.validate_experiment_spec(valid_spec)
    assert is_valid

    invalid_spec = ExperimentSpecRecord(
        spec_id="test_E2",
        hypothesis_id="test_H2",
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["NonExistentMetabolite"],
        target="Outcome"
    )
    is_valid, msg = controller.validate_experiment_spec(invalid_spec)
    assert not is_valid
    assert "not found in dataset" in msg
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_controller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_controller'`

**Step 3: Write minimal implementation**
In `modules/experiment_controller.py`:
- `ExperimentController`:
  - `__init__(config: StoppingConditionConfig, memory: Optional[ExperimentMemory] = None, df: Optional[pd.DataFrame] = None, target: str = "Outcome", output_dir: Optional[str] = None)`
  - `validate_experiment_spec(spec: ExperimentSpecRecord) -> Tuple[bool, str]`:
    Checks target existence, binary target validity, column membership in `df.columns`, supported method/model type.
  - `should_stop() -> Tuple[bool, str]`:
    Checks:
    1. `experiments_completed >= config.max_experiments`
    2. `iterations_completed >= config.max_iterations`
    3. `elapsed_minutes >= config.max_runtime_minutes`
    4. `consecutive_no_improvement >= config.no_improvement_tolerance`
    5. `best_score >= config.target_roc_auc`
    6. `exhausted_candidates`
  - `update_progress(result: StandardExperimentResult) -> None`:
    Updates best AUC, checks whether `gain >= config.min_meaningful_gain`, updates `consecutive_no_improvement`, saves `autonomous_state.json`.
  - `generate_final_report() -> Tuple[str, Dict[str, Any]]`:
    Synthesizes markdown publication report (`final_scientific_report.md`) and JSON summary (`final_scientific_report.json`), summarizing all hypotheses tested, supported vs refuted ratio, top predictive model, baseline comparison rankings, and clinical conclusions.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_controller.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add modules/experiment_controller.py tests/test_experiment_controller.py
```

---

### Task 5: Autonomous Discovery Loop Orchestrator & Final Scientific Report

**Files:**
- Modify: `modules/experiment_controller.py`
- Test: `tests/test_autonomous_loop.py`

**Step 1: Write the failing test**
In `tests/test_autonomous_loop.py`:
```python
import os
import json
import pytest
import pandas as pd
from modules.schemas import StoppingConditionConfig
from modules.experiment_controller import run_autonomous_scientist

def test_autonomous_scientist_loop_executes_and_generates_final_report(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    df = pd.read_csv("diabetes.csv")
    
    class MockOllamaResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": json.dumps({
                    "hypothesis_status": "supported_by_current_evidence",
                    "interpretation": "Combination shows significant predictive synergy.",
                    "observations": ["ROC-AUC improved", "p-value < 0.05"],
                    "limitations": ["Requires external validation"],
                    "next_hypothesis": "Glucose, BMI, and Age together provide highest discrimination.",
                    "proposed_experiment": {
                        "features": ["Glucose", "BMI", "Age"],
                        "model": "sklearn.linear_model.LogisticRegression",
                        "metric": "ROC_AUC"
                    }
                })
            }
    import requests
    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockOllamaResponse())

    config = StoppingConditionConfig(
        max_experiments=3,
        max_iterations=2,
        no_improvement_tolerance=2
    )

    initial_hypotheses = [
        {
            "hypothesis_id": "iter1_H1",
            "iteration_no": 1,
            "hypothesis_type": "individual_dependency",
            "hypothesis": "Glucose positively associates with Outcome.",
            "features_involved": ["Glucose"],
            "evidence_from_data": "Pearson r = 0.4665",
            "proposed_experiment": "Logistic regression on Glucose."
        },
        {
            "hypothesis_id": "iter1_H2",
            "iteration_no": 1,
            "hypothesis_type": "feature_combination_synergy",
            "hypothesis": "Glucose and BMI jointly predict diabetes.",
            "features_involved": ["Glucose", "BMI"],
            "evidence_from_data": "Synergy gain = 0.0223",
            "proposed_experiment": "Paired CV comparison."
        }
    ]

    report_md, report_data = run_autonomous_scientist(
        initial_hypotheses=initial_hypotheses,
        df=df,
        target="Outcome",
        config=config,
        output_dir=out_dir,
        n_permutations=20,
        n_bootstrap=20
    )

    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.md"))
    assert os.path.exists(os.path.join(out_dir, "final_scientific_report.json"))
    assert os.path.exists(os.path.join(out_dir, "autonomous_state.json"))
    assert report_data["total_experiments"] >= 2
    assert "Executive Summary" in report_md or "Confirmed Discoveries" in report_md
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_autonomous_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_autonomous_scientist'`

**Step 3: Write minimal implementation**
In `modules/experiment_controller.py`:
- Implement `run_autonomous_scientist(...)`:
  1. Initialize `ExperimentMemory`, `ExperimentController`, and candidate hypothesis pool with `initial_hypotheses`.
  2. Enter the while loop while not `controller.should_stop()`:
     - Select highest-ranked candidate using `rank_and_select_candidates`.
     - Validate candidate hypothesis using `validate_hypotheses`.
     - If rejected or invalid: retry or fetch next candidate.
     - Plan experiment specification using `plan_experiments`.
     - Check duplicate: if `memory.is_duplicate_experiment(spec)`, reject and request alternative or advance queue.
     - Execute experiment using `execute_all_experiments`.
     - Evaluate result using `evaluate_all_results`.
     - Interpret result & generate refined next hypothesis using `interpret_and_refine_all`.
     - Sync all records to `ExperimentMemory` and MongoDB collections.
     - Update controller state & loop metrics (`best_score`, `consecutive_no_improvement`).
     - Feed refined hypotheses into the candidate pool for the next iteration.
  3. Compile final comprehensive report:
     - Writes `final_scientific_report.md` (structured publication format: Executive Summary, Discovery Pipeline, Tested Hypotheses & Evidence Verdicts, Top Predictive Combinations Table, Limitations, and Lineage Graph).
     - Writes `final_scientific_report.json`.
     - Persists final report record to MongoDB `ai_scientist_memory.final_reports`.
     - Returns `report_md` and `report_data`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_autonomous_loop.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add modules/experiment_controller.py tests/test_autonomous_loop.py
```

---

### Task 6: Pipeline Integration in `mains.py` and CLI Options

**Files:**
- Modify: `mains.py`
- Modify: `README.md`
- Test: `tests/test_pipeline_e2e.py`

**Step 1: Write the failing test**
In `tests/test_pipeline_e2e.py`, add assertion verifying that the autonomous loop controller executes successfully and outputs `final_scientific_report.md`.

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_e2e.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
1. In `mains.py`:
   - Add CLI argument `--autonomous` (defaulting to standard single-pass or multi-iteration autonomous loop mode).
   - Wire `run_autonomous_scientist` when running in autonomous mode.
   - Display clear progress banner showing current iteration, experiment count, best ROC-AUC, and stopping criteria.
   - Print path to `final_scientific_report.md` on completion.
2. In `README.md`:
   - Document Batch 6 (Autonomous Loop & Experiment Controller) and Batch 7 (MongoDB Experiment Memory).
   - Document `final_scientific_report.md` and `final_scientific_report.json`.
   - Add stopping condition configurations table.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_e2e.py -v`
Expected: PASS

**Step 5: Staging**
```bash
git add mains.py README.md tests/test_pipeline_e2e.py
```

---

### Task 7: Full Test Suite Verification & Live Validation

**Step 1: Run complete test suite**
Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: All 45+ unit and integration tests PASS.

**Step 2: Run live autonomous execution**
Run: `.\.venv\Scripts\python.exe mains.py --autonomous`
Expected: Full autonomous loop runs across iterations, enforces duplicate prevention, checks stopping conditions, syncs to MongoDB memory, and generates `final_scientific_report.md`.
