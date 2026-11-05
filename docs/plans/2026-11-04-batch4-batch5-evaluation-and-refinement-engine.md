# Batch 4 & Batch 5 — Result Evaluation and LLM Interpretation Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Result Evaluation Engine (Batch 4) to assess statistical significance, uncertainty, and baseline superiority across all executed experiments, and the LLM Interpretation & Hypothesis Refinement Engine (Batch 5) to reason over experimental findings, learn from outcomes, and generate testable follow-up hypotheses for subsequent iterations.

**Architecture:** 
- `modules/result_evaluator.py`: Standardizes every experiment output into a common outer evaluation schema (`StandardExperimentResult`), implements dedicated evaluators (H1/H2 odds ratios & CIs, H3/H4 synergy gains & paired t-tests, H5 interaction LRT & bootstrap CIs, H6 adjusted regression deltas), and builds a cross-experiment baseline comparison matrix.
- `modules/llm_interpreter.py`: Prompts the local LLM (`qwen3:4b` on Ollama) with structured evidence, configurations, and results to generate rigorous interpretations, observations, limitations, and refined follow-up hypotheses adhering to the JSON contract.
- Persistent MongoDB storage segregates data on port 27017: `ai_scientist_evaluations` (collection `evaluations`) and `ai_scientist_refinements` (collections `interpretations` and `refined_hypotheses`).

**Tech Stack:** Python 3.13, Pandas, NumPy, Scikit-learn, SciPy, Statsmodels, Requests, Pydantic v2, PyMongo, Ollama (`qwen3:4b`), Pytest.

---

### Task 1: Schema & Central Configuration Updates
**Files:**
- Modify: `modules/schemas.py`
- Modify: `modules/config.py`
- Test: `tests/test_schemas.py`

**Step 1: Write the failing test**
In `tests/test_schemas.py`:
```python
def test_evaluation_and_refinement_schemas():
    from modules.schemas import (
        StandardExperimentResult,
        CrossExperimentComparisonRecord,
        LLMInterpretationRecord,
        RefinedHypothesisRecord
    )
    eval_rec = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={
            "features": ["Glucose", "BMI"],
            "model": "LogisticRegression"
        },
        results={
            "individual_auc": {"Glucose": 0.79, "BMI": 0.686},
            "combined_auc": 0.817,
            "gain": 0.0274
        },
        statistical_validation={
            "comparison_method": "paired_cross_validation",
            "confidence_interval": [0.012, 0.043],
            "p_value": 0.0026
        },
        evaluation={
            "verdict": "supported_by_current_evidence",
            "is_statistically_significant": True,
            "practical_significance": "meaningful",
            "uncertainty_level": "low",
            "summary": "Combining Glucose and BMI significantly improved ROC-AUC by 0.0274."
        }
    )
    assert eval_rec.experiment_id == "iter1_E3"
    assert eval_rec.results["gain"] == 0.0274
    assert eval_rec.evaluation["is_statistically_significant"] is True

    interp_rec = LLMInterpretationRecord(
        interpretation_id="INT-iter1_E3",
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_status="supported_by_current_evidence",
        interpretation="The combined feature model outperformed individual feature models.",
        observations=["Glucose was the strongest predictor.", "BMI added complementary information."],
        limitations=["Causal mediation between Glucose and BMI has not been established."],
        next_hypothesis="Adding Age to Glucose and BMI will further improve diabetes prediction.",
        proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}
    )
    assert interp_rec.hypothesis_status == "supported_by_current_evidence"
    assert "Glucose" in interp_rec.proposed_experiment["features"]
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py::test_evaluation_and_refinement_schemas -v`
Expected: FAIL with `cannot import name 'StandardExperimentResult'`

**Step 3: Implement minimal code**
In `modules/schemas.py`:
- Add `StandardExperimentResult` with fields:
  - `experiment_id: str`
  - `hypothesis_id: str`
  - `iteration_no: int = 1`
  - `status: str = "completed"`
  - `experiment_type: str`
  - `configuration: Dict[str, Any]`
  - `results: Dict[str, Any]`
  - `statistical_validation: Dict[str, Any]`
  - `evaluation: Dict[str, Any]`
  - `timestamp: str`
- Add `CrossExperimentComparisonRecord` with fields:
  - `comparison_id: str`
  - `iteration_no: int = 1`
  - `individual_baselines: Dict[str, float]`
  - `combination_scores: Dict[str, float]`
  - `synergy_gains: Dict[str, float]`
  - `ranking: List[Dict[str, Any]]`
  - `summary_findings: List[str]`
- Add `LLMInterpretationRecord` with fields:
  - `interpretation_id: str`
  - `experiment_id: str`
  - `hypothesis_id: str`
  - `iteration_no: int = 1`
  - `hypothesis_status: str`
  - `interpretation: str`
  - `observations: List[str]`
  - `limitations: List[str]`
  - `next_hypothesis: str`
  - `proposed_experiment: Dict[str, Any]`
  - `timestamp: str`
- Add `RefinedHypothesisRecord` with fields:
  - `hypothesis_id: str`
  - `parent_hypothesis_id: str`
  - `iteration_no: int`
  - `hypothesis: str`
  - `rationale: str`
  - `proposed_experiment: Dict[str, Any]`
  - `features_involved: List[str]`

In `modules/config.py`:
- Add `DB_EVALUATIONS = "ai_scientist_evaluations"`
- Add `DB_REFINEMENTS = "ai_scientist_refinements"`
- Add `EXPERIMENT_EVALUATIONS_PATH = os.path.join(OUTPUT_DIR, "experiment_evaluations.json")`
- Add `CROSS_EXPERIMENT_COMPARISON_PATH = os.path.join(OUTPUT_DIR, "cross_experiment_comparison.json")`
- Add `LLM_INTERPRETATIONS_PATH = os.path.join(OUTPUT_DIR, "llm_interpretations.json")`
- Add `REFINED_HYPOTHESES_PATH = os.path.join(OUTPUT_DIR, "refined_next_hypotheses.json")`

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py::test_evaluation_and_refinement_schemas -v`
Expected: PASS

---

### Task 2: Standard Result Structure Converter & Individual Evaluators (Batch 4.3 & 4.4)
**Files:**
- Create: `modules/result_evaluator.py`
- Test: `tests/test_result_evaluator.py`

**Step 1: Write the failing test**
In `tests/test_result_evaluator.py`:
```python
def test_evaluate_individual_dependency_result():
    from modules.schemas import ExperimentResultRecord
    from modules.result_evaluator import evaluate_experiment_result
    raw_res = ExperimentResultRecord(
        result_id="RES-iter1_E1",
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        metrics={
            "coefficient": 1.242,
            "odds_ratio": 3.46,
            "confidence_interval": [2.82, 4.23],
            "p_value": 0.00001,
            "permutation_p_value": 0.000,
            "mean_auc": 0.790,
            "std_auc": 0.021
        },
        details={"fold_auc": [0.78, 0.81, 0.77, 0.80, 0.79]},
        verdict="SUPPORTED",
        summary="Glucose strongly supported"
    )
    standard_res = evaluate_experiment_result(raw_res)
    assert standard_res.experiment_id == "iter1_E1"
    assert standard_res.results["odds_ratio"] == 3.46
    assert standard_res.evaluation["verdict"] == "supported_by_current_evidence"
    assert standard_res.evaluation["is_statistically_significant"] is True
    assert standard_res.evaluation["uncertainty_level"] == "low"
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_evaluate_individual_dependency_result -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.result_evaluator'`

**Step 3: Implement minimal code**
In `modules/result_evaluator.py`:
- Implement specialized evaluators for each experiment type:
  - `_evaluate_individual_dependency(result: ExperimentResultRecord) -> StandardExperimentResult`:
    - Checks odds ratio $> 1.0$, $95\%$ CI strictly excludes $1.0$.
    - Checks parametric $p < 0.05$ and permutation $p < 0.05$.
    - Evaluates uncertainty level based on CI width and fold AUC standard deviation.
  - `_evaluate_feature_synergy(result: ExperimentResultRecord) -> StandardExperimentResult`:
    - Checks $\text{gain} > 0$ and paired t-test $p < 0.05$.
    - Sets practical significance: `meaningful` if $\text{gain} \ge 0.02$, `marginal` if $\text{gain} \in (0, 0.02)$, `negligible_or_negative` if $\le 0$.
  - `_evaluate_feature_interaction(result: ExperimentResultRecord) -> StandardExperimentResult`:
    - Checks LRT $\chi^2$ p-value $< 0.05$ and bootstrap 95% CI strictly excluding $0$.
  - `_evaluate_adjusted_regression(result: ExperimentResultRecord) -> StandardExperimentResult`:
    - Assesses coefficient shift $\Delta\%$ against $10\%$ threshold and covariate significance.
  - `evaluate_experiment_result(result: ExperimentResultRecord) -> StandardExperimentResult`:
    - Routes to the appropriate evaluator and returns `StandardExperimentResult`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_evaluate_individual_dependency_result -v`
Expected: PASS

---

### Task 3: Cross-Experiment Baseline Comparison Matrix (Batch 4.5)
**Files:**
- Modify: `modules/result_evaluator.py`
- Test: `tests/test_result_evaluator.py`

**Step 1: Write the failing test**
In `tests/test_result_evaluator.py`:
```python
def test_build_cross_experiment_comparison():
    from modules.schemas import ExperimentResultRecord
    from modules.result_evaluator import build_cross_experiment_comparison
    raw_results = [
        ExperimentResultRecord(
            result_id="RES-iter1_E1",
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome",
            metrics={"mean_auc": 0.790, "odds_ratio": 3.46, "p_value": 0.0001},
            verdict="SUPPORTED",
            summary="Glucose"
        ),
        ExperimentResultRecord(
            result_id="RES-iter1_E2",
            experiment_id="iter1_E2",
            hypothesis_id="iter1_H2",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["BMI"],
            target="Outcome",
            metrics={"mean_auc": 0.686, "odds_ratio": 2.03, "p_value": 0.0001},
            verdict="SUPPORTED",
            summary="BMI"
        ),
        ExperimentResultRecord(
            result_id="RES-iter1_E3",
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            experiment_type="feature_combination_synergy",
            method="paired_model_comparison",
            model_type="sklearn.linear_model.LogisticRegression",
            features=["Glucose", "BMI"],
            target="Outcome",
            metrics={"combined_mean_auc": 0.817, "gain_over_best_individual": 0.0274, "paired_test_p_value": 0.0026},
            verdict="SUPPORTED",
            summary="Glucose + BMI"
        )
    ]
    comparison = build_cross_experiment_comparison(raw_results, iteration_no=1)
    assert comparison.iteration_no == 1
    assert comparison.individual_baselines["Glucose"] == 0.790
    assert comparison.combination_scores["Glucose + BMI"] == 0.817
    assert comparison.synergy_gains["Glucose + BMI"] == 0.0274
    assert len(comparison.ranking) == 3
    assert comparison.ranking[0]["name"] == "Glucose + BMI"  # Top performer
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_build_cross_experiment_comparison -v`
Expected: FAIL with `cannot import name 'build_cross_experiment_comparison'`

**Step 3: Implement minimal code**
In `modules/result_evaluator.py`:
- Implement `build_cross_experiment_comparison(results: List[ExperimentResultRecord], iteration_no: int = 1) -> CrossExperimentComparisonRecord`:
  - Extracts all individual AUC baselines.
  - Extracts all combination AUC scores and synergy gains.
  - Compares Model A vs Model B vs Model C.
  - Sorts all models into a descending ranking by performance (ROC-AUC).
  - Formulates high-level summary observation statements (e.g. "Combining Glucose and BMI achieves the highest overall predictive performance of 0.817, improving by 0.0274 AUC over top individual baseline Glucose").

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_build_cross_experiment_comparison -v`
Expected: PASS

---

### Task 4: Evaluation Orchestrator & MongoDB Persistence (`ai_scientist_evaluations`)
**Files:**
- Modify: `modules/result_evaluator.py`
- Test: `tests/test_result_evaluator.py`

**Step 1: Write the failing test**
In `tests/test_result_evaluator.py`:
```python
def test_evaluate_all_results_and_save(tmp_path):
    from modules.schemas import ExperimentResultRecord
    from modules.result_evaluator import evaluate_all_results
    sample_results = [
        ExperimentResultRecord(
            result_id="RES-iter1_E1",
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome",
            metrics={"odds_ratio": 3.46, "p_value": 0.0001, "permutation_p_value": 0.000, "mean_auc": 0.790, "confidence_interval": [2.8, 4.2]},
            verdict="SUPPORTED",
            summary="Glucose"
        )
    ]
    eval_list, comparison = evaluate_all_results(
        results=sample_results,
        iteration_no=1,
        output_dir=str(tmp_path)
    )
    assert len(eval_list) == 1
    assert (tmp_path / "experiment_evaluations.json").exists()
    assert (tmp_path / "cross_experiment_comparison.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "experiment_evaluations.json").exists()
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_evaluate_all_results_and_save -v`
Expected: FAIL with `cannot import name 'evaluate_all_results'`

**Step 3: Implement minimal code**
In `modules/result_evaluator.py`:
- Implement `evaluate_all_results(results: List[ExperimentResultRecord], iteration_no: int = 1, output_dir: Optional[str] = None) -> Tuple[List[StandardExperimentResult], CrossExperimentComparisonRecord]`:
  - Evaluates each result into `StandardExperimentResult`.
  - Builds `CrossExperimentComparisonRecord`.
  - Saves JSON files in run directory and iteration archive.
  - Persists evaluated records to MongoDB cluster: `ai_scientist_evaluations.evaluations`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_result_evaluator.py::test_evaluate_all_results_and_save -v`
Expected: PASS

---

### Task 5: LLM Interpretation Engine & Prompt Builder (Batch 5.1, 5.2, 5.3)
**Files:**
- Create: `modules/llm_interpreter.py`
- Test: `tests/test_llm_interpreter.py`

**Step 1: Write the failing test**
In `tests/test_llm_interpreter.py`:
```python
def test_build_interpretation_prompt():
    from modules.schemas import StandardExperimentResult, HypothesisRecord
    from modules.llm_interpreter import build_interpretation_prompt
    std_res = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
        statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
    )
    hypo = HypothesisRecord(
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_type="feature_combination_synergy",
        hypothesis="Combining Glucose and BMI provides greater predictive power for diabetes than either feature alone.",
        features_involved=["Glucose", "BMI"],
        evidence_from_data="Joint MI = 0.18, Synergy Gain = +0.027",
        proposed_experiment="Run 5-fold Stratified CV comparing individual vs combined models."
    )
    prompt = build_interpretation_prompt(std_res, hypo)
    assert "Role" in prompt
    assert "Original Hypothesis" in prompt
    assert "Experiment Configuration" in prompt
    assert "Experimental Results" in prompt
    assert "Instructions" in prompt
    assert "hypothesis_status" in prompt
    assert "next_hypothesis" in prompt
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_build_interpretation_prompt -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.llm_interpreter'`

**Step 3: Implement minimal code**
In `modules/llm_interpreter.py`:
- Implement `build_interpretation_prompt(result: StandardExperimentResult, hypothesis: Optional[HypothesisRecord] = None) -> str`:
  - Formulates the exact prompt structure specified in Batch 5.2:
    - **Role**: You are the interpretation component of an autonomous AI Scientist.
    - **Original Hypothesis**: State hypothesis text and ID.
    - **Experiment Configuration**: Model, features tested, cross-validation setup, evaluation metric.
    - **Experimental Results**: Individual scores, combined score, gain, statistical test p-value, odds ratios, confidence intervals.
    - **Instructions**:
      1. Interpret the experimental findings.
      2. Determine whether the hypothesis is supported, not supported, or inconclusive.
      3. Identify the strongest observations.
      4. Do not invent statistical significance or causal relationships.
      5. Suggest one or more testable follow-up hypotheses.
      6. Propose a suitable next experiment.
    - **JSON Schema Requirement**: Output strictly valid JSON with keys: `hypothesis_status`, `interpretation`, `observations`, `limitations`, `next_hypothesis`, `proposed_experiment`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_build_interpretation_prompt -v`
Expected: PASS

---

### Task 4: LLM Interpretation Execution with Strict Ollama Requirement (Batch 5.1 & 5.2)
**Files:**
- Modify: `modules/llm_interpreter.py`
- Test: `tests/test_llm_interpreter.py`

**Step 1: Write the failing test**
In `tests/test_llm_interpreter.py`:
```python
def test_interpret_experiment_with_mocked_ollama(monkeypatch):
    from modules.schemas import StandardExperimentResult, HypothesisRecord
    from modules.llm_interpreter import interpret_single_experiment
    import requests

    std_res = StandardExperimentResult(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        status="completed",
        experiment_type="feature_combination_synergy",
        configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
        results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
        statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
        evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
    )
    hypo = HypothesisRecord(
        hypothesis_id="iter1_H3",
        iteration_no=1,
        hypothesis_type="feature_combination_synergy",
        hypothesis="Combining Glucose and BMI provides greater predictive power for diabetes.",
        features_involved=["Glucose", "BMI"],
        evidence_from_data="Synergy Gain = +0.027",
        proposed_experiment="5-fold CV"
    )

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": '{"hypothesis_status": "supported_by_current_evidence", "interpretation": "Combined model outperforms baselines.", "observations": ["Glucose is strongest baseline", "BMI adds synergy"], "limitations": ["Requires validation on external cohorts"], "next_hypothesis": "Adding Age to Glucose and BMI will further improve diabetes prediction.", "proposed_experiment": {"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}}'
            }

    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockResponse())

    interp = interpret_single_experiment(std_res, hypo)
    assert interp.hypothesis_status == "supported_by_current_evidence"
    assert interp.experiment_id == "iter1_E3"
    assert "Age" in interp.proposed_experiment["features"]
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_interpret_experiment_with_mocked_ollama -v`
Expected: FAIL with `cannot import name 'interpret_single_experiment'`

**Step 3: Implement minimal code**
In `modules/llm_interpreter.py`:
- Implement `interpret_single_experiment(eval_result: StandardExperimentResult, hypothesis: Optional[HypothesisRecord] = None, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL) -> LLMInterpretationRecord`:
  - Sends formatted prompt to Ollama endpoint.
  - If Ollama is unreachable (`requests.exceptions.ConnectionError`, `Timeout`, or non-200), raises `ConnectionError` as strictly required.
  - Robustly extracts JSON block from Ollama response using regex/json parsing.
  - Validates and returns `LLMInterpretationRecord`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_interpret_experiment_with_mocked_ollama -v`
Expected: PASS

---

### Task 7: Hypothesis Refinement Loop & Registration for Iteration 2 (Batch 5.3 & 5.4)
**Files:**
- Modify: `modules/llm_interpreter.py`
- Test: `tests/test_llm_interpreter.py`

**Step 1: Write the failing test**
In `tests/test_llm_interpreter.py`:
```python
def test_format_and_register_refined_hypotheses(tmp_path):
    from modules.schemas import LLMInterpretationRecord
    from modules.hypothesis_registry import HypothesisRegistry
    from modules.llm_interpreter import register_refined_hypotheses
    interps = [
        LLMInterpretationRecord(
            interpretation_id="INT-iter1_E3",
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            hypothesis_status="supported_by_current_evidence",
            interpretation="Combined model improved performance.",
            observations=["Glucose and BMI synergistic"],
            limitations=["Need Age evaluation"],
            next_hypothesis="Adding Age to Glucose and BMI will further improve diabetes prediction beyond the 0.817 ROC-AUC achieved by Glucose and BMI.",
            proposed_experiment={"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}
        )
    ]
    registry = HypothesisRegistry(registry_path=str(tmp_path / "hypo_reg.json"))
    refined_records = register_refined_hypotheses(interps, next_iteration_no=2, registry=registry)
    assert len(refined_records) == 1
    assert refined_records[0].hypothesis_id == "iter2_H1"
    assert refined_records[0].parent_hypothesis_id == "iter1_H3"
    assert "Age" in refined_records[0].features_involved
    # Verify registered in registry
    assert not registry.is_novel("Adding Age to Glucose and BMI will further improve diabetes prediction beyond the 0.817 ROC-AUC achieved by Glucose and BMI.")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_format_and_register_refined_hypotheses -v`
Expected: FAIL with `cannot import name 'register_refined_hypotheses'`

**Step 3: Implement minimal code**
In `modules/llm_interpreter.py`:
- Implement `register_refined_hypotheses(interpretations: List[LLMInterpretationRecord], next_iteration_no: int = 2, registry: Optional[HypothesisRegistry] = None) -> List[RefinedHypothesisRecord]`:
  - Iterates over all interpretation records.
  - Assigns clean next-iteration identifiers: `iter<next_iteration_no>_H1`, `iter<next_iteration_no>_H2`, etc.
  - Links `parent_hypothesis_id` for lineage tracking.
  - Registers each hypothesis with `HypothesisRegistry` so that future generations check for duplicates.
  - Returns `List[RefinedHypothesisRecord]`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_format_and_register_refined_hypotheses -v`
Expected: PASS

---

### Task 8: Batch 5 Orchestrator & Multi-Database Persistence (`ai_scientist_refinements`)
**Files:**
- Modify: `modules/llm_interpreter.py`
- Test: `tests/test_llm_interpreter.py`

**Step 1: Write the failing test**
In `tests/test_llm_interpreter.py`:
```python
def test_interpret_and_refine_all_experiments(monkeypatch, tmp_path):
    from modules.schemas import StandardExperimentResult, HypothesisRecord
    from modules.llm_interpreter import interpret_and_refine_all
    import requests

    eval_results = [
        StandardExperimentResult(
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            status="completed",
            experiment_type="feature_combination_synergy",
            configuration={"features": ["Glucose", "BMI"], "model": "LogisticRegression"},
            results={"individual_auc": {"Glucose": 0.79, "BMI": 0.686}, "combined_auc": 0.817, "gain": 0.0274},
            statistical_validation={"comparison_method": "paired_cross_validation", "confidence_interval": [0.012, 0.043], "p_value": 0.0026},
            evaluation={"verdict": "supported_by_current_evidence", "is_statistically_significant": True, "practical_significance": "meaningful", "uncertainty_level": "low", "summary": "Gain of 0.0274."}
        )
    ]
    hypotheses = [
        HypothesisRecord(
            hypothesis_id="iter1_H3",
            iteration_no=1,
            hypothesis_type="feature_combination_synergy",
            hypothesis="Glucose and BMI synergy.",
            features_involved=["Glucose", "BMI"],
            evidence_from_data="gain=0.027",
            proposed_experiment="5-fold CV"
        )
    ]

    class MockResponse:
        def raise_for_status(self): pass
        def json(self):
            return {
                "response": '{"hypothesis_status": "supported_by_current_evidence", "interpretation": "Combined model outperforms baselines.", "observations": ["Glucose is strongest baseline"], "limitations": ["Requires external cohort"], "next_hypothesis": "Add Age to Glucose + BMI.", "proposed_experiment": {"features": ["Glucose", "BMI", "Age"], "model": "LogisticRegression", "metric": "ROC_AUC"}}'
            }

    monkeypatch.setattr(requests, "post", lambda url, json, **kwargs: MockResponse())

    interps, refined = interpret_and_refine_all(
        eval_results=eval_results,
        hypotheses=hypotheses,
        iteration_no=1,
        output_dir=str(tmp_path)
    )
    assert len(interps) == 1
    assert len(refined) == 1
    assert (tmp_path / "llm_interpretations.json").exists()
    assert (tmp_path / "refined_next_hypotheses.json").exists()
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_interpret_and_refine_all_experiments -v`
Expected: FAIL with `cannot import name 'interpret_and_refine_all'`

**Step 3: Implement minimal code**
In `modules/llm_interpreter.py`:
- Implement `interpret_and_refine_all(eval_results: List[StandardExperimentResult], hypotheses: List[HypothesisRecord], iteration_no: int = 1, output_dir: Optional[str] = None) -> Tuple[List[LLMInterpretationRecord], List[RefinedHypothesisRecord]]`:
  - Maps `eval_results` to matching original hypotheses.
  - Calls `interpret_single_experiment` for each evaluated experiment.
  - Calls `register_refined_hypotheses` to formulate and register next iteration candidates (`iter<N+1>_H<M>`).
  - Saves `llm_interpretations.json` and `refined_next_hypotheses.json` locally and under `iterations/iter_<N>/`.
  - Persists interpretations to `ai_scientist_refinements.interpretations` and refined hypotheses to `ai_scientist_refinements.refined_hypotheses` in MongoDB cluster.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_llm_interpreter.py::test_interpret_and_refine_all_experiments -v`
Expected: PASS

---

### Task 9: Pipeline Integration in `mains.py`, End-to-End Verification & Documentation
**Files:**
- Modify: `mains.py`
- Modify: `README.md`
- Test: Full pytest suite (`python -m pytest -v`)

**Step 1: Wire Stage 7 & Stage 8 into mains.py**
In `mains.py`:
- After Stage 6 Experiment Execution:
  ```python
  # Stage 7: Batch 4 — Result Evaluation & Cross-Experiment Comparison
  eval_results, cross_comparison = evaluate_all_results(
      results=experiment_results,
      iteration_no=iteration_no,
      output_dir=run_output_dir
  )

  # Stage 8: Batch 5 — LLM Interpretation & Hypothesis Refinement Engine
  interpretations, refined_hypotheses = interpret_and_refine_all(
      eval_results=eval_results,
      hypotheses=hypotheses,
      iteration_no=iteration_no,
      output_dir=run_output_dir
  )
  ```
- Print formatted evaluation findings, statistical significance, comparative baselines, LLM interpretations, limitations, and refined next hypotheses.
- Report cluster persistence to `ai_scientist_evaluations.evaluations` and `ai_scientist_refinements.interpretations`.

**Step 2: Run all unit and integration tests**
Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: All tests pass (100% pass rate).

**Step 3: Execute `mains.py` end-to-end**
Run: `.\.venv\Scripts\python.exe mains.py`
Expected: Full 8-stage automated research lifecycle runs smoothly from cached dataset through LLM hypothesis interpretation and refinement.

**Step 4: Update `README.md`**
Update `README.md` to document Batch 4 (Result Evaluation), Batch 5 (LLM Interpretation & Refinement), standard result outer schemas, multi-database persistence (`ai_scientist_evaluations` and `ai_scientist_refinements`), and the iterative loop.
