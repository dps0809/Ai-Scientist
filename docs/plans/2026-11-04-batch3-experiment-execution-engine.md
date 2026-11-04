# Batch 3 — Experiment Execution Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Batch 3 Experiment Execution to execute structured experiment specifications (`iter<N>_E1` .. `iter<N>_EN`) on prepared data, computing inferential and predictive metrics (odds ratios, 95% CIs, permutation tests, 5-fold CV AUCs, 3-model synergy gains, centered interaction likelihood-ratio tests, and adjusted regression deltas), saving standardized records to disk and a dedicated `ai_scientist_results` MongoDB database.

**Architecture:** A modular execution engine in `modules/experiment_executor.py` driven by `prepare_dataset_for_experiments` and specialized experiment runners for all 4 supported categories in `EXPERIMENT_REGISTRY`. Results adhere to a strict Pydantic contract (`ExperimentResultRecord`), preserving full audit trails across iterations (`iter<N>_E<M>`), with database segregation across the existing port 27017 cluster.

**Tech Stack:** Python 3.13, Pandas, NumPy, Scikit-learn (Pipelines, StratifiedKFold, SimpleImputer, StandardScaler, LogisticRegression, roc_auc_score), SciPy (chi2, stats), Statsmodels (Logit, add_constant), Pydantic v2, PyMongo, Pytest.

---

### Task 1: Schema & Central Configuration Updates
**Files:**
- Modify: `modules/schemas.py`
- Modify: `modules/config.py`
- Test: `tests/test_schemas.py`

**Step 1: Write the failing test**
In `tests/test_schemas.py`:
```python
def test_experiment_result_schema():
    from modules.schemas import ExperimentResultRecord
    record = ExperimentResultRecord(
        result_id="RES-iter1_E1",
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        metrics={"odds_ratio": 1.78, "p_value": 0.0001, "permutation_p_value": 0.001},
        details={"fold_auc": [0.75, 0.78, 0.74, 0.81, 0.76], "mean_auc": 0.768},
        verdict="SUPPORTED",
        summary="Glucose is significantly associated with diabetes odds."
    )
    assert record.result_id == "RES-iter1_E1"
    assert record.metrics["odds_ratio"] == 1.78
    assert record.verdict == "SUPPORTED"
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py::test_experiment_result_schema -v`
Expected: FAIL with `cannot import name 'ExperimentResultRecord'`

**Step 3: Implement minimal code**
In `modules/schemas.py`:
- Add `ExperimentResultRecord` schema with fields:
  - `result_id: str`
  - `experiment_id: str`
  - `hypothesis_id: str`
  - `iteration_no: int = 1`
  - `experiment_type: str`
  - `method: str`
  - `model_type: str`
  - `features: List[str]`
  - `target: str`
  - `metrics: Dict[str, Any]`
  - `details: Dict[str, Any] = Field(default_factory=dict)`
  - `verdict: str` (`SUPPORTED`, `REFUTED`, `INCONCLUSIVE`)
  - `summary: str`
  - `timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())`

In `modules/config.py`:
- Add `DB_RESULTS = "ai_scientist_results"`
- Add `EXPERIMENT_RESULTS_PATH = os.path.join(OUTPUT_DIR, "experiment_results.json")`

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py::test_experiment_result_schema -v`
Expected: PASS

---

### Task 2: Dataset Preparation for Experiment Execution
**Files:**
- Create: `modules/experiment_executor.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_prepare_dataset_replaces_invalid_zeros():
    from modules.experiment_executor import prepare_dataset_for_experiments
    raw_df = pd.DataFrame({
        "Glucose": [0, 120, 140],
        "BMI": [25.0, 0, 30.0],
        "BloodPressure": [70, 80, 0],
        "Insulin": [0, 100, 150],
        "SkinThickness": [20, 0, 30],
        "Pregnancies": [0, 2, 4], # 0 pregnancies is valid observation!
        "Outcome": [0, 1, 1]
    })
    prep_df = prepare_dataset_for_experiments(raw_df)
    assert pd.isna(prep_df.loc[0, "Glucose"])
    assert pd.isna(prep_df.loc[1, "BMI"])
    assert pd.isna(prep_df.loc[2, "BloodPressure"])
    assert pd.isna(prep_df.loc[0, "Insulin"])
    assert pd.isna(prep_df.loc[1, "SkinThickness"])
    assert prep_df.loc[0, "Pregnancies"] == 0 # Preserved!
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_prepare_dataset_replaces_invalid_zeros -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.experiment_executor'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `prepare_dataset_for_experiments(df: pd.DataFrame) -> pd.DataFrame`:
  - Identifies Pima zero-missing columns: `["Glucose", "BMI", "BloodPressure", "Insulin", "SkinThickness"]`.
  - Replaces `0` with `np.nan` only for existing columns in that list, preserving true zeros in columns like `Pregnancies` or binary targets.
  - Returns a clean copy ready for fold-based imputation and standalone analyses.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_prepare_dataset_replaces_invalid_zeros -v`
Expected: PASS

---

### Task 3: Experiment E1 & E2 — Individual Dependency Runners (Statsmodels Logit + Permutation Test + 5-Fold Stratified CV)
**Files:**
- Modify: `modules/experiment_executor.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_run_individual_dependency_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_individual_dependency_experiment
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E1",
        hypothesis_id="iter1_H1",
        iteration_no=1,
        experiment_type="individual_dependency",
        method="logistic_regression",
        model_type="statsmodels.Logit",
        features=["Glucose"],
        target="Outcome",
        parameters={"fit_intercept": True, "maxiter": 100},
        evaluation={"odds_ratio": True, "confidence_interval": True, "permutation_iterations": 100}
    )
    result = run_individual_dependency_experiment(sample_diabetes_df, spec, n_permutations=100)
    assert result.experiment_id == "iter1_E1"
    assert "odds_ratio" in result.metrics
    assert "confidence_interval" in result.metrics
    assert "p_value" in result.metrics
    assert "permutation_p_value" in result.metrics
    assert "mean_auc" in result.details
    assert len(result.details["fold_auc"]) == 5
    assert result.verdict in ("SUPPORTED", "REFUTED")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_individual_dependency_experiment -v`
Expected: FAIL with `cannot import name 'run_individual_dependency_experiment'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `run_individual_dependency_experiment(df: pd.DataFrame, spec: ExperimentSpecRecord, n_permutations: int = 1000) -> ExperimentResultRecord`:
  1. **Inferential Logistic Regression (Statsmodels Logit)**:
     - Drops rows missing target or the single feature.
     - Standardizes predictor using `StandardScaler` for numerical stability.
     - Adds constant term (`sm.add_constant`).
     - Fits `sm.Logit(y, X_model).fit(disp=False)`.
     - Extracts $\beta_1$, odds ratio $e^{\beta_1}$, 95% confidence interval $[e^{\text{CI}_{\text{lower}}}, e^{\text{CI}_{\text{upper}}}]$, and model $p$-value.
  2. **Permutation Test**:
     - Shuffles target labels $N$ times (default 1000, configurable in spec/test, random seed 42).
     - Fits null models and counts proportion of absolute null coefficients $\ge |\beta_{\text{observed}}|$.
     - Computes two-sided permutation $p$-value.
  3. **Predictive Cross-Validation (Scikit-learn StratifiedKFold)**:
     - 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
     - In each fold, fits pipeline `make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))`.
     - Predicts test fold probabilities and computes ROC-AUC.
     - Records `fold_auc`, `mean_auc`, and `std_auc`.
  4. **Verdict Synthesis**:
     - `SUPPORTED` if $p < 0.05$ and odds ratio aligns with positive dependency hypothesis.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_individual_dependency_experiment -v`
Expected: PASS

---

### Task 4: Experiment E3 — Feature Combination Synergy Runner (3 Models, Identical Folds, Gain & Significance)
**Files:**
- Modify: `modules/experiment_executor.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_run_feature_combination_synergy_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_feature_combination_synergy_experiment
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E3",
        hypothesis_id="iter1_H3",
        iteration_no=1,
        experiment_type="feature_combination_synergy",
        method="paired_model_comparison",
        model_type="sklearn.linear_model.LogisticRegression",
        features=["Glucose", "BMI"],
        target="Outcome",
        comparison=[["Glucose"], ["BMI"], ["Glucose", "BMI"]],
        parameters={"solver": "lbfgs", "max_iter": 500},
        evaluation={"metric": "roc_auc", "folds": 5}
    )
    result = run_feature_combination_synergy_experiment(sample_diabetes_df, spec)
    assert result.experiment_id == "iter1_E3"
    assert "combined_mean_auc" in result.metrics
    assert "gain_over_best_individual" in result.metrics
    assert "individual_scores" in result.details
    assert "paired_test_p_value" in result.metrics
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_feature_combination_synergy_experiment -v`
Expected: FAIL with `cannot import name 'run_feature_combination_synergy_experiment'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `run_feature_combination_synergy_experiment(df: pd.DataFrame, spec: ExperimentSpecRecord) -> ExperimentResultRecord`:
  1. Define 3 models: Model A (`feat_a`), Model B (`feat_b`), Model C (`feat_a + feat_b`).
  2. Generate single `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` across `df`.
  3. Evaluate Model A, Model B, and Model C on the **exact same fold splits**:
     - Pipeline: `SimpleImputer(strategy="median")` + `StandardScaler()` + `LogisticRegression(max_iter=1000)`.
     - Collect 5 fold AUCs for each model.
  4. Compute `mean_auc_A`, `mean_auc_B`, `mean_auc_C`.
  5. Compute `gain_over_best_individual = mean_auc_C - max(mean_auc_A, mean_auc_B)`.
  6. Perform paired statistical test (`scipy.stats.ttest_rel` on 5-fold AUC differences between Model C and best individual model).
  7. Assign verdict:
     - `SUPPORTED`: `gain > 0` and paired test $p < 0.05$.
     - `INCONCLUSIVE`: `gain > 0` but paired test $p \ge 0.05$ (practical improvement without strict statistical certainty).
     - `REFUTED`: `gain \le 0`.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_feature_combination_synergy_experiment -v`
Expected: PASS

---

### Task 5: Experiment E4 — Feature Interaction & Moderation Runner (Centered Features, Likelihood-Ratio Test, Bootstrap)
**Files:**
- Modify: `modules/experiment_executor.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_run_feature_interaction_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_feature_interaction_experiment
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E4",
        hypothesis_id="iter1_H4",
        iteration_no=1,
        experiment_type="feature_interaction_moderation",
        method="interaction_regression",
        model_type="statsmodels.Logit",
        features=["Glucose", "Age"],
        target="Outcome",
        parameters={"include_interaction_term": True, "maxiter": 100},
        evaluation={"interaction_term": True, "likelihood_ratio_test": True, "bootstrap_samples": 50}
    )
    result = run_feature_interaction_experiment(sample_diabetes_df, spec, n_bootstrap=50)
    assert result.experiment_id == "iter1_E4"
    assert "interaction_coefficient" in result.metrics
    assert "interaction_p_value" in result.metrics
    assert "lrt_statistic" in result.metrics
    assert "lrt_p_value" in result.metrics
    assert "bootstrap_ci" in result.metrics
    assert result.verdict in ("SUPPORTED", "REFUTED")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_feature_interaction_experiment -v`
Expected: FAIL with `cannot import name 'run_feature_interaction_experiment'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `run_feature_interaction_experiment(df: pd.DataFrame, spec: ExperimentSpecRecord, n_bootstrap: int = 500) -> ExperimentResultRecord`:
  1. Drops missing rows in `[feat_a, feat_b, target]`.
  2. Centers features around empirical means: $G = feat_a - \bar{feat_a}$, $A = feat_b - \bar{feat_b}$.
  3. Constructs interaction product term $G \times A$.
  4. Fits Additive Model: `sm.Logit(y, sm.add_constant(df[[G, A]])).fit(disp=False)`.
  5. Fits Interaction Model: `sm.Logit(y, sm.add_constant(df[[G, A, G_A]])).fit(disp=False)`.
  6. Likelihood-Ratio Test:
     - $LR = 2 \times (\text{LL}_{\text{interaction}} - \text{LL}_{\text{additive}})$.
     - $p_{\text{LRT}} = \chi^2_{\text{sf}}(LR, df=1)$.
  7. Bootstrap Resampling ($B=500$ resamples with replacement, seed 42):
     - Refits interaction model on resamples, computes 2.5% and 97.5% empirical quantiles for interaction coefficient $\beta_3$.
  8. Verdict Synthesis:
     - `SUPPORTED` if $p_{\text{LRT}} < 0.05$ (or interaction $p < 0.05$) and bootstrap 95% CI does not overlap 0.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_feature_interaction_experiment -v`
Expected: PASS

---

### Task 6: Experiment E5 — Adjusted Relationship & Possible Confounding Runner
**Files:**
- Modify: `modules/experiment_executor.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_run_adjusted_regression_experiment(sample_diabetes_df):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import run_adjusted_regression_experiment
    spec = ExperimentSpecRecord(
        experiment_id="iter1_E5",
        hypothesis_id="iter1_H5",
        iteration_no=1,
        experiment_type="comparative_or_confounding_relationship",
        method="adjusted_regression",
        model_type="statsmodels.Logit",
        features=["Pregnancies", "Age"],
        target="Outcome",
        parameters={"adjust_for_covariates": True, "maxiter": 100},
        evaluation={"adjusted_odds_ratio": True, "unadjusted_odds_ratio": True, "delta_coefficient": True}
    )
    result = run_adjusted_regression_experiment(sample_diabetes_df, spec)
    assert result.experiment_id == "iter1_E5"
    assert "unadjusted_coefficient" in result.metrics
    assert "adjusted_coefficient" in result.metrics
    assert "delta_coefficient" in result.metrics
    assert "percent_change" in result.metrics
    assert "unadjusted_or" in result.metrics
    assert "adjusted_or" in result.metrics
    assert result.verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE")
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_adjusted_regression_experiment -v`
Expected: FAIL with `cannot import name 'run_adjusted_regression_experiment'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `run_adjusted_regression_experiment(df: pd.DataFrame, spec: ExperimentSpecRecord) -> ExperimentResultRecord`:
  1. Standardizes predictors $Z_1, Z_2$ ($Z$-scored) on complete cases.
  2. Fits Model A (Unadjusted): `sm.Logit(y, sm.add_constant(df[[Z1]])).fit(disp=False)`.
  3. Fits Model B (Adjusted): `sm.Logit(y, sm.add_constant(df[[Z1, Z2]])).fit(disp=False)`.
  4. Computes:
     - $\beta_{\text{unadj}}$ and $\text{OR}_{\text{unadj}} = e^{\beta_{\text{unadj}}}$.
     - $\beta_{\text{adj}}$ and $\text{OR}_{\text{adj}} = e^{\beta_{\text{adj}}}$.
     - $\Delta \beta = \beta_{\text{adj}} - \beta_{\text{unadj}}$.
     - $\text{Percent Change} = \frac{\beta_{\text{adj}} - \beta_{\text{unadj}}}{\beta_{\text{unadj}}} \times 100\%$.
     - Covariate coefficient $\beta_{\text{covar}}$ and $p$-values.
  5. Confounding / Mediation Diagnostic:
     - If $|\text{Percent Change}| \ge 10\%$, flags meaningful adjustment/confounding effect.
     - Documents causal assumption limitations (requires formal temporality and no unmeasured confounding).
  6. Verdict Synthesis:
     - `SUPPORTED`: If adjustment produces $\ge 10\%$ shift or significant covariate effect as hypothesized.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_run_adjusted_regression_experiment -v`
Expected: PASS

---

### Task 7: Orchestrator Engine, Result Archival, and MongoDB Persistence
**Files:**
- Modify: `modules/experiment_executor.py`
- Modify: `modules/mongo_cluster.py`
- Test: `tests/test_experiment_executor.py`

**Step 1: Write the failing test**
In `tests/test_experiment_executor.py`:
```python
def test_execute_all_experiments_and_persistence(sample_diabetes_df, tmp_path):
    from modules.schemas import ExperimentSpecRecord
    from modules.experiment_executor import execute_all_experiments
    specs = [
        ExperimentSpecRecord(
            experiment_id="iter1_E1",
            hypothesis_id="iter1_H1",
            iteration_no=1,
            experiment_type="individual_dependency",
            method="logistic_regression",
            model_type="statsmodels.Logit",
            features=["Glucose"],
            target="Outcome"
        ),
        ExperimentSpecRecord(
            experiment_id="iter1_E3",
            hypothesis_id="iter1_H3",
            iteration_no=1,
            experiment_type="feature_combination_synergy",
            method="paired_model_comparison",
            model_type="sklearn.linear_model.LogisticRegression",
            features=["Glucose", "BMI"],
            target="Outcome",
            comparison=[["Glucose"], ["BMI"], ["Glucose", "BMI"]]
        )
    ]
    results = execute_all_experiments(
        specs=specs,
        df=sample_diabetes_df,
        iteration_no=1,
        output_dir=str(tmp_path),
        n_permutations=50,
        n_bootstrap=50
    )
    assert len(results) == 2
    assert (tmp_path / "experiment_results.json").exists()
    assert (tmp_path / "iterations" / "iter_1" / "experiment_results.json").exists()
```

**Step 2: Run test to verify it fails**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_execute_all_experiments_and_persistence -v`
Expected: FAIL with `cannot import name 'execute_all_experiments'`

**Step 3: Implement minimal code**
In `modules/experiment_executor.py`:
- Implement `execute_all_experiments(specs: List[ExperimentSpecRecord], df: pd.DataFrame, iteration_no: int = 1, output_dir: Optional[str] = None, n_permutations: int = 1000, n_bootstrap: int = 500) -> List[ExperimentResultRecord]`:
  1. Prepares dataset using `prepare_dataset_for_experiments(df)`.
  2. Routes each spec by `method` or `experiment_type`:
     - `logistic_regression` -> `run_individual_dependency_experiment`
     - `paired_model_comparison` -> `run_feature_combination_synergy_experiment`
     - `interaction_regression` -> `run_feature_interaction_experiment`
     - `adjusted_regression` -> `run_adjusted_regression_experiment`
  3. Collects all `ExperimentResultRecord` objects.
  4. Saves results to `os.path.join(out_dir, "experiment_results.json")` and `os.path.join(out_dir, "iterations", f"iter_{iteration_no}", "experiment_results.json")`.
  5. Persists to MongoDB cluster database `ai_scientist_results.results` with unique `result_id` indexing.

**Step 4: Run test to verify it passes**
Run: `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_executor.py::test_execute_all_experiments_and_persistence -v`
Expected: PASS

---

### Task 8: Pipeline Integration in mains.py & Verification
**Files:**
- Modify: `mains.py`
- Modify: `README.md`
- Test: Full pytest suite (`python -m pytest -v`)

**Step 1: Wire Stage 6 into mains.py**
- Import `execute_all_experiments` from `modules.experiment_executor`.
- In `mains.py`, after Stage 5 planning:
  ```python
  # Stage 6: Batch 3 — Experiment Execution Engine
  experiment_results = execute_all_experiments(
      specs=experiment_specs,
      df=df,
      iteration_no=iteration_no,
      output_dir=run_output_dir
  )
  ```
- Print formatted execution outcomes, odds ratios, synergy gains, LRT stats, and verdicts.
- Log database persistence to `ai_scientist_results.results`.

**Step 2: Run all tests**
Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: ALL tests pass (100% pass rate).

**Step 3: Execute mains.py live end-to-end**
Run: `.\.venv\Scripts\python.exe mains.py`
Expected: Full pipeline executes from Cached Dataset -> Stage 3 (LLM) -> Stage 4 (Validation) -> Stage 5 (Planning) -> Stage 6 (Execution of all 6 experiments) with 100% success and database persistence.

**Step 4: Update README.md**
Update `README.md` with Batch 3 execution engine details, statistical methods, output schemas, and `ai_scientist_results` database documentation.
