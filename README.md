# AI Scientist

AI Scientist is a modular, reproducible research pipeline that discovers statistical relationships, generates testable hypotheses, and produces visual reports (charts, tables) from a dataset. The codebase is organized under the `modules/` package and driven by the `mains.py` entry point.

## Features
- Load and profile datasets
- Clean data with robust handling of missing/invalid values
- Numerical pattern discovery with Pearson correlation, Z‑score, and nicely formatted p‑values
- Categorical pattern discovery (group differences, chi‑square)
- Feature importance via Random Forest
- Automatic EDA visualizations (histograms, box‑plots, correlation heatmap) saved as PNG
- Hypothesis generation using a local LLM (or placeholder) and JSON/Markdown reports
- All results packaged as tables and figures in `ai_scientist_output/`

## Quick start

```powershell
# Resolve dependencies (uv is required)
uv sync

# Run the full pipeline on the built‑in diabetes dataset
.venv\Scripts\python.exe mains.py
```

The script writes:
- `ai_scientist_output/llm_context.json` – JSON context passed to the LLM
- `ai_scientist_output/eda_report.html` – interactive HTML with charts
- `ai_scientist_output/scientific_report.md` – human‑readable Markdown summary
- `ai_scientist_output/scientific_report.json` – machine‑readable record

## Repository layout

C:/Coding/Ai_Scientist/
├─ modules/
│   ├─ __init__.py
│   ├─ config.py                # central configuration (paths, model settings, DB names, DATASET_PROCESSED_DIR)
│   ├─ schemas.py               # centralized Pydantic contracts (Input, Pattern, Hypothesis, Validation, Spec, Fingerprint, Report)
│   ├─ mongo_cluster.py         # multi-database cluster manager (port 27017 with graceful fallback)
│   ├─ load_dataset.py          # dataset loading
│   ├─ profile_dataset.py       # basic statistics, missing‑value summary
│   ├─ clean_dataset.py         # imputation & type coercion
│   ├─ generate_eda.py          # charts & tables (histograms, box‑plots, heatmap)
│   ├─ identify_target.py       # target column inference
│   ├─ individual_dependency_analysis.py # Stage 1 dependency analysis
│   ├─ feature_combination_analysis.py  # Stage 2 synergy & interaction analysis
│   ├─ categorical_patterns.py # t‑test & chi‑square for categorical features
│   ├─ feature_importance.py    # RandomForest feature importance
│   ├─ save_patterns.py         # writes pattern records to DB & JSON
│   ├─ create_llm_context.py    # builds the LLM prompt context
│   ├─ generate_hypotheses_with_llm.py # LLM hypothesis generation (iter<N>_H<M>)
│   ├─ hypothesis_validator.py  # Batch 1 validation & evidence reconciliation
│   ├─ experiment_planner.py    # Batch 2 experiment registry & specification (iter<N>_E<M>)
│   ├─ experiment_executor.py   # Batch 3 statistical experiment execution engine (E1-E5)
│   ├─ result_evaluator.py      # Batch 4 statistical significance & cross-experiment comparison engine
│   ├─ llm_interpreter.py       # Batch 5 LLM interpretation & hypothesis refinement engine
│   ├─ experiment_selector.py   # Batch 6 Component A: candidate scoring, hallucination filtering & synergy ranking
│   ├─ experiment_controller.py # Batch 6 Component B: plan validation, stopping criteria, closed-loop orchestrator & report generator
│   └─ experiment_memory.py     # Batch 7: normalized fingerprint deduplication & scientific history memory
├─ mains.py                     # orchestrates two-tier pipeline (standard single-run or autonomous closed loop)
├─ README.md                    # this file
└─ ai_scientist_output/
    ├─ dataset_processed/       # Shared processed dataset data (reused across all runs)
    │   ├─ cleaned_dataset.csv
    │   ├─ dataset_profile.csv
    │   ├─ correlation_matrix.csv
    │   ├─ individual_dependencies.csv
    │   ├─ stage1_data.json
    │   ├─ feature_combinations.csv
    │   ├─ stage2_data.json
    │   ├─ feature_importance.csv
    │   ├─ discovered_patterns.json
    │   └─ plots/
    └─ <YYYYMMDD_HHMMSS>/       # Run-specific hypothesis & experiment iterations
        ├─ llm_context.json
        ├─ generated_hypotheses.json
        ├─ validated_hypotheses.json
        ├─ experiment_specifications.json
        ├─ experiment_results.json
        ├─ experiment_evaluations.json
        ├─ cross_experiment_comparison.json
        ├─ llm_interpretations.json
        ├─ refined_next_hypotheses.json
        ├─ experiment_memory.json
        ├─ autonomous_state.json
        ├─ final_scientific_report.md
        ├─ final_scientific_report.json
        └─ iterations/iter_<N>/
```

## Multi-Database MongoDB Cluster Architecture
The pipeline connects to the active MongoDB port (`localhost:27017`) and segregates domain records across dedicated databases:
- `ai_scientist_inputs` – Dataset ingestion records, schemas, and `input_id` metadata.
- `ai_scientist_patterns` – Stage 1 and Stage 2 discovered patterns tagged with `pattern_id`.
- `ai_scientist_hypotheses` – LLM contexts, generated hypotheses with `iteration_no`, and deduplication index.
- `ai_scientist_validation` – Batch 1 validation reports, discrepancy checks, and reconciled evidence.
- `ai_scientist_experiments` – Batch 2 structured experiment specifications.
- `ai_scientist_results` – Batch 3 statistical experiment results, odds ratios, CIs, p-values, synergy gains, interaction tests, bootstrap intervals, and verdicts.
- `ai_scientist_evaluations` – Batch 4 standardized evaluations (`evaluations`) and cross-experiment baseline comparison matrices (`comparisons`).
- `ai_scientist_refinements` – Batch 5 LLM experiment interpretations (`interpretations`) and refined candidate hypotheses for subsequent iterations (`refined_hypotheses`).
- `ai_scientist_memory` – Batch 7 normalized experiment fingerprints (`fingerprints`) and cumulative scientific history records (`experiments`).

If MongoDB is offline or unavailable, `modules.mongo_cluster.MongoClusterManager` gracefully logs a notice and allows the system to continue operating in standalone file-based mode.

## Pipeline steps
1. **Load dataset with unique ID** – `modules.load_dataset.load_dataset_with_id` assigns an identifiable `input_id` (e.g. `INP-YYYYMMDD-HHMMSS`) and persists schema metadata to `ai_scientist_inputs.inputs`.
2. **Profile** – overview statistics printed to console and saved.
3. **Clean** – missing values imputed, numeric coercion.
4. **EDA** – charts & tables automatically saved in `ai_scientist_output/`.
5. **Pattern discovery** – numerical and categorical relationships with unique `pattern_id` (`PAT-001`, `PAT-002`, ...) linked to `input_id` and saved in `ai_scientist_patterns.patterns`:
   - correlation coefficient
   - Z‑score (Fisher transformation)
   - formatted p‑value (`< 0.0001` when appropriate)
   - direction (positive/negative)
6. **Feature importance** – RandomForest scores.
7. **Context creation** – all evidence merged into a JSON block for the LLM.
8. **Hypothesis generation & de-duplication** – `modules.generate_hypotheses_with_llm.generate_hypotheses_with_llm` uses `modules.hypothesis_registry.HypothesisRegistry` to track iterations (`iteration_no`), inject historical fingerprints into prompts, avoid redundant hypotheses, and synchronize results to `ai_scientist_hypotheses.hypotheses`. Hypotheses strictly use iteration-prefixed IDs numbered 1..N within their iteration (`iter1_H1`, `iter1_H2`, `iter1_H3`... in iteration 1; `iter2_H1`, `iter2_H2`... in iteration 2).
9. **Batch 1 — Hypothesis validation & reconciliation** – `modules.hypothesis_validator.validate_hypotheses` validates schema compliance, target binary criteria, feature numeric checks, reconciles numerical discrepancies (e.g., H4 interaction delta contradiction), assigns statuses (`VALIDATED`, `NEEDS_REVIEW`, `REJECTED`), tags validation records with `VAL-iter<N>_H<M>`, and records audit trails to `ai_scientist_validation.validations`.
10. **Batch 2 — Experiment planning** – `modules.experiment_planner.plan_experiments` maps validated hypotheses to executable statistical experiment specifications with `iter<N>_E<M>` IDs, routing H5 to adjusted regression.
11. **Batch 3 — Statistical experiment execution** – `modules.experiment_executor.execute_all_experiments` prepares data by handling physiological invalid zeros (`Glucose`, `BMI`, `BloodPressure`, `Insulin`, `SkinThickness` -> `np.nan`), applies within-fold median imputation during 5-fold Stratified CV, computes inferential odds ratios with 95% CIs and permutation tests (1000 iter), tests feature synergy gains over best individuals using paired fold comparisons, performs Likelihood Ratio Tests with 500-sample bootstrap CIs for interaction terms, and computes unadjusted vs adjusted regression diagnostics for confounding. Saves records to `experiment_results.json` and syncs to `ai_scientist_results.results`.
12. **Batch 4 — Result evaluation & comparison matrix** – `modules.result_evaluator.evaluate_all_results` evaluates statistical significance, practical impact, uncertainty levels, and hypothesis support across all experiment types. Constructs an aggregated cross-experiment baseline comparison matrix ranking individual baselines against combinations, saving to `experiment_evaluations.json`, `cross_experiment_comparison.json`, and `ai_scientist_evaluations`.
13. **Batch 5 — LLM interpretation & hypothesis refinement** – `modules.llm_interpreter.interpret_and_refine_all` synthesizes scientific verdicts via the local Ollama LLM (`qwen3:4b`), extracts key clinical/statistical observations and limitations, formulates `next_hypothesis` specifications for Iteration 2 (`iter2_H1`, `iter2_H2`, ...), and indexes them with `HypothesisRegistry` deduplication. Saves to `llm_interpretations.json`, `refined_next_hypotheses.json`, and `ai_scientist_refinements`.
14. **Batch 6 & 7 — Autonomous Discovery Loop & Experiment Memory** –
    - **Component A (LLM Experiment Selector)**: Analyzes statistical signals, filters unmeasured/hallucinated features (e.g. `HOMA-IR`), prioritizes multi-feature synergies combining top-ranked individual baselines, and prompts the local LLM for valid alternatives if an experiment is rejected.
    - **Component B (Experiment Controller)**: The central authority that validates proposed configurations, checks normalized fingerprints (`sorted_features|model|target|metric`) against memory to guarantee zero duplicate executions, monitors multi-criteria stopping conditions (max experiments, max iterations, max runtime, no-improvement tolerance, target ROC-AUC), manages candidate pools, and compiles `final_scientific_report.md` / `final_scientific_report.json`.
    - **Batch 7 (MongoDB Experiment Memory)**: Synchronizes all historical specifications, results, and normalized fingerprints across `ai_scientist_memory`, `ai_scientist_experiments`, and `ai_scientist_results`, with offline JSON fallback.
15. **Two-Tier Storage & Run Archival** –
    - **Dataset Processing Layer (`ai_scientist_output/dataset_processed/`)**: Contains all dataset-specific artifacts (`cleaned_dataset.csv`, `dataset_profile.csv`, `correlation_matrix.csv`, `individual_dependencies.csv`, `stage1_data.json`, `feature_combinations.csv`, `stage2_data.json`, `feature_importance.csv`, `discovered_patterns.json`, and `plots/`). On startup, `mains.py` seeks these files first. If found, it reuses them immediately (`[CACHE HIT]`), bypassing expensive recomputation. If not found, it computes, profiles, and caches them for all future runs.
    - **Run-Specific Hypothesis & Experiment Output (`ai_scientist_output/<YYYYMMDD_HHMMSS>/`)**: Isolates hypothesis-cycle artifacts (`llm_context.json`, `generated_hypotheses.json`, `validated_hypotheses.json`, `experiment_specifications.json`, `experiment_results.json`, `experiment_evaluations.json`, `cross_experiment_comparison.json`, `llm_interpretations.json`, `refined_next_hypotheses.json`, `experiment_memory.json`, `autonomous_state.json`, `final_scientific_report.md`, and `iterations/iter_<N>/`) for each execution without polluting the root output directory.

## Predefined Experiment Registry
Supported hypothesis categories and their statistical execution mappings:
| Hypothesis Type | Experiment Method | Statistical Model | Key Evaluation Outputs |
|---|---|---|---|
| `individual_dependency` | `logistic_regression` | `statsmodels.Logit` & `sklearn.linear_model.LogisticRegression` | Odds ratios per SD, 95% CI, p-values, permutation tests (1000 iter), 5-fold Stratified CV ROC-AUC |
| `feature_combination_synergy` | `paired_model_comparison` | `sklearn.linear_model.LogisticRegression` | 5-fold Stratified CV ROC-AUC ($X_A$ vs $X_B$ vs $X_A+X_B$), synergy gain over best individual, paired t-test p-value |
| `feature_interaction_moderation` | `interaction_regression` | `statsmodels.Logit` | Mean-centered main effects + product term ($G \times A$), Likelihood Ratio Test $\chi^2$ and p-value, 500-sample bootstrap 95% CI |
| `comparative_or_confounding_relationship` | `adjusted_regression` | `statsmodels.Logit` | Unadjusted vs adjusted standardized coefficients, percentage shift $\Delta\%$, confounding diagnosis |

## Running the Pipeline

### Mode 1: Standard Single-Run (Iteration 1)
```powershell
python mains.py
```
Executes Iteration 1 hypothesis generation, validation, planning, execution, evaluation, and refinement.

### Mode 2: Autonomous Closed-Loop AI Scientist (Batches 6 & 7)
```powershell
# Run the autonomous multi-iteration research loop with default budgets
python mains.py --autonomous

# Run with custom budget and performance thresholds
python mains.py --autonomous --max-experiments 12 --max-iterations 3 --target-auc 0.85 --max-runtime-minutes 15
```

Available CLI Options:
- `--autonomous`: Enable closed-loop autonomous multi-iteration discovery.
- `--max-experiments <int>`: Total experiment budget across all iterations (default: 15).
- `--max-iterations <int>`: Maximum autonomous iterations (default: 3).
- `--max-runtime-minutes <int>`: Timeout budget in minutes (default: 20).
- `--target-auc <float>`: Target ROC-AUC stopping threshold (default: 0.85).
- `--no-improvement-tolerance <int>`: Consecutive iterations without gain tolerance (default: 3).
- `--n-permutations <int>`: Number of permutation test iterations (default: 1000).
- `--n-bootstrap <int>`: Number of bootstrap iterations for CI (default: 500).
- `--no-mongo`: Disable MongoDB sync and run in pure offline JSON file mode.

## Testing
Run the complete unit and integration test suite:
```powershell
python -m pytest -v
```

## License
MIT – feel free to fork, adapt, and integrate into your own research pipelines.

---
*This README reflects the current modular code base and highlights the built‑in visual reporting capabilities (charts, tables) produced by the pipeline.*
