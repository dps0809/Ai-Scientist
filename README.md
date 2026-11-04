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
│   ├─ schemas.py               # centralized Pydantic contracts (Input, Pattern, Hypothesis, Validation, Spec)
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
│   └─ experiment_planner.py    # Batch 2 experiment registry & specification (iter<N>_E<M>)
├─ mains.py                     # orchestrates two-tier pipeline (cached dataset -> run hypotheses)
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
        └─ iterations/iter_<N>/
```

## Multi-Database MongoDB Cluster Architecture
The pipeline connects to the active MongoDB port (`localhost:27017`) and segregates domain records across dedicated databases:
- `ai_scientist_inputs` – Dataset ingestion records, schemas, and `input_id` metadata.
- `ai_scientist_patterns` – Stage 1 and Stage 2 discovered patterns tagged with `pattern_id`.
- `ai_scientist_hypotheses` – LLM contexts, generated hypotheses with `iteration_no`, and deduplication index.
- `ai_scientist_validation` – Batch 1 validation reports, discrepancy checks, and reconciled evidence.
- `ai_scientist_experiments` – Batch 2 structured experiment specifications.

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
11. **Two-Tier Storage & Run Archival** –
    - **Dataset Processing Layer (`ai_scientist_output/dataset_processed/`)**: Contains all dataset-specific artifacts (`cleaned_dataset.csv`, `dataset_profile.csv`, `correlation_matrix.csv`, `individual_dependencies.csv`, `stage1_data.json`, `feature_combinations.csv`, `stage2_data.json`, `feature_importance.csv`, `discovered_patterns.json`, and `plots/`). On startup, `mains.py` seeks these files first. If found, it reuses them immediately (`[CACHE HIT]`), bypassing expensive recomputation. If not found, it computes, profiles, and caches them for all future runs.
    - **Run-Specific Hypothesis & Experiment Output (`ai_scientist_output/<YYYYMMDD_HHMMSS>/`)**: Isolates hypothesis-cycle artifacts (`llm_context.json`, `generated_hypotheses.json`, `validated_hypotheses.json`, `experiment_specifications.json`, and `iterations/iter_<N>/`) for each execution without polluting the root output directory.

## Predefined Experiment Registry
Supported hypothesis categories and their statistical execution mappings:
| Hypothesis Type | Experiment Method | Statistical Model | Key Evaluation Outputs |
|---|---|---|---|
| `individual_dependency` | `logistic_regression` | `statsmodels.Logit` | Odds ratios, 95% CI, p-values, permutation tests (1000 iter) |
| `feature_combination_synergy` | `paired_model_comparison` | `sklearn.linear_model.LogisticRegression` | 5-fold Stratified CV ROC-AUC comparison ($X_A$ vs $X_B$ vs $X_A+X_B$) |
| `feature_interaction_moderation` | `interaction_regression` | `statsmodels.Logit` | Interaction term product $X_A \times X_B$, likelihood ratio test, 500 bootstrap samples |
| `comparative_or_confounding_relationship` | `adjusted_regression` | `statsmodels.Logit` | Adjusted vs unadjusted odds ratios, confounding coefficient deltas |

## Charts & tables
`generate_eda.py` creates:
- Histogram of each numeric feature.
- Box‑plot grouped by the target.
- Correlation heatmap.
- Summary tables of missing values and basic statistics.
All figures are saved as PNG files in `ai_scientist_output/` and referenced from the Markdown report.

## Extending the project
- Replace the placeholder LLM call in `generate_hypotheses_with_llm.py` with any local LLM API.
- Add new pattern modules (e.g., time‑series, survival analysis) and expose them in `mains.py`.
- Swap the synthetic dataset for a real CSV by editing `modules/config.py → DATASET_PATH`.

## Running the Full Pipeline
```powershell
python mains.py
```
This executes the full automated research lifecycle:
1. Data Ingestion & Profiling (assigned `input_id` saved to `ai_scientist_inputs.inputs`)
2. Stage 1 Individual Dependency Analysis
3. Stage 2 Feature Combination & Interaction Discovery (assigned `pattern_id` saved to `ai_scientist_patterns.patterns`)
4. Stage 3 LLM Hypothesis Generation & Novelty De-duplication (saved to `ai_scientist_hypotheses.hypotheses`)
5. Stage 4 Batch 1 Hypothesis Validation & Evidence Reconciliation (saved to `ai_scientist_validation.validations`)
6. Stage 5 Batch 2 Experiment Registry Specifications (saved to `ai_scientist_experiments.specifications`)

## Testing
Run the complete unit and integration test suite:
```powershell
python -m pytest -v
```

## License
MIT – feel free to fork, adapt, and integrate into your own research pipelines.

---
*This README reflects the current modular code base and highlights the built‑in visual reporting capabilities (charts, tables) produced by the pipeline.*
