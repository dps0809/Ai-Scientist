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

```
C:/Coding/Ai_Scientist/
├─ modules/
│   ├─ __init__.py
│   ├─ config.py                # central configuration (paths, model settings)
│   ├─ load_dataset.py          # dataset loading
│   ├─ profile_dataset.py       # basic statistics, missing‑value summary
│   ├─ clean_dataset.py         # imputation & type coercion
│   ├─ generate_eda.py          # charts & tables (histograms, box‑plots, heatmap)
│   ├─ identify_target.py       # target column inference
│   ├─ numerical_patterns.py    # Pearson correlation → correlation, Z‑score, p‑value
│   ├─ categorical_patterns.py # t‑test & chi‑square for categorical features
│   ├─ feature_importance.py    # RandomForest feature importance
│   ├─ save_patterns.py         # writes Section 8 JSON + prints summary
│   ├─ create_llm_context.py    # builds the LLM prompt context
│   └─ generate_hypotheses_with_llm.py # placeholder LLM call
├─ mains.py                     # orchestrates the whole pipeline
├─ README.md                    # this file
└─ ai_scientist_output/         # generated artifacts (charts, tables, reports)
```

## Pipeline steps
1. **Load dataset** – `modules.load_dataset.load_dataset`
2. **Profile** – overview statistics printed to console and saved.
3. **Clean** – missing values imputed, numeric coercion.
4. **EDA** – charts & tables automatically saved in `ai_scientist_output/`.
5. **Pattern discovery** – numerical and categorical relationships, each with:
   - correlation coefficient
   - Z‑score (Fisher transformation)
   - formatted p‑value (`< 0.0001` when appropriate)
   - direction (positive/negative)
6. **Feature importance** – RandomForest scores.
7. **Context creation** – all evidence merged into a JSON block for the LLM.
8. **Hypothesis generation** – produces three testable, evidence‑based hypotheses.
9. **Reporting** – JSON and Markdown reports, plus the EDA visual assets.

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

## Testing
```powershell
uv run pytest           # unit and integration tests
uv run python -m pip install -e .   # editable install for IDE support
```

## License
MIT – feel free to fork, adapt, and integrate into your own research pipelines.

---
*This README reflects the current modular code base and highlights the built‑in visual reporting capabilities (charts, tables) produced by the pipeline.*
