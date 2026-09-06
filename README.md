# AI Scientist

AI Scientist is a reproducible research system for testing this question:

> Can an AI system autonomously discover useful hypotheses and improve model performance through iterative experimentation?

The project is designed to produce evidence, not just explanations. Its core loop is:

```text
dataset
	-> data profiler
	-> pattern discovery
	-> hypothesis generator
	-> experiment planner
	-> experiment runner
	-> evaluator
	-> scientific report
	-> next experiment
```

## Current MVP

The first slice runs entirely locally with a deterministic synthetic dataset:

1. Profile a feature and target.
2. Discover a correlation pattern.
3. Generate a falsifiable hypothesis.
4. Compare a mean-only human baseline with a linear treatment model.
5. Classify the hypothesis as supported, rejected, or inconclusive.
6. Write machine-readable JSON and human-readable Markdown evidence.

The implementation deliberately avoids an LLM and network calls at this stage. The pipeline stages are small modules so an LLM hypothesis generator, real datasets, and more models can be added behind the same boundaries.

## Quick start

Install and resolve the `uv` environment:

```powershell
uv sync
```

Run the experiment:

```powershell
uv run ai-scientist demo
```

Reports are written to `artifacts/scientific_report.json` and `artifacts/scientific_report.md`. Use another output directory or seed when comparing runs:

```powershell
uv run ai-scientist demo --output-dir artifacts/run-2 --seed 11
```

Run the tests and build the package:

```powershell
uv run pytest
uv build
```

## Repository layout

```text
src/ai_scientist/
├── models.py                 # Typed scientific records
├── pipeline.py               # End-to-end orchestration
├── cli.py                    # `ai-scientist demo`
├── data/                     # Dataset provider and profiler
├── patterns/                 # Evidence-based pattern discovery
├── hypotheses/               # Falsifiable hypothesis generation
├── experiments/              # Planning and model execution
├── evaluation/               # Threshold-based evidence evaluation
└── reporting/                # JSON and Markdown reports
tests/                        # Unit and end-to-end checks
configs/demo.toml             # Reproducible demo settings
```

## Research protocol

The system should be compared against a fixed human-designed baseline under the same dataset, split, metric, seed, and compute budget. Every experiment should record:

- model performance
- experiment count
- hypothesis quality and falsifiability
- convergence or stopping reason
- runtime
- estimated compute cost
- seed and configuration for reproducibility
- evidence and limitations

The MVP uses explicit hypothesis-quality proxies: a predeclared success criterion, a measurable treatment/control comparison, evidence completeness, and an outcome classification. Correlation discovery is not treated as proof of causation.

## Execution roadmap

### Phase 1: Foundation

- Add typed domain records and interfaces for each pipeline stage.
- Keep configuration for seeds, metrics, iteration limits, and budgets.
- Expand tests around serialization, thresholds, failures, and stop conditions.

### Phase 2: Real experiments

- Add pluggable real dataset providers.
- Add train/validation/test splits and repeated trials.
- Add model adapters and confidence intervals.
- Record dependency versions, source revision, dataset identity, and resource usage in an experiment manifest.

### Phase 3: Autonomous iteration

- Add multiple candidate hypotheses per iteration.
- Rank the next experiment using prior evidence and budget.
- Stop on convergence, budget exhaustion, no viable hypothesis, or repeated failure.
- Preserve the complete experiment history in every report.

### Phase 4: AI scientist comparison

- Add an LLM-backed hypothesis generator behind the existing interface.
- Compare rule-based, LLM-generated, and human-designed experiment policies.
- Use repeated seeds and expert review to assess hypothesis quality.
- Report performance, efficiency, reproducibility, and cost rather than only the best score.

## Design principles

- Evidence is stored before prose is generated.
- Every hypothesis has a predicted outcome and success criterion.
- Control and treatment are explicit.
- Experiments are bounded by seed, iteration count, and compute budget.
- Reports state uncertainty and limitations.
- External services remain optional adapters, not hidden requirements.
