import os

# ============================================================
# CONFIGURATION  (edit values here — all modules pick them up)
# ============================================================

DATASET_PATH = "diabetes.csv"
ACTIVE_DATASET = os.getenv("AI_SCIENTIST_DATASET", "diabetes")

# If your dataset has a target column, put its name here.
# If None, the program will try to identify one automatically.
TARGET_COLUMN = None

# Local Ollama model
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Feature Combination & CV parameters
TOP_K_CANDIDATES = 6
CV_FOLDS = 5

OUTPUT_DIR = "ai_scientist_output"
# DEPRECATED: legacy path kept for backward compatibility.
# New code should use get_dataset_cache_dir() for per-dataset isolation.
DATASET_PROCESSED_DIR = os.path.join(OUTPUT_DIR, "dataset_processed")
CACHE_ROOT = os.path.join(OUTPUT_DIR, "cache")

# MongoDB & Multi-Database Cluster Settings (port 27017)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_TIMEOUT_MS = 2000

DB_INPUTS = "ai_scientist_inputs"
DB_PATTERNS = "ai_scientist_patterns"
DB_HYPOTHESES = "ai_scientist_hypotheses"
DB_VALIDATION = "ai_scientist_validation"
DB_EXPERIMENTS = "ai_scientist_experiments"
DB_RESULTS = "ai_scientist_results"
DB_EVALUATIONS = "ai_scientist_evaluations"
DB_REFINEMENTS = "ai_scientist_refinements"
DB_MEMORY = "ai_scientist_memory"

VALIDATED_HYPOTHESES_PATH = os.path.join(OUTPUT_DIR, "validated_hypotheses.json")
EXPERIMENT_SPECS_PATH = os.path.join(OUTPUT_DIR, "experiment_specifications.json")
EXPERIMENT_RESULTS_PATH = os.path.join(OUTPUT_DIR, "experiment_results.json")
EXPERIMENT_EVALUATIONS_PATH = os.path.join(OUTPUT_DIR, "experiment_evaluations.json")
CROSS_EXPERIMENT_COMPARISON_PATH = os.path.join(OUTPUT_DIR, "cross_experiment_comparison.json")
LLM_INTERPRETATIONS_PATH = os.path.join(OUTPUT_DIR, "llm_interpretations.json")
REFINED_HYPOTHESES_PATH = os.path.join(OUTPUT_DIR, "refined_next_hypotheses.json")
HYPOTHESIS_REGISTRY_PATH = os.path.join(OUTPUT_DIR, "hypothesis_registry.json")
AUTONOMOUS_STATE_PATH = os.path.join(OUTPUT_DIR, "autonomous_state.json")
FINAL_REPORT_MD_PATH = os.path.join(OUTPUT_DIR, "final_scientific_report.md")
FINAL_REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "final_scientific_report.json")

# Autonomous Loop Budgets (Batch 6.4)
MAX_AUTONOMOUS_EXPERIMENTS = 15
MAX_AUTONOMOUS_ITERATIONS = 3
MAX_RUNTIME_MINUTES = 20


# ----- Per-dataset cache isolation -----

def get_dataset_cache_dir(dataset_id: str) -> str:
    """Return (and create) a cache directory for *dataset_id*.

    Layout: ``<CACHE_ROOT>/<dataset_id>/``  (with a ``plots/`` sub-dir).
    """
    path = os.path.join(CACHE_ROOT, dataset_id)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    return path


# ----- Ensure directories exist at import time -----

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_PROCESSED_DIR, "plots"), exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

