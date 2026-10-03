import os

# ============================================================
# CONFIGURATION  (edit values here — all modules pick them up)
# ============================================================

DATASET_PATH = "diabetes.csv"

# If your dataset has a target column, put its name here.
# If None, the program will try to identify one automatically.
TARGET_COLUMN = None

# Local Ollama model
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

OUTPUT_DIR = "ai_scientist_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
