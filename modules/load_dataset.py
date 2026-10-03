import os
import pandas as pd

def load_dataset(path):
    print("\n" + "=" * 70)
    print("1. LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDataset not found: {path}\n"
            f"Put your CSV file in the same folder as this script "
            f"or change DATASET_PATH."
        )

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raise ValueError("Only CSV and Excel files are supported.")

    print(f"Rows       : {df.shape[0]}")
    print(f"Columns    : {df.shape[1]}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    return df
