import os
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def clean_dataset(df, output_dir=None, force_recompute: bool = False):
    out_dir = output_dir or OUTPUT_DIR
    cleaned_path = os.path.join(out_dir, "cleaned_dataset.csv")
    if os.path.exists(cleaned_path) and not force_recompute:
        try:
            cached_df = pd.read_csv(cleaned_path)
            if set(cached_df.columns) == set(df.columns):
                print(f"[CACHE HIT] Loaded cleaned dataset from: {cleaned_path} ({len(cached_df)} rows)")
                return cached_df
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("3. DATA CLEANING")
    print("=" * 70)

    original_rows = len(df)
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    print(f"Original rows : {original_rows}")
    print(f"Final rows    : {len(df)}")
    print(f"Removed       : {original_rows - len(df)}")

    numerical = df.select_dtypes(include=np.number).columns
    for col in numerical:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in categorical:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cleaned_dataset.csv"), index=False)
    print("\nCleaned dataset saved.")
    return df
