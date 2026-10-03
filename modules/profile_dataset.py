import os
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def profile_dataset(df):
    print("\n" + "=" * 70)
    print("2. DATA PROFILING")
    print("=" * 70)

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_%": (df.isnull().mean() * 100).round(2),
        "unique_values": df.nunique()
    })

    print("\nDataset Profile:")
    print(profile)

    profile.to_csv(os.path.join(OUTPUT_DIR, "dataset_profile.csv"))

    numerical = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\nNumerical columns:")
    print(numerical)
    print("\nCategorical columns:")
    print(categorical)

    return numerical, categorical
