import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR

def generate_eda(df):
    print("\n" + "=" * 70)
    print("4. AUTOMATED EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    numerical = df.select_dtypes(include=np.number).columns.tolist()

    if len(numerical) >= 2:
        corr = df[numerical].corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "correlation_matrix.png"))
        plt.close()
        corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
        print("Correlation matrix generated.")

    for col in numerical[:10]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        safe_name = str(col).replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{safe_name}_distribution.png"))
        plt.close()

    print("Distribution plots generated.")
