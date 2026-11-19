# pyrefly: ignore [missing-import]
from .config import TARGET_COLUMN

def identify_target(df, target_column=None):
    # 1. Explicit parameter override takes highest priority
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in dataset.")
        return target_column

    # 2. Config-level override
    if TARGET_COLUMN is not None:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in dataset.")
        return TARGET_COLUMN

    # 3. Auto-detection fallback
    possible_targets = [
        "target", "label", "class", "outcome", "response", "y", "prediction", "result",
        "output"
    ]
    lower_map = {str(col).lower(): col for col in df.columns}

    for name in possible_targets:
        if name in lower_map:
            return lower_map[name]

    print("\nNo obvious target column detected.")
    print(f"Using the last column as target: {df.columns[-1]}")
    return df.columns[-1]
