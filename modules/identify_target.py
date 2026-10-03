# pyrefly: ignore [missing-import]
from .config import TARGET_COLUMN

def identify_target(df):
    if TARGET_COLUMN is not None:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in dataset.")
        return TARGET_COLUMN

    possible_targets = [
        "target", "label", "class", "outcome", "response", "y", "prediction", "result"
    ]
    lower_map = {str(col).lower(): col for col in df.columns}

    for name in possible_targets:
        if name in lower_map:
            return lower_map[name]

    print("\nNo obvious target column detected.")
    print(f"Using the last column as target: {df.columns[-1]}")
    return df.columns[-1]
