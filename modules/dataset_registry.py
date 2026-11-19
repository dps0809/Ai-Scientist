import os
import hashlib
import tomllib
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd

# pyrefly: ignore [missing-import]
from .config import OUTPUT_DIR, CACHE_ROOT, get_dataset_cache_dir

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "datasets.toml"
)


def load_dataset_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Reads datasets.toml and returns the configuration dictionary."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset configuration file not found: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def list_datasets(config_path: Optional[str] = None) -> List[str]:
    """Returns a list of all registered dataset names."""
    cfg = load_dataset_config(config_path)
    return list(cfg.get("datasets", {}).keys())


def compute_dataset_hash(*paths: str) -> str:
    """
    Computes a deterministic SHA-256 hash across sorted file contents.
    Returns the first 8 hex characters.
    """
    hasher = hashlib.sha256()
    for p in sorted(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found for hash calculation: {p}")
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()[:8]


def get_cache_dir(dataset_id: str) -> str:
    """
    Returns the isolated cache directory path for the given dataset_id:
    ai_scientist_output/cache/<dataset_id>/
    """
    return get_dataset_cache_dir(dataset_id)


def load_registered_dataset(
    name: str,
    config_path: Optional[str] = None
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Loads a dataset by its registered name from datasets.toml.
    Supports single_file and split_input_output datasets.

    Returns:
        (df, dataset_id, ds_config)
    """
    cfg = load_dataset_config(config_path)
    datasets = cfg.get("datasets", {})
    if name not in datasets:
        available = ", ".join(datasets.keys())
        raise KeyError(f"Dataset '{name}' not found in registry. Available datasets: {available}")

    ds_config = datasets[name]
    ds_type = ds_config.get("type", "single_file")

    print("\n" + "=" * 70)
    print(f"LOADING REGISTERED DATASET: '{name}' (Type: {ds_type})")
    print("=" * 70)

    if ds_type == "single_file":
        path = ds_config.get("path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file path '{path}' does not exist.")

        data_hash = compute_dataset_hash(path)
        dataset_id = f"{name}_{data_hash}"

        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format for path: {path}")

        print(f"Loaded single-file dataset: {path}")
        print(f"Assigned Dataset ID: {dataset_id}")
        print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        return df, dataset_id, ds_config

    elif ds_type == "split_input_output":
        input_path = ds_config.get("input_path")
        output_path = ds_config.get("output_path")

        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file path '{input_path}' does not exist.")
        if not output_path or not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file path '{output_path}' does not exist.")

        data_hash = compute_dataset_hash(input_path, output_path)
        dataset_id = f"{name}_{data_hash}"

        df_in = pd.read_csv(input_path)
        df_out = pd.read_csv(output_path)

        if ds_config.get("drop_unnamed", False):
            df_in = df_in.loc[:, ~df_in.columns.str.startswith("Unnamed")]
            df_out = df_out.loc[:, ~df_out.columns.str.startswith("Unnamed")]

        # Determine features from input
        input_features = ds_config.get("input_features", [])
        if input_features:
            missing_feats = [f for f in input_features if f not in df_in.columns]
            if missing_feats:
                raise KeyError(f"Features {missing_feats} not found in input file {input_path}")
            features_df = df_in[input_features]
        else:
            features_df = df_in

        # Extra input columns (e.g. classification labels G, C, B, A)
        extra_cols = ds_config.get("extra_input_columns", [])
        extra_df = pd.DataFrame()
        if extra_cols:
            valid_extra = [c for c in extra_cols if c in df_in.columns and c not in input_features]
            if valid_extra:
                extra_df = df_in[valid_extra]

        # Target from output
        output_target = ds_config.get("output_target")
        if not output_target or output_target not in df_out.columns:
            raise KeyError(f"Target column '{output_target}' not found in output file {output_path}")
        target_df = df_out[[output_target]]

        # Row-wise alignment (truncate to shorter)
        min_rows = min(len(df_in), len(df_out))
        print(f"Aligning split dataset: input rows={len(df_in)}, output rows={len(df_out)} -> aligned to {min_rows} rows")

        dfs_to_concat = []
        if not extra_df.empty:
            dfs_to_concat.append(extra_df.iloc[:min_rows].reset_index(drop=True))
        dfs_to_concat.append(features_df.iloc[:min_rows].reset_index(drop=True))
        dfs_to_concat.append(target_df.iloc[:min_rows].reset_index(drop=True))

        merged_df = pd.concat(dfs_to_concat, axis=1)

        print(f"Merged split dataset: {input_path} + {output_path}")
        print(f"Assigned Dataset ID: {dataset_id}")
        print(f"Final Merged Shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        print(f"Columns: {list(merged_df.columns)}")
        return merged_df, dataset_id, ds_config

    else:
        raise ValueError(f"Unknown dataset type: '{ds_type}'")
