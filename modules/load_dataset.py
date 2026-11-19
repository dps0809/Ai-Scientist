import os
from datetime import datetime, timezone
from typing import Tuple, Union, Optional
import pandas as pd
from .schemas import InputRecord
from .mongo_cluster import MongoClusterManager
from .config import DB_INPUTS


def create_input_record(
    df: pd.DataFrame,
    source: str,
    metadata: Optional[dict] = None
) -> InputRecord:
    """
    Creates an InputRecord for an in-memory DataFrame and synchronizes it to MongoDB.
    """
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    input_id = f"INP-{timestamp_str}"

    meta = metadata or {}
    input_record = InputRecord(
        input_id=input_id,
        source=str(source),
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        features=[str(c) for c in df.columns],
        metadata=meta
    )

    cluster = MongoClusterManager()
    cluster.save_document(
        db_name=DB_INPUTS,
        collection_name="inputs",
        doc=input_record.model_dump(),
        id_key="input_id"
    )

    print(f"Assigned Input ID : {input_id}")
    print(f"Source            : {source}")
    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    return input_record


def load_dataset_with_id(
    path_or_df: Union[str, pd.DataFrame],
    source: Optional[str] = None
) -> Tuple[pd.DataFrame, InputRecord]:
    """
    Loads dataset, attaches an identifiable input_id, records schema metadata,
    and synchronizes the input record to MongoDB cluster (ai_scientist_inputs.inputs).
    Supports either a file path string or a pre-loaded pandas DataFrame.
    """
    print("\n" + "=" * 70)
    print("1. LOADING DATASET WITH UNIQUE INPUT IDENTIFIER")
    print("=" * 70)

    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
        src = source or "in_memory_dataframe"
        meta = {}
        input_record = create_input_record(df, source=src, metadata=meta)
        return df, input_record

    path = str(path_or_df)
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

    file_size = os.path.getsize(path) if os.path.exists(path) else 0
    input_record = create_input_record(
        df,
        source=str(path),
        metadata={"file_size_bytes": file_size}
    )

    return df, input_record


def load_dataset(path: str) -> pd.DataFrame:
    df, _ = load_dataset_with_id(path)
    return df
