import os
from datetime import datetime, timezone
from typing import Tuple
import pandas as pd
from .schemas import InputRecord
from .mongo_cluster import MongoClusterManager
from .config import DB_INPUTS

def load_dataset_with_id(path: str) -> Tuple[pd.DataFrame, InputRecord]:
    """
    Loads dataset, attaches an identifiable input_id, records schema metadata,
    and synchronizes the input record to MongoDB cluster (ai_scientist_inputs.inputs).
    """
    print("\n" + "=" * 70)
    print("1. LOADING DATASET WITH UNIQUE INPUT IDENTIFIER")
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

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    input_id = f"INP-{timestamp_str}"

    print(f"Assigned Input ID : {input_id}")
    print(f"Source file       : {os.path.basename(path)}")
    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    input_record = InputRecord(
        input_id=input_id,
        source=str(path),
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        features=[str(c) for c in df.columns],
        metadata={"file_size_bytes": os.path.getsize(path)}
    )

    cluster = MongoClusterManager()
    cluster.save_document(
        db_name=DB_INPUTS,
        collection_name="inputs",
        doc=input_record.model_dump(),
        id_key="input_id"
    )

    return df, input_record

def load_dataset(path: str) -> pd.DataFrame:
    df, _ = load_dataset_with_id(path)
    return df
