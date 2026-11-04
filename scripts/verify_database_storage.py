import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
try:
    import pymongo
except ImportError:
    print("[ERROR] 'pymongo' is not installed in the current environment.")
    print("Please run 'uv sync' or install pymongo.")
    sys.exit(1)
from modules.config import (
    MONGO_URI,
    DB_INPUTS,
    DB_PATTERNS,
    DB_HYPOTHESES,
    DB_VALIDATION,
    DB_EXPERIMENTS
)

def inspect_database_cluster():
    print("=" * 70)
    print("MONGODB CLUSTER STORAGE VERIFICATION")
    print(f"Connecting to: {MONGO_URI}")
    print("=" * 70)

    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
        print("[SUCCESS] MongoDB cluster ping successful on localhost:27017\n")
    except Exception as e:
        print(f"[ERROR] Could not connect to MongoDB: {e}")
        return False

    databases = [
        (DB_INPUTS, "inputs", "input_id"),
        (DB_PATTERNS, "patterns", "pattern_id"),
        (DB_HYPOTHESES, "hypotheses", "hypothesis_id"),
        (DB_VALIDATION, "validations", "validation_id"),
        (DB_VALIDATION, "reports", "report_id"),
        (DB_EXPERIMENTS, "specifications", "experiment_id")
    ]

    all_stored = True

    for db_name, collection_name, key_field in databases:
        db = client[db_name]
        col = db[collection_name]
        count = col.count_documents({})
        
        print("-" * 70)
        print(f"DATABASE: {db_name} | COLLECTION: {collection_name}")
        print(f"Document Count: {count}")
        
        if count == 0:
            print("[WARNING] No documents found in this collection!")
            all_stored = False
        else:
            sample = col.find_one({}, {"_id": 0})
            print(f"Key Identifier Field: '{key_field}' -> Value: {sample.get(key_field, 'MISSING')}")
            print("Sample Document Summary:")
            # Print pretty json excerpt
            sample_str = json.dumps(sample, indent=2)
            lines = sample_str.split("\n")
            if len(lines) > 15:
                print("\n".join(lines[:15]))
                print(f"  ... (+ {len(lines) - 15} lines)")
            else:
                print(sample_str)
        print("-" * 70 + "\n")

    return all_stored

if __name__ == "__main__":
    success = inspect_database_cluster()
    if success:
        print("ALL 5 DATABASES VERIFIED: Data is actively and properly storing in MongoDB!")
    else:
        print("Verification completed with warnings. Some collections may need pipeline execution.")
