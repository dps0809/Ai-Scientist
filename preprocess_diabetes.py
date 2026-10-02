import csv
from pathlib import Path

base_dir = Path(__file__).resolve().parent
input_path = base_dir / "diabetes.csv"
output_path = base_dir / "diabetes_preprocessed.csv"

keep_columns = ["Pregnancies", "Glucose", "BMI", "Age", "Outcome"]

with input_path.open("r", newline="") as infile, output_path.open("w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=keep_columns)
    writer.writeheader()

    cleaned_rows = 0
    for row in reader:
        cleaned = {col: row.get(col, "") for col in keep_columns}

        # Drop rows with missing values in key feature columns.
        if any(cleaned[col] in ("", None) for col in ["Pregnancies", "Glucose", "BMI", "Age", "Outcome"]):
            continue

        try:
            float(cleaned["Pregnancies"])
            float(cleaned["Glucose"])
            float(cleaned["BMI"])
            float(cleaned["Age"])
            float(cleaned["Outcome"])
        except ValueError:
            continue

        writer.writerow(cleaned)
        cleaned_rows += 1

print(f"Saved cleaned dataset to: {output_path}")
print(f"Rows kept: {cleaned_rows}")
print(f"Columns kept: {keep_columns}")
