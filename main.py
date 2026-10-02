import os
import json
import warnings
import requests
from typing import cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, ttest_ind, chi2_contingency

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "diabetes.csv"

# If your dataset has a target column, put its name here.
# If None, the program will try to identify one automatically.
TARGET_COLUMN = None

# Local Ollama model
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

OUTPUT_DIR = "ai_scientist_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(
    os.path.join(OUTPUT_DIR, "plots"),
    exist_ok=True
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset(path):

    print("\n" + "=" * 70)
    print("1. LOADING DATASET")
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
        raise ValueError(
            "Only CSV and Excel files are supported."
        )

    print(f"Rows       : {df.shape[0]}")
    print(f"Columns    : {df.shape[1]}")

    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col}")

    return df


# ============================================================
# 2. BASIC DATA PROFILING
# ============================================================

def profile_dataset(df):

    print("\n" + "=" * 70)
    print("2. DATA PROFILING")
    print("=" * 70)

    profile = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_%": (
            df.isnull().mean() * 100
        ).round(2),
        "unique_values": df.nunique()
    })

    print("\nDataset Profile:")
    print(profile)

    profile.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dataset_profile.csv"
        )
    )

    numerical = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("\nNumerical columns:")
    print(numerical)

    print("\nCategorical columns:")
    print(categorical)

    return numerical, categorical


# ============================================================
# 3. DATA CLEANING
# ============================================================

def clean_dataset(df):

    print("\n" + "=" * 70)
    print("3. DATA CLEANING")
    print("=" * 70)

    original_rows = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove completely empty columns
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove completely empty rows
    df = df.dropna(
        axis=0,
        how="all"
    )

    print(f"Original rows : {original_rows}")
    print(f"Final rows    : {len(df)}")
    print(
        f"Removed       : "
        f"{original_rows - len(df)}"
    )

    # Fill numerical missing values
    numerical = df.select_dtypes(
        include=np.number
    ).columns

    for col in numerical:

        if df[col].isnull().sum() > 0:

            df[col] = df[col].fillna(
                df[col].median()
            )

    # Fill categorical missing values
    categorical = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for col in categorical:

        if df[col].isnull().sum() > 0:

            mode = df[col].mode()

            if len(mode) > 0:
                df[col] = df[col].fillna(
                    mode.iloc[0]
                )

            else:
                df[col] = df[col].fillna(
                    "Unknown"
                )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "cleaned_dataset.csv"
        ),
        index=False
    )

    print("\nCleaned dataset saved.")

    return df


# ============================================================
# 4. AUTOMATIC EDA
# ============================================================

def generate_eda(df):

    print("\n" + "=" * 70)
    print("4. AUTOMATED EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    numerical = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numerical) >= 2:

        # ----------------------------------------------------
        # Correlation matrix
        # ----------------------------------------------------

        corr = df[numerical].corr()

        plt.figure(
            figsize=(12, 9)
        )

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title(
            "Feature Correlation Matrix"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "plots",
                "correlation_matrix.png"
            )
        )

        plt.close()

        corr.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "correlation_matrix.csv"
            )
        )

        print(
            "Correlation matrix generated."
        )

    # --------------------------------------------------------
    # Numerical distributions
    # --------------------------------------------------------

    for col in numerical[:10]:

        plt.figure(
            figsize=(8, 5)
        )

        sns.histplot(
            df[col],
            kde=True
        )

        plt.title(
            f"Distribution of {col}"
        )

        plt.tight_layout()

        safe_name = (
            str(col)
            .replace("/", "_")
            .replace(" ", "_")
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "plots",
                f"{safe_name}_distribution.png"
            )
        )

        plt.close()

    print(
        "Distribution plots generated."
    )


# ============================================================
# 5. TARGET IDENTIFICATION
# ============================================================

def identify_target(df):

    if TARGET_COLUMN is not None:

        if TARGET_COLUMN not in df.columns:

            raise ValueError(
                f"TARGET_COLUMN "
                f"'{TARGET_COLUMN}' "
                f"not found in dataset."
            )

        return TARGET_COLUMN

    # Common target names
    possible_targets = [
        "target",
        "label",
        "class",
        "outcome",
        "response",
        "y",
        "prediction",
        "result"
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in possible_targets:

        if name in lower_map:

            return lower_map[name]

    print(
        "\nNo obvious target column detected."
    )

    print(
        "Using the last column as target:"
        f" {df.columns[-1]}"
    )

    return df.columns[-1]


# ============================================================
# 6. CORRELATION-BASED PATTERN DISCOVERY
# ============================================================

def numerical_patterns(df, target):

    print("\n" + "=" * 70)
    print("5. NUMERICAL RELATIONSHIP DISCOVERY")
    print("=" * 70)

    patterns = []

    numerical = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if target not in numerical:

        print(
            "Target is not numerical. "
            "Skipping Pearson correlation."
        )

        return patterns

    for feature in numerical:

        if feature == target:
            continue

        x = df[feature]
        y = df[target]

        if (
            x.nunique() < 2
            or y.nunique() < 2
        ):
            continue

        try:

            correlation_result = cast(
                tuple[float, float],
                pearsonr(x, y)
            )
            correlation = float(correlation_result[0])
            p_value = float(correlation_result[1])

            if (
                abs(correlation) >= 0.3
                and p_value < 0.05
            ):

                direction = (
                    "positive"
                    if correlation > 0
                    else "negative"
                )

                pattern = {
                    "type": "correlation",
                    "feature": str(feature),
                    "target": str(target),
                    "correlation": round(
                        float(correlation),
                        4
                    ),
                    "p_value": round(
                        float(p_value),
                        6
                    ),
                    "direction": direction
                }

                patterns.append(pattern)

                print(
                    f"{feature} → {target} | "
                    f"r={correlation:.3f} | "
                    f"p={p_value:.5f}"
                )

        except Exception:
            pass

    return patterns


# ============================================================
# 7. CATEGORICAL GROUP DIFFERENCES
# ============================================================

def categorical_patterns(df, target):

    print("\n" + "=" * 70)
    print("6. CATEGORICAL RELATIONSHIP DISCOVERY")
    print("=" * 70)

    patterns = []

    categorical = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    numerical = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # --------------------------------------------------------
    # Categorical feature vs numerical target
    # --------------------------------------------------------

    if target in numerical:

        for feature in categorical:

            if df[feature].nunique() > 15:
                continue

            groups = [
                group[target].dropna().values
                for _, group
                in df.groupby(feature)
            ]

            if len(groups) != 2:
                continue

            try:

                test_result = cast(
                    tuple[float, float],
                    ttest_ind(
                        groups[0],
                        groups[1],
                        equal_var=False
                    )
                )
                p_value = float(test_result[1])

                means = (
                    df.groupby(feature)[target]
                    .mean()
                )

                if p_value < 0.05:

                    pattern = {
                        "type":
                            "group_difference",

                        "feature":
                            str(feature),

                        "target":
                            str(target),

                        "group_means": {
                            str(k):
                                round(
                                    float(v),
                                    4
                                )
                            for k, v
                            in means.items()
                        },

                        "p_value":
                            round(
                                float(p_value),
                                6
                            )
                    }

                    patterns.append(
                        pattern
                    )

                    print(
                        f"{feature} differs "
                        f"significantly across "
                        f"groups | "
                        f"p={p_value:.5f}"
                    )

            except Exception:
                pass

    # --------------------------------------------------------
    # Categorical feature vs categorical target
    # --------------------------------------------------------

    if target in categorical:

        for feature in categorical:

            if feature == target:
                continue

            if df[feature].nunique() > 20:
                continue

            try:

                table = pd.crosstab(
                    df[feature],
                    df[target]
                )

                chi_square_result = cast(
                    tuple[float, float, object, object],
                    chi2_contingency(table)
                )
                chi2 = float(chi_square_result[0])
                p_value = float(chi_square_result[1])

                if p_value < 0.05:

                    pattern = {
                        "type":
                            "categorical_association",

                        "feature":
                            str(feature),

                        "target":
                            str(target),

                        "chi_square":
                            round(
                                float(chi2),
                                4
                            ),

                        "p_value":
                            round(
                                float(p_value),
                                6
                            )
                    }

                    patterns.append(
                        pattern
                    )

                    print(
                        f"{feature} ↔ {target} | "
                        f"chi²={chi2:.3f} | "
                        f"p={p_value:.5f}"
                    )

            except Exception:
                pass

    return patterns


# ============================================================
# 8. MACHINE LEARNING FEATURE IMPORTANCE
# ============================================================

def feature_importance(df, target):

    print("\n" + "=" * 70)
    print("7. MACHINE LEARNING FEATURE IMPORTANCE")
    print("=" * 70)

    patterns = []

    try:

        X = df.drop(
            columns=[target]
        ).copy()

        y = df[target].copy()

        # Encode categorical features
        for col in X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns:

            X[col] = LabelEncoder().fit_transform(
                X[col].astype(str)
            )

        # Fill remaining missing values
        X = X.fillna(0)

        # Encode target if categorical
        if (
            y.dtype == "object"
            or str(y.dtype) == "category"
        ):

            y = LabelEncoder().fit_transform(
                y.astype(str)
            )

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

        else:

            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )

        model.fit(
            X,
            y
        )

        importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(
            ascending=False
        )

        print("\nTop features:")

        for feature, importance in (
            importances.head(10).items()
        ):

            print(
                f"{feature:<30} "
                f"{importance:.4f}"
            )

            if importance >= 0.05:

                patterns.append({
                    "type":
                        "ml_feature_importance",

                    "feature":
                        str(feature),

                    "importance":
                        round(
                            float(importance),
                            4
                        )
                })

        importances.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "feature_importance.csv"
            )
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 6)
        )

        (
            importances.head(10)
            .sort_values()
            .plot(kind="barh")
        )

        plt.title(
            "Top Feature Importances"
        )

        plt.xlabel(
            "Importance"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "plots",
                "feature_importance.png"
            )
        )

        plt.close()

    except Exception as e:

        print(
            "Feature importance could not "
            "be calculated:"
        )

        print(e)

    return patterns


# ============================================================
# 9. COMBINE ALL DISCOVERED PATTERNS
# ============================================================

def save_patterns(patterns):

    print("\n" + "=" * 70)
    print("8. DISCOVERED PATTERNS")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "discovered_patterns.json"
    )

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            patterns,
            f,
            indent=4
        )

    print(
        f"\nTotal discovered patterns: "
        f"{len(patterns)}"
    )

    for i, pattern in enumerate(
        patterns,
        1
    ):

        print(
            f"\nPattern {i}:"
        )

        print(
            json.dumps(
                pattern,
                indent=2
            )
        )

    return patterns


# ============================================================
# 10. PREPARE SMALL LLM EVIDENCE
# ============================================================

def create_llm_context(
    df,
    target,
    patterns
):

    # --------------------------------------------------------
    # IMPORTANT:
    # We DO NOT send the complete dataset to the LLM.
    #
    # Python has already performed the statistical analysis.
    # We only give the LLM the evidence it needs to formulate
    # hypotheses.
    # --------------------------------------------------------

    evidence = []

    for pattern in patterns:

        if pattern.get("type") == "correlation":

            evidence.append({
                "type":
                    "statistical_relationship",

                "feature":
                    pattern.get("feature"),

                "target":
                    pattern.get("target"),

                "correlation":
                    pattern.get("correlation"),

                "p_value":
                    pattern.get("p_value"),

                "direction":
                    pattern.get("direction")
            })

        elif (
            pattern.get("type")
            == "ml_feature_importance"
        ):

            evidence.append({
                "type":
                    "feature_importance",

                "feature":
                    pattern.get("feature"),

                "importance":
                    pattern.get("importance")
            })

        elif (
            pattern.get("type")
            == "group_difference"
        ):

            evidence.append({
                "type":
                    "group_difference",

                "feature":
                    pattern.get("feature"),

                "target":
                    pattern.get("target"),

                "group_means":
                    pattern.get(
                        "group_means"
                    ),

                "p_value":
                    pattern.get("p_value")
            })

        elif (
            pattern.get("type")
            == "categorical_association"
        ):

            evidence.append({
                "type":
                    "categorical_association",

                "feature":
                    pattern.get("feature"),

                "target":
                    pattern.get("target"),

                "chi_square":
                    pattern.get(
                        "chi_square"
                    ),

                "p_value":
                    pattern.get("p_value")
            })

    # Keep the context small
    evidence = evidence[:8]

    context = {
        "dataset_rows":
            int(df.shape[0]),

        "dataset_columns":
            int(df.shape[1]),

        "target":
            str(target),

        "discovered_evidence":
            evidence
    }

    # Save context so we can inspect it
    with open(
        os.path.join(
            OUTPUT_DIR,
            "llm_context.json"
        ),
        "w"
    ) as f:

        json.dump(
            context,
            f,
            indent=4
        )

    return context


# ============================================================
# 11. LOCAL LLM HYPOTHESIS GENERATION
# ============================================================

def generate_hypotheses_with_llm(
    context
):

    print("\n" + "=" * 70)
    print("9. LOCAL LLM HYPOTHESIS GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Extract ONLY discovered evidence
    # --------------------------------------------------------

    evidence = context.get(
        "discovered_evidence",
        []
    )

    evidence_text = json.dumps(
        evidence,
        indent=2
    )

    # --------------------------------------------------------
    # SMALL CONTROLLED PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a scientific hypothesis generation assistant.

A Python analysis system has already analyzed a
diabetes dataset.

The system discovered the following evidence:

{evidence_text}

Generate exactly 3 testable hypotheses.

IMPORTANT RULES:

1. Use ONLY the evidence provided above.
2. Do NOT invent relationships.
3. Do NOT claim causation.
4. Each hypothesis must be specific.
5. Each hypothesis must be testable using the dataset.
6. Keep each hypothesis concise.
7. Explain the evidence supporting each hypothesis.
8. Suggest one simple experiment to test each hypothesis.
9. Return ONLY valid JSON.
10. Do NOT use markdown.
11. Do NOT write anything before or after the JSON.

Return exactly this structure:

[
  {{
    "hypothesis_id": "H1",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }},
  {{
    "hypothesis_id": "H2",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }},
  {{
    "hypothesis_id": "H3",
    "hypothesis": "One specific testable hypothesis.",
    "evidence_from_data": "Evidence supporting the hypothesis.",
    "proposed_experiment": "A simple experiment that can test it."
  }}
]
"""

    # --------------------------------------------------------
    # Ollama request
    # --------------------------------------------------------

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,

        "options": {
            # Low temperature = more deterministic
            "temperature": 0.2,

            # Limit output length
            "num_predict": 1500
        }
    }

    text = ""

    try:

        print(
            "\nSending discovered evidence "
            "to local LLM..."
        )

        print(
            "Waiting for LLM response..."
        )

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        result = response.json()

        text = result.get(
            "response",
            ""
        ).strip()

        if not text:

            print(
                "\nLLM returned an empty response."
            )

            return []

        print(
            "\nRaw LLM response:"
        )

        print(text)

        # ----------------------------------------------------
        # Remove markdown fences
        # ----------------------------------------------------

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        hypotheses = json.loads(
            text
        )

        # Make sure we actually received a list
        if not isinstance(
            hypotheses,
            list
        ):

            print(
                "\nLLM response was not a JSON list."
            )

            return []

        # ----------------------------------------------------
        # Display hypotheses
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("GENERATED HYPOTHESES")
        print("=" * 70)

        for hypothesis in hypotheses:

            print(
                f"\n{hypothesis.get('hypothesis_id', 'H?')}"
            )

            print(
                "-" * 70
            )

            print(
                "Hypothesis:"
            )

            print(
                hypothesis.get(
                    "hypothesis",
                    "N/A"
                )
            )

            print(
                "\nEvidence:"
            )

            print(
                hypothesis.get(
                    "evidence_from_data",
                    "N/A"
                )
            )

            print(
                "\nProposed Experiment:"
            )

            print(
                hypothesis.get(
                    "proposed_experiment",
                    "N/A"
                )
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            "generated_hypotheses.json"
        )

        with open(
            output_path,
            "w"
        ) as f:

            json.dump(
                hypotheses,
                f,
                indent=4
            )

        print(
            "\nHypotheses saved successfully:"
        )

        print(
            output_path
        )

        return hypotheses

    except requests.exceptions.Timeout:

        print(
            "\nLLM generation timed out."
        )

        print(
            "Ollama is running, but the model "
            "did not respond within the timeout."
        )

        return []

    except requests.exceptions.ConnectionError:

        print(
            "\nCould not connect to Ollama."
        )

        print(
            "Make sure Ollama is running and "
            f"'{OLLAMA_MODEL}' is installed."
        )

        return []

    except json.JSONDecodeError:

        print(
            "\nLLM returned invalid JSON."
        )

        print(
            "\nRaw response:"
        )

        print(text)

        return []

    except Exception as e:

        print(
            "\nLLM generation failed:"
        )

        print(e)

        return []


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "              LIGHTWEIGHT AI SCIENTIST"
    )

    print(
        "        DATA-DRIVEN HYPOTHESIS DISCOVERY"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    numerical, categorical = (
        profile_dataset(df)
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_dataset(
        df
    )

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    generate_eda(
        df
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = identify_target(
        df
    )

    print(
        f"\nSelected target variable: "
        f"{target}"
    )

    # --------------------------------------------------------
    # Pattern discovery
    # --------------------------------------------------------

    patterns = []

    patterns.extend(
        numerical_patterns(
            df,
            target
        )
    )

    patterns.extend(
        categorical_patterns(
            df,
            target
        )
    )

    patterns.extend(
        feature_importance(
            df,
            target
        )
    )

    # --------------------------------------------------------
    # Save patterns
    # --------------------------------------------------------

    patterns = save_patterns(
        patterns
    )

    # --------------------------------------------------------
    # Create SMALL LLM context
    # --------------------------------------------------------

    context = create_llm_context(
        df,
        target,
        patterns
    )

    # --------------------------------------------------------
    # Generate hypotheses
    # --------------------------------------------------------

    hypotheses = (
        generate_hypotheses_with_llm(
            context
        )
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCurrent implemented flow:"
    )

    print(
        "Dataset"
        " → Data Profiling"
        " → Cleaning"
        " → EDA"
        " → Pattern Discovery"
        " → Local LLM"
        " → Hypotheses"
    )

    print(
        "\nOutput files are available in:"
    )

    print(
        f"  {OUTPUT_DIR}/"
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()