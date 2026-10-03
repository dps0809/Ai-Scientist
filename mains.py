import warnings

# pyrefly: ignore [missing-import]
from modules.config import DATASET_PATH, OUTPUT_DIR
# pyrefly: ignore [missing-import]
from modules.load_dataset import load_dataset
# pyrefly: ignore [missing-import]
from modules.profile_dataset import profile_dataset
# pyrefly: ignore [missing-import]
from modules.clean_dataset import clean_dataset
# pyrefly: ignore [missing-import]
from modules.generate_eda import generate_eda
# pyrefly: ignore [missing-import]
from modules.identify_target import identify_target
# pyrefly: ignore [missing-import]
from modules.numerical_patterns import numerical_patterns
# pyrefly: ignore [missing-import]
from modules.categorical_patterns import categorical_patterns
# pyrefly: ignore [missing-import]
from modules.feature_importance import feature_importance
# pyrefly: ignore [missing-import]
from modules.save_patterns import save_patterns
# pyrefly: ignore [missing-import]
from modules.create_llm_context import create_llm_context
# pyrefly: ignore [missing-import]
from modules.generate_hypotheses_with_llm import generate_hypotheses_with_llm

warnings.filterwarnings("ignore")


def main():
    print("\n")
    print("=" * 70)
    print("              LIGHTWEIGHT AI SCIENTIST")
    print("        DATA-DRIVEN HYPOTHESIS DISCOVERY")
    print("=" * 70)

    df = load_dataset(DATASET_PATH)
    numerical, categorical = profile_dataset(df)
    df = clean_dataset(df)
    generate_eda(df)
    target = identify_target(df)

    print(f"\nSelected target variable: {target}")

    patterns = []
    patterns.extend(numerical_patterns(df, target))
    patterns.extend(categorical_patterns(df, target))
    patterns.extend(feature_importance(df, target))

    patterns = save_patterns(patterns)
    context = create_llm_context(df, target, patterns)
    hypotheses = generate_hypotheses_with_llm(context)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nCurrent implemented flow:")
    print("Dataset -> Data Profiling -> Cleaning -> EDA -> Pattern Discovery -> Local LLM -> Hypotheses")
    print("\nOutput files are available in:")
    print(f"  {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
