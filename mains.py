import warnings

# pyrefly: ignore [missing-import]
from modules.config import DATASET_PATH, OUTPUT_DIR, TOP_K_CANDIDATES, CV_FOLDS
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
from modules.individual_dependency_analysis import individual_dependency_analysis
# pyrefly: ignore [missing-import]
from modules.feature_combination_analysis import feature_combination_analysis
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

    # Ingestion & Profiling Layer
    df = load_dataset(DATASET_PATH)
    numerical, categorical = profile_dataset(df)
    df = clean_dataset(df)
    generate_eda(df)
    target = identify_target(df)

    print(f"\nSelected target variable: {target}")

    patterns = []

    # Stage 1: Individual Feature Dependency Analysis
    stage1 = individual_dependency_analysis(df, target)
    patterns.extend(stage1.get("patterns", []))

    # Stage 2: Feature Combination Discovery & Interaction Analysis
    stage2 = feature_combination_analysis(
        df=df,
        target=target,
        top_candidates=stage1.get("ranked_features", []),
        max_candidates=TOP_K_CANDIDATES,
        cv_folds=CV_FOLDS
    )
    patterns.extend(stage2.get("patterns", []))

    # Complementary Categorical and Tree Feature Importance signals
    patterns.extend(categorical_patterns(df, target))
    patterns.extend(feature_importance(df, target))

    # Save Patterns & Build LLM Context
    patterns = save_patterns(patterns)
    context = create_llm_context(
        df=df,
        target=target,
        patterns=patterns,
        stage1_data=stage1,
        stage2_data=stage2
    )

    # Stage 3: LLM Hypothesis Generation
    hypotheses = generate_hypotheses_with_llm(context)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nCurrent implemented flow:")
    print("Dataset -> Profiling -> Cleaning -> EDA -> Stage 1 (Individual Dependency) -> Stage 2 (Feature Combination & Interaction) -> Local LLM -> Hypotheses")
    print("\nOutput files are available in:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  - {OUTPUT_DIR}/individual_dependencies.csv")
    print(f"  - {OUTPUT_DIR}/feature_combinations.csv")
    print(f"  - {OUTPUT_DIR}/llm_context.json")
    print(f"  - {OUTPUT_DIR}/generated_hypotheses.json")


if __name__ == "__main__":
    main()
