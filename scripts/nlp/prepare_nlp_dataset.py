"""Clean the AbdomenAtlas CSV into data/processed/nlp/liver_nlp_clean.csv (the raw CSV is not modified)."""

from pathlib import Path

import pandas as pd

from liversira_ai.nlp.data.raw_csv import (
    ATTENUATION,
    COUNT,
    DIAMETER,
    ID,
    LOCATION,
    TEXT,
    clean_report,
)

INPUT_PATH = Path("data/raw/nlp/AbdomenAtlas3.0MiniWithMeta.csv")
OUTPUT_PATH = Path("data/processed/nlp/liver_nlp_clean.csv")


def prepare_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the full preparation pipeline and return the cleaned DataFrame."""
    # 1. Load and validate
    df = pd.read_csv(input_path)

    required = [ID, TEXT, COUNT, LOCATION, DIAMETER, ATTENUATION]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    original_rows = len(df)

    # 2. Preserve and clean report text
    df["narrative_original"] = df[TEXT]
    df["narrative_clean"] = df[TEXT].apply(clean_report)

    # 3. Exclude records without usable report text
    missing_text = df["narrative_clean"].isna()
    excluded_ids = df.loc[missing_text, ID].tolist()
    df = df.loc[~missing_text].copy()

    # 4. Standardize missing liver-lesion values
    count = pd.to_numeric(df[COUNT], errors="coerce")

    for column in [LOCATION, DIAMETER, ATTENUATION]:
        df[column] = df[column].astype("string")

    no_lesions = count.eq(0)
    has_lesions = count.gt(0)

    # No liver lesions -> largest-lesion fields do not apply.
    df.loc[no_lesions, LOCATION] = "NO_LESION"
    df.loc[no_lesions, DIAMETER] = "NOT_APPLICABLE"
    df.loc[no_lesions, ATTENUATION] = "NOT_APPLICABLE"

    # Lesions exist but a target value is missing.
    for column in [LOCATION, DIAMETER, ATTENUATION]:
        missing_value = has_lesions & df[column].isna()
        df.loc[missing_value, column] = "UNKNOWN"

    # 5. Keep only columns needed for this NLP task
    output_columns = [
        ID,
        "narrative_original",
        "narrative_clean",
        COUNT,
        LOCATION,
        DIAMETER,
        ATTENUATION,
    ]

    nlp_df = df[output_columns].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nlp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("LIVER NLP DATASET SUMMARY")

    print(f"Original rows: {original_rows:,}")
    print(f"Rows excluded due to missing report: {len(excluded_ids)}")
    print(f"Rows in output: {len(nlp_df):,}")
    print(f"Output columns: {len(nlp_df.columns)}")

    if excluded_ids:
        print(f"Excluded IDs: {excluded_ids}")

    print("\nPlaceholder counts:")
    for column in [LOCATION, DIAMETER, ATTENUATION]:
        print(f"\n{column}:")
        print(nlp_df[column].value_counts(dropna=False).to_string())

    print("\nRemaining missing values:")
    for column in [COUNT, LOCATION, DIAMETER, ATTENUATION]:
        print(f"{column}: {nlp_df[column].isna().sum():,}")

    print(f"\nSaved to: {output_path}")
    print("Original CSV was not modified.")

    return nlp_df


def main() -> None:
    """Entry point."""
    prepare_dataset(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
