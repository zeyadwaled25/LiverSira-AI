"""Generate a stratified review cohort for human clinical adjudication.

Samples:
- 30 randomly selected NO_LESION studies.
- 30 randomly selected positive lesion studies with documented Couinaud location.
- All 19 positive lesion cases where Couinaud location is missing/undocumented.
Total: 79 cases saved to data/processed/target_validation_review.csv.

Used by reviewers to audit target labels before upgrading validation_status
from 'pending' to 'verified'.
"""

from pathlib import Path
import pandas as pd

# 1. Paths
INPUT_PATH = Path("data/processed/liver_nlp_clean.csv")
OUTPUT_PATH = Path("data/processed/target_validation_review.csv")

ID = "BDMAP ID"
TEXT = "narrative_original"
COUNT = "number of liver lesion instances"
LOCATION = "largest liver lesion location (1-8)"
DIAMETER = "largest liver lesion diameter (cm)"
ATTENUATION = (
    "largest liver lesion attenuation "
    "(hyperattenuating, isoattenuating, hypoattenuating)"
)


def create_validation_sample(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Extract reproducible review sample and write CSV."""
    df = pd.read_csv(input_path)

    required = [ID, TEXT, COUNT, LOCATION, DIAMETER, ATTENUATION]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    count = pd.to_numeric(df[COUNT], errors="coerce")

    # Divide records into review groups
    no_lesion = df[count.eq(0)]
    lesion_with_location = df[
        count.gt(0) & df[LOCATION].notna() & df[LOCATION].ne("UNKNOWN")
    ]
    lesion_missing_location = df[
        count.gt(0) & df[LOCATION].eq("UNKNOWN")
    ]

    # Select reproducible sample
    sample_no_lesion = no_lesion.sample(
        n=min(30, len(no_lesion)),
        random_state=42,
    )
    sample_with_location = lesion_with_location.sample(
        n=min(30, len(lesion_with_location)),
        random_state=42,
    )
    # Include all cases with missing location
    sample_missing_location = lesion_missing_location.copy()

    review = pd.concat(
        [
            sample_no_lesion,
            sample_with_location,
            sample_missing_location,
        ],
        ignore_index=True,
    )

    # Review audit fields
    review["review_group"] = (
        ["NO_LESION"] * len(sample_no_lesion)
        + ["LESION_WITH_LOCATION"] * len(sample_with_location)
        + ["LESION_MISSING_LOCATION"] * len(sample_missing_location)
    )
    review["review_presence"] = ""
    review["reviewed_count"] = ""
    review["reviewed_location"] = ""
    review["reviewed_diameter_cm"] = ""
    review["reviewed_attenuation"] = ""
    review["targets_match_report"] = ""
    review["review_notes"] = ""

    columns = [
        ID,
        "review_group",
        TEXT,
        COUNT,
        LOCATION,
        DIAMETER,
        ATTENUATION,
        "review_presence",
        "reviewed_count",
        "reviewed_location",
        "reviewed_diameter_cm",
        "reviewed_attenuation",
        "targets_match_report",
        "review_notes",
    ]

    review = review[columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 55)
    print("TARGET VALIDATION SAMPLE")
    print("=" * 55)
    print(f"No-lesion cases: {len(sample_no_lesion)}")
    print(f"Lesion cases with location: {len(sample_with_location)}")
    print(f"Lesion cases with missing location: {len(sample_missing_location)}")
    print(f"Total review cases: {len(review)}")
    print(f"\nSaved to: {output_path}")
    print("=" * 55)

    return review


def main() -> None:
    """Run validation sample extraction."""
    create_validation_sample(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
