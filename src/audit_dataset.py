"""Audit the AbdomenAtlas CSV for data-quality issues.

Prints a structured report covering:
1. Dataset overview (rows, duplicates, missing IDs/reports).
2. Columns ranked by missingness.
3. Liver-lesion target consistency checks.
4. Narrative-report text quality.
5. Specific case IDs that need manual review.

The original CSV is **never** modified.
"""

from pathlib import Path

import pandas as pd

# ── column name constants ──────────────────────────────────────
ID = "BDMAP ID"
TEXT = "narrative report"
COUNT = "number of liver lesion instances"
LOCATION = "largest liver lesion location (1-8)"
DIAMETER = "largest liver lesion diameter (cm)"
ATTENUATION = (
    "largest liver lesion attenuation "
    "(hyperattenuating, isoattenuating, hypoattenuating)"
)


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and ensure all required columns exist."""
    df = pd.read_csv(csv_path)

    required = [ID, TEXT, COUNT, LOCATION, DIAMETER, ATTENUATION]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def print_overview(df: pd.DataFrame) -> None:
    """Section 1 – basic dataset statistics."""
    text = df[TEXT].astype("string")

    print("\n" + "=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate IDs: {df[ID].duplicated().sum()}")
    print(f"Missing IDs: {df[ID].isna().sum()}")
    print(f"Missing reports: {text.isna().sum()}")
    print(f"Empty reports: {text.str.strip().eq('').fillna(False).sum()}")


def print_missing_values(df: pd.DataFrame) -> None:
    """Section 2 – columns ranked by percentage of missing values."""
    print("\n" + "=" * 60)
    print("2. COLUMNS WITH MOST MISSING VALUES")
    print("=" * 60)

    stats = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    ).sort_values("missing_%", ascending=False)

    print(stats.head(10).to_string())
    print("\nHigh missingness does not automatically mean deletion.")


def print_target_checks(df: pd.DataFrame) -> None:
    """Section 3 – consistency between lesion count and target columns."""
    count = pd.to_numeric(df[COUNT], errors="coerce")
    location = df[LOCATION]
    diameter = pd.to_numeric(df[DIAMETER], errors="coerce")
    attenuation = df[ATTENUATION]

    no_lesions = count.eq(0)
    has_lesions = count.gt(0)

    print("\n" + "=" * 60)
    print("3. LIVER TARGET RELATIONSHIPS")
    print("=" * 60)

    checks = {
        "Rows with zero lesions": no_lesions.sum(),
        "Rows with one or more lesions": has_lesions.sum(),
        "Missing lesion count": count.isna().sum(),
        "Negative lesion counts": count.lt(0).sum(),
        "Lesions present but location missing": (has_lesions & location.isna()).sum(),
        "No lesions but location present": (no_lesions & location.notna()).sum(),
        "Lesions present but diameter missing": (has_lesions & diameter.isna()).sum(),
        "Lesions present but diameter zero": (has_lesions & diameter.eq(0)).sum(),
        "Negative diameters": diameter.lt(0).sum(),
        "No lesions but nonzero diameter": (
            no_lesions & diameter.notna() & diameter.ne(0)
        ).sum(),
        "Lesions present but attenuation missing": (
            has_lesions & attenuation.isna()
        ).sum(),
    }

    for label, value in checks.items():
        print(f"{label}: {int(value)}")


def print_text_quality(df: pd.DataFrame) -> None:
    """Section 4 – narrative-report length and duplicate stats."""
    text = df[TEXT].astype("string")

    print("\n" + "=" * 60)
    print("4. REPORT TEXT QUALITY")
    print("=" * 60)

    lengths = text.str.len()

    print(f"Average characters: {lengths.mean():.1f}")
    print(f"Median characters: {lengths.median():.0f}")
    print(f"Shortest report: {lengths.min()}")
    print(f"Longest report: {lengths.max()}")

    normalized = text.str.strip().str.lower()
    duplicates = normalized.notna() & normalized.duplicated(keep=False)

    print(f"Rows with duplicated report text: {duplicates.sum()}")


def print_review_cases(df: pd.DataFrame) -> None:
    """Section 5 – specific IDs that need manual review."""
    text = df[TEXT].astype("string")
    count = pd.to_numeric(df[COUNT], errors="coerce")
    location = df[LOCATION]
    has_lesions = count.gt(0)

    print("\n" + "=" * 60)
    print("5. CASES TO REVIEW")
    print("=" * 60)

    missing_report = df[text.isna() | text.str.strip().eq("").fillna(False)]

    print("\nIDs with missing reports:")
    print(missing_report[ID].to_list() if not missing_report.empty else "None")

    missing_location = df[has_lesions & location.isna()]

    print("\nLesions present but location missing (first 10):")

    if missing_location.empty:
        print("None")
    else:
        review = missing_location[[ID, COUNT, LOCATION, DIAMETER, TEXT]].copy()
        review[TEXT] = review[TEXT].astype("string").str.slice(0, 180)
        print(review.head(10).to_string(index=False))


def main() -> None:
    """Run the full audit pipeline."""
    csv_path = Path("data/raw/AbdomenAtlas3.0MiniWithMeta.csv")
    df = load_and_validate(csv_path)

    print_overview(df)
    print_missing_values(df)
    print_target_checks(df)
    print_text_quality(df)
    print_review_cases(df)

    print("\n" + "=" * 60)
    print("AUDIT FINISHED - ORIGINAL CSV WAS NOT MODIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()
