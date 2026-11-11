"""Column names, loading and text cleaning for the raw AbdomenAtlas CSV."""

import re
from pathlib import Path

import pandas as pd

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


def clean_report(value: object) -> str | pd.NA:
    """Normalize whitespace and strip a single report string."""
    if pd.isna(value):
        return pd.NA

    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else pd.NA
