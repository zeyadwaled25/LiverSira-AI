"""Apply documented human adjudication to the validation cohort.

The review CSV is an input artifact owned by the reviewer. This workflow never
fills or rewrites it. Only rows with complete review values and attributable,
parseable provenance can promote matching Ground Truth records to VERIFIED.
"""

from datetime import datetime
import os
from pathlib import Path
import re
import tempfile

import pandas as pd

from liversira_ai.nlp.schemas.ground_truth_schema import (
    GroundTruthRecord,
    ValidationStatus,
)

REVIEW_CSV = Path("data/processed/nlp/target_validation_review.csv")
GT_JSONL = Path("data/processed/nlp/ground_truth.jsonl")

REQUIRED_REVIEW_COLUMNS = (
    "BDMAP ID",
    "review_group",
    "review_presence",
    "reviewed_count",
    "reviewed_location",
    "reviewed_diameter_cm",
    "reviewed_attenuation",
    "targets_match_report",
    "review_notes",
    "reviewer_id",
    "reviewed_at",
)

ATTENUATIONS = {
    "hyperattenuating",
    "isoattenuating",
    "hypoattenuating",
}
REVIEW_GROUPS = {
    "NO_LESION",
    "LESION_WITH_LOCATION",
    "LESION_MISSING_LOCATION",
}


def _is_blank(value: object) -> bool:
    return pd.isna(value) or not str(value).strip()


def _parse_reviewed_at(value: object) -> datetime:
    if _is_blank(value):
        raise ValueError("reviewed_at is required")
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be a valid ISO-8601 datetime") from exc
    return parsed


def _parse_positive_float(value: object, field: str) -> float:
    if _is_blank(value):
        raise ValueError(f"{field} is required")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _parse_nonnegative_int(value: object, field: str) -> int:
    if _is_blank(value):
        raise ValueError(f"{field} is required")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not parsed.is_integer() or parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(parsed)


def _validate_review_row(row: pd.Series) -> None:
    group = str(row["review_group"]).strip()
    if group not in REVIEW_GROUPS:
        raise ValueError(f"review_group has unsupported value: {group!r}")

    presence = str(row["review_presence"]).strip()
    if presence not in {"YES", "NO"}:
        raise ValueError("review_presence must be YES or NO")

    count = _parse_nonnegative_int(row["reviewed_count"], "reviewed_count")
    location = str(row["reviewed_location"]).strip()
    diameter = str(row["reviewed_diameter_cm"]).strip()
    attenuation = str(row["reviewed_attenuation"]).strip()

    if presence == "NO":
        if count != 0 or location != "NOT_APPLICABLE":
            raise ValueError("NO review must have zero count and NOT_APPLICABLE location")
        if diameter != "NOT_APPLICABLE" or attenuation != "NOT_APPLICABLE":
            raise ValueError("NO review must mark diameter and attenuation NOT_APPLICABLE")
    else:
        if count <= 0:
            raise ValueError("YES review must have a positive lesion count")
        if location == "UNDOCUMENTED":
            pass
        elif not re.fullmatch(r"[1-8](?:\s*,\s*[1-8])*", location):
            raise ValueError("reviewed_location must contain Couinaud segments 1-8")
        _parse_positive_float(diameter, "reviewed_diameter_cm")
        if attenuation not in ATTENUATIONS:
            raise ValueError("reviewed_attenuation has an unsupported value")

    if str(row["targets_match_report"]).strip() not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError("targets_match_report must be YES, NO, or UNCERTAIN")
    if _is_blank(row["review_notes"]):
        raise ValueError("review_notes is required")


def _provenance_is_missing(row: pd.Series) -> bool:
    return _is_blank(row["reviewer_id"]) or _is_blank(row["reviewed_at"])


def _write_ground_truth(path: Path, records: list[GroundTruthRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            for record in records:
                temporary.write(record.model_dump_json() + "\n")
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        raise


def adjudicate_cohort(review_csv: Path, gt_jsonl: Path) -> int:
    """Promote only complete, attributable human reviews.

    Rows without reviewer provenance are explicitly marked
    ``needs_adjudication`` and are never promoted. Review fields are consumed
    as entered by a reviewer; Source Label columns are never copied into them.
    """
    df_review = pd.read_csv(review_csv, dtype=object, keep_default_na=False)
    missing_columns = [
        column for column in REQUIRED_REVIEW_COLUMNS if column not in df_review.columns
    ]
    if missing_columns:
        raise ValueError(f"Review CSV is missing required columns: {missing_columns}")

    review_ids = [str(value).strip() for value in df_review["BDMAP ID"]]
    if any(not value for value in review_ids):
        raise ValueError("Review CSV contains an empty BDMAP ID")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Review CSV contains duplicate BDMAP IDs")

    with gt_jsonl.open("r", encoding="utf-8") as f_in:
        records = [GroundTruthRecord.model_validate_json(line) for line in f_in if line.strip()]
    records_by_id = {record.id: record for record in records}
    unknown_ids = sorted(set(review_ids) - set(records_by_id))
    if unknown_ids:
        raise ValueError(f"Review CSV contains IDs absent from Ground Truth: {unknown_ids}")

    verified_count = 0
    for idx, row in df_review.iterrows():
        record = records_by_id[review_ids[idx]]
        if _provenance_is_missing(row):
            record.validation.status = ValidationStatus.NEEDS_ADJUDICATION
            record.validation.verified_fields = []
            record.validation.unverified_fields = [
                "has_lesion",
                "lesion_count",
                "largest_lesion.location_segments",
                "largest_lesion.source_diameter_cm",
                "largest_lesion.attenuation",
            ]
            record.validation.review_notes = [
                "Human reviewer ID and reviewed_at provenance are missing."
            ]
            continue

        _parse_reviewed_at(row["reviewed_at"])
        if not str(row["reviewer_id"]).strip():
            raise ValueError(f"Row {idx} has an empty reviewer_id")
        _validate_review_row(row)

        record.validation.status = ValidationStatus.VERIFIED
        if str(row["review_presence"]).strip() == "NO":
            record.validation.verified_fields = ["has_lesion", "lesion_count"]
            record.validation.unverified_fields = []
        else:
            record.validation.verified_fields = [
                "has_lesion",
                "lesion_count",
                "largest_lesion.location_segments",
                "largest_lesion.source_diameter_cm",
                "largest_lesion.attenuation",
            ]
            record.validation.unverified_fields = []
        record.validation.review_notes = [
            f"Human adjudication recorded by reviewer {str(row['reviewer_id']).strip()} "
            f"at {str(row['reviewed_at']).strip()}: {str(row['review_notes']).strip()}"
        ]
        verified_count += 1

    _write_ground_truth(gt_jsonl, list(records_by_id.values()))
    return verified_count


def main() -> None:
    """Execute the human adjudication workflow."""
    adjudicate_cohort(REVIEW_CSV, GT_JSONL)


if __name__ == "__main__":
    main()
