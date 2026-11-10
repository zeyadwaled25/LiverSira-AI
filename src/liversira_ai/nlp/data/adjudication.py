"""Apply documented human adjudication to the validation cohort.

The review CSV is an input artifact owned by the reviewer. This workflow never
fills or rewrites it. Only rows with complete review values and attributable,
parseable provenance change a Ground Truth record's validation status, and the
status reflects whether the reviewed values *agree* with the source label:

* every reviewed field equals the source label -> ``verified``
* presence agrees but some fields differ or are undocumented -> ``partially_verified``
* presence disagrees -> ``needs_adjudication``

Source labels themselves are never overwritten. The reviewed values are written
separately as human gold labels (``GoldStandardRecord``) when ``gold_jsonl`` is
given.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import tempfile
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from liversira_ai.nlp.schemas.ground_truth_schema import (
    GroundTruthRecord,
    ValidationStatus,
)
from liversira_ai.nlp.schemas.gold_standard_schema import (
    GoldLabels,
    GoldLargestLesion,
    GoldMethod,
    GoldProvenance,
    GoldStandardRecord,
)

REVIEW_CSV = Path("data/processed/nlp/target_validation_review.csv")
GT_JSONL = Path("data/processed/nlp/ground_truth.jsonl")
GOLD_JSONL = Path("data/processed/nlp/gold_standard.jsonl")

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
# Optional columns written by the two-annotator gold-set workflow.
OPTIONAL_REVIEW_COLUMNS = ("review_method", "annotator_ids", "evidence_quote")

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
UNDOCUMENTED = "UNDOCUMENTED"
NOT_APPLICABLE = "NOT_APPLICABLE"
DIAMETER_MATCH_TOLERANCE_CM = 0.05
ALL_FIELDS = [
    "has_lesion",
    "lesion_count",
    "largest_lesion.location_segments",
    "largest_lesion.source_diameter_cm",
    "largest_lesion.attenuation",
]


@dataclass(frozen=True)
class ReviewedValues:
    """Parsed reviewer values; ``None`` means the report does not state the field."""

    has_lesion: bool
    lesion_count: Optional[int]
    location_segments: Optional[list[int]]
    diameter_cm: Optional[float]
    attenuation: Optional[str]


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


def parse_review_values(
    presence: object,
    count: object,
    location: object,
    diameter: object,
    attenuation: object,
) -> ReviewedValues:
    """Validate and parse one set of reviewed values.

    ``presence`` is YES or NO. For NO, every other field must be NOT_APPLICABLE
    (count may be 0). For YES, each field holds a value or UNDOCUMENTED.
    """
    presence_text = str(presence).strip()
    if presence_text not in {"YES", "NO"}:
        raise ValueError("review_presence must be YES or NO")
    count_text = str(count).strip()
    location_text = str(location).strip()
    diameter_text = str(diameter).strip()
    attenuation_text = str(attenuation).strip()

    if presence_text == "NO":
        if count_text not in {"0", NOT_APPLICABLE} and _parse_nonnegative_int(count, "reviewed_count") != 0:
            raise ValueError("NO review must have zero count")
        if location_text != NOT_APPLICABLE:
            raise ValueError("NO review must have zero count and NOT_APPLICABLE location")
        if diameter_text != NOT_APPLICABLE or attenuation_text != NOT_APPLICABLE:
            raise ValueError("NO review must mark diameter and attenuation NOT_APPLICABLE")
        return ReviewedValues(False, 0, None, None, None)

    if count_text == UNDOCUMENTED:
        parsed_count = None
    else:
        parsed_count = _parse_nonnegative_int(count, "reviewed_count")
        if parsed_count <= 0:
            raise ValueError("YES review must have a positive lesion count or UNDOCUMENTED")

    if location_text == UNDOCUMENTED:
        segments = None
    elif re.fullmatch(r"[1-8](?:\s*,\s*[1-8])*", location_text):
        segments = sorted({int(value) for value in re.findall(r"[1-8]", location_text)})
    else:
        raise ValueError("reviewed_location must contain Couinaud segments 1-8 or UNDOCUMENTED")

    parsed_diameter = (
        None
        if diameter_text == UNDOCUMENTED
        else _parse_positive_float(diameter, "reviewed_diameter_cm")
    )

    if attenuation_text == UNDOCUMENTED:
        parsed_attenuation = None
    elif attenuation_text in ATTENUATIONS:
        parsed_attenuation = attenuation_text
    else:
        raise ValueError("reviewed_attenuation has an unsupported value")

    return ReviewedValues(True, parsed_count, segments, parsed_diameter, parsed_attenuation)


def _validate_review_row(row: pd.Series) -> ReviewedValues:
    group = str(row["review_group"]).strip()
    if group not in REVIEW_GROUPS:
        raise ValueError(f"review_group has unsupported value: {group!r}")
    values = parse_review_values(
        row["review_presence"],
        row["reviewed_count"],
        row["reviewed_location"],
        row["reviewed_diameter_cm"],
        row["reviewed_attenuation"],
    )
    if str(row["targets_match_report"]).strip() not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError("targets_match_report must be YES, NO, or UNCERTAIN")
    if _is_blank(row["review_notes"]):
        raise ValueError("review_notes is required")
    return values


def _provenance_is_missing(row: pd.Series) -> bool:
    return _is_blank(row["reviewer_id"]) or _is_blank(row["reviewed_at"])


def compare_with_source(record: GroundTruthRecord, values: ReviewedValues) -> dict[str, bool]:
    """Per field: does the reviewed value equal the source-derived label?"""
    truth = record.ground_truth
    agreement = {"has_lesion": values.has_lesion == truth.has_lesion}
    if not (values.has_lesion and truth.has_lesion):
        agreement["lesion_count"] = values.lesion_count == truth.lesion_count
        return agreement
    largest = truth.largest_lesion
    agreement["lesion_count"] = values.lesion_count == truth.lesion_count
    agreement["largest_lesion.location_segments"] = (values.location_segments or None) == (
        largest.location_segments or None
    )
    agreement["largest_lesion.source_diameter_cm"] = (
        values.diameter_cm is not None
        and largest.source_diameter_cm is not None
        and abs(values.diameter_cm - largest.source_diameter_cm) <= DIAMETER_MATCH_TOLERANCE_CM
    )
    source_attenuation = largest.attenuation.value if largest.attenuation else None
    agreement["largest_lesion.attenuation"] = values.attenuation == source_attenuation
    return agreement


def _gold_record(row: pd.Series, record_id: str, values: ReviewedValues, agreement: dict[str, bool]) -> GoldStandardRecord:
    method_text = "" if _is_blank(row.get("review_method", "")) else str(row["review_method"]).strip()
    method = GoldMethod(method_text) if method_text else GoldMethod.SINGLE_REVIEW
    reviewer = str(row["reviewer_id"]).strip()
    annotators_text = "" if _is_blank(row.get("annotator_ids", "")) else str(row["annotator_ids"])
    annotators = [item.strip() for item in annotators_text.split(";") if item.strip()] or [reviewer]
    evidence = None if _is_blank(row.get("evidence_quote", "")) else str(row["evidence_quote"]).strip()
    largest = (
        GoldLargestLesion(
            location_segments=values.location_segments,
            diameter_cm=values.diameter_cm,
            attenuation=values.attenuation,
        )
        if values.has_lesion
        else None
    )
    return GoldStandardRecord(
        id=record_id,
        gold=GoldLabels(
            has_lesion=values.has_lesion,
            lesion_count=values.lesion_count,
            largest_lesion=largest,
        ),
        provenance=GoldProvenance(
            method=method,
            annotators=annotators,
            adjudicator=reviewer if method == GoldMethod.CLINICIAN_ADJUDICATION else None,
            reviewed_at=str(row["reviewed_at"]).strip(),
            notes=str(row["review_notes"]).strip(),
            evidence_quote=evidence,
            source_label_agreement=agreement,
        ),
    )


def _write_jsonl_atomic(path: Path, records: list[BaseModel]) -> None:
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


def adjudicate_cohort(review_csv: Path, gt_jsonl: Path, gold_jsonl: Path | None = None) -> int:
    """Apply complete, attributable human reviews; return the number fully verified.

    Rows without reviewer provenance are explicitly marked
    ``needs_adjudication`` and are never promoted. Review fields are consumed
    as entered by a reviewer; Source Label columns are never copied into them.
    When ``gold_jsonl`` is given, every attributable review is also written as
    a human gold label (the file is rebuilt from this CSV on each run).
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

    # Validate every attributable row before changing anything.
    parsed: dict[int, ReviewedValues] = {}
    for idx, row in df_review.iterrows():
        if _provenance_is_missing(row):
            continue
        _parse_reviewed_at(row["reviewed_at"])
        parsed[idx] = _validate_review_row(row)

    verified_count = 0
    gold_records: list[GoldStandardRecord] = []
    for idx, row in df_review.iterrows():
        record = records_by_id[review_ids[idx]]
        if idx not in parsed:
            record.validation.status = ValidationStatus.NEEDS_ADJUDICATION
            record.validation.verified_fields = []
            record.validation.unverified_fields = list(ALL_FIELDS)
            record.validation.review_notes = [
                "Human reviewer ID and reviewed_at provenance are missing."
            ]
            continue

        values = parsed[idx]
        agreement = compare_with_source(record, values)
        verified = [field for field, agrees in agreement.items() if agrees]
        unverified = [field for field, agrees in agreement.items() if not agrees]
        if not agreement["has_lesion"]:
            status = ValidationStatus.NEEDS_ADJUDICATION
        elif unverified:
            status = ValidationStatus.PARTIALLY_VERIFIED
        else:
            status = ValidationStatus.VERIFIED
            verified_count += 1

        reviewer = str(row["reviewer_id"]).strip()
        notes = [
            f"Human adjudication recorded by reviewer {reviewer} "
            f"at {str(row['reviewed_at']).strip()}: {str(row['review_notes']).strip()}"
        ]
        if unverified:
            notes.append(
                "Reviewed values differ from the source label for: "
                + ", ".join(unverified)
                + ". Source label left unchanged; see the gold standard."
            )
        record.validation.status = status
        record.validation.verified_fields = verified
        record.validation.unverified_fields = unverified
        record.validation.review_notes = notes
        gold_records.append(_gold_record(row, record.id, values, agreement))

    _write_jsonl_atomic(gt_jsonl, list(records_by_id.values()))
    if gold_jsonl is not None:
        _write_jsonl_atomic(gold_jsonl, gold_records)
    return verified_count


def main() -> None:
    """Execute the human adjudication workflow."""
    adjudicate_cohort(REVIEW_CSV, GT_JSONL, GOLD_JSONL)


if __name__ == "__main__":
    main()
