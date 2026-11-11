"""Safety tests for the human adjudication workflow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from liversira_ai.nlp.schemas.ground_truth_schema import (
    GroundTruthLabels,
    GroundTruthRecord,
    LargestLesionGT,
    ValidationMetadata,
    ValidationStatus,
)
from liversira_ai.nlp.data.adjudication import adjudicate_cohort


def _write_gt(path: Path, *, positive: bool = True) -> None:
    lesion = (
        LargestLesionGT(
            location_segments=[5],
            source_diameter_cm=3.0,
            attenuation="hypoattenuating",
        )
        if positive
        else None
    )
    record = GroundTruthRecord(
        id="A-1",
        ground_truth=GroundTruthLabels(
            has_lesion=positive,
            lesion_count=1 if positive else 0,
            largest_lesion=lesion,
        ),
        validation=ValidationMetadata(
            status=ValidationStatus.PENDING,
            unverified_fields=["has_lesion"],
        ),
    )
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")


def _write_review(path: Path, **overrides: object) -> None:
    row = {
        "BDMAP ID": "A-1",
        "review_group": "LESION_WITH_LOCATION",
        "number of liver lesion instances": "99",
        "largest liver lesion location (1-8)": "8",
        "largest liver lesion diameter (cm)": "99",
        "largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)": "hyperattenuating",
        "review_presence": "YES",
        "reviewed_count": "1",
        "reviewed_location": "5",
        "reviewed_diameter_cm": "3.0",
        "reviewed_attenuation": "hypoattenuating",
        "targets_match_report": "YES",
        "review_notes": "Reviewed against narrative.",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-01-15T10:30:00Z",
    }
    row.update(overrides)
    pd.DataFrame([row]).to_csv(path, index=False)


def _status(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])["validation"]


def test_complete_attributable_review_promotes_without_rewriting_csv(tmp_path: Path):
    review = tmp_path / "target_validation_review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review)
    _write_gt(ground_truth)
    before = review.read_bytes()

    assert adjudicate_cohort(review, ground_truth) == 1
    assert review.read_bytes() == before
    assert _status(ground_truth)["status"] == "verified"


def test_missing_provenance_becomes_needs_adjudication(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review, reviewer_id="", reviewed_at="")
    _write_gt(ground_truth)

    assert adjudicate_cohort(review, ground_truth) == 0
    status = _status(ground_truth)
    assert status["status"] == "needs_adjudication"
    assert status["verified_fields"] == []


def test_missing_provenance_downgrades_legacy_verified_status(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review, reviewer_id="", reviewed_at="")
    _write_gt(ground_truth)
    record = json.loads(ground_truth.read_text(encoding="utf-8"))
    record["validation"]["status"] = "verified"
    ground_truth.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert adjudicate_cohort(review, ground_truth) == 0
    assert _status(ground_truth)["status"] == "needs_adjudication"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewed_at", "not-a-date"),
        ("reviewed_count", "one"),
        ("reviewed_diameter_cm", "-1"),
        ("reviewed_attenuation", "unknown"),
        ("review_notes", ""),
    ],
)
def test_invalid_review_fields_are_rejected(tmp_path: Path, field: str, value: str):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review, **{field: value})
    _write_gt(ground_truth)

    with pytest.raises(ValueError):
        adjudicate_cohort(review, ground_truth)
    assert _status(ground_truth)["status"] == "pending"


def _gt_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_reviewed_values_that_differ_are_not_marked_verified(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review, reviewed_count="2", reviewed_diameter_cm="4.0")
    _write_gt(ground_truth)

    assert adjudicate_cohort(review, ground_truth) == 0
    validation = _status(ground_truth)
    assert validation["status"] == "partially_verified"
    assert "lesion_count" in validation["unverified_fields"]
    assert "largest_lesion.source_diameter_cm" in validation["unverified_fields"]
    assert "largest_lesion.attenuation" in validation["verified_fields"]
    # Source labels are never overwritten by review values.
    assert _gt_record(ground_truth)["ground_truth"]["lesion_count"] == 1


def test_presence_disagreement_needs_adjudication(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(
        review,
        review_presence="NO",
        reviewed_count="0",
        reviewed_location="NOT_APPLICABLE",
        reviewed_diameter_cm="NOT_APPLICABLE",
        reviewed_attenuation="NOT_APPLICABLE",
    )
    _write_gt(ground_truth)

    assert adjudicate_cohort(review, ground_truth) == 0
    assert _status(ground_truth)["status"] == "needs_adjudication"


def test_undocumented_fields_are_accepted_and_written_to_gold(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write_review(
        review,
        reviewed_count="UNDOCUMENTED",
        reviewed_diameter_cm="UNDOCUMENTED",
        evidence_quote="A liver lesion is seen in segment 5.",
    )
    _write_gt(ground_truth)

    adjudicate_cohort(review, ground_truth, gold)
    record = json.loads(gold.read_text(encoding="utf-8").splitlines()[0])
    assert record["gold"] == {
        "has_lesion": True,
        "lesion_count": None,
        "largest_lesion": {
            "location_segments": [5],
            "diameter_cm": None,
            "attenuation": "hypoattenuating",
        },
    }
    assert record["provenance"]["method"] == "single_review"
    assert record["provenance"]["evidence_quote"] == "A liver lesion is seen in segment 5."


def test_source_columns_are_not_copied_into_review_fields(tmp_path: Path):
    review = tmp_path / "review.csv"
    ground_truth = tmp_path / "ground_truth.jsonl"
    _write_review(review, reviewed_count="", reviewed_diameter_cm="")
    _write_gt(ground_truth)

    with pytest.raises(ValueError):
        adjudicate_cohort(review, ground_truth)
    source = pd.read_csv(review, dtype=object, keep_default_na=False)
    assert source.loc[0, "reviewed_count"] == ""
    assert source.loc[0, "reviewed_diameter_cm"] == ""
