"""Unit tests for the automated target consistency audit logic."""

import json
from pathlib import Path

import pytest

from liversira_ai.nlp.data.audit import (
    audit_single_record,
    run_consistency_audit,
    split_sentences,
)


def test_split_sentences_decimal_protection():
    """Decimal points in numbers (e.g. '86.6') should not cause split."""
    text = "The liver demonstrates a mean HU value of 86.6 +/- 19.1. Spleen is normal."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert "86.6" in sents[0]


def test_audit_single_record_match():
    """Clean positive record matching GT returns MATCH across all fields."""
    master = {
        "id": "TEST_001",
        "reports": {
            "narrative_clean": (
                "The patient has a normal-sized spleen. However, there are two hypoattenuating "
                "liver lesions identified. The largest lesion is located in hepatic segment 8, "
                "measuring 2.5 x 1.8 cm."
            )
        },
    }
    gt = {
        "id": "TEST_001",
        "ground_truth": {
            "has_lesion": True,
            "lesion_count": 2,
            "largest_lesion": {
                "location_segments": [8],
                "source_diameter_cm": 2.5,
                "attenuation": "hypoattenuating",
            },
        },
    }
    result = audit_single_record(master, gt)
    assert result["has_lesion_status"] == "MATCH"
    assert result["count_status"] == "MATCH"
    assert result["location_status"] == "MATCH"
    assert result["diameter_status"] == "MATCH"
    assert result["attenuation_status"] == "MATCH"
    assert result["overall_status"] == "MATCH"


def test_audit_single_record_count_mismatch():
    """Count mismatch should flag count_status and overall_status as MISMATCH."""
    master = {
        "id": "TEST_002",
        "reports": {
            "narrative_clean": (
                "There is a single hypoattenuating liver lesion in segment 2, measuring 1.5 cm."
            )
        },
    }
    gt = {
        "id": "TEST_002",
        "ground_truth": {
            "has_lesion": True,
            "lesion_count": 5,  # Mismatch with text (which says single / 1)
            "largest_lesion": {
                "location_segments": [2],
                "source_diameter_cm": 1.5,
                "attenuation": "hypoattenuating",
            },
        },
    }
    result = audit_single_record(master, gt)
    assert result["count_status"] == "MISMATCH"
    assert result["overall_status"] == "MISMATCH"
    assert "Count mismatch" in result["discrepancy_details"]


def test_audit_single_record_negative_case():
    """Negative findings in both GT and text must return MATCH."""
    master = {
        "id": "TEST_003",
        "reports": {
            "narrative_clean": (
                "The spleen is normal. The liver is normal in size and appearance. "
                "No focal lesions are observed in the liver."
            )
        },
    }
    gt = {
        "id": "TEST_003",
        "ground_truth": {
            "has_lesion": False,
            "lesion_count": 0,
            "largest_lesion": None,
        },
    }
    result = audit_single_record(master, gt)
    assert result["has_lesion_status"] == "MATCH"
    assert result["count_status"] == "MATCH"
    assert result["overall_status"] == "MATCH"


NEGATIVE_GT = {
    "has_lesion": False,
    "lesion_count": 0,
    "largest_lesion": None,
}


def test_negative_label_with_unmentioned_presence_is_uncertain_not_mismatch():
    """A negative label whose text never states presence is UNCERTAIN overall."""
    master = {
        "id": "TEST_004",
        "reports": {
            "narrative_clean": (
                "The spleen is normal in size. The liver is normal in size, "
                "with a volume of 1500 cc."
            )
        },
    }
    result = audit_single_record(master, {"id": "TEST_004", "ground_truth": NEGATIVE_GT})
    assert result["has_lesion_status"] == "UNCERTAIN"
    assert result["overall_status"] == "UNCERTAIN"


def test_negative_label_with_affirmed_lesion_is_mismatch():
    """A negative label contradicted by an affirmative liver lesion is MISMATCH."""
    master = {
        "id": "TEST_005",
        "reports": {
            "narrative_clean": (
                "A hypoattenuating liver lesion is identified in hepatic segment 7, "
                "measuring 2.0 cm."
            )
        },
    }
    result = audit_single_record(master, {"id": "TEST_005", "ground_truth": NEGATIVE_GT})
    assert result["has_lesion_status"] == "MISMATCH"
    assert result["overall_status"] == "MISMATCH"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_audit_joins_master_and_ground_truth_by_id_not_line_order(tmp_path: Path):
    """Records in different file orders must still be paired by ID."""
    positive_text = (
        "There are two hypoattenuating liver lesions. The largest lesion is located "
        "in hepatic segment 8, measuring 2.5 x 1.8 cm."
    )
    negative_text = "No focal lesions are observed in the liver."
    masters = [
        {"id": "A", "reports": {"narrative_clean": positive_text}},
        {"id": "B", "reports": {"narrative_clean": negative_text}},
    ]
    ground_truth = [
        {"id": "B", "ground_truth": NEGATIVE_GT},
        {
            "id": "A",
            "ground_truth": {
                "has_lesion": True,
                "lesion_count": 2,
                "largest_lesion": {
                    "location_segments": [8],
                    "source_diameter_cm": 2.5,
                    "attenuation": "hypoattenuating",
                },
            },
        },
    ]
    _write_jsonl(tmp_path / "master.jsonl", masters)
    _write_jsonl(tmp_path / "gt.jsonl", ground_truth)

    audit = run_consistency_audit(
        tmp_path / "master.jsonl", tmp_path / "gt.jsonl", tmp_path / "audit.csv"
    ).set_index("id")

    assert audit.loc["A", "has_lesion_gt"]
    assert audit.loc["A", "overall_status"] == "MATCH"
    assert not audit.loc["B", "has_lesion_gt"]
    assert audit.loc["B", "overall_status"] == "MATCH"


def test_audit_rejects_mismatched_id_sets(tmp_path: Path):
    """An ID present in only one file must fail instead of being skipped."""
    _write_jsonl(
        tmp_path / "master.jsonl",
        [{"id": "A", "reports": {"narrative_clean": "No liver lesion."}}],
    )
    _write_jsonl(tmp_path / "gt.jsonl", [{"id": "B", "ground_truth": NEGATIVE_GT}])

    with pytest.raises(ValueError, match="IDs differ"):
        run_consistency_audit(
            tmp_path / "master.jsonl", tmp_path / "gt.jsonl", tmp_path / "audit.csv"
        )
