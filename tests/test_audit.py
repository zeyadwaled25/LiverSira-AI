"""Unit tests for the automated target consistency audit logic."""

from src.audit_target_consistency import (
    audit_single_record,
    get_liver_sentences,
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

