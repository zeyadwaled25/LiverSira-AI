"""Unit tests for the rule-based baseline extraction logic."""

from pathlib import Path
import json

from liversira_ai.nlp.baseline import (
    extract_liver_lesion_features,
    run_baseline_inference,
)


def test_baseline_negative_case():
    """Extracting from normal report yields has_lesion=False and count=0."""
    text = "The spleen is normal. The liver is normal in size and appearance. No focal liver lesions."
    features = extract_liver_lesion_features(text)
    assert features["has_lesion"] is False
    assert features["lesion_count"] == 0
    assert features["location_segments"] is None
    assert features["derived_max_diameter_cm"] is None


def test_baseline_positive_single_lesion():
    """Extracting single lesion with Couinaud location and dimensions."""
    text = (
        "The patient has a normal spleen. However, a small hypoattenuating liver lesion "
        "is identified in hepatic segment 4, measuring 1.2 x 0.8 cm."
    )
    features = extract_liver_lesion_features(text)
    assert features["has_lesion"] is True
    assert features["lesion_count"] == 1
    assert features["location_segments"] == [4]
    assert features["derived_max_diameter_cm"] == 1.2
    assert features["attenuation"] == "hypoattenuating"


def test_baseline_positive_multi_lesions_multi_segments():
    """Extracting multiple lesions across several segments."""
    text = (
        "There are multiple liver lesions identified, with a total of 15 lesions present. "
        "The largest lesion is located in hepatic segments 1 and 5, measuring 6.5 x 4.2 cm. "
        "All lesions are hypoattenuating."
    )
    features = extract_liver_lesion_features(text)
    assert features["has_lesion"] is True
    assert features["lesion_count"] == 15
    assert features["location_segments"] == [1, 5]
    assert features["derived_max_diameter_cm"] == 6.5
    assert features["attenuation"] == "hypoattenuating"


def test_baseline_size_only_description_is_unclear_not_negative():
    """A size statement does not rule out a lesion (v1.1)."""
    features = extract_liver_lesion_features(
        "The liver is normal in size, with a volume of 1500.0 cc."
    )

    assert features["has_lesion"] is None
    assert features["lesion_count"] is None
    assert features["missing_or_unclear"] == [
        {
            "field": "liver_lesions.has_lesion",
            "reason": "not_mentioned",
        }
    ]


def test_baseline_normal_appearance_is_negative():
    """v1.1: a normal-appearance statement is an explicit negative."""
    features = extract_liver_lesion_features("The liver is normal in size and appearance.")

    assert features["has_lesion"] is False
    assert features["lesion_count"] == 0
    assert features["evidence"][0]["text"] == "The liver is normal in size and appearance."


def test_baseline_does_not_impute_unstated_count():
    """v1.0 set "multiple lesions" to 1; v1.1 leaves it unquantified."""
    features = extract_liver_lesion_features(
        "There are multiple hypoattenuating liver lesions. The largest lesion is in "
        "hepatic segment 2, measuring 3.0 x 2.0 cm."
    )

    assert features["has_lesion"] is True
    assert features["lesion_count"] is None
    assert {"field": "liver_lesions.lesion_count", "reason": "unquantified"} in features[
        "missing_or_unclear"
    ]


def test_baseline_missing_narrative_is_unclear(tmp_path: Path):
    master_path = tmp_path / "master.jsonl"
    output_path = tmp_path / "inferences.jsonl"
    master_path.write_text(
        json.dumps(
            {
                "id": "MISSING_TEXT",
                "reports": {"narrative_clean": None},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = run_baseline_inference(master_path, output_path)

    assert records[0].liver_lesions.has_lesion is None
    assert records[0].liver_lesions.lesion_count is None
    assert records[0].missing_or_unclear[0].field == "liver_lesions.has_lesion"
    assert records[0].missing_or_unclear[0].reason == "not_mentioned"


def test_baseline_explicit_negative_is_false():
    features = extract_liver_lesion_features(
        "The liver is normal. No focal liver lesions are identified."
    )

    assert features["has_lesion"] is False
    assert features["lesion_count"] == 0
    assert features["missing_or_unclear"] == []


def test_inference_record_preserves_evidence_offsets(tmp_path: Path):
    text = "No focal liver lesions are identified."
    master_path = tmp_path / "master.jsonl"
    output_path = tmp_path / "inferences.jsonl"
    master_path.write_text(
        json.dumps({"id": "NEGATIVE", "reports": {"narrative_clean": text}})
        + "\n",
        encoding="utf-8",
    )

    records = run_baseline_inference(master_path, output_path)
    evidence = records[0].evidence_spans[0]

    assert evidence.target_field == "liver_lesions.has_lesion"
    assert text[evidence.start:evidence.end] == evidence.text
    assert evidence.text == text
