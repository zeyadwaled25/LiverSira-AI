"""Unit tests for the rule-based baseline extraction logic."""

from src.baseline_extractor import extract_liver_lesion_features


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

