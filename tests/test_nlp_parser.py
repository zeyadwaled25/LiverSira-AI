import pytest

from src.data_prep.audit_target_consistency import audit_single_record
from src.baseline_extractor import extract_liver_lesion_features
from src.nlp_parser import parse_liver_report


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The liver is normal. No focal liver lesions are identified.",
            {
                "has_mention": False,
                "lesion_count": None,
                "location_segments": None,
                "reported_dimensions_cm": None,
                "derived_max_diameter_cm": None,
                "attenuation": None,
            },
        ),
        (
            "A single hypoattenuating liver lesion is present in hepatic segment 2, "
            "measuring 1.5 cm.",
            {
                "has_mention": True,
                "lesion_count": 1,
                "location_segments": [2],
                "reported_dimensions_cm": [1.5],
                "derived_max_diameter_cm": 1.5,
                "attenuation": "hypoattenuating",
            },
        ),
        (
            "There are three additional liver lesions. The largest lesion is in "
            "hepatic segments 1 and 5, measuring 6.5 x 4.2 cm.",
            {
                "has_mention": True,
                "lesion_count": 4,
                "location_segments": [1, 5],
                "reported_dimensions_cm": [6.5, 4.2],
                "derived_max_diameter_cm": 6.5,
                "attenuation": None,
            },
        ),
        (
            "The liver has a total of 0 lesions.",
            {
                "has_mention": True,
                "lesion_count": 0,
                "location_segments": None,
                "reported_dimensions_cm": None,
                "derived_max_diameter_cm": None,
                "attenuation": None,
            },
        ),
    ],
)
def test_shared_parser_handles_core_report_variants(text, expected):
    parsed = parse_liver_report(text)

    for field, value in expected.items():
        assert getattr(parsed, field) == value


def test_shared_parser_preserves_baseline_attenuation_fallback():
    text = (
        "There are multiple liver lesions. All lesions are hyperattenuating "
        "relative to the liver."
    )

    audit_parser = parse_liver_report(text)
    baseline = extract_liver_lesion_features(text)

    assert audit_parser.attenuation is None
    assert baseline["attenuation"] == "hyperattenuating"


@pytest.mark.parametrize(
    ("text", "presence", "reason"),
    [
        ("A small liver lesion is present.", True, None),
        ("No focal liver lesions are identified.", False, None),
        ("The liver is normal in size and appearance.", None, "not_mentioned"),
        ("A lesion cannot be excluded in the liver.", None, "ambiguous"),
        (
            "The liver has no lesions but a hepatic mass is suspected.",
            None,
            "ambiguous",
        ),
    ],
)
def test_shared_parser_presence_semantics(text, presence, reason):
    parsed = parse_liver_report(text)

    assert parsed.presence is presence
    assert parsed.presence_reason == reason


def test_positive_evidence_has_exact_offsets_for_all_supported_fields():
    text = (
        "There are two hypoattenuating liver lesions. The largest lesion is "
        "located in hepatic segments 1 and 5, measuring 6.5 x 4.2 cm."
    )
    parsed = parse_liver_report(text, include_broad_attenuation_fallback=True)

    fields = {item.target_field for item in parsed.evidence}
    assert {
        "liver_lesions.has_lesion",
        "liver_lesions.lesion_count",
        "liver_lesions.largest_lesion.location_segments",
        "liver_lesions.largest_lesion.reported_dimensions_cm",
        "liver_lesions.largest_lesion.derived_max_diameter_cm",
        "liver_lesions.largest_lesion.attenuation",
    } <= fields
    for item in parsed.evidence:
        assert text[item.start:item.end] == item.text


def test_negative_evidence_keeps_negation_context():
    text = "The liver is normal. No focal liver lesions are identified."
    features = extract_liver_lesion_features(text)

    presence = [
        item for item in features["evidence"]
        if item["target_field"] == "liver_lesions.has_lesion"
    ]
    assert len(presence) == 1
    assert presence[0]["text"] == "No focal liver lesions are identified."
    assert "No" in presence[0]["text"]


def test_ambiguous_and_contradictory_evidence_is_retained_without_false_choice():
    text = (
        "No focal liver lesions are identified. However, a hepatic mass is "
        "suspected in segment 4."
    )
    features = extract_liver_lesion_features(text)
    presence = [
        item for item in features["evidence"]
        if item["target_field"] == "liver_lesions.has_lesion"
    ]

    assert features["has_lesion"] is None
    assert features["missing_or_unclear"][0]["reason"] == "conflicting_terms"
    assert len(presence) == 2
    assert "No focal liver lesions" in presence[0]["text"]
    assert "hepatic mass" in presence[1]["text"]


def test_no_evidence_for_missing_or_unmentioned_text():
    assert extract_liver_lesion_features(None)["evidence"] == []
    assert extract_liver_lesion_features(
        "The liver is normal in size and appearance."
    )["evidence"] == []


def test_baseline_and_audit_use_same_core_parse_values():
    text = (
        "There are two hypoattenuating liver lesions. The largest lesion is "
        "located in hepatic segment 8, measuring 2.5 x 1.8 cm."
    )
    parsed = parse_liver_report(text)
    baseline = extract_liver_lesion_features(text)
    audit = audit_single_record(
        {"id": "TEST_SHARED", "reports": {"narrative_clean": text}},
        {
            "id": "TEST_SHARED",
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
    )

    assert baseline["lesion_count"] == parsed.lesion_count
    assert baseline["location_segments"] == parsed.location_segments
    assert baseline["reported_dimensions_cm"] == parsed.reported_dimensions_cm
    assert baseline["attenuation"] == parsed.attenuation
    assert audit["count_text"] == parsed.lesion_count
    assert audit["location_text"] == "8"
    assert audit["diameter_text"] == parsed.derived_max_diameter_cm
    assert audit["attenuation_text"] == parsed.attenuation


def test_shared_parser_handles_decimal_sentence_boundaries_and_ambiguity():
    text = (
        "The liver demonstrates 86.6 HU. A lesion cannot be excluded in "
        "hepatic segment 4, measuring 0.8 x 0.6 cm."
    )
    parsed = parse_liver_report(text)

    assert len(parsed.liver_sentences) == 2
    assert parsed.has_mention is False
    assert parsed.presence is None
    assert parsed.presence_reason == "ambiguous"
    assert parsed.location_segments == [4]
    assert parsed.derived_max_diameter_cm is None
