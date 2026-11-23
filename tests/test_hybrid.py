"""Parser v1.2 field status and the hybrid rules H / H2 (Amendment 1)."""

from liversira_ai.nlp.baseline import extract_liver_lesion_features, extract_liver_lesion_features_v1_2
from liversira_ai.nlp.hybrid import hybrid_h, hybrid_h2

MODEL = {"has_lesion": True, "lesion_count": 7, "location_segments": [3], "derived_max_diameter_cm": 9.9,
         "attenuation": "hyperattenuating"}


def test_status_is_v1_2_only_and_explains_empty_fields():
    assert extract_liver_lesion_features("A hypoattenuating liver lesion of 2.0 cm.")["field_status"] is None
    stated = extract_liver_lesion_features_v1_2("A hypoattenuating liver lesion in segment 6 measures 2.0 x 1.0 cm.")
    assert set(stated["field_status"].values()) == {"stated"}
    unquantified = extract_liver_lesion_features_v1_2("Innumerable hypoattenuating liver lesions are present.")
    assert unquantified["lesion_count"] is None and unquantified["field_status"]["lesion_count"] == "not_mentioned"
    assert unquantified["field_status"]["location_segments"] == "not_mentioned"


def test_negative_and_unknown_presence():
    negative = extract_liver_lesion_features_v1_2("No focal liver lesion is seen.")
    assert negative["field_status"]["lesion_count"] == "stated"
    assert negative["field_status"]["attenuation"] == "not_mentioned"
    unknown = extract_liver_lesion_features_v1_2("The liver has a mean HU value of 60.")
    assert set(unknown["field_status"].values()) == {"unparsed"}


def test_h_and_h2_differ_only_on_not_mentioned():
    parser = extract_liver_lesion_features_v1_2("Innumerable hypoattenuating liver lesions are present.")
    h, h2 = hybrid_h(parser, MODEL), hybrid_h2(parser, MODEL)
    assert h["lesion_count"] == 7 and h2["lesion_count"] is None  # stated as unquantified: H2 keeps it empty
    assert h["attenuation"] == h2["attenuation"] == "hypoattenuating"  # parser value wins in both
    unknown = extract_liver_lesion_features_v1_2("The liver has a mean HU value of 60.")
    assert hybrid_h(unknown, MODEL) == hybrid_h2(unknown, MODEL)  # all unparsed: both take the model
    negative = extract_liver_lesion_features_v1_2("No focal liver lesion is seen.")
    assert hybrid_h2(negative, MODEL)["lesion_count"] == 0 == hybrid_h(negative, MODEL)["lesion_count"]
