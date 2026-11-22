"""Tests for the liver-only text filter."""

import pytest

from liversira_ai.nlp.liver_text import extract_liver_text, organs_in

REPORT = (
    "The patient has a normal-sized spleen, measuring 134.9 cc in volume, with a mean HU value of 124.6 +/- 35.2. "
    "The liver is also normal in size, with a volume of 1291.3 cc and a mean HU value of 111.4 +/- 17.3. "
    "However, there are multiple hypoattenuating liver lesions identified, with a total of 25 lesions present. "
    "The largest lesion is located in hepatic segment 2 and measures 3.1 x 1.6 cm. "
    "The total volume of all liver masses is 19.4 cc. "
    "The pancreas is normal in size, with a volume of 80.0 cc and a mean HU value of 104.9 +/- 28.6. "
    "The kidneys are also normal in size, with a right kidney volume of 166.6 cc. "
    "The mean HU value for the kidneys is 127.2 +/- 53.1. "
    "IMPRESSION: Multiple hypoattenuating liver masses are present, with the largest lesion measuring "
    "3.1 x 1.6 cm in hepatic segment 2. The total volume of all liver masses is 19.4 cc."
)


def liver(text: str) -> str:
    return extract_liver_text(text).text


def test_keeps_only_liver_sentences_of_a_full_report():
    out = liver(REPORT)
    assert out.startswith("The liver is also normal in size")
    assert "25 lesions" in out and "hepatic segment 2 and measures 3.1 x 1.6 cm" in out
    assert "IMPRESSION: Multiple hypoattenuating liver masses" in out
    for word in ("spleen", "pancreas", "kidney", "124.6", "80.0"):
        assert word not in out


def test_spans_point_at_verbatim_text():
    result = extract_liver_text(REPORT)
    pieces = [REPORT[s:e] for s, e in result.spans]
    assert all(piece in result.text for piece in pieces)
    assert result.text == " ".join(pieces)


def test_drops_other_organ_mass_and_its_follow_up_sentence():
    text = (
        "The liver is normal in size. A colon mass is identified, measuring 7.3 x 2.9 cm. "
        "This mass is hypoattenuating, with an HU value of 59.0 +/- 32.4 relative to the surrounding colon."
    )
    assert liver(text) == "The liver is normal in size."


def test_anaphoric_sentence_follows_a_liver_lesion():
    text = "A liver lesion is seen in hepatic segment 4. This lesion is hypoattenuating. The spleen is normal."
    assert liver(text) == "A liver lesion is seen in hepatic segment 4. This lesion is hypoattenuating."


def test_splits_mixed_sentence_with_other_organ_size():
    text = (
        "An 8.2 cm cyst is seen in the pancreatic tail, and a hyperattenuating liver lesion "
        "measuring 5.0 x 3.1 cm is identified in hepatic segment 8."
    )
    assert liver(text) == "a hyperattenuating liver lesion measuring 5.0 x 3.1 cm is identified in hepatic segment 8."


def test_splits_volume_list_by_organ():
    text = (
        "The spleen measures 150.0 cc in volume, the liver measures 1216.0 cc, "
        "the pancreas measures 70.0 cc, and the combined kidney volume is 300.0 cc."
    )
    assert liver(text) == "the liver measures 1216.0 cc"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "The patient has an enlarged spleen and multiple hypoattenuating liver lesions, "
            "with the largest one measuring 8.6 x 6.2 cm.",
            "multiple hypoattenuating liver lesions, with the largest one measuring 8.6 x 6.2 cm.",
        ),
        (
            "The largest liver mass measures 3.6 x 3.2 cm, and the largest kidney mass measures 1.3 x 1.1 cm.",
            "The largest liver mass measures 3.6 x 3.2 cm",
        ),
        (
            "IMPRESSION: The patient has an enlarged spleen, multiple liver masses, with the largest one measuring "
            "1.0 x 0.7 cm in hepatic segment 2, an enlarged pancreas, and enlarged kidneys.",
            "IMPRESSION: multiple liver masses, with the largest one measuring 1.0 x 0.7 cm in hepatic segment 2",
        ),
    ],
)
def test_splits_liver_clause_from_other_organs(text, expected):
    assert liver(text) == expected


def test_keeps_organ_lists_whole_and_their_follow_up():
    text = (
        "The spleen, liver, pancreas, and kidneys are all within normal limits. "
        "No masses or lesions are identified in these organs."
    )
    assert liver(text) == text


def test_keeps_liver_and_spleen_sentence_whole():
    text = "The liver and spleen are normal in size, with volumes of 1500.0 cc and 150.0 cc, respectively."
    assert liver(text) == text


def test_numbered_impression_items():
    text = (
        "The liver is normal. IMPRESSION: 1. Enlarged spleen. 2. A hypoattenuating liver mass in hepatic "
        "segment 6, measuring 2.0 x 1.5 cm. 3. Enlarged pancreas."
    )
    assert liver(text) == (
        "The liver is normal. IMPRESSION: 2. A hypoattenuating liver mass in hepatic segment 6, measuring 2.0 x 1.5 cm."
    )


def test_bullets_keep_liver_items_only():
    text = "RESULTS: - Spleen: Normal size (137.4 cc). - Liver: Normal size (1128.7 cc). - Pancreas: Normal size."
    assert liver(text) == "RESULTS: - Liver: Normal size (1128.7 cc)."


def test_sentence_after_a_header_has_no_inherited_organ():
    text = "A liver lesion is seen in hepatic segment 2. IMPRESSION: Clinical stage: T2NxMx."
    assert liver(text) == "A liver lesion is seen in hepatic segment 2."


def test_hepatic_artery_is_not_the_liver():
    assert organs_in("The tumor does not contact the common hepatic artery (CHA).") == {"other"}
    assert organs_in("No focal hepatic lesions are seen.") == {"liver"}


def test_reference_organ_in_a_comparison_is_ignored():
    assert organs_in("The mass is hypoattenuating relative to the kidney.") == frozenset()
    assert organs_in("A liver lesion of lower attenuation than the surrounding liver parenchyma.") == {"liver"}


@pytest.mark.parametrize("value", [None, ""])
def test_empty_input(value):
    assert extract_liver_text(value).text == ""
    assert extract_liver_text(value).spans == []
