"""Tests for the 6.2 extraction targets and the training-set generator."""

import random

import pytest

from liversira_ai.nlp import extraction_targets as et
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import build_training_set as ts

POSITIVE = (
    "The liver is normal in size. However, there are multiple hypoattenuating liver lesions identified, "
    "with a total of 25 lesions present. The largest lesion is located in hepatic segment 2 and measures "
    "3.1 x 1.6 cm. IMPRESSION: Multiple hypoattenuating liver masses, the largest measuring 3.1 x 1.6 cm."
)
BASE = {"id": "S001", "liver_text": "The liver is also normal in size, with a volume of 2860.9 cc."}


# ── targets from source labels ──────────────────────────────────────────────
def test_negative_report():
    t = et.targets_from_source("The liver is normal.", 0, "NO_LESION", "NOT_APPLICABLE", "NOT_APPLICABLE")
    assert t == {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}


def test_stated_values_become_targets():
    t = et.targets_from_source(POSITIVE, 25, "2", "3.1", "hypoattenuating")
    assert t["count"] == "10+"
    assert t["segments"] == [2]
    assert t["attenuation"] == "hypoattenuating"
    assert t["diameter_cm"] == 3.1
    assert [POSITIVE[s:e] for s, e in t["diameter_spans"]] == ["3.1 x 1.6 cm", "3.1 x 1.6 cm"]


def test_unstated_values_are_none():
    text = "Multiple liver lesions are seen in the right lobe, the largest measuring 2.0 cm."
    t = et.targets_from_source(text, 4, "6,7", "2.5", "hyperattenuating")
    assert t["count"] is None  # "multiple" without a number
    assert t["segments"] is None  # segments not written
    assert t["attenuation"] is None  # attenuation not written
    assert t["diameter_cm"] is None and t["diameter_spans"] == []  # 2.5 not written


def test_single_lesion_count():
    text = "A hypoattenuating liver lesion is identified in hepatic segment 5, measuring 12 x 8 mm."
    t = et.targets_from_source(text, 1, "5", "1.2", "hypoattenuating")
    assert t["count"] == "1"
    assert t["diameter_cm"] == 1.2  # mm converted
    assert et.targets_from_source("Multiple liver lesions.", 1, "5", "1.2", "hypoattenuating")["count"] is None


@pytest.mark.parametrize(
    "text,expected",
    [("3.1 x 1.6 cm", 3.1), ("82 mm", 8.2), ("5.1 x 4.0 centimeters", 5.1), ("8.2 by 4.3 cm", 8.2)],
)
def test_size_phrases(text, expected):
    [(start, end, value)] = et.size_phrases(text)
    assert (start, end) == (0, len(text)) and value == pytest.approx(expected)


def test_count_classes():
    assert [et.count_class(n) for n in (0, 1, 9, 10, 176)] == ["0", "1", "9", "10+", "10+"]
    assert et.count_class(None) is None


# ── generator ───────────────────────────────────────────────────────────────
def test_unmark_round_trip():
    text, spans = ts.unmark(f"a {ts.OPEN}3.1 x 1.6 cm{ts.CLOSE} lesion")
    assert text == "a 3.1 x 1.6 cm lesion" and [text[s:e] for s, e in spans] == ["3.1 x 1.6 cm"]
    assert ts.unmark(ts.mark_spans(text, spans)) == (text, spans)


@pytest.mark.parametrize("category,subtype,fn", ts.GENERATORS)
def test_generators_are_deterministic_and_consistent(category, subtype, fn):
    for seed in range(30):
        block, expected = fn(random.Random(f"t:{seed}"))
        assert (block, expected) == fn(random.Random(f"t:{seed}"))
        assert block.count(ts.OPEN) == block.count(ts.CLOSE) == 1
        text, spans = ts.unmark(block)
        assert et.diameter_from_spans(text, spans) == pytest.approx(expected["derived_max_diameter_cm"])
        targets = et.targets_from_expected(expected, spans)
        assert targets["count"] is None or targets["count"] in et.COUNT_CLASSES
        assert targets["attenuation"] in et.ATTENUATIONS


def test_generated_item_with_ocr_keeps_span():
    for k in range(30):
        item = ts.generated_item(BASE, "size", "cm_by_cm", ts.size_cm_by_cm, random.Random(f"o:{k}"), ocr=True)
        t = item["targets"]
        assert ts.OPEN not in item["text"] and item["category"] == "ocr"
        assert et.diameter_from_spans(item["text"], t["diameter_spans"]) == pytest.approx(t["diameter_cm"])


def test_real_ocr_item_keeps_targets():
    row = {"id": "R1", "liver_text": POSITIVE, "count": 25, "location": "2", "diameter": "3.1",
           "attenuation": "hypoattenuating"}
    clean, noisy = ts.real_item(row), ts.real_item(row, random.Random("x"))
    assert noisy["id"] == "ocr:R1" and noisy["text"] != clean["text"]
    assert {k: v for k, v in noisy["targets"].items() if k != "diameter_spans"} == {
        k: v for k, v in clean["targets"].items() if k != "diameter_spans"
    }
    assert et.diameter_from_spans(noisy["text"], noisy["targets"]["diameter_spans"]) == 3.1


# ── disjointness ────────────────────────────────────────────────────────────
def test_ngrams_mask_numbers_and_attenuation():
    assert ts.ngrams("A hypoattenuating lesion in segment 4a measures 3.1 cm.") == ts.ngrams(
        "A hyperattenuating lesion in segment IV measures 9.9 cm."
    )


def test_overlap_flags_only_challenge_specific_grams():
    challenge = ["However, a liver lesion is identified in the fourth hepatic segment, measuring 6.0 x 3.4 cm."]
    real = ["However, a liver lesion is identified in hepatic segment 2, measuring 1.0 x 0.5 cm."]
    specific = ts.challenge_specific(challenge, real)
    copied = {"source": "generated", "inserted_text": "A lesion lies in the second hepatic segment today."}
    reworded = {"source": "generated", "inserted_text": "A lesion lies in the 2nd segment of the liver."}
    assert ts.overlap([copied], specific)
    assert not ts.overlap([reworded], specific)


def test_generated_categories_leave_out_distractors():
    assert "distractor" not in ts.CATEGORIES
    assert set(cs.CATEGORIES) - {"control", "distractor", "count_format", "size_format", "segment_format",
                                 "attenuation_format", "largest_order", "ocr_noise", "paraphrase"} == set()
