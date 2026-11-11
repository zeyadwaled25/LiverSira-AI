"""Tests for the Phase 6 Challenge Set transforms and training helpers (no model downloads)."""

import random

import pytest

from scripts.nlp.transformers.presence import build_challenge_set as cs

REPORT = (
    "The spleen is normal, measuring 134.9 cc in volume. "
    "The liver is normal in size, with a volume of 1291.3 cc. "
    "However, there is a liver lesion in hepatic segment 2. "
    "This lesion is hypoattenuating, measuring 3.1 x 1.6 cm. "
    "The kidneys are normal in size. "
    "IMPRESSION: Hypoattenuating liver mass in hepatic segment 2."
)


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def test_sentences_do_not_split_decimals():
    assert cs.split_sentences("Volume 1487.6 cc. The liver is normal.") == [
        "Volume 1487.6 cc.",
        "The liver is normal.",
    ]


def test_liver_block_includes_following_anaphor_but_not_other_organs():
    sentences = cs.split_sentences(REPORT)
    assert [sentences[i] for i in cs._liver_indices(sentences)] == [
        sentences[1], sentences[2], sentences[3], sentences[5]
    ]


def test_negation_replaces_whole_liver_block():
    text = cs.negation(REPORT, rng())
    assert "This lesion" not in text and "segment 2" not in text
    assert "spleen" in text and "kidneys" in text
    assert any(template in text for template in cs.NEGATED)


def test_liver_removed_keeps_other_organ_lesions_and_rejects_orphans():
    kidney = "The liver has a 2 cm lesion. The left kidney has a small lesion. The spleen is normal."
    assert cs.liver_removed(kidney, rng()) == "The left kidney has a small lesion. The spleen is normal."
    orphan = "The liver is normal. The spleen is normal. A 2 cm lesion is seen."
    assert cs.liver_removed(orphan, rng()) is None


def test_liver_last_moves_block_in_order():
    text = cs.liver_last(REPORT, rng())
    sentences = cs.split_sentences(text)
    assert sentences[:2] == ["The spleen is normal, measuring 134.9 cc in volume.", "The kidneys are normal in size."]
    assert sentences[-1].startswith("IMPRESSION")
    assert sentences[3].startswith("However") and sentences[4].startswith("This lesion")


def test_paraphrase_swaps_lesion_words_only_in_liver_sentences():
    text = "Liver lesions are identified. The kidney lesion is stable."
    out = cs.paraphrase(text, rng(1))
    assert out is not None and out.endswith("The kidney lesion is stable.")
    assert out.split(".")[0] != "Liver lesions are identified"
    assert cs.paraphrase("The spleen is normal.", rng()) is None


def test_ocr_noise_is_deterministic_and_changes_text():
    assert cs.ocr_noise(REPORT, rng(7)) == cs.ocr_noise(REPORT, rng(7)) != REPORT


def test_other_organ_lesion_ignores_negated_sentences():
    assert cs.is_other_organ_lesion("The liver is normal. There are two kidney lesions identified.")
    assert not cs.is_other_organ_lesion("No evidence of tumor in the spleen or kidneys.")


def test_build_applies_label_rules_and_quotas():
    examples = [{"id": f"P{i}", "text": REPORT, "label": True} for i in range(3)]
    examples += [
        {"id": f"N{i}", "text": "The spleen is normal. The liver is normal in size. The left kidney has a small lesion.",
         "label": False}
        for i in range(3)
    ]
    records = cs.build(examples, lambda t: str(len(t)))
    by_category = {}
    for r in records:
        by_category.setdefault(r["category"], []).append(r)
    assert {r["label"] for r in by_category["negation"]} == {False}
    assert {r["label"] for r in by_category["liver_removed"]} == {False}
    assert all(r["label"] == r["base_label"] for r in by_category["ocr_noise"])
    assert {r["label"] for r in by_category["hedged"]} == {None}
    assert not any(r["scored"] for r in by_category["hedged"])
    assert len({r["challenge_id"] for r in records}) == len(records)


def test_class_weights_and_summary():
    train = pytest.importorskip("scripts.nlp.transformers.presence.train")
    weights = train.class_weights([True, False, False, False])
    assert weights.tolist() == pytest.approx([4 / 6, 2.0])
    summary = train.summarize([True, False, True, False], [0.9, 0.2, 0.4, 0.6], 0.5)
    assert summary["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert summary["f1"] == 0.5 and summary["specificity"] == 0.5
