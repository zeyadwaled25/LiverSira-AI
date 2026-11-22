"""Tests for per-lesion labels, the per-lesion training generator, the per-lesion model
and extraction_challenge_v3 (no checkpoint download)."""

import random
import re

import pytest
import torch

from liversira_ai.nlp import lesion_model as lm
from liversira_ai.nlp import lesion_targets as lt
from liversira_ai.nlp.extraction_targets import count_class
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import build_challenge_v3 as v3
from scripts.nlp.transformers.extraction import build_training_set_lesion as tl

BASE = {"id": "S001", "liver_text": "The liver is also normal in size, with a volume of 2860.9 cc."}


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=512, return_offsets_mapping=False):
        pieces = [(m.start(), m.end()) for m in re.finditer(r"\d+\.\d+|\w+|[^\w\s]", text)][: max_length - 2]
        offsets = [(0, 0), *pieces, (0, 0)]
        return {"input_ids": list(range(1, len(offsets) + 1)), "attention_mask": [1] * len(offsets),
                "offset_mapping": offsets}


# ── labels ──────────────────────────────────────────────────────────────────
def test_dimensions_and_counts():
    assert lt.dimensions("3.1 x 1.6 cm") == (3.1, 1.6)
    assert lt.dimensions("82 x 43 mm") == pytest.approx((8.2, 4.3))
    assert lt.parse_count("Four") == 4 and lt.parse_count("12") == 12 and lt.parse_count("a") is None
    text = "There are three hypoattenuating liver lesions."
    assert text[slice(*lt.count_span(text, 3))] == "three"


def test_range_sizes_are_not_lesions():
    text = "Lesions ranging in size from 0.6 x 0.3 cm to 6.3 x 4.0 cm. Others measure 0.5 to 0.8 cm."
    assert lt.lesion_size_phrases(text) == []


def test_real_labels_repeat_and_largest_by_source():
    text = ("There are two liver lesions. The first, in hepatic segment 5, measures 5.5 x 4.2 cm and is hypoattenuating. "
            "The second, in hepatic segment 8, measures 6.9 x 3.5 cm. IMPRESSION: the largest measures 5.5 x 4.2 cm.")
    labels = lt.real_labels(text, 2, "5", "5.5", "hypoattenuating")
    spans = [text[slice(*lesion["span"])] for lesion in labels["lesions"]]
    assert spans == ["5.5 x 4.2 cm", "6.9 x 3.5 cm", "5.5 x 4.2 cm"]
    assert [lesion["repeat"] for lesion in labels["lesions"]] == [False, False, True]
    assert [lesion["largest"] for lesion in labels["lesions"]] == [True, False, True]  # by source (volume), not length
    assert labels["count_source"] == "number"
    assert lt.decode_labels(text, labels) == {"count": "2", "segments": [5], "attenuation": "hypoattenuating",
                                              "diameter_cm": 5.5}


def test_real_labels_single_and_negative():
    text = "A hypoattenuating liver lesion in hepatic segment 2 measures 1.2 x 0.8 cm."
    labels = lt.real_labels(text, 1, "2", "1.2", "hypoattenuating")
    assert labels["count_source"] == "enumerated"
    assert lt.decode_labels(text, labels)["count"] == "1"
    assert lt.real_labels("The liver is normal.", 0, "NO_LESION", "NOT_APPLICABLE", "NOT_APPLICABLE")["count_source"] == "none"


# ── per-lesion training generator ───────────────────────────────────────────
def _expected(spec, source, extra):
    return {
        "count": None if source == "unstated" else count_class(len(spec.lesions)),
        "segments": sorted(spec.largest.segments) if extra.get("segments_written", True) else None,
        "attenuation": spec.largest.attenuation,
        "diameter_cm": spec.largest.dims[0],
    }


@pytest.mark.parametrize("category,subtype,fn", tl.GENERATORS)
def test_generator_labels_decode_to_construction(category, subtype, fn):
    for k in range(25):
        item = tl.generated_item(BASE, category, subtype, fn, random.Random(f"t:{k}"), ocr=False)
        block, spec, source, extra = fn(random.Random(f"t:{k}"))
        assert tl.OPEN not in item["text"] and tl.CLOSE not in item["text"]
        decoded = lt.decode_labels(item["text"], item["labels"])
        expected = _expected(spec, source, extra)
        assert decoded["count"] == expected["count"]
        assert decoded["segments"] == expected["segments"]
        assert decoded["attenuation"] == expected["attenuation"]
        assert decoded["diameter_cm"] == pytest.approx(expected["diameter_cm"])


def test_ocr_items_keep_consistent_labels():
    for k in range(40):
        item = tl.generated_item(BASE, "count", "present", tl.count_present, random.Random(f"o:{k}"), ocr=True)
        labels = item["labels"]
        if labels["count_source"] == "number":
            assert lt.parse_count(item["text"][slice(*labels["count_spans"][0])]) is not None
        for lesion in labels["lesions"]:
            assert lt.dimensions(item["text"][slice(*lesion["span"])])


def test_non_lesion_sizes_are_not_labelled():
    item = tl.generated_item(BASE, "non_lesion", "node", tl.non_lesion_node, random.Random("n"), ocr=False)
    labelled = [item["text"][slice(*lesion["span"])] for lesion in item["labels"]["lesions"]]
    assert len(labelled) == 1 and "node" in item["text"]


# ── per-lesion model ────────────────────────────────────────────────────────
TEXT = "Three lesions. One in segment 2 measures 1.1 x 0.8 cm. The largest in segment 6 measures 4.2 x 3.0 cm."


def _labels():
    a, b = TEXT.index("1.1 x 0.8 cm"), TEXT.index("4.2 x 3.0 cm")
    return {"count_source": "number", "count_spans": [[0, 5]],
            "lesions": [{"span": [a, a + 12], "repeat": False, "largest": False, "known": True, "segments": [2],
                         "attenuation": None},
                        {"span": [b, b + 12], "repeat": False, "largest": True, "known": True, "segments": [6],
                         "attenuation": "hypoattenuating"}]}


def test_bio_and_tagged_spans_round_trip():
    enc = lm.encode_item(FakeTokenizer(), TEXT, _labels(), 128)
    logits = torch.full((len(enc["offsets"]), 3), -5.0)
    for k, tag in enumerate(enc["size_tags"]):
        logits[k, 0 if tag == lm.IGNORE else tag] = 5.0
    spans = lm.tagged_spans(enc["offsets"], logits)
    assert [TEXT[s:e] for s, e in spans] == ["1.1 x 0.8 cm", "4.2 x 3.0 cm"]
    assert enc["span_features"] == [[0.0, pytest.approx(1.1 / 4.2)], [1.0, 1.0]]
    assert enc["count_tags"][1] == lm.B


def test_collate_and_losses():
    enc = lm.encode_item(FakeTokenizer(), TEXT, _labels(), 128)
    empty = lm.encode_item(FakeTokenizer(), "The liver is normal.",
                           {"count_source": "none", "count_spans": [], "lesions": []}, 128)
    batch = lm.collate([enc, empty], pad_id=0)
    assert batch["span_batch"].tolist() == [0, 0] and batch["span_mask"].shape[0] == 2
    width = batch["input_ids"].shape[1]
    out = {"size_tag": torch.randn(2, width, 3), "count_tag": torch.randn(2, width, 3),
           "source": torch.randn(2, len(lt.COUNT_SOURCES)), "span_segments": torch.randn(2, lm.N_SEGMENTS + 1),
           "span_attenuation": torch.randn(2, 4), "span_repeat": torch.randn(2), "span_largest": torch.randn(2)}
    parts = lm.losses(out, batch)
    assert set(parts) == {"size_tag", "count_tag", "source", "span"}
    assert all(torch.isfinite(v) for v in parts.values())


# ── Challenge v3 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("category,subtype", [(c, s) for c, subs in v3.CATEGORIES.items() for s in subs])
def test_v3_generators_are_deterministic(category, subtype):
    fn = v3.CATEGORIES[category][subtype]
    for k in range(20):
        first = fn(random.Random(f"v3:{k}"))
        assert first == fn(random.Random(f"v3:{k}"))
        block, tail, expected, targets = first
        assert set(expected) == set(cs.FIELDS)
        assert cs.fmt(expected["derived_max_diameter_cm"]) in block
