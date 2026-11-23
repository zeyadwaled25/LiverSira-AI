"""Tests for the 6.2 extraction model's encoding, losses and decoding (no checkpoint download)."""

import re

import pytest
import torch

from liversira_ai.nlp import extraction_model as em
from scripts.nlp.transformers.extraction.compare import parser_targets
from scripts.nlp.transformers.extraction.train import field_correct, target_accuracy

TEXT = "A hypoattenuating lesion in segment 4 measures 3.1 x 1.6 cm."
SPAN = (TEXT.index("3.1"), TEXT.index("cm") + 2)
TARGETS = {"count": "1", "segments": [4], "attenuation": "hypoattenuating", "diameter_cm": 3.1,
           "diameter_spans": [SPAN]}


class FakeTokenizer:
    """Whitespace/punctuation tokenizer with [CLS]/[SEP] and character offsets."""

    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=512, return_offsets_mapping=False):
        pieces = [(m.start(), m.end()) for m in re.finditer(r"\d+\.\d+|\w+|[^\w\s]", text)]
        pieces = pieces[: max_length - 2]
        offsets = [(0, 0), *pieces, (0, 0)]
        return {"input_ids": list(range(1, len(offsets) + 1)), "attention_mask": [1] * len(offsets),
                "offset_mapping": offsets}


def test_encode_item_labels():
    enc = em.encode_item(FakeTokenizer(), TEXT, TARGETS, max_length=64)
    tagged = [TEXT[s:e] for (s, e), t in zip(enc["offsets"], enc["diameter_tags"]) if t == 1]
    assert tagged == ["3.1", "x", "1.6", "cm"]
    assert enc["diameter_tags"][0] == enc["diameter_tags"][-1] == em.IGNORE
    assert em.COUNT_LABELS[enc["count"]] == "1"
    assert enc["segments_stated"] == 1.0 and enc["segments"][3] == 1.0 and sum(enc["segments"]) == 1.0
    assert em.ATTENUATION_LABELS[enc["attenuation"]] == "hypoattenuating"
    assert not enc["truncated_span"]


def test_encode_unstated_and_truncation():
    unstated = {"count": None, "segments": None, "attenuation": None, "diameter_cm": 3.1, "diameter_spans": [SPAN]}
    enc = em.encode_item(FakeTokenizer(), TEXT, unstated, max_length=6)
    assert em.COUNT_LABELS[enc["count"]] == em.UNSTATED
    assert em.ATTENUATION_LABELS[enc["attenuation"]] == em.UNSTATED
    assert enc["segments_stated"] == 0.0 and enc["truncated_span"]


def _outputs(enc, count="1", segments=(4,), attenuation="hypoattenuating", tag_tokens=("3.1", "x", "1.6", "cm")):
    n = len(enc["input_ids"])
    out = {
        "count": torch.full((len(em.COUNT_LABELS),), -5.0),
        "segments": torch.full((em.N_SEGMENTS + 1,), -5.0),
        "attenuation": torch.full((len(em.ATTENUATION_LABELS),), -5.0),
        "diameter": torch.zeros(n, 2),
    }
    out["count"][em.COUNT_LABELS.index(count)] = 5.0
    out["attenuation"][em.ATTENUATION_LABELS.index(attenuation)] = 5.0
    if segments:
        out["segments"][em.N_SEGMENTS] = 5.0
        for s in segments:
            out["segments"][s - 1] = 5.0
    for k, (s, e) in enumerate(enc["offsets"]):
        out["diameter"][k] = torch.tensor([-5.0, 5.0]) if s != e and TEXT[s:e] in tag_tokens else torch.tensor([5.0, -5.0])
    return out


def test_decode_round_trip():
    enc = em.encode_item(FakeTokenizer(), TEXT, TARGETS, max_length=64)
    d = em.decode(TEXT, enc["offsets"], _outputs(enc))
    assert (d.count, d.segments, d.attenuation, d.diameter_cm) == ("1", [4], "hypoattenuating", 3.1)
    assert d.diameter_spans == [SPAN]


def test_decode_unstated():
    enc = em.encode_item(FakeTokenizer(), TEXT, TARGETS, max_length=64)
    d = em.decode(TEXT, enc["offsets"], _outputs(enc, count=em.UNSTATED, segments=(), attenuation=em.UNSTATED,
                                                tag_tokens=()))
    assert (d.count, d.segments, d.attenuation, d.diameter_cm) == (None, None, None, None)


def test_losses_are_finite_and_mask_segments():
    enc = em.encode_item(FakeTokenizer(), TEXT, TARGETS, max_length=64)
    unstated = em.encode_item(FakeTokenizer(), TEXT, {**TARGETS, "segments": None}, max_length=64)
    batch = em.collate([enc, unstated], pad_id=0)
    n = batch["input_ids"].shape[1]
    outputs = {"count": torch.randn(2, len(em.COUNT_LABELS)), "segments": torch.randn(2, em.N_SEGMENTS + 1),
               "attenuation": torch.randn(2, len(em.ATTENUATION_LABELS)), "diameter": torch.randn(2, n, 2)}
    parts = em.losses(outputs, batch)
    assert set(parts) == {"count", "segments", "attenuation", "diameter"}
    assert all(torch.isfinite(v) for v in parts.values())


@pytest.mark.parametrize(
    "count,expected",
    [("0", {"has_lesion": False, "lesion_count": 0}), ("3", {"has_lesion": True, "lesion_count": 3}),
     ("10+", {"has_lesion": True, "lesion_count": 10}), (None, {"has_lesion": True, "lesion_count": None})],
)
def test_to_parser_fields(count, expected):
    d = em.Decoded(count, [4], "hypoattenuating", 3.1, [SPAN], {})
    fields = em.to_parser_fields(d)
    assert {k: fields[k] for k in expected} == expected
    if count == "0":
        assert fields["derived_max_diameter_cm"] is None


def test_field_correct_and_accuracy():
    assert field_correct("segments", [5, 4], [4, 5])
    assert field_correct("diameter_cm", 3.12, 3.1) and not field_correct("diameter_cm", 3.2, 3.1)
    assert field_correct("count", None, None) and not field_correct("count", "2", None)
    items = [{"targets": {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None}},
             {"targets": {"count": "2", "segments": [4], "attenuation": "hypoattenuating", "diameter_cm": 3.1}}]
    preds = [{"count": "0", "segments": None, "attenuation": None, "diameter_cm": None},
             {"count": "2", "segments": [5], "attenuation": "hypoattenuating", "diameter_cm": 3.1}]
    acc = target_accuracy(items, preds)
    assert acc["all"]["segments"] == 0.5
    assert {k: acc["positive"][k] for k in ("n", "count", "segments", "attenuation", "diameter_cm")} == {
        "n": 1, "count": 1.0, "segments": 0.0, "attenuation": 1.0, "diameter_cm": 1.0}
    # A real report with a lesion and a null count target: the count is unverifiable and not scored.
    items.append({"category": "real", "targets": {"count": None, "segments": [4], "attenuation": None,
                                                  "diameter_cm": 2.0}})
    preds.append({"count": "7", "segments": [4], "attenuation": "hypoattenuating", "diameter_cm": 2.0})
    acc = target_accuracy(items, preds)
    assert acc["positive"]["n_count"] == 1 and acc["positive"]["unverifiable"] == {
        "count": 1, "segments": 0, "attenuation": 1, "diameter_cm": 0}
    assert acc["positive"]["count"] == 1.0 and acc["positive"]["n_segments"] == 2


def test_parser_targets_mapping():
    assert parser_targets({"has_lesion": False})["count"] == "0"
    assert parser_targets({"has_lesion": None}) == {"count": None, "segments": None, "attenuation": None,
                                                    "diameter_cm": None}
    t = parser_targets({"has_lesion": True, "lesion_count": 12, "location_segments": [7, 6],
                        "attenuation": "hypoattenuating", "derived_max_diameter_cm": 2.0})
    assert t == {"count": "10+", "segments": [6, 7], "attenuation": "hypoattenuating", "diameter_cm": 2.0}


# ── Amendment 3: unverifiable fields ────────────────────────────────────────
def test_unverifiable_fields_only_for_real_reports_with_a_lesion():
    from liversira_ai.nlp.extraction_targets import unverifiable_fields

    targets = {"count": None, "segments": [5], "attenuation": None, "diameter_cm": 3.1, "diameter_spans": []}
    assert unverifiable_fields({"category": "real", "subtype": "real", "targets": targets}) == ["count", "attenuation"]
    assert unverifiable_fields({"category": "ocr", "subtype": "real", "targets": targets}) == ["count", "attenuation"]
    assert unverifiable_fields({"category": "count", "subtype": "vague", "targets": targets}) == []  # by construction
    assert unverifiable_fields({"category": "ocr", "subtype": "count/vague", "targets": targets}) == []
    negative = {**targets, "count": "0", "segments": None, "diameter_cm": None}
    assert unverifiable_fields({"category": "real", "targets": negative}) == []


def test_masked_fields_carry_no_label_and_no_loss():
    import torch

    from liversira_ai.nlp import extraction_model as em

    class Tok:
        pad_token_id = 0

        def __call__(self, text, truncation=True, max_length=512, return_offsets_mapping=True):
            import re
            pieces = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            offsets = [(0, 0), *pieces, (0, 0)]
            return {"input_ids": list(range(1, len(offsets) + 1)), "attention_mask": [1] * len(offsets),
                    "offset_mapping": offsets}

    text = "A large mass of 9.6 x 6.8 cm and six smaller lesions."
    targets = {"count": None, "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}
    masked = em.encode_item(Tok(), text, {**targets, "unverifiable": ["count", "segments", "attenuation", "diameter_cm"]}, 64)
    plain = em.encode_item(Tok(), text, targets, 64)
    assert masked["count"] == em.IGNORE and masked["attenuation"] == em.IGNORE and masked["segments_mask"] == 0.0
    assert set(masked["diameter_tags"]) == {em.IGNORE}
    assert plain["count"] == em.COUNT_LABELS.index(em.UNSTATED) and plain["segments_mask"] == 1.0

    batch = em.collate([masked], pad_id=0)
    width = batch["input_ids"].shape[1]
    out = {"count": torch.randn(1, len(em.COUNT_LABELS), requires_grad=True), "segments": torch.randn(1, 9),
           "attenuation": torch.randn(1, len(em.ATTENUATION_LABELS)), "diameter": torch.randn(1, width, 2)}
    parts = em.losses(out, batch)
    assert all(torch.isfinite(v) and float(v) == 0.0 for v in parts.values())


def test_losses_without_mask_are_unchanged():
    import torch

    from liversira_ai.nlp import extraction_model as em

    torch.manual_seed(0)
    batch = {"count": torch.tensor([1, 3]), "attenuation": torch.tensor([0, 3]),
             "segments_stated": torch.tensor([1.0, 0.0]), "segments": torch.rand(2, 8).round(),
             "diameter_tags": torch.tensor([[em.IGNORE, 1, 0], [em.IGNORE, 0, 0]])}
    out = {"count": torch.randn(2, len(em.COUNT_LABELS)), "segments": torch.randn(2, 9),
           "attenuation": torch.randn(2, len(em.ATTENUATION_LABELS)), "diameter": torch.randn(2, 3, 2)}
    ce, bce = torch.nn.functional.cross_entropy, torch.nn.functional.binary_cross_entropy_with_logits
    bits = bce(out["segments"][:, :8], batch["segments"], reduction="none").mean(dim=1)
    expected = {"count": ce(out["count"], batch["count"]), "attenuation": ce(out["attenuation"], batch["attenuation"]),
                "segments": bce(out["segments"][:, 8], batch["segments_stated"]) + bits[0],
                "diameter": ce(out["diameter"].reshape(-1, 2), batch["diameter_tags"].reshape(-1), ignore_index=em.IGNORE)}
    parts = em.losses(out, batch)
    for k, v in expected.items():
        assert torch.allclose(parts[k], v), k
