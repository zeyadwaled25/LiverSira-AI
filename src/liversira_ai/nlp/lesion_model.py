"""Per-lesion extraction model for Phase 6.2 (config_lesion.json).

One encoder with:
- token heads: BIO tags for lesion size phrases and for the written count;
- a document head (on [CLS]): count source (none / number / enumerated / unstated);
- span heads on each lesion size span (mean of its tokens plus two features from
  the spans' parsed dimensions): segments (8 + stated), attenuation (3 + unstated),
  repeat, largest.

Decoding: the lesion is the span with the highest "largest" probability (if at
least 0.5); diameter is the longest dimension written in it; segments and
attenuation come from its span heads; the count is read from the tagged count
(number), counted over non-repeat spans (enumerated), or left unstated.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from liversira_ai.nlp.extraction_model import ATTENUATION_LABELS, UNSTATED, Decoded
from liversira_ai.nlp.extraction_targets import count_class
from liversira_ai.nlp.lesion_targets import COUNT_SOURCES, dimensions, parse_count

N_SEGMENTS = 8
IGNORE = -100
O, B, I = 0, 1, 2
N_FEATURES = 2


# ── Encoding ────────────────────────────────────────────────────────────────
def _bio(offsets: list[tuple[int, int]], spans: list[list[int]]) -> list[int]:
    """BIO tag per token: B on a span's first token, I on the rest, IGNORE on special tokens."""
    tags, previous = [], None
    for start, end in offsets:
        if start == end:
            tags.append(IGNORE)
            continue
        inside = next((k for k, (s, e) in enumerate(spans) if start < e and end > s), None)
        tags.append(O if inside is None else (I if inside == previous else B))
        previous = inside
    return tags


def _span_tokens(offsets: list[tuple[int, int]], span: list[int]) -> list[int]:
    s, e = span
    return [k for k, (a, b) in enumerate(offsets) if a != b and a < e and b > s]


def span_features(text: str, spans: list[list[int]]) -> list[list[float]]:
    """[is the longest dimension among spans, dimension / longest] per span."""
    longest = [(dimensions(text[s:e]) or (0.0,))[0] for s, e in spans]
    top = max(longest, default=0.0)
    return [[float(top > 0 and abs(v - top) <= 0.05), (v / top) if top > 0 else 0.0] for v in longest]


def encode_item(tokenizer, text: str, labels: dict[str, Any] | None, max_length: int) -> dict[str, Any]:
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    item: dict[str, Any] = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "offsets": offsets}
    if labels is None:
        return item
    lesions = [lesion for lesion in labels["lesions"] if _span_tokens(offsets, lesion["span"])]
    spans = [lesion["span"] for lesion in lesions]
    item.update({
        "size_tags": _bio(offsets, [lesion["span"] for lesion in labels["lesions"]]),
        "count_tags": _bio(offsets, labels["count_spans"]),
        "source": COUNT_SOURCES.index(labels["count_source"]),
        "span_tokens": [_span_tokens(offsets, span) for span in spans],
        "span_features": span_features(text, spans),
        "span_largest": [float(lesion["largest"]) for lesion in lesions],
        "span_repeat": [float(lesion["repeat"]) for lesion in lesions],
        "span_known": [float(lesion["known"]) for lesion in lesions],
        "span_seg_stated": [float(lesion["segments"] is not None) for lesion in lesions],
        "span_segments": [[float(lesion["segments"] is not None and k + 1 in lesion["segments"]) for k in range(N_SEGMENTS)]
                          for lesion in lesions],
        "span_attenuation": [ATTENUATION_LABELS.index(lesion["attenuation"] or UNSTATED) if lesion["known"] else IGNORE
                             for lesion in lesions],
        "dropped_spans": len(labels["lesions"]) - len(lesions),
    })
    return item


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

    out = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0)}
    if "size_tags" not in batch[0]:
        return out
    out.update({"size_tags": pad("size_tags", IGNORE), "count_tags": pad("count_tags", IGNORE),
                "source": torch.tensor([b["source"] for b in batch])})
    rows, masks = [], []
    for i, b in enumerate(batch):
        for tokens in b["span_tokens"]:
            mask = torch.zeros(width)
            mask[tokens] = 1.0
            rows.append(i)
            masks.append(mask)
    out["span_batch"] = torch.tensor(rows, dtype=torch.long)
    out["span_mask"] = torch.stack(masks) if masks else torch.zeros(0, width)

    def flat(key: str, dtype=torch.float32) -> torch.Tensor:
        values = [v for b in batch for v in b[key]]
        return torch.tensor(values, dtype=dtype) if values else torch.zeros(0, dtype=dtype)

    out["span_features"] = flat("span_features").reshape(-1, N_FEATURES)
    for key in ("span_largest", "span_repeat", "span_known", "span_seg_stated"):
        out[key] = flat(key)
    out["span_segments"] = flat("span_segments").reshape(-1, N_SEGMENTS)
    out["span_attenuation"] = flat("span_attenuation", torch.long)
    return out


# ── Model ───────────────────────────────────────────────────────────────────
class LesionModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.size_tag = nn.Linear(hidden, 3)
        self.count_tag = nn.Linear(hidden, 3)
        self.source = nn.Linear(hidden, len(COUNT_SOURCES))
        span_in = hidden + N_FEATURES
        self.segments = nn.Linear(span_in, N_SEGMENTS + 1)
        self.attenuation = nn.Linear(span_in, len(ATTENUATION_LABELS))
        self.repeat = nn.Linear(span_in, 1)
        self.largest = nn.Linear(span_in, 1)

    @classmethod
    def from_checkpoint(cls, repo_id: str, revision: str) -> "LesionModel":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(repo_id, revision=revision))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        return {"hidden": hidden, "size_tag": self.size_tag(hidden), "count_tag": self.count_tag(hidden),
                "source": self.source(hidden[:, 0])}

    def spans(self, hidden: torch.Tensor, span_batch: torch.Tensor, span_mask: torch.Tensor,
              features: torch.Tensor) -> dict[str, torch.Tensor]:
        if span_batch.numel() == 0:
            empty = hidden.new_zeros(0)
            return {"segments": empty.reshape(0, N_SEGMENTS + 1), "attenuation": empty.reshape(0, len(ATTENUATION_LABELS)),
                    "repeat": empty, "largest": empty}
        rows = hidden[span_batch]  # (spans, seq, hidden)
        mask = span_mask.to(rows.dtype).unsqueeze(-1)
        pooled = (rows * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([pooled, features.to(pooled.dtype)], dim=-1)
        return {"segments": self.segments(x), "attenuation": self.attenuation(x),
                "repeat": self.repeat(x).squeeze(-1), "largest": self.largest(x).squeeze(-1)}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.encode(batch["input_ids"], batch["attention_mask"])
        out.update({f"span_{k}": v for k, v in
                    self.spans(out["hidden"], batch["span_batch"], batch["span_mask"], batch["span_features"]).items()})
        return out


def losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ce = nn.functional.cross_entropy
    bce = nn.functional.binary_cross_entropy_with_logits
    parts = {
        "size_tag": ce(out["size_tag"].float().reshape(-1, 3), batch["size_tags"].reshape(-1), ignore_index=IGNORE),
        "count_tag": ce(out["count_tag"].float().reshape(-1, 3), batch["count_tags"].reshape(-1), ignore_index=IGNORE),
        "source": ce(out["source"].float(), batch["source"]),
    }
    zero = out["source"].float().sum() * 0.0
    if batch["span_batch"].numel() == 0:
        return {**parts, "span": zero}
    known = batch["span_known"]
    seg = out["span_segments"].float()
    stated = batch["span_seg_stated"] * known
    seg_loss = (bce(seg[:, N_SEGMENTS], batch["span_seg_stated"], reduction="none") * known).sum() / known.sum().clamp(min=1)
    bits = bce(seg[:, :N_SEGMENTS], batch["span_segments"], reduction="none").mean(1)
    seg_loss = seg_loss + (bits * stated).sum() / stated.sum().clamp(min=1)
    att = batch["span_attenuation"]
    att_loss = ce(out["span_attenuation"].float(), att, ignore_index=IGNORE) if (att != IGNORE).any() else zero
    parts["span"] = (
        bce(out["span_largest"].float(), batch["span_largest"])
        + bce(out["span_repeat"].float(), batch["span_repeat"])
        + seg_loss
        + att_loss
    )
    return parts


# ── Decoding ────────────────────────────────────────────────────────────────
def tagged_spans(offsets: list[tuple[int, int]], logits: torch.Tensor) -> list[list[int]]:
    """Character spans from BIO logits (an I without a preceding B starts a span)."""
    tags = logits.argmax(-1).tolist()
    spans: list[list[int]] = []
    current = None
    for (start, end), tag in zip(offsets, tags):
        if start == end:
            continue
        if tag == B or (tag == I and current is None):
            if current:
                spans.append(current)
            current = [start, end]
        elif tag == I:
            current[1] = end
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)
    return spans


@torch.no_grad()
def decode_item(model: LesionModel, hidden: torch.Tensor, enc_out: dict[str, torch.Tensor], text: str,
                offsets: list[tuple[int, int]]) -> tuple[Decoded, dict[str, Any]]:
    """Decode one item from its encoder outputs (unbatched: hidden is (seq, hidden))."""
    length = len(offsets)
    source_p = torch.softmax(enc_out["source"].float(), -1)
    source = COUNT_SOURCES[int(source_p.argmax())]
    size_spans = [s for s in tagged_spans(offsets, enc_out["size_tag"][:length].float()) if dimensions(text[s[0]:s[1]])]
    count_spans = tagged_spans(offsets, enc_out["count_tag"][:length].float())
    detail: dict[str, Any] = {"count_source": source, "size_spans": size_spans, "count_spans": count_spans}
    confidence = {"count_source": float(source_p.max())}
    if source == "none":
        return Decoded("0", None, None, None, [], confidence), detail

    lesions = []
    if size_spans:
        masks = torch.zeros(len(size_spans), hidden.shape[0], device=hidden.device)
        for k, span in enumerate(size_spans):
            masks[k, _span_tokens(offsets, span)] = 1.0
        feats = torch.tensor(span_features(text, size_spans), device=hidden.device)
        rows = torch.zeros(len(size_spans), dtype=torch.long, device=hidden.device)
        heads = model.spans(hidden.unsqueeze(0), rows, masks, feats)
        for k, span in enumerate(size_spans):
            seg_p = torch.sigmoid(heads["segments"][k].float())
            lesions.append({
                "span": span,
                "largest_p": float(torch.sigmoid(heads["largest"][k].float())),
                "repeat": float(torch.sigmoid(heads["repeat"][k].float())) >= 0.5,
                "seg_p": seg_p,
                "att_p": torch.softmax(heads["attenuation"][k].float(), -1),
            })
    detail["lesions"] = [{"span": l["span"], "largest_p": round(l["largest_p"], 4), "repeat": l["repeat"]} for l in lesions]

    count = None
    if source == "number" and count_spans:
        count = parse_count(text[count_spans[0][0]:count_spans[0][1]])
    elif source == "enumerated":
        count = sum(not l["repeat"] for l in lesions) or None

    pick = max(lesions, key=lambda l: l["largest_p"]) if lesions else None
    if pick is None or pick["largest_p"] < 0.5:
        confidence["largest"] = pick["largest_p"] if pick else 0.0
        return Decoded(count_class(count), None, None, None, [], confidence), detail
    seg_p, att_p = pick["seg_p"], pick["att_p"]
    segments = None
    if float(seg_p[N_SEGMENTS]) >= 0.5:
        bits = seg_p[:N_SEGMENTS]
        segments = [k + 1 for k in range(N_SEGMENTS) if bits[k] >= 0.5] or [int(bits.argmax()) + 1]
    attenuation = ATTENUATION_LABELS[int(att_p.argmax())]
    span = pick["span"]
    confidence.update({"largest": pick["largest_p"], "attenuation": float(att_p.max())})
    return (
        Decoded(count_class(count), segments, None if attenuation == UNSTATED else attenuation,
                round(dimensions(text[span[0]:span[1]])[0], 2), [tuple(span)], confidence),
        detail,
    )
