"""Multi-head transformer for Phase 6.2 extraction, with target encoding and decoding.

One encoder, four heads, all trained together on liver-only text:
- count: softmax over COUNT_LABELS (COUNT_CLASSES + "unstated");
- segments: 8 sigmoid outputs (segment 1-8) plus one "stated" sigmoid;
- attenuation: softmax over ATTENUATION_LABELS (three terms + "unstated");
- diameter: per-token tag (inside the largest lesion's size phrase or not).

``decode`` turns outputs back into target-shaped fields (``None`` = unstated), and
``to_parser_fields`` into the field names used by the parser and the Challenge Set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from liversira_ai.nlp.extraction_targets import ATTENUATIONS, COUNT_CLASSES, diameter_from_spans

UNSTATED = "unstated"
COUNT_LABELS = (*COUNT_CLASSES, UNSTATED)
ATTENUATION_LABELS = (*ATTENUATIONS, UNSTATED)
N_SEGMENTS = 8
IGNORE = -100


# ── Encoding ────────────────────────────────────────────────────────────────
def encode_item(tokenizer, text: str, targets: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Token ids plus head labels for one item; ``truncated_span`` flags a size span cut off by truncation.

    Fields listed in ``targets["unverifiable"]`` (Amendment 3) get no label: their
    loss is masked, so the model is not taught "unstated" where the text may state it.
    """
    masked = set(targets.get("unverifiable", ()))
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    spans = [tuple(s) for s in targets["diameter_spans"]]
    tags = []
    for start, end in enc["offset_mapping"]:
        if start == end or "diameter_cm" in masked:  # special token, or no verified size
            tags.append(IGNORE)
        else:
            tags.append(int(any(start < e and end > s for s, e in spans)))
    segments = targets["segments"]
    count = COUNT_LABELS.index(targets["count"] if targets["count"] is not None else UNSTATED)
    attenuation = ATTENUATION_LABELS.index(targets["attenuation"] or UNSTATED)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "offsets": [tuple(o) for o in enc["offset_mapping"]],
        "count": IGNORE if "count" in masked else count,
        "segments_stated": float(segments is not None),
        "segments_mask": 0.0 if "segments" in masked else 1.0,
        "segments": [float(segments is not None and k + 1 in segments) for k in range(N_SEGMENTS)],
        "attenuation": IGNORE if "attenuation" in masked else attenuation,
        "diameter_tags": tags,
        "truncated_span": bool(spans) and 1 not in tags,
    }


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, value: int) -> torch.Tensor:
        return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "count": torch.tensor([b["count"] for b in batch]),
        "segments_stated": torch.tensor([b["segments_stated"] for b in batch]),
        "segments_mask": torch.tensor([b.get("segments_mask", 1.0) for b in batch]),
        "segments": torch.tensor([b["segments"] for b in batch]),
        "attenuation": torch.tensor([b["attenuation"] for b in batch]),
        "diameter_tags": pad("diameter_tags", IGNORE),
    }


# ── Model ───────────────────────────────────────────────────────────────────
class ExtractionModel(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.count = nn.Linear(hidden, len(COUNT_LABELS))
        self.segments = nn.Linear(hidden, N_SEGMENTS + 1)  # last unit: segments stated
        self.attenuation = nn.Linear(hidden, len(ATTENUATION_LABELS))
        self.diameter = nn.Linear(hidden, 2)

    @classmethod
    def from_checkpoint(cls, repo_id: str, revision: str) -> "ExtractionModel":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(repo_id, revision=revision))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        cls = hidden[:, 0]
        return {
            "count": self.count(cls),
            "segments": self.segments(cls),
            "attenuation": self.attenuation(cls),
            "diameter": self.diameter(hidden),
        }


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over labelled rows; zero (not NaN) when every row is masked."""
    if not bool((labels != IGNORE).any()):
        return logits.float().sum() * 0.0
    return nn.functional.cross_entropy(logits.float(), labels, ignore_index=IGNORE)


def losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """One loss per head; segment bits are scored only where segments are stated.

    Masked (unverifiable) fields carry IGNORE labels or a zero segments mask. With no
    mask the losses are the same as before Amendment 3.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    seg_logits = outputs["segments"].float()
    stated = batch["segments_stated"]
    mask = batch.get("segments_mask", torch.ones_like(stated))
    bits = bce(seg_logits[:, :N_SEGMENTS], batch["segments"], reduction="none").mean(dim=1)
    stated_loss = bce(seg_logits[:, N_SEGMENTS], stated, reduction="none")
    used = stated * mask
    return {
        "count": _ce(outputs["count"], batch["count"]),
        "segments": (stated_loss * mask).sum() / mask.sum().clamp(min=1) + (bits * used).sum() / used.sum().clamp(min=1),
        "attenuation": _ce(outputs["attenuation"], batch["attenuation"]),
        "diameter": _ce(outputs["diameter"].reshape(-1, 2), batch["diameter_tags"].reshape(-1)),
    }


# ── Decoding ────────────────────────────────────────────────────────────────
@dataclass
class Decoded:
    count: str | None
    segments: list[int] | None
    attenuation: str | None
    diameter_cm: float | None
    diameter_spans: list[tuple[int, int]]
    confidence: dict[str, float]


def _tag_spans(text: str, offsets: list[tuple[int, int]], probs: list[float]) -> list[tuple[tuple[int, int], float]]:
    """Contiguous runs of tokens tagged inside (p >= 0.5), as character spans with mean probability."""
    runs, current = [], []
    for (start, end), p in zip(offsets, probs):
        if start == end:
            continue
        if p >= 0.5:
            current.append((start, end, p))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [((run[0][0], run[-1][1]), sum(p for _, _, p in run) / len(run)) for run in runs]


def decode(text: str, offsets: list[tuple[int, int]], outputs: dict[str, torch.Tensor]) -> Decoded:
    """Decode one item's head outputs (unbatched tensors on CPU, float)."""
    count_p = torch.softmax(outputs["count"], -1)
    att_p = torch.softmax(outputs["attenuation"], -1)
    seg_p = torch.sigmoid(outputs["segments"])
    tag_p = torch.softmax(outputs["diameter"], -1)[:, 1].tolist()

    count = COUNT_LABELS[int(count_p.argmax())]
    attenuation = ATTENUATION_LABELS[int(att_p.argmax())]
    stated = float(seg_p[N_SEGMENTS])
    segments = None
    if stated >= 0.5:
        bits = seg_p[:N_SEGMENTS]
        chosen = [k + 1 for k in range(N_SEGMENTS) if bits[k] >= 0.5] or [int(bits.argmax()) + 1]
        segments = chosen

    runs = _tag_spans(text, offsets, tag_p)
    spans, diameter, diameter_conf = [], None, 0.0
    for span, conf in sorted(runs, key=lambda r: -r[1]):
        value = diameter_from_spans(text, [span])
        if value is not None:
            spans, diameter, diameter_conf = [span], value, conf
            break
    return Decoded(
        count=None if count == UNSTATED else count,
        segments=segments,
        attenuation=None if attenuation == UNSTATED else attenuation,
        diameter_cm=diameter,
        diameter_spans=spans,
        confidence={
            "count": float(count_p.max()),
            "segments": stated if segments is not None else 1 - stated,
            "attenuation": float(att_p.max()),
            "diameter": diameter_conf if diameter is not None else 1 - max(tag_p, default=0.0),
        },
    )


def to_parser_fields(decoded: Decoded) -> dict[str, Any]:
    """Map to the parser / Challenge Set field names. "10+" maps to 10."""
    count = decoded.count
    lesion_count = None if count is None else (10 if count == "10+" else int(count))
    if lesion_count == 0:
        return {"has_lesion": False, "lesion_count": 0, "location_segments": None,
                "derived_max_diameter_cm": None, "attenuation": None}
    described = any(v is not None for v in (decoded.segments, decoded.attenuation, decoded.diameter_cm))
    return {
        "has_lesion": True if lesion_count or described else None,
        "lesion_count": lesion_count,
        "location_segments": decoded.segments,
        "derived_max_diameter_cm": decoded.diameter_cm,
        "attenuation": decoded.attenuation,
    }
