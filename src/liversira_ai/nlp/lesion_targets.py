"""Per-lesion labels for the 6.2 per-lesion model, and how they decode to targets.

Labels per item:
- ``count_source``: "none" (no lesion), "number" (the count is written; read the
  tagged ``count_spans``), "enumerated" (count the distinct lesions described), or
  "unstated".
- ``lesions``: one entry per lesion size phrase: ``span``, ``repeat`` (restates a
  lesion already described), ``largest`` (a phrase of the largest lesion), and,
  when ``known``, ``segments`` and ``attenuation`` (``None`` = not stated).
  The source labels pick the largest lesion by volume, which is not always the
  longest written dimension, so ``largest`` is a label, not a computed value.

Real reports: every size phrase in the liver-only text of a lesion-positive report
is a lesion size (checked on Train: negatives have 2 size phrases in 5,453
reports), except sizes inside a range ("ranging from 0.6 to 6.3 cm"), which
describe a group. A phrase repeats an earlier one when its dimensions are the same.
Attributes are known only for the largest lesion's phrases (the source labels
describe only that lesion), and only where the text states them.
"""

from __future__ import annotations

import re
from typing import Any

from liversira_ai.nlp.extraction_targets import (
    _LESION_NOUN,
    _MULTIPLE,
    count_class,
    size_phrases,
    targets_from_source,
)
from liversira_ai.nlp.parser_v1_1 import NUMBER_WORDS

LESION_TARGETS_VERSION = "1.0.0"
COUNT_SOURCES = ("none", "number", "enumerated", "unstated")
DIAMETER_TOLERANCE_CM = 0.05


def dimensions(text: str) -> tuple[float, ...]:
    """Dimensions (cm) written in one size phrase, longest first."""
    phrases = size_phrases(text)
    if not phrases:
        return ()
    factor = 10.0 if re.search(r"\bmm\b|millimet", text, re.I) else 1.0
    return tuple(sorted((float(n) / factor for n in re.findall(r"\d+(?:\.\d+)?", text)), reverse=True))


def parse_count(text: str) -> int | None:
    """Read a tagged count: digits or a number word."""
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return NUMBER_WORDS.get(text) if text not in ("a", "an") else None


def count_span(text: str, count: int) -> tuple[int, int] | None:
    """Span of the written count before a plural lesion noun (as in ``count_stated``)."""
    words = [str(count)] + [w for w, n in NUMBER_WORDS.items() if n == count and len(w) > 2]
    match = re.search(rf"\b({'|'.join(words)})\s+{_LESION_NOUN}\b", text, re.I)
    return match.span(1) if match else None


_RANGE_BEFORE = re.compile(r"\b(?:rang\w*|from)\b[^.;]*$|\bto\s+$", re.I)


def in_range_phrase(text: str, start: int) -> bool:
    """A size inside "ranging (in size) from X to Y" or "X to Y" describes a group, not one lesion."""
    return bool(_RANGE_BEFORE.search(text[max(0, start - 45) : start]))


def lesion_size_phrases(text: str) -> list[tuple[int, int, float]]:
    return [p for p in size_phrases(text) if not in_range_phrase(text, p[0])]


def mark_repeats(lesions: list[dict[str, Any]], text: str) -> None:
    seen: set[tuple[float, ...]] = set()
    for lesion in lesions:
        dims = dimensions(text[slice(*lesion["span"])])
        lesion["repeat"] = dims in seen
        seen.add(dims)


def real_labels(text: str, count: int, location: str, diameter: str, attenuation: str) -> dict[str, Any]:
    if count == 0:
        return {"count_source": "none", "count_spans": [], "lesions": []}
    targets = targets_from_source(text, count, location, diameter, attenuation)
    largest = {tuple(s) for s in targets["diameter_spans"]}
    lesions = []
    for start, end, _ in lesion_size_phrases(text):
        known = (start, end) in largest
        lesions.append({
            "span": [start, end],
            "largest": known,
            "known": known,
            "segments": targets["segments"] if known else None,
            "attenuation": targets["attenuation"] if known else None,
        })
    mark_repeats(lesions, text)
    distinct = sum(not lesion["repeat"] for lesion in lesions)

    span = count_span(text, count) if count >= 2 else None
    if span is not None:
        return {"count_source": "number", "count_spans": [list(span)], "lesions": lesions}
    if count == 1 and not _MULTIPLE.search(text) and distinct == 1:
        return {"count_source": "enumerated", "count_spans": [], "lesions": lesions}
    return {"count_source": "unstated", "count_spans": [], "lesions": lesions}


def decode_labels(text: str, labels: dict[str, Any]) -> dict[str, Any]:
    """Targets implied by per-lesion labels (used to check labels and to decode predictions)."""
    source = labels["count_source"]
    if source == "none":
        return {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None}
    lesions = [dict(lesion, dims=dimensions(text[slice(*lesion["span"])])) for lesion in labels["lesions"]]
    lesions = [lesion for lesion in lesions if lesion["dims"]]
    count = None
    if source == "number" and labels["count_spans"]:
        count = parse_count(text[slice(*labels["count_spans"][0])])
    elif source == "enumerated":
        count = sum(not lesion["repeat"] for lesion in lesions) or None
    if not lesions:
        return {"count": count_class(count), "segments": None, "attenuation": None, "diameter_cm": None}
    flagged = [lesion for lesion in lesions if lesion.get("largest")]
    if not flagged:  # the largest lesion's size is not written
        return {"count": count_class(count), "segments": None, "attenuation": None, "diameter_cm": None}
    pick = next((lesion for lesion in flagged if lesion.get("known", True)), flagged[0])
    return {
        "count": count_class(count),
        "segments": sorted(pick["segments"]) if pick.get("segments") else None,
        "attenuation": pick.get("attenuation"),
        "diameter_cm": round(pick["dims"][0], 2),
    }
