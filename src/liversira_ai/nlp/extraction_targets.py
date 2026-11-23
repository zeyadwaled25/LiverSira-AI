"""Phase 6.2 extraction targets, from source labels and the liver-only report text.

A source value becomes a target only when the text states it; otherwise the
target is ``None`` ("unstated"), which is also the output a model should give.
The text decides only *whether* a value is written, never the value itself, so
targets do not come from the parser.

Targets per study:
- ``count``: a class in COUNT_CLASSES ("0" = no lesion), or None.
- ``segments``: sorted Couinaud segments (1-8) of the largest lesion, or None.
- ``attenuation``: one of ATTENUATIONS for the largest lesion, or None.
- ``diameter_cm``: longest dimension of the largest lesion, or None.
- ``diameter_spans``: character spans of the size phrase(s) giving it, for
  token tagging; empty when unstated.
"""

from __future__ import annotations

import re
from typing import Any

from liversira_ai.nlp.parser_v1_1 import NUMBER_WORDS

TARGETS_VERSION = "1.0.0"
COUNT_CLASSES = tuple(str(n) for n in range(10)) + ("10+",)
ATTENUATIONS = ("hypoattenuating", "hyperattenuating", "isoattenuating")
DIAMETER_TOLERANCE_CM = 0.05

# A size phrase: "3.1 x 1.6 cm", "8.2 by 4.3 cm", "82 mm", "5.1 x 4.0 centimeters".
SIZE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*(?:x|by)\s*\d+(?:\.\d+)?)*\s*"
    r"(cm|mm|centimet(?:er|re)s?|millimet(?:er|re)s?)\b",
    re.I,
)
_SEGMENT_LIST = re.compile(r"segments?\s+((?:[1-8](?:\s*(?:/|,|-|and|,\s*and)\s*)?)+)", re.I)
_MULTIPLE = re.compile(r"\b(?:multiple|several|numerous|innumerable|many)\b", re.I)
_LESION_NOUN = r"(?:\w+\s+){0,3}?(?:liver\s+|hepatic\s+)?(?:lesions|masses|nodules|tumou?rs|foci|metastases)"


def count_class(count: int | None) -> str | None:
    if count is None:
        return None
    return "10+" if count >= 10 else str(count)


def size_phrases(text: str) -> list[tuple[int, int, float]]:
    """(start, end, longest dimension in cm) for every size phrase."""
    out = []
    for match in SIZE.finditer(text):
        factor = 10.0 if match.group(1).lower().startswith("m") else 1.0
        numbers = [float(n) / factor for n in re.findall(r"\d+(?:\.\d+)?", match.group(0))]
        out.append((match.start(), match.end(), max(numbers)))
    return out


def diameter_from_spans(text: str, spans: list[tuple[int, int]]) -> float | None:
    """Longest dimension (cm) written inside the given spans: how tagged spans are read back."""
    values = [m for s, e in spans for _, _, m in size_phrases(text[s:e])]
    return round(max(values), 2) if values else None


def stated_segments(text: str) -> set[int]:
    return {int(d) for group in _SEGMENT_LIST.findall(text) for d in re.findall(r"[1-8]", group)}


def count_stated(text: str, count: int) -> bool:
    """The count is written as a number or number word before a plural lesion noun."""
    words = [str(count)] + [w for w, n in NUMBER_WORDS.items() if n == count and len(w) > 2]
    return bool(re.search(rf"\b(?:{'|'.join(words)})\s+{_LESION_NOUN}\b", text, re.I))


def _parse_segments(value: str) -> list[int] | None:
    parts = [p.strip() for p in str(value).split(",")]
    if not parts or not all(p.isdigit() and 1 <= int(p) <= 8 for p in parts):
        return None
    return sorted(int(p) for p in parts)


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def targets_from_source(
    text: str, count: int, location: str, diameter: str, attenuation: str
) -> dict[str, Any]:
    """Targets for a real report from its source labels, kept only where the text states them."""
    if count == 0:
        return {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}

    if count == 1:
        target_count = None if _MULTIPLE.search(text) else "1"
    else:
        target_count = count_class(count) if count_stated(text, count) else None

    segments = _parse_segments(location)
    if segments is not None and not set(segments) <= stated_segments(text):
        segments = None

    target_attenuation = attenuation if attenuation in ATTENUATIONS and attenuation in text.lower() else None

    value = _parse_float(diameter)
    spans = []
    if value is not None:
        spans = [(s, e) for s, e, m in size_phrases(text) if abs(m - value) <= DIAMETER_TOLERANCE_CM]
    return {
        "count": target_count,
        "segments": segments,
        "attenuation": target_attenuation,
        "diameter_cm": value if spans else None,
        "diameter_spans": spans,
    }


def targets_from_expected(expected: dict[str, Any], diameter_spans: list[tuple[int, int]]) -> dict[str, Any]:
    """Targets for a generated item from its by-construction values (Challenge Set field names)."""
    segments = expected.get("location_segments")
    return {
        "count": count_class(expected.get("lesion_count")),
        "segments": sorted(segments) if segments else None,
        "attenuation": expected.get("attenuation"),
        "diameter_cm": expected.get("derived_max_diameter_cm") if diameter_spans else None,
        "diameter_spans": diameter_spans,
    }


TARGET_FIELDS = ("count", "segments", "attenuation", "diameter_cm")


def unverifiable_fields(item: dict[str, Any]) -> list[str]:
    """Fields that cannot be scored or trained on automatically (Phase 8 plan, Amendment 3).

    For a real report (or its OCR copy) that has a lesion, a null target means the
    source value was not found in the text, or the source has no usable value. It
    does not mean the text leaves the field unstated: "a large mass ... six smaller
    lesions" states 7 lesions, yet its count target is null when the source says 8.
    Generated items and lesion-free reports have nulls by construction, so they are
    verified and kept.
    """
    category = item.get("category", "real")
    real = category == "real" or (category == "ocr" and item.get("subtype") == "real")
    targets = item["targets"]
    if not real or targets["count"] == "0":
        return []
    return [f for f in TARGET_FIELDS if targets[f] is None]
