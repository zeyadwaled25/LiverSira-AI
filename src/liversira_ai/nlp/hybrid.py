"""Hybrid extraction rules for Phase 8: parser v1.2 first, the first extraction model as fallback.

Both take parser features (``baseline.extract_liver_lesion_features_v1_2``) and
the model's output in the same field form (``extraction_model.to_parser_fields``).

- ``hybrid_h`` (C5, the primary hybrid): a field the parser leaves empty is
  taken from the model, whatever the reason it is empty. It ignores the status.
- ``hybrid_h2`` (C6, exploratory, Amendment 1; never eligible for the Phase 8
  decision): a field the parser leaves empty is taken from the model only when
  the parser's status for it is ``unparsed``; ``not_mentioned`` stays empty.
  On Train-aug it was worse than H (613 vs 934 of 938 fields where they differ).

No thresholds, no averaging: values pass through unchanged.
"""

from typing import Any

FIELDS = ("lesion_count", "location_segments", "derived_max_diameter_cm", "attenuation")
NO_LESION = {"has_lesion": False, "lesion_count": 0, "location_segments": None,
             "derived_max_diameter_cm": None, "attenuation": None}


def hybrid_h(parser: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    if parser["has_lesion"] is False:
        return dict(NO_LESION)
    return {"has_lesion": True, **{f: parser[f] if parser[f] is not None else model.get(f) for f in FIELDS}}


def hybrid_h2(parser: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    if parser["has_lesion"] is False:
        return dict(NO_LESION)
    status = parser.get("field_status")
    if status is None:
        raise ValueError("hybrid_h2 needs parser v1.2 features (field_status)")
    out: dict[str, Any] = {"has_lesion": True}
    for f in FIELDS:
        if parser[f] is not None:
            out[f] = parser[f]
        else:
            out[f] = model.get(f) if status[f] == "unparsed" else None
    return out
