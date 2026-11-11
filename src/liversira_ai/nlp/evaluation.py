"""Scoring helpers shared by the NLP evaluation scripts.

Paths are relative to the repository root, like every script that uses them.
"""

import json
from pathlib import Path
from typing import Any

from liversira_ai.nlp.metrics import field_report

GROUND_TRUTH_PATH = Path("data/processed/nlp/ground_truth.jsonl")
GOLD_PATH = Path("data/processed/nlp/gold_standard.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
DIAMETER_TOLERANCE_CM = 0.5


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for record in map(json.loads, handle) if record}


def reference_labels(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {record_id: record[key] for record_id, record in load_jsonl(path).items()}


def baseline_fields(
    ids: list[str],
    reference: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    options: dict[str, int],
) -> dict[str, Any]:
    """Field metrics over reference-positive studies; missing predictions lower coverage."""
    count_exact, count_error = [], []
    location_exact, location_iou = [], []
    diameter_error, diameter_within = [], []
    attenuation_exact = []
    for record_id in ids:
        truth = reference[record_id]
        if not truth["has_lesion"]:
            continue
        prediction = baseline[record_id]["liver_lesions"]
        largest_truth = truth.get("largest_lesion") or {}
        largest_pred = prediction.get("largest_lesion") or {}

        # Gold labels may leave a field unstated (None); such studies are not
        # eligible for that field.
        if truth.get("lesion_count") is not None:
            count = prediction.get("lesion_count")
            count_exact.append(None if count is None else float(count == truth["lesion_count"]))
            count_error.append(None if count is None else abs(count - truth["lesion_count"]))

        if largest_truth.get("location_segments"):
            segments = largest_pred.get("location_segments")
            if segments is None:
                location_exact.append(None)
                location_iou.append(None)
            else:
                true_set, pred_set = set(largest_truth["location_segments"]), set(segments)
                location_exact.append(float(true_set == pred_set))
                location_iou.append(len(true_set & pred_set) / len(true_set | pred_set))

        diameter = largest_pred.get("derived_max_diameter_cm")
        true_diameter = largest_truth.get("source_diameter_cm", largest_truth.get("diameter_cm"))
        if true_diameter is not None:
            error = None if diameter is None else abs(diameter - true_diameter)
            diameter_error.append(error)
            diameter_within.append(
                None if error is None else float(error <= DIAMETER_TOLERANCE_CM)
            )

        if largest_truth.get("attenuation"):
            attenuation = largest_pred.get("attenuation")
            attenuation_exact.append(
                None
                if attenuation is None
                else float(str(attenuation).lower() == str(largest_truth["attenuation"]).lower())
            )

    return {
        "lesion_count_exact": field_report(count_exact, **options),
        "lesion_count_abs_error": field_report(count_error, **options),
        "location_exact": field_report(location_exact, **options),
        "location_iou": field_report(location_iou, **options),
        "diameter_abs_error_cm": field_report(diameter_error, **options),
        f"diameter_within_{DIAMETER_TOLERANCE_CM}_cm": field_report(diameter_within, **options),
        "attenuation_exact": field_report(attenuation_exact, **options),
    }
