"""Produce auditable error and abstention summaries for the frozen baseline."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

GROUND_TRUTH_PATH = Path("data/processed/nlp/ground_truth.jsonl")
INFERENCES_PATH = Path("evaluation/baseline/baseline_inferences_updated.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
OUTPUT_JSON = Path("evaluation/baseline/baseline_error_analysis.json")
OUTPUT_REPORT = Path("evaluation/baseline/baseline_error_analysis.md")


def _load(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for record in map(json.loads, handle)}


def run_error_analysis() -> dict[str, Any]:
    gt = _load(GROUND_TRUTH_PATH)
    inf = _load(INFERENCES_PATH)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"baseline_version": "rule-based-v1.0.0", "splits": {}}
    for split_name in ("val", "test"):
        counts = Counter()
        examples: dict[str, list[str]] = {}
        for record_id in splits[split_name]:
            if record_id not in gt or record_id not in inf:
                counts["unpaired"] += 1
                continue
            truth = gt[record_id]["ground_truth"]
            prediction = inf[record_id]["liver_lesions"]
            if prediction["has_lesion"] is None:
                counts["presence_abstention"] += 1
                examples.setdefault("presence_abstention", []).append(record_id)
            elif bool(prediction["has_lesion"]) != bool(truth["has_lesion"]):
                key = "false_positive" if prediction["has_lesion"] else "false_negative"
                counts[key] += 1
                examples.setdefault(key, []).append(record_id)
            if truth["has_lesion"]:
                largest_truth = truth.get("largest_lesion") or {}
                largest_prediction = prediction.get("largest_lesion") or {}
                fields = (
                    ("lesion_count", truth.get("lesion_count"), prediction.get("lesion_count")),
                    ("location_segments", largest_truth.get("location_segments"), largest_prediction.get("location_segments")),
                    ("diameter_cm", largest_truth.get("source_diameter_cm"), largest_prediction.get("derived_max_diameter_cm")),
                    ("attenuation", largest_truth.get("attenuation"), largest_prediction.get("attenuation")),
                )
                for field, expected, actual in fields:
                    if expected is None or actual is None:
                        counts[f"{field}_excluded"] += 1
                    elif field == "diameter_cm":
                        if abs(float(expected) - float(actual)) > 0.5:
                            counts[f"{field}_error"] += 1
                            examples.setdefault(f"{field}_error", []).append(record_id)
                    elif (set(expected) if field == "location_segments" else expected) != (
                        set(actual) if field == "location_segments" else actual
                    ):
                        counts[f"{field}_error"] += 1
                        examples.setdefault(f"{field}_error", []).append(record_id)
        result["splits"][split_name] = {
            "counts": dict(sorted(counts.items())),
            "examples": {key: values[:20] for key, values in examples.items()},
        }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Baseline Error Analysis", "", "Frozen rule-based-v1.0.0; examples are capped at 20 IDs per category.", ""]
    for split, item in result["splits"].items():
        lines += [f"## {split}", "", "| Category | Count |", "|---|---:|"]
        lines += [f"| {key} | {value} |" for key, value in item["counts"].items()]
        lines.append("")
    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] Error analysis: {OUTPUT_JSON}")
    print(f"[DONE] Error analysis report: {OUTPUT_REPORT}")
    return result


if __name__ == "__main__":
    run_error_analysis()
