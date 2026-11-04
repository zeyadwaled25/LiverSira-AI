"""Evaluate the rule-based baseline without treating abstentions as errors.

The source-derived labels are a comparison reference, not a clinical gold
standard.  Fields are evaluated only when the target is applicable and the
prediction is present; excluded records are reported with explicit reasons.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

GROUND_TRUTH_PATH = Path("data/processed/nlp/ground_truth.jsonl")
INFERENCES_PATH = Path("evaluation/baseline/baseline_inferences_updated.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
AUDIT_PATH = Path("data/processed/nlp/target_consistency_audit.csv")
METRICS_PATH = Path("evaluation/baseline/baseline_metrics_updated.json")
REPORT_PATH = Path("evaluation/baseline/baseline_evaluation_report_updated.md")

FIELD_NAMES = (
    "has_lesion",
    "lesion_count",
    "location_segments",
    "diameter_cm",
    "attenuation",
)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _field_stats(
    eligible: int, evaluated: int, reasons: Counter[str], not_applicable: int = 0
) -> dict[str, Any]:
    return {
        "eligible": eligible,
        "evaluated": evaluated,
        "excluded": sum(reasons.values()),
        "not_applicable": not_applicable,
        "exclusion_reasons": dict(sorted(reasons.items())),
    }


def _location_metrics(
    true_values: list[set[int]], pred_values: list[set[int]]
) -> dict[str, float | None]:
    ious: list[float] = []
    exact: list[float] = []
    for truth, prediction in zip(true_values, pred_values):
        union = truth | prediction
        ious.append(1.0 if not union else len(truth & prediction) / len(union))
        exact.append(float(truth == prediction))
    return {"iou": _mean(ious), "exact_match": _mean(exact), "denominator": len(ious)}


def _presence_metrics(
    y_true: list[bool], y_pred: list[bool], uncertain: int
) -> dict[str, Any]:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    return {
        "confusion_matrix": {
            "labels": [False, True],
            "rows_true": {
                "false": {"predicted_false": tn, "predicted_true": fp},
                "true": {"predicted_false": fn, "predicted_true": tp},
            },
        },
        "accuracy": _rate(tp + tn, len(y_true)),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
        "specificity": _rate(tn, tn + fp),
        "evaluated": len(y_true),
        "denominator": len(y_true),
        "abstained_predictions": uncertain,
    }


def _load_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record_id = record["id"]
            if record_id in records:
                duplicates.append(record_id)
            records[record_id] = record
    return records, duplicates


def _audit_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    import csv

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["id"]: row["overall_status"]
            for row in csv.DictReader(handle)
            if row.get("id") and row.get("overall_status")
        }


def _evaluate_records(
    gt_records: dict[str, dict[str, Any]],
    inf_records: dict[str, dict[str, Any]],
    split_ids: list[str],
    audit_status: dict[str, str],
) -> dict[str, Any]:
    paired_ids = [record_id for record_id in split_ids if record_id in gt_records and record_id in inf_records]
    missing_gt = [record_id for record_id in split_ids if record_id not in gt_records]
    missing_inference = [record_id for record_id in split_ids if record_id not in inf_records]

    presence_true: list[bool] = []
    presence_pred: list[bool] = []
    presence_uncertain = 0
    count_true: list[int] = []
    count_pred: list[int] = []
    location_true: list[set[int]] = []
    location_pred: list[set[int]] = []
    diameter_true: list[float] = []
    diameter_pred: list[float] = []
    attenuation_true: list[str] = []
    attenuation_pred: list[str] = []
    eligible = Counter()
    evaluated = Counter()
    excluded: dict[str, Counter[str]] = {field: Counter() for field in FIELD_NAMES}
    not_applicable = Counter()
    evidence_hits = Counter()
    evidence_record_ids: dict[str, set[str]] = {field: set() for field in FIELD_NAMES}
    eligible_record_ids: dict[str, set[str]] = {field: set() for field in FIELD_NAMES}
    uncertainty = Counter()
    group_counts = Counter()

    for record_id in paired_ids:
        gt = gt_records[record_id]["ground_truth"]
        inf = inf_records[record_id]["liver_lesions"]
        largest_gt = gt.get("largest_lesion") or {}
        largest_inf = inf.get("largest_lesion") or {}
        group_counts[audit_status.get(record_id, "UNAVAILABLE")] += 1
        for span in inf_records[record_id].get("evidence_spans", []):
            evidence_hits[span["target_field"]] += 1
            for field, target in {
                "has_lesion": "liver_lesions.has_lesion",
                "lesion_count": "liver_lesions.lesion_count",
                "location_segments": "liver_lesions.largest_lesion.location_segments",
                "diameter_cm": "liver_lesions.largest_lesion.derived_max_diameter_cm",
                "attenuation": "liver_lesions.largest_lesion.attenuation",
            }.items():
                if span["target_field"] == target:
                    evidence_record_ids[field].add(record_id)
        for item in inf_records[record_id].get("missing_or_unclear", []):
            uncertainty[item["field"]] += 1

        # Presence is applicable to every paired source-label record.
        eligible["has_lesion"] += 1
        eligible_record_ids["has_lesion"].add(record_id)
        predicted_presence = inf.get("has_lesion")
        if predicted_presence is None:
            presence_uncertain += 1
            excluded["has_lesion"]["prediction_uncertain"] += 1
        else:
            evaluated["has_lesion"] += 1
            presence_true.append(bool(gt["has_lesion"]))
            presence_pred.append(bool(predicted_presence))

        if not gt["has_lesion"]:
            for field in FIELD_NAMES[1:]:
                not_applicable[field] += 1
            continue

        target_specs = (
            ("lesion_count", gt.get("lesion_count"), inf.get("lesion_count")),
            (
                "location_segments",
                largest_gt.get("location_segments"),
                largest_inf.get("location_segments"),
            ),
            (
                "diameter_cm",
                largest_gt.get("source_diameter_cm"),
                largest_inf.get("derived_max_diameter_cm"),
            ),
            ("attenuation", largest_gt.get("attenuation"), largest_inf.get("attenuation")),
        )
        for field, truth, prediction in target_specs:
            eligible[field] += 1
            if truth is not None:
                eligible_record_ids[field].add(record_id)
            if truth is None:
                excluded[field]["ground_truth_missing"] += 1
                continue
            if prediction is None:
                excluded[field]["prediction_missing"] += 1
                continue
            evaluated[field] += 1
            if field == "lesion_count":
                count_true.append(int(truth))
                count_pred.append(int(prediction))
            elif field == "location_segments":
                location_true.append(set(truth))
                location_pred.append(set(prediction))
            elif field == "diameter_cm":
                diameter_true.append(float(truth))
                diameter_pred.append(float(prediction))
            else:
                attenuation_true.append(str(truth).lower())
                attenuation_pred.append(str(prediction).lower())

    metrics: dict[str, Any] = {
        "studies_in_split": len(split_ids),
        "paired_studies": len(paired_ids),
        "missing_ground_truth_ids": missing_gt,
        "missing_inference_ids": missing_inference,
        "group_counts": dict(sorted(group_counts.items())),
        "presence": _presence_metrics(presence_true, presence_pred, presence_uncertain),
        "lesion_count": {
            "exact_match": _rate(sum(t == p for t, p in zip(count_true, count_pred)), len(count_true)),
            "mae": _mean([abs(t - p) for t, p in zip(count_true, count_pred)]),
            "denominator": len(count_true),
        },
        "location": _location_metrics(location_true, location_pred),
        "diameter": {
            "mae_cm": _mean([abs(t - p) for t, p in zip(diameter_true, diameter_pred)]),
            "tolerance_cm": 0.5,
            "tolerance_accuracy": _rate(
                sum(abs(t - p) <= 0.5 for t, p in zip(diameter_true, diameter_pred)),
                len(diameter_true),
            ),
            "denominator": len(diameter_true),
        },
        "attenuation": {
            "accuracy": _rate(sum(t == p for t, p in zip(attenuation_true, attenuation_pred)), len(attenuation_true)),
            "denominator": len(attenuation_true),
        },
        "per_field": {},
        "evidence_coverage": {},
        "uncertainty": {
            "records_with_any_missing_or_unclear": sum(
                1 for record_id in paired_ids if inf_records[record_id].get("missing_or_unclear")
            ),
            "by_field": dict(sorted(uncertainty.items())),
            "presence_abstentions": presence_uncertain,
        },
        "metric_denominators": {
            "presence": len(presence_true),
            "lesion_count": len(count_true),
            "location": len(location_true),
            "diameter": len(diameter_true),
            "attenuation": len(attenuation_true),
        },
    }
    for field in FIELD_NAMES:
        metrics["per_field"][field] = _field_stats(
            eligible[field], evaluated[field], excluded[field], not_applicable[field]
        )
    evidence_targets = {
        "has_lesion": "liver_lesions.has_lesion",
        "lesion_count": "liver_lesions.lesion_count",
        "location_segments": "liver_lesions.largest_lesion.location_segments",
        "diameter_cm": "liver_lesions.largest_lesion.derived_max_diameter_cm",
        "attenuation": "liver_lesions.largest_lesion.attenuation",
    }
    for field, target in evidence_targets.items():
        denominator = metrics["per_field"][field]["eligible"]
        metrics["evidence_coverage"][field] = {
            "records_with_evidence": len(
                evidence_record_ids[field] & eligible_record_ids[field]
            ),
            "eligible_records": denominator,
            "coverage": _rate(
                len(evidence_record_ids[field] & eligible_record_ids[field]),
                denominator,
            ),
            "note": "Evidence coverage only; no independent evidence annotations exist, so evidence accuracy is not claimed.",
        }
    return metrics


def evaluate_split(
    gt_records: dict[str, dict[str, Any]],
    inf_records: dict[str, dict[str, Any]],
    split_ids: list[str],
    split_name: str,
    audit_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate one split; retained as a public helper for focused tests."""
    result = _evaluate_records(gt_records, inf_records, split_ids, audit_status or {})
    result["split"] = split_name
    return result


def _validate_ids(
    gt_records: dict[str, dict[str, Any]],
    inf_records: dict[str, dict[str, Any]],
    split_data: dict[str, list[str]],
    gt_duplicates: list[str],
    inf_duplicates: list[str],
) -> dict[str, Any]:
    split_sets = {name: set(ids) for name, ids in split_data.items()}
    overlap = {
        f"{left}_vs_{right}": sorted(split_sets[left] & split_sets[right])
        for left in split_sets
        for right in split_sets
        if left < right and split_sets[left] & split_sets[right]
    }
    all_split_ids = set().union(*split_sets.values())
    return {
        "ground_truth_count": len(gt_records),
        "inference_count": len(inf_records),
        "ground_truth_duplicates": sorted(gt_duplicates),
        "inference_duplicates": sorted(inf_duplicates),
        "split_overlap": overlap,
        "split_ids_missing_ground_truth": sorted(all_split_ids - set(gt_records)),
        "split_ids_missing_inference": sorted(all_split_ids - set(inf_records)),
        "ground_truth_ids_missing_from_splits": sorted(set(gt_records) - all_split_ids),
        "inference_ids_missing_from_splits": sorted(set(inf_records) - all_split_ids),
        "split_leakage_free": not overlap,
    }


def _comparison_groups(
    gt_records: dict[str, dict[str, Any]],
    inf_records: dict[str, dict[str, Any]],
    split_ids: list[str],
    statuses: dict[str, str],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for status in ("MATCH", "MISMATCH", "UNCERTAIN"):
        ids = [record_id for record_id in split_ids if statuses.get(record_id) == status]
        groups[status] = evaluate_split(gt_records, inf_records, ids, status, statuses)
    return groups


def _markdown_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Updated NLP Baseline Evaluation",
        "",
        "These are rule-based comparisons against source-derived labels. "
        "`MATCH`, `MISMATCH`, and `UNCERTAIN` come from the text-vs-source-label "
        "consistency audit; they are not clinical gold-standard groups.",
        "",
        "No new model was trained. Existing schemas, source data, and split IDs were not modified.",
        "",
        "## ID and split integrity",
        f"- Leakage-free: **{metrics['id_validation']['split_leakage_free']}**",
        f"- Ground-truth IDs: {metrics['id_validation']['ground_truth_count']:,}; "
        f"inference IDs: {metrics['id_validation']['inference_count']:,}",
        "",
        "## Validation/Test comparison",
        "| Split | Paired | Presence F1 | Count exact | Location IoU | Diameter MAE (cm) | Attenuation accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "test"):
        item = metrics["splits"][name]
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {name} | {item['paired_studies']} | {fmt(item['presence']['f1'])} | "
            f"{fmt(item['lesion_count']['exact_match'])} | {fmt(item['location']['iou'])} | "
            f"{fmt(item['diameter']['mae_cm'])} | {fmt(item['attenuation']['accuracy'])} |"
        )
    lines += [
        "",
        "## Per-field eligibility and exclusions",
        "| Split | Field | Eligible | Evaluated | Excluded | Not applicable | Reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name in ("validation", "test"):
        for field, stat in metrics["splits"][name]["per_field"].items():
            lines.append(
                f"| {name} | {field} | {stat['eligible']} | {stat['evaluated']} | "
                f"{stat['excluded']} | {stat['not_applicable']} | "
                f"{stat['exclusion_reasons'] or '—'} |"
            )
    lines += [
        "",
        "## Target/prediction definitions",
        "Every metric denominator is the number of paired records with both an "
        "applicable ground-truth value and a non-missing prediction. Presence "
        "uses evaluated non-abstaining predictions; lesion count, location, "
        "diameter, and attenuation use their respective field-level evaluated "
        "counts. Abstentions, missing targets, and non-applicable negative "
        "studies are excluded and reported separately.",
        "",
        "- Lesion count: `ground_truth.lesion_count` vs `liver_lesions.lesion_count`.",
        "- Location: `ground_truth.largest_lesion.location_segments` vs "
        "`liver_lesions.largest_lesion.location_segments`.",
        "- Diameter: `ground_truth.largest_lesion.source_diameter_cm` vs "
        "`liver_lesions.largest_lesion.derived_max_diameter_cm` (maximum reported dimension).",
        "- Evidence coverage is reported, but evidence accuracy is not claimed because there are no independent evidence annotations.",
        "",
        "## Comparison-audit groups",
        "| Split | Audit group | Studies | Presence F1 | Diameter MAE (cm) |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ("validation", "test"):
        groups = metrics["splits"][name]["comparison_groups"]
        for group in ("MATCH", "MISMATCH", "UNCERTAIN"):
            item = groups[group]
            presence_f1 = item["presence"]["f1"]
            diameter_mae = item["diameter"]["mae_cm"]
            f1_text = "—" if presence_f1 is None else f"{presence_f1:.4f}"
            diameter_text = "—" if diameter_mae is None else f"{diameter_mae:.4f}"
            lines.append(
                f"| {name} | {group} | {item['studies_in_split']} | "
                f"{f1_text} | {diameter_text} |"
            )
    lines += [
        "",
        "## Readiness",
        "**Not ready as a clinical validation benchmark.** It is usable as a transparent "
        "source-label reference only after documenting the reported exclusions, abstentions, "
        "and unresolved source-label inconsistencies. The 79 reviewed-sample records are "
        "not treated as clinical verification because the repository does not establish "
        "documented human-review provenance for these outputs.",
    ]
    return "\n".join(lines) + "\n"


def run_evaluation() -> dict[str, Any]:
    """Load records, validate IDs, and write machine-readable and Markdown reports."""
    gt_records, gt_duplicates = _load_jsonl(GROUND_TRUTH_PATH)
    inf_records, inf_duplicates = _load_jsonl(INFERENCES_PATH)
    with SPLITS_PATH.open(encoding="utf-8") as handle:
        splits = json.load(handle)
    audit_status = _audit_statuses(AUDIT_PATH)
    result = {
        "comparison_basis": "source_labels_not_clinical_gold_standard",
        "inference_file": str(INFERENCES_PATH),
        "splits_file": str(SPLITS_PATH),
        "id_validation": _validate_ids(
            gt_records, inf_records, splits, gt_duplicates, inf_duplicates
        ),
        "splits": {},
    }
    for split_key, report_name in (("val", "validation"), ("test", "test")):
        report = evaluate_split(
            gt_records, inf_records, splits[split_key], report_name, audit_status
        )
        report["comparison_groups"] = _comparison_groups(
            gt_records, inf_records, splits[split_key], audit_status
        )
        result["splits"][report_name] = report
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_markdown_report(result), encoding="utf-8")
    print(f"[DONE] Metrics: {METRICS_PATH}")
    print(f"[DONE] Report: {REPORT_PATH}")
    return result


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
