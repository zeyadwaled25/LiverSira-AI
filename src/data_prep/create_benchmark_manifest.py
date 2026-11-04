"""Create a reproducible eligibility manifest for baseline benchmarking."""

import csv
import json
from pathlib import Path
from typing import Any

MASTER_PATH = Path("data/processed/master_records.jsonl")
GROUND_TRUTH_PATH = Path("data/processed/ground_truth.jsonl")
SPLITS_PATH = Path("data/processed/splits.json")
AUDIT_PATH = Path("data/processed/target_consistency_audit.csv")
OUTPUT_PATH = Path("evaluation/baseline/benchmark_eligibility_manifest.csv")
SUMMARY_PATH = Path("evaluation/baseline/benchmark_eligibility_summary.json")


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for record in map(json.loads, handle)}


def create_manifest() -> list[dict[str, Any]]:
    masters = _load_jsonl(MASTER_PATH)
    ground_truth = _load_jsonl(GROUND_TRUTH_PATH)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    statuses: dict[str, str] = {}
    with AUDIT_PATH.open(encoding="utf-8-sig", newline="") as handle:
        statuses = {
            row["id"]: row.get("overall_status", "UNAVAILABLE")
            for row in csv.DictReader(handle)
            if row.get("id")
        }

    split_by_id = {record_id: split for split, ids in splits.items() for record_id in ids}
    rows: list[dict[str, Any]] = []
    for record_id in sorted(set(masters) | set(ground_truth) | set(split_by_id)):
        reasons: list[str] = []
        master = masters.get(record_id)
        if record_id not in split_by_id:
            reasons.append("not_in_frozen_split")
        if master is None:
            reasons.append("missing_master_record")
        narrative = (master or {}).get("reports", {}).get("narrative_clean")
        if not isinstance(narrative, str) or not narrative.strip():
            reasons.append("missing_narrative")
        if record_id not in ground_truth:
            reasons.append("missing_ground_truth")
        rows.append(
            {
                "id": record_id,
                "split": split_by_id.get(record_id, ""),
                "eligible": not reasons,
                "exclusion_reasons": "|".join(reasons),
                "audit_status": statuses.get(record_id, "UNAVAILABLE"),
                "patient_level_status": "not_assessable_no_patient_id",
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "1.0",
        "eligibility_rule": "frozen split membership + Master narrative_clean + GroundTruthRecord",
        "patient_level_leakage": "not_assessable_no_patient_id",
        "counts": {
            split: {
                "eligible": sum(row["eligible"] and row["split"] == split for row in rows),
                "excluded": sum((not row["eligible"]) and row["split"] == split for row in rows),
            }
            for split in ("train", "val", "test")
        },
        "exclusions": {
            reason: sum(reason in row["exclusion_reasons"].split("|") for row in rows)
            for reason in sorted(
                {
                    reason
                    for row in rows
                    for reason in row["exclusion_reasons"].split("|")
                    if reason
                }
            )
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Eligibility manifest: {OUTPUT_PATH}")
    print(f"[DONE] Eligibility summary: {SUMMARY_PATH}")
    return rows


if __name__ == "__main__":
    create_manifest()
