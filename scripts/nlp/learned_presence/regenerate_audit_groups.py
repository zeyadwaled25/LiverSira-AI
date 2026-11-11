"""Regenerate audit-group tables from saved predictions without refitting.

The MATCH / MISMATCH / UNCERTAIN groups come from the target-consistency audit.
When the audit is corrected, the saved model predictions do not change; only
their group labels, group metrics, and error-example categories do.  This
script rebuilds those parts from the persisted prediction files, so no model
is refitted and no new Test prediction is made.

Usage (from the repository root):

    python -m scripts.nlp.learned_presence.regenerate_audit_groups            # write updated outputs
    python -m scripts.nlp.learned_presence.regenerate_audit_groups --check \
        --audit <old_audit.csv>                              # verify against existing outputs
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from liversira_ai.nlp.learned_presence import load_jsonl
from scripts.nlp import evaluate_baseline
from scripts.nlp.learned_presence import compare_learned_presence as compare
from scripts.nlp.learned_presence import final_evaluate_learned_presence as final
from scripts.nlp.learned_presence import train_learned_presence as train

AUDIT_PATH = Path("data/processed/nlp/target_consistency_audit.csv")
MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
PREDICTIONS_DIR = Path("results/nlp/predictions")
GROUPS = ("MATCH", "MISMATCH", "UNCERTAIN")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def _load_statuses(path: Path) -> dict[str, str]:
    import csv

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row["overall_status"] for row in csv.DictReader(handle)}


def _relabel(rows: list[dict[str, Any]], audit: dict[str, str]) -> None:
    for row in rows:
        row["audit_group"] = audit.get(row["id"], "UNAVAILABLE")


def _arrays(rows: list[dict[str, Any]]) -> tuple[list[str], list[bool], np.ndarray]:
    ids = [row["id"] for row in rows]
    labels = [bool(row["source_label"]) for row in rows]
    predictions = np.array([bool(row["prediction"]) for row in rows])
    return ids, labels, predictions


def _validation_experiment(
    audit: dict[str, str], master: dict[str, dict[str, Any]]
) -> dict[Path, str]:
    predictions_path = PREDICTIONS_DIR / "learned_presence_validation_predictions.jsonl"
    rows = _read_rows(predictions_path)
    _relabel(rows, audit)
    result = json.loads(train.METRICS_PATH.read_text(encoding="utf-8"))
    result["comparison_on_common_eligible_records"] = train._compare_baseline(rows, master)
    result["audit_group_metrics"] = {
        group: train._group_metrics(rows, group) for group in GROUPS
    }
    return {
        predictions_path: _rows_text(rows),
        train.METRICS_PATH: _json_text(result),
        train.REPORT_PATH: train._render_report(result),
    }


def _architecture_comparison(
    audit: dict[str, str], master: dict[str, dict[str, Any]]
) -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    result = json.loads(compare.METRICS_PATH.read_text(encoding="utf-8"))
    for name, item in result["models"].items():
        predictions_path = PREDICTIONS_DIR / f"{name}_validation_predictions.jsonl"
        if not predictions_path.exists():
            # The selected word model's validation rows live under its own name.
            raise FileNotFoundError(predictions_path)
        rows = _read_rows(predictions_path)
        _relabel(rows, audit)
        ids, labels, predictions = _arrays(rows)
        item["audit_groups"] = compare._group_metrics(ids, labels, predictions, audit)
        item["error_examples"] = compare._error_examples(
            ids, labels, predictions, master, audit
        )
        item["validation_predictions"] = str(predictions_path)
        outputs[predictions_path] = _rows_text(rows)
    outputs[compare.METRICS_PATH] = _json_text(result)
    outputs[compare.REPORT_PATH] = compare._report(result)
    return outputs


def _final_test(audit: dict[str, str], master: dict[str, dict[str, Any]]) -> dict[Path, str]:
    predictions_path = PREDICTIONS_DIR / "learned_presence_test_predictions.jsonl"
    rows = _read_rows(predictions_path)
    _relabel(rows, audit)
    ids, labels, predictions = _arrays(rows)
    result = json.loads(final.METRICS_PATH.read_text(encoding="utf-8"))
    result["audit_groups"] = final._group_metrics(ids, labels, predictions, audit)
    result["error_examples"] = final._error_examples(ids, labels, predictions, master, audit)
    result["artifacts"]["predictions"] = str(predictions_path)
    return {
        predictions_path: _rows_text(rows),
        final.METRICS_PATH: _json_text(result),
        final.REPORT_PATH: final._render_report(result),
    }


def _baseline(audit_path: Path) -> dict[Path, str]:
    """Re-run the (fitting-free) baseline evaluation into a temporary folder."""
    originals = (
        evaluate_baseline.AUDIT_PATH,
        evaluate_baseline.METRICS_PATH,
        evaluate_baseline.REPORT_PATH,
    )
    with tempfile.TemporaryDirectory() as tmp:
        evaluate_baseline.AUDIT_PATH = audit_path
        evaluate_baseline.METRICS_PATH = Path(tmp) / "metrics.json"
        evaluate_baseline.REPORT_PATH = Path(tmp) / "report.md"
        try:
            evaluate_baseline.run_evaluation()
            return {
                originals[1]: evaluate_baseline.METRICS_PATH.read_text(encoding="utf-8"),
                originals[2]: evaluate_baseline.REPORT_PATH.read_text(encoding="utf-8"),
            }
        finally:
            (
                evaluate_baseline.AUDIT_PATH,
                evaluate_baseline.METRICS_PATH,
                evaluate_baseline.REPORT_PATH,
            ) = originals


def regenerate(audit_path: Path, check: bool) -> int:
    audit = _load_statuses(audit_path)
    master = load_jsonl(MASTER_PATH)
    outputs: dict[Path, str] = {}
    outputs.update(_baseline(audit_path))
    outputs.update(_validation_experiment(audit, master))
    outputs.update(_architecture_comparison(audit, master))
    outputs.update(_final_test(audit, master))

    differing = [
        path
        for path, text in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    if check:
        for path in sorted(outputs):
            print(f"{'DIFFERS' if path in differing else 'same   '} {path}")
        return 1 if differing else 0
    for path in differing:
        path.write_text(text := outputs[path], encoding="utf-8")
        print(f"[UPDATED] {path}")
    for path in sorted(set(outputs) - set(differing)):
        print(f"[UNCHANGED] {path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report which outputs would change; exit 1 if any differ.",
    )
    args = parser.parse_args()
    sys.exit(regenerate(args.audit, args.check))


if __name__ == "__main__":
    main()
