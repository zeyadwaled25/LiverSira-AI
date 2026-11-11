"""Reliability report: bootstrap CIs and coverage-adjusted metrics.

Scores the saved predictions of the rule-based baseline and the frozen learned
presence model; nothing is refitted and no new prediction is made.

References:

* ``source``: the source-derived labels in ``ground_truth.jsonl`` (all studies).
* ``gold``: human-adjudicated labels in ``gold_standard.jsonl``, used
  automatically once that file exists (only the studies it contains).

Usage (from the repository root):

    python scripts/nlp/reliability_report.py [--resamples 2000] [--seed 42]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from liversira_ai.nlp.metrics import (  # noqa: E402
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    field_report,
    format_ci,
    paired_comparison,
    presence_report,
)

GROUND_TRUTH_PATH = Path("data/processed/nlp/ground_truth.jsonl")
GOLD_PATH = Path("data/processed/nlp/gold_standard.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
BASELINE_PATH = Path("results/nlp/predictions/baseline_inferences_updated.jsonl")
LEARNED_PATHS = {
    # Selected architecture, fitted on Train only (validation-only experiment).
    "val": Path("results/nlp/predictions/word_logistic_regression_validation_predictions.jsonl"),
    # Frozen model, fitted on Train + Validation, evaluated once on Test.
    "test": Path("results/nlp/predictions/learned_presence_test_predictions.jsonl"),
}
OUTPUT_DIR = Path("results/nlp/reliability")
METRICS_PATH = OUTPUT_DIR / "reliability_metrics.json"
REPORT_PATH = OUTPUT_DIR / "reliability_report.md"
SPLIT_NAMES = {"val": "validation", "test": "test"}
DIAMETER_TOLERANCE_CM = 0.5


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for record in map(json.loads, handle) if record}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _baseline_fields(
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


def _evaluate_split(
    ids: list[str],
    reference: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    learned: dict[str, dict[str, Any]] | None,
    options: dict[str, int],
) -> dict[str, Any]:
    ids = [record_id for record_id in ids if record_id in reference and record_id in baseline]
    truth = [bool(reference[record_id]["has_lesion"]) for record_id in ids]
    baseline_presence = [baseline[record_id]["liver_lesions"].get("has_lesion") for record_id in ids]
    result: dict[str, Any] = {
        "studies": len(ids),
        "baseline": {
            "presence": presence_report(truth, baseline_presence, **options),
            "fields_on_reference_positives": _baseline_fields(ids, reference, baseline, options),
        },
    }
    if learned is not None:
        learned_ids = [record_id for record_id in ids if record_id in learned]
        if len(learned_ids) != len(ids):
            raise ValueError("Learned predictions do not cover every evaluated study")
        learned_presence = [bool(learned[record_id]["prediction"]) for record_id in ids]
        result["learned_presence"] = presence_report(truth, learned_presence, **options)
        common = [index for index, value in enumerate(baseline_presence) if value is not None]
        result["paired_learned_vs_baseline_on_baseline_answered"] = paired_comparison(
            [truth[index] for index in common],
            [learned_presence[index] for index in common],
            [bool(baseline_presence[index]) for index in common],
            **options,
        )
    return result


def _reference_labels(path: Path, key: str) -> dict[str, dict[str, Any]]:
    return {record_id: record[key] for record_id, record in _load_jsonl(path).items()}


def build_report(n_resamples: int, seed: int) -> dict[str, Any]:
    options = {"n_resamples": n_resamples, "seed": seed}
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    baseline = _load_jsonl(BASELINE_PATH)
    learned = {split: _load_jsonl(path) for split, path in LEARNED_PATHS.items()}
    references = {"source": _reference_labels(GROUND_TRUTH_PATH, "ground_truth")}
    if GOLD_PATH.exists():
        references["gold"] = _reference_labels(GOLD_PATH, "gold")

    inputs = [GROUND_TRUTH_PATH, SPLITS_PATH, BASELINE_PATH, *LEARNED_PATHS.values()]
    if GOLD_PATH.exists():
        inputs.append(GOLD_PATH)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "method": {
            "interval": "percentile bootstrap over studies",
            "confidence": 0.95,
            "n_resamples": n_resamples,
            "seed": seed,
            "presence_views": {
                "selective": "answered studies only",
                "end_to_end": "all studies; an abstention counts as an error",
                "abstain_as_negative": "all studies; an abstention is read as no lesion",
            },
            "field_metrics": "over reference-positive studies; mean over predicted studies, reported with coverage",
        },
        "learned_predictions": {SPLIT_NAMES[k]: str(v.as_posix()) for k, v in LEARNED_PATHS.items()},
        "inputs_sha256": {path.as_posix(): _sha256(path) for path in inputs},
        "git_commit": _git_commit(),
        "references": {},
    }
    for reference_name, labels in references.items():
        result["references"][reference_name] = {
            SPLIT_NAMES[split]: _evaluate_split(
                splits[split], labels, baseline, learned[split], options
            )
            for split in ("val", "test")
        }
    return result


def _presence_rows(name: str, report: dict[str, Any]) -> list[str]:
    rows = [f"| {name} | Coverage | {format_ci(report['coverage']['coverage'])} |"]
    for view in ("selective", "end_to_end", "abstain_as_negative"):
        for metric, item in report[view].items():
            rows.append(f"| {name} | {view.replace('_', ' ')} {metric} | {format_ci(item)} |")
    return rows


def render_markdown(result: dict[str, Any]) -> str:
    method = result["method"]
    lines = [
        "# NLP Reliability Report",
        "",
        f"95% percentile bootstrap intervals over studies ({method['n_resamples']} resamples, "
        f"seed {method['seed']}). Saved predictions only; nothing was refitted.",
        "",
        "Presence is reported in three views so abstentions are never hidden:",
        "- **selective**: answered studies only (the view earlier reports used);",
        "- **end to end**: every study counts and an abstention is an error;",
        "- **abstain as negative**: an abstention is read as “no lesion”.",
        "",
        "Learned model: the selected word TF-IDF + logistic regression. Validation rows come from "
        "the Train-only fit; Test rows come from the frozen Train + Validation fit.",
    ]
    for reference_name, splits in result["references"].items():
        title = (
            "Source-derived labels (not a clinical gold standard)"
            if reference_name == "source"
            else "Human gold standard"
        )
        lines += ["", f"## Reference: {title}"]
        if reference_name == "gold":
            lines += [
                "",
                "Gold-set estimates are unweighted over a stratified sample that "
                "over-represents label conflicts and hard cases; weight by the key's "
                "`inclusion_probability` for population estimates. Fields the report "
                "does not state are excluded from that field's metrics.",
            ]
        for split_name, split in splits.items():
            lines += [
                "",
                f"### {split_name.capitalize()} ({split['studies']} studies)",
                "",
                "| System | Metric | Estimate [95% CI] |",
                "|---|---|---|",
            ]
            lines += _presence_rows("Rule-based", split["baseline"]["presence"])
            if "learned_presence" in split:
                lines += _presence_rows("Learned", split["learned_presence"])
            paired = split.get("paired_learned_vs_baseline_on_baseline_answered")
            if paired:
                lines += [
                    "",
                    f"Paired comparison on the {paired['studies']} studies the rule-based "
                    "baseline answered (learned minus rule-based):",
                    f"- F1 difference: {format_ci(paired['f1_difference'], 4)}",
                    f"- Accuracy difference: {format_ci(paired['accuracy_difference'], 4)}",
                    f"- Discordant studies: learned right / rule-based wrong "
                    f"{paired['first_correct_second_wrong']}, rule-based right / learned wrong "
                    f"{paired['second_correct_first_wrong']}; exact McNemar p = "
                    f"{paired['mcnemar_exact_p']:.3g}",
                ]
            fields = split["baseline"]["fields_on_reference_positives"]
            lines += [
                "",
                "Rule-based field extraction on reference-positive studies:",
                "",
                "| Field metric | Eligible | Coverage [95% CI] | Mean on predicted [95% CI] |",
                "|---|---:|---|---|",
            ]
            for name, item in fields.items():
                lines.append(
                    f"| {name.replace('_', ' ')} | {item['eligible']} | "
                    f"{format_ci(item['coverage'])} | {format_ci(item['mean'])} |"
                )
    lines += [
        "",
        "## Reading these numbers",
        "- Interval width reflects sampling uncertainty only; it does not cover label noise "
        "in the source-derived reference.",
        "- Overlapping validation and test intervals mean an observed gap may be noise.",
        "- Field means are computed only where a prediction exists; read them with coverage.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = build_report(args.resamples, args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_markdown(result), encoding="utf-8")
    print(f"[DONE] Metrics: {METRICS_PATH}")
    print(f"[DONE] Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
