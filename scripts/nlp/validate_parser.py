"""Validation-only comparison of the rule-based parser v1.1 with frozen v1.0.

* v1.1 is run on Validation studies only (``run_baseline_inference`` with the
  Validation IDs); Test reports are never parsed or scored here.
* v1.0 is read from its saved predictions, so it is exactly the frozen baseline.
* Both are scored with the same bootstrap metrics as the reliability report,
  against source-derived labels and, once it exists, the Validation part of the
  human gold standard.

Usage (from the repository root):

    python scripts/nlp/validate_parser.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reliability_report as rr  # noqa: E402
from liversira_ai.nlp.baseline import BASELINE_VERSION, run_baseline_inference  # noqa: E402
from liversira_ai.nlp.metrics import format_ci, paired_comparison  # noqa: E402
from liversira_ai.nlp.parser import PARSER_VERSION  # noqa: E402

V10_PATH = Path("results/nlp/predictions/baseline_inferences_updated.jsonl")
V11_PATH = Path("results/nlp/predictions/baseline_v1_1_validation_inferences.jsonl")
V11_MANIFEST = Path("results/nlp/predictions/baseline_v1_1_validation_manifest.json")
OUTPUT_DIR = Path("results/nlp/parser_v1_1")
METRICS_PATH = OUTPUT_DIR / "validation_comparison.json"
REPORT_PATH = OUTPUT_DIR / "validation_comparison.md"
RESAMPLES, SEED = 2000, 42


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _score(
    ids: list[str], reference: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, Any]:
    options = {"n_resamples": RESAMPLES, "seed": SEED}
    ids = [i for i in ids if i in reference]
    truth = [bool(reference[i]["has_lesion"]) for i in ids]
    presence = [predictions[i]["liver_lesions"].get("has_lesion") for i in ids]
    return {
        "studies": len(ids),
        "presence": rr.presence_report(truth, presence, **options),
        "fields_on_reference_positives": rr._baseline_fields(ids, reference, predictions, options),
    }


def _paired_presence(
    ids: list[str], reference: dict[str, Any], v10: dict[str, Any], v11: dict[str, Any]
) -> dict[str, Any]:
    both = [
        i
        for i in ids
        if i in reference
        and v10[i]["liver_lesions"].get("has_lesion") is not None
        and v11[i]["liver_lesions"].get("has_lesion") is not None
    ]
    return paired_comparison(
        [bool(reference[i]["has_lesion"]) for i in both],
        [bool(v11[i]["liver_lesions"]["has_lesion"]) for i in both],
        [bool(v10[i]["liver_lesions"]["has_lesion"]) for i in both],
        n_resamples=RESAMPLES,
        seed=SEED,
    )


def run() -> dict[str, Any]:
    splits = json.loads(rr.SPLITS_PATH.read_text(encoding="utf-8"))
    validation = list(splits["val"])
    test_ids = set(splits["test"])
    if test_ids & set(validation):
        raise ValueError("Validation and Test overlap")

    run_baseline_inference(rr.GROUND_TRUTH_PATH.parent / "master_records.jsonl", V11_PATH, set(validation), V11_MANIFEST)
    v11 = rr._load_jsonl(V11_PATH)
    if set(v11) & test_ids or set(v11) != set(validation):
        raise ValueError("v1.1 predictions must cover exactly the Validation split")
    v10 = {i: r for i, r in rr._load_jsonl(V10_PATH).items() if i in set(validation)}

    references = {"source": rr._reference_labels(rr.GROUND_TRUTH_PATH, "ground_truth")}
    if rr.GOLD_PATH.exists():
        gold = rr._reference_labels(rr.GOLD_PATH, "gold")
        references["gold"] = {i: g for i, g in gold.items() if i in set(validation)}

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "split": "validation",
        "test_policy": "Test reports were not parsed or scored.",
        "versions": {"v1.0": "rule-based-v1.0.0 (saved predictions)", "v1.1": f"{BASELINE_VERSION} / parser {PARSER_VERSION}"},
        "inputs_sha256": {
            p.as_posix(): _sha(p) for p in (rr.GROUND_TRUTH_PATH, rr.SPLITS_PATH, V10_PATH, V11_PATH)
        },
        "git_commit": _git_commit(),
        "method": {"interval": "percentile bootstrap over studies", "n_resamples": RESAMPLES, "seed": SEED},
        "references": {},
    }
    for name, reference in references.items():
        result["references"][name] = {
            "v1.0": _score(validation, reference, v10),
            "v1.1": _score(validation, reference, v11),
            "paired_presence_v11_minus_v10_on_both_answered": _paired_presence(validation, reference, v10, v11),
        }
    return result


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Parser v1.1 vs v1.0: Validation",
        "",
        f"{result['versions']['v1.1']} against the frozen {result['versions']['v1.0']}. "
        "Validation split only; Test reports were not parsed. 95% bootstrap intervals over studies "
        f"({result['method']['n_resamples']} resamples, seed {result['method']['seed']}).",
    ]
    rows = [
        ("Presence coverage", ("presence", "coverage", "coverage")),
        ("Presence selective F1", ("presence", "selective", "f1")),
        ("Presence selective precision", ("presence", "selective", "precision")),
        ("Presence end-to-end accuracy", ("presence", "end_to_end", "accuracy")),
        ("Presence end-to-end specificity", ("presence", "end_to_end", "specificity")),
        ("Presence abstain-as-negative F1", ("presence", "abstain_as_negative", "f1")),
    ]
    field_rows = [
        ("Count exact", "lesion_count_exact"),
        ("Count abs error", "lesion_count_abs_error"),
        ("Location exact", "location_exact"),
        ("Diameter abs error (cm)", "diameter_abs_error_cm"),
        ("Diameter within 0.5 cm", "diameter_within_0.5_cm"),
        ("Attenuation exact", "attenuation_exact"),
    ]
    for name, block in result["references"].items():
        title = "source-derived labels" if name == "source" else "human gold standard (Validation part)"
        v10, v11 = block["v1.0"], block["v1.1"]
        lines += [
            "",
            f"## Against {title}",
            "",
            "| Metric | v1.0 | v1.1 |",
            "|---|---|---|",
        ]
        for label, (a, b, c) in rows:
            lines.append(f"| {label} | {format_ci(v10[a][b][c])} | {format_ci(v11[a][b][c])} |")
        lines += [
            "",
            "Field extraction on reference-positive studies (coverage, then mean where predicted):",
            "",
            "| Field | v1.0 coverage | v1.0 mean | v1.1 coverage | v1.1 mean |",
            "|---|---|---|---|---|",
        ]
        for label, key in field_rows:
            f10 = v10["fields_on_reference_positives"][key]
            f11 = v11["fields_on_reference_positives"][key]
            lines.append(
                f"| {label} | {format_ci(f10['coverage'])} | {format_ci(f10['mean'])} | "
                f"{format_ci(f11['coverage'])} | {format_ci(f11['mean'])} |"
            )
        paired = block["paired_presence_v11_minus_v10_on_both_answered"]
        lines += [
            "",
            f"Presence on the {paired['studies']} studies both versions answered (v1.1 minus v1.0): "
            f"F1 difference {format_ci(paired['f1_difference'], 4)}; v1.1 right / v1.0 wrong "
            f"{paired['first_correct_second_wrong']}, v1.0 right / v1.1 wrong "
            f"{paired['second_correct_first_wrong']}; exact McNemar p = {paired['mcnemar_exact_p']:.3g}.",
        ]
    lines += [
        "",
        "## Notes",
        "- **These are development estimates.** v1.1 was built by inspecting Validation errors, "
        "so its Validation scores are optimistic. Unbiased estimates need the human gold standard "
        "or a single Test evaluation after v1.1 is frozen.",
        "- v1.0 set an unstated count to 1; v1.1 leaves \"multiple lesions\" unquantified, so its "
        "count coverage is lower by design and its count accuracy is measured only where a count is stated.",
        "- Remaining validation disagreements were reviewed: most are cases where the report text "
        "and the source label disagree (for example the text states a different attenuation for the "
        "largest lesion). They are not tuned away.",
        "- The frozen audit, split and gold-set strata still use parser v1.0.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(f"[DONE] {METRICS_PATH}\n[DONE] {REPORT_PATH}")


if __name__ == "__main__":
    main()
