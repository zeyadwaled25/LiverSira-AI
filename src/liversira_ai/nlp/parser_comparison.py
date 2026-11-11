"""Scoring and report layout shared by the parser v1.1 vs v1.0 evaluations.

Used by the Validation comparison and by the single-use Test evaluation, so
both report the same metrics in the same layout.
"""

from pathlib import Path
from typing import Any

from liversira_ai.nlp.evaluation import baseline_fields
from liversira_ai.nlp.metrics import format_ci, paired_comparison, presence_report

V10_PATH = Path("results/nlp/predictions/baseline_inferences_updated.jsonl")
OUTPUT_DIR = Path("results/nlp/parser_v1_1")
RESAMPLES, SEED = 2000, 42


def score(
    ids: list[str], reference: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, Any]:
    options = {"n_resamples": RESAMPLES, "seed": SEED}
    ids = [i for i in ids if i in reference]
    truth = [bool(reference[i]["has_lesion"]) for i in ids]
    presence = [predictions[i]["liver_lesions"].get("has_lesion") for i in ids]
    return {
        "studies": len(ids),
        "presence": presence_report(truth, presence, **options),
        "fields_on_reference_positives": baseline_fields(ids, reference, predictions, options),
    }


def paired_presence(
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
