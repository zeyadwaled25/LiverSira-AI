"""Compare the 6.2 extraction model runs with parser v1.1 using the criteria in config.json.

Per field, the model "adds value" only if, for every seed: (1) on
extraction_challenge_v2_liver its pooled accuracy over the subtypes that target
the field beats the parser with exact McNemar p < 0.05, and (2) on lesion-positive
Validation studies its accuracy against the extraction targets is at most 0.01
below the parser's. Test is not read.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from liversira_ai.nlp.baseline import extract_liver_lesion_features
from liversira_ai.nlp.extraction_targets import count_class
from liversira_ai.nlp.metrics import paired_comparison
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.train import (
    CONFIG_PATH,
    RESULTS_ROOT,
    TARGET_FIELDS,
    field_correct,
    read_jsonl,
)

PARSER_DIR = Path("results/nlp/transformers/extraction")
FIELD_MAP = {  # target field -> Challenge Set field
    "count": "lesion_count",
    "segments": "location_segments",
    "attenuation": "attenuation",
    "diameter_cm": "derived_max_diameter_cm",
}
MCNEMAR_ALPHA = 0.05
REGRESSION_MARGIN = 0.01


def parser_targets(features: dict[str, Any]) -> dict[str, Any]:
    """Parser output in extraction-target form."""
    if features["has_lesion"] is False:
        return {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None}
    if features["has_lesion"] is None:
        return {f: None for f in TARGET_FIELDS}
    segments = features["location_segments"]
    return {
        "count": count_class(features["lesion_count"]),
        "segments": sorted(segments) if segments else None,
        "attenuation": features["attenuation"],
        "diameter_cm": features["derived_max_diameter_cm"],
    }


def correct_vector(rows: dict[str, dict[str, Any]], ids: list[str], field: str) -> list[bool]:
    return [rows[i]["outcomes"][field] == "correct" for i in ids]


def targeted_ids(rows: list[dict[str, Any]], field: str) -> list[str]:
    return [r["challenge_id"] for r in rows if field in r["targets"] and r["category"] != "control"]


def mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1) if len(values) > 1 else 0.0:.3f}"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    val = read_jsonl(Path(config["selection_data"]))
    positives = [i for i in val if i["targets"]["count"] != "0"]
    parser_val = {i["id"]: parser_targets(extract_liver_lesion_features(i["text"])) for i in positives}
    parser_v2 = {r["challenge_id"]: r for r in read_jsonl(PARSER_DIR / "parser_challenge_v2_liver_predictions.jsonl")}
    parser_distractor = {
        r["challenge_id"]: r
        for r in read_jsonl(PARSER_DIR / "parser_challenge_liver_only_predictions.jsonl")
        if r["category"] == "distractor"
    }

    runs = []
    for model in config["models"]:
        for seed in config["seeds"]:
            run_dir = RESULTS_ROOT / model / f"seed{seed}"
            if not (run_dir / "run_manifest.json").exists():
                raise SystemExit(f"missing run: {run_dir}")
            runs.append({
                "model": model,
                "seed": seed,
                "manifest": json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")),
                "val": {r["id"]: r["predicted"] for r in read_jsonl(run_dir / "val_predictions.jsonl")},
                "v2": {r["challenge_id"]: r for r in read_jsonl(run_dir / "challenge_v2_liver_predictions.jsonl")},
                "distractor": {
                    r["challenge_id"]: r for r in read_jsonl(run_dir / "challenge_v1_distractor_predictions.jsonl")
                },
            })

    v2_rows = list(parser_v2.values())
    result: dict[str, Any] = {"criteria": config["selection_criteria"], "fields": {}, "runs": []}
    for field in TARGET_FIELDS:
        cfield = FIELD_MAP[field]
        ids = targeted_ids(v2_rows, cfield)
        parser_challenge = correct_vector(parser_v2, ids, cfield)
        parser_val_acc = float(np.mean([field_correct(field, parser_val[i["id"]][field], i["targets"][field])
                                        for i in positives]))
        seeds = []
        for run in runs:
            model_challenge = correct_vector(run["v2"], ids, cfield)
            test = paired_comparison([True] * len(ids), model_challenge, parser_challenge)
            model_val_acc = float(np.mean([field_correct(field, run["val"][i["id"]][field], i["targets"][field])
                                           for i in positives]))
            better = float(np.mean(model_challenge)) > float(np.mean(parser_challenge))
            seeds.append({
                "seed": run["seed"],
                "challenge_accuracy": round(float(np.mean(model_challenge)), 6),
                "model_right_parser_wrong": test["first_correct_second_wrong"],
                "parser_right_model_wrong": test["second_correct_first_wrong"],
                "mcnemar_p": test["mcnemar_exact_p"],
                "val_positive_accuracy": round(model_val_acc, 6),
                "challenge_pass": better and test["mcnemar_exact_p"] < MCNEMAR_ALPHA,
                "val_pass": model_val_acc >= parser_val_acc - REGRESSION_MARGIN,
            })
        result["fields"][field] = {
            "challenge_items": len(ids),
            "parser_challenge_accuracy": round(float(np.mean(parser_challenge)), 6),
            "parser_val_positive_accuracy": round(parser_val_acc, 6),
            "seeds": seeds,
            "adds_value": all(s["challenge_pass"] and s["val_pass"] for s in seeds),
        }

    subtypes = sorted({(r["category"], r["subtype"]) for r in v2_rows})
    distractor_subtypes = sorted({(r["category"], r["subtype"]) for r in parser_distractor.values()})

    def subtype_table(parser_rows: dict[str, Any], key: str, keys: list[tuple[str, str]]) -> list[str]:
        lines = []
        for category, subtype in keys:
            ids = [i for i, r in parser_rows.items() if (r["category"], r["subtype"]) == (category, subtype)]
            targets = parser_rows[ids[0]]["targets"]
            cells = []
            for field in TARGET_FIELDS:
                cfield = FIELD_MAP[field]
                p = np.mean(correct_vector(parser_rows, ids, cfield))
                m = [np.mean(correct_vector(run[key], ids, cfield)) for run in runs]
                cell = f"{p:.3f} → {mean_sd(m)}"
                cells.append(f"**{cell}**" if cfield in targets and len(targets) < len(FIELD_MAP) else cell)
            lines.append(f"| {category} | {subtype} | {len(ids)} | " + " | ".join(cells) + " |")
        return lines

    for run in runs:
        m = run["manifest"]
        result["runs"].append({"model": run["model"], "seed": run["seed"], "best_epoch": m["best_epoch"],
                               "best_val_loss": m["best_val_loss"], "train_seconds": m["train_seconds"],
                               "weights_sha256": m["weights"]["sha256"]})

    out = RESULTS_ROOT / "comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    header = "| Category | Subtype | n | " + " | ".join(TARGET_FIELDS) + " |"
    rule = "|---|---|---|" + "---|" * len(TARGET_FIELDS)
    lines = [
        "# Extraction model vs parser v1.1",
        "",
        f"Model: {', '.join(config['models'])}, seeds {config['seeds']}, trained on `extraction_train_v1` "
        "(Train only), epoch chosen by Validation loss. Parser: frozen v1.1. Both read the liver-only text. "
        "Test was not read. The decision rule was fixed in `config.json` before the first run.",
        "",
        "## Decision per field",
        "",
        f"Adds value = for every seed, (1) higher pooled accuracy than the parser on the `extraction_challenge_v2_liver` "
        f"subtypes that target the field, with exact McNemar p < {MCNEMAR_ALPHA}, and (2) lesion-positive Validation "
        f"accuracy at most {REGRESSION_MARGIN} below the parser's.",
        "",
        "| Field | Challenge: parser | Challenge: model (per seed) | McNemar p (per seed) | Val positive: parser "
        "| Val positive: model (per seed) | Adds value |",
        "|---|---|---|---|---|---|---|",
    ]
    for field, f in result["fields"].items():
        seeds = f["seeds"]
        lines.append(
            f"| {field} | {f['parser_challenge_accuracy']:.3f} "
            f"| {' / '.join(f'{s['challenge_accuracy']:.3f}' for s in seeds)} "
            f"| {' / '.join(f'{s['mcnemar_p']:.2g}' for s in seeds)} "
            f"| {f['parser_val_positive_accuracy']:.3f} "
            f"| {' / '.join(f'{s['val_positive_accuracy']:.3f}' for s in seeds)} "
            f"| {'**yes**' if f['adds_value'] else 'no'} |"
        )
    lines += [
        "",
        f"Validation: {len(positives)} lesion-positive studies; a field is correct when it equals the target, "
        "and `null` (unstated) must match `null`.",
        "",
        "## Challenge v2 by subtype",
        "",
        "Cells: parser → model (mean ± SD over seeds). Bold = a field the subtype targets.",
        "",
        header,
        rule,
        *subtype_table(parser_v2, "v2", subtypes),
        "",
        "## Distractor check (`extraction_challenge_v1`, liver-only filter applied; not part of the decision)",
        "",
        header,
        rule,
        *subtype_table(parser_distractor, "distractor", distractor_subtypes),
        "",
        "## Runs",
        "",
        "| Model | Seed | Best epoch | Best Val loss | Train seconds |",
        "|---|---|---|---|---|",
        *[f"| {r['model']} | {r['seed']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | {r['train_seconds']:.0f} |"
          for r in result["runs"]],
        "",
        "## Notes",
        "",
        "- The Challenge Sets are synthetic in their lesion text; they show which phrasings each system handles, "
        "not error rates on real reports.",
        "- Training items cover the same phenomena as the Challenge Sets with a separate lexicon "
        "(`extraction_train_v1.md`), so these scores test generalization across wording, not memorized templates.",
        "- Counts of 10 or more are one class (`10+`) for the model.",
        "",
    ]
    (RESULTS_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_ROOT / 'comparison.md'} and {out}")


if __name__ == "__main__":
    main()
