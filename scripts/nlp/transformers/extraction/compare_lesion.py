"""Apply the registered decision rule of config_lesion.json: per-lesion model vs the
first extraction model (same seed) and parser v1.1. Test is not read."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from liversira_ai.nlp.baseline import extract_liver_lesion_features
from liversira_ai.nlp.metrics import paired_comparison
from scripts.nlp.transformers.extraction.compare import FIELD_MAP, mean_sd, parser_targets
from scripts.nlp.transformers.extraction.train import RESULTS_ROOT as FIRST_ROOT, TARGET_FIELDS, field_correct, read_jsonl
from scripts.nlp.transformers.extraction.train_lesion import CONFIG_PATH, RESULTS_ROOT

PARSER_DIR = Path("results/nlp/transformers/extraction")
MARGIN = 0.01
ALPHA = 0.05
SUBTYPE_FLOOR = 0.95
MUST_IMPROVE = ("count", "segments", "diameter_cm")


def by_id(path: Path, key: str = "challenge_id") -> dict[str, dict[str, Any]]:
    return {r[key]: r for r in read_jsonl(path)}


def correct(rows: dict[str, dict[str, Any]], ids: list[str], cfield: str) -> list[bool]:
    return [rows[i]["outcomes"][cfield] == "correct" for i in ids]


def targeted(rows: dict[str, dict[str, Any]], cfield: str) -> list[str]:
    return [i for i, r in rows.items() if cfield in r["targets"] and r["category"] != "control"]


def val_accuracy(predicted: dict[str, dict[str, Any]], items: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([field_correct(field, predicted[i["id"]][field], i["targets"][field]) for i in items]))


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    val = read_jsonl(Path(config["selection_data"]))
    positives = [i for i in val if i["targets"]["count"] != "0"]
    parser_val = {i["id"]: parser_targets(extract_liver_lesion_features(i["text"])) for i in positives}
    parser = {"v2": by_id(PARSER_DIR / "parser_challenge_v2_liver_predictions.jsonl"),
              "v3": by_id(PARSER_DIR / "parser_challenge_v3_predictions.jsonl")}
    distractor_parser = {i: r for i, r in by_id(PARSER_DIR / "parser_challenge_liver_only_predictions.jsonl").items()
                         if r["category"] == "distractor"}

    runs = []
    for model in config["models"]:
        for seed in config["seeds"]:
            lesion_dir, first_dir = RESULTS_ROOT / model / f"seed{seed}", FIRST_ROOT / model / f"seed{seed}"
            for d in (lesion_dir, first_dir):
                if not (d / "run_manifest.json").exists():
                    raise SystemExit(f"missing run: {d}")
            runs.append({
                "seed": seed,
                "manifest": json.loads((lesion_dir / "run_manifest.json").read_text(encoding="utf-8")),
                "lesion": {"v2": by_id(lesion_dir / "challenge_v2_liver_predictions.jsonl"),
                           "v3": by_id(lesion_dir / "challenge_v3_predictions.jsonl"),
                           "distractor": by_id(lesion_dir / "challenge_v1_distractor_predictions.jsonl"),
                           "val": {r["id"]: r["predicted"] for r in read_jsonl(lesion_dir / "val_predictions.jsonl")}},
                "first": {"v2": by_id(first_dir / "challenge_v2_liver_predictions.jsonl"),
                          "v3": by_id(first_dir / "challenge_v3_predictions.jsonl"),
                          "distractor": by_id(first_dir / "challenge_v1_distractor_predictions.jsonl"),
                          "val": {r["id"]: r["predicted"] for r in read_jsonl(first_dir / "val_predictions.jsonl")}},
            })

    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        cfield = FIELD_MAP[field]
        ids3, ids2 = targeted(parser["v3"], cfield), targeted(parser["v2"], cfield)
        p3, p2 = np.mean(correct(parser["v3"], ids3, cfield)), np.mean(correct(parser["v2"], ids2, cfield))
        pv = val_accuracy(parser_val, positives, field)
        seeds = []
        for run in runs:
            l3, f3 = correct(run["lesion"]["v3"], ids3, cfield), correct(run["first"]["v3"], ids3, cfield)
            test = paired_comparison([True] * len(ids3), l3, f3)
            l2 = np.mean(correct(run["lesion"]["v2"], ids2, cfield))
            f2 = np.mean(correct(run["first"]["v2"], ids2, cfield))
            lv, fv = val_accuracy(run["lesion"]["val"], positives, field), val_accuracy(run["first"]["val"], positives, field)
            primary = bool(np.mean(l3) >= p3 and np.mean(l3) >= np.mean(f3) - MARGIN)
            if field in MUST_IMPROVE:
                primary = bool(primary and np.mean(l3) > np.mean(f3) and test["mcnemar_exact_p"] < ALPHA)
            seeds.append({
                "seed": run["seed"],
                "v3_lesion": round(float(np.mean(l3)), 6), "v3_first": round(float(np.mean(f3)), 6),
                "v3_mcnemar_p": test["mcnemar_exact_p"],
                "v3_lesion_right_first_wrong": test["first_correct_second_wrong"],
                "v3_first_right_lesion_wrong": test["second_correct_first_wrong"],
                "v2_lesion": round(float(l2), 6), "v2_first": round(float(f2), 6),
                "val_lesion": round(lv, 6), "val_first": round(fv, 6),
                "primary_v3": primary,
                "no_regression_v2": bool(l2 >= f2 - MARGIN),
                "no_regression_val": bool(lv >= pv - MARGIN and lv >= fv - MARGIN),
            })
        fields[field] = {"v3_items": len(ids3), "v3_parser": round(float(p3), 6), "v2_parser": round(float(p2), 6),
                         "val_parser": round(pv, 6), "seeds": seeds}

    hidden = []
    for name in ("v2", "v3"):
        rows = parser[name]
        for key in sorted({(r["category"], r["subtype"]) for r in rows.values()}):
            ids = [i for i, r in rows.items() if (r["category"], r["subtype"]) == key]
            for cfield in rows[ids[0]]["targets"]:
                if np.mean(correct(rows, ids, cfield)) >= SUBTYPE_FLOOR:
                    mean = float(np.mean([np.mean(correct(run["lesion"][name], ids, cfield)) for run in runs]))
                    if mean < SUBTYPE_FLOOR:
                        hidden.append({"set": name, "subtype": "/".join(key), "field": cfield, "mean_accuracy": round(mean, 4)})

    criteria = {
        "primary_v3": all(s["primary_v3"] for f in fields.values() for s in f["seeds"]),
        "no_regression_v2": all(s["no_regression_v2"] for f in fields.values() for s in f["seeds"]),
        "no_regression_val": all(s["no_regression_val"] for f in fields.values() for s in f["seeds"]),
        "no_hidden_regression": not hidden,
    }
    verdict = "model-only candidate" if all(criteria.values()) else "hybrid kept for Phase 8"
    result = {"criteria_text": config["selection_criteria"], "criteria": criteria, "verdict": verdict,
              "fields": fields, "hidden_regressions": hidden,
              "runs": [{"seed": r["seed"], "best_epoch": r["manifest"]["best_epoch"],
                        "best_val_loss": r["manifest"]["best_val_loss"], "train_seconds": r["manifest"]["train_seconds"],
                        "weights_sha256": r["manifest"]["weights"]["sha256"]} for r in runs]}
    (RESULTS_ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def per_seed(f: dict[str, Any], key: str, digits: str = ".3f") -> str:
        return " / ".join(format(s[key], digits) for s in f["seeds"])

    def yes(ok: bool) -> str:
        return "pass" if ok else "**fail**"

    def subtype_rows(name: str, rows: dict[str, Any], key: str) -> list[str]:
        out = []
        for sub in sorted({(r["category"], r["subtype"]) for r in rows.values()}):
            ids = [i for i, r in rows.items() if (r["category"], r["subtype"]) == sub]
            tg = rows[ids[0]]["targets"]
            cells = []
            for field in TARGET_FIELDS:
                cfield = FIELD_MAP[field]
                p = np.mean(correct(rows, ids, cfield))
                first = [np.mean(correct(run["first"][key], ids, cfield)) for run in runs]
                lesion = [np.mean(correct(run["lesion"][key], ids, cfield)) for run in runs]
                cell = f"{p:.2f} / {np.mean(first):.2f} / {mean_sd(lesion)}"
                cells.append(f"**{cell}**" if cfield in tg and len(tg) < len(FIELD_MAP) else cell)
            out.append(f"| {sub[0]} | {sub[1]} | {len(ids)} | " + " | ".join(cells) + " |")
        return out

    head = "| Category | Subtype | n | " + " | ".join(TARGET_FIELDS) + " |"
    rule = "|---|---|---|" + "---|" * len(TARGET_FIELDS)
    lines = [
        "# Per-lesion extraction model: registered comparison",
        "",
        f"Per-lesion model (`config_lesion.json`, seeds {config['seeds']}) vs the first extraction model (same seed) "
        "and parser v1.1. All read the liver-only text; epochs chosen by Validation loss; Test was not read. "
        "The rule was registered before the training data, model and runs (one amendment, recorded in the config, "
        "also before any of them).",
        "",
        f"## Verdict: {verdict}",
        "",
        "| Criterion | Result |",
        "|---|---|",
        *[f"| {k} | {yes(v)} |" for k, v in criteria.items()],
        "",
        "## Per field (seeds 13 / 42 / 87)",
        "",
        "| Field | v3 parser | v3 first model | v3 per-lesion | v3 McNemar p (vs first) | v2 first | v2 per-lesion "
        "| Val parser | Val first | Val per-lesion | primary / v2 / val |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for field, f in fields.items():
        flags = " ".join(
            "/".join("✓" if s[k] else "✗" for k in ("primary_v3", "no_regression_v2", "no_regression_val"))
            for s in f["seeds"]
        )
        lines.append(
            f"| {field} | {f['v3_parser']:.3f} | {per_seed(f, 'v3_first')} | {per_seed(f, 'v3_lesion')} "
            f"| {per_seed(f, 'v3_mcnemar_p', '.2g')} | {per_seed(f, 'v2_first')} | {per_seed(f, 'v2_lesion')} "
            f"| {f['val_parser']:.3f} | {per_seed(f, 'val_first')} | {per_seed(f, 'val_lesion')} | {flags} |"
        )
    lines += [
        "",
        f"Validation: {len(positives)} lesion-positive studies, scored against the source-derived extraction targets.",
        "",
        f"## Hidden regressions (parser ≥ {SUBTYPE_FLOOR} on a targeted field, per-lesion mean < {SUBTYPE_FLOOR})",
        "",
        *([f"- `{h['set']}` {h['subtype']} {h['field']}: {h['mean_accuracy']:.3f}" for h in hidden] or ["None."]),
        "",
        "## extraction_challenge_v3 by subtype",
        "",
        "Cells: parser / first model (mean) / per-lesion (mean ± SD). Bold = targeted field.",
        "",
        head, rule, *subtype_rows("v3", parser["v3"], "v3"),
        "",
        "## extraction_challenge_v2_liver by subtype",
        "",
        head, rule, *subtype_rows("v2", parser["v2"], "v2"),
        "",
        "## Distractor check (v1, liver-only filter applied; not part of the rule)",
        "",
        head, rule, *subtype_rows("distractor", distractor_parser, "distractor"),
        "",
        "## Runs",
        "",
        "| Seed | Best epoch | Best Val loss | Train seconds |",
        "|---|---|---|---|",
        *[f"| {r['seed']} | {r['best_epoch']} | {r['best_val_loss']:.4f} | {r['train_seconds']:.0f} |" for r in result["runs"]],
        "",
    ]
    (RESULTS_ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_ROOT / 'comparison.md'}: {verdict}")


if __name__ == "__main__":
    main()
