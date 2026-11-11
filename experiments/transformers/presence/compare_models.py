"""Compare presence systems on Validation and the Challenge Set (Phase 6.1).

Systems:

* ``lr``      the frozen word TF-IDF + logistic regression configuration, re-fitted
              on Train only (the frozen final model saw Validation, which the
              Challenge Set is built from). The refit must reproduce the saved
              Validation predictions exactly, otherwise the script stops.
* ``parser``  rule-based parser v1.1, abstentions counted as negative.
* ``bert`` / ``biobert`` / ``clinicalbert``  the runs written by ``train.py``;
              every seed in ``config.json`` must be present.

Transformer results are reported as mean +/- SD over seeds, and every seed is
compared with ``lr`` on the same studies (exact McNemar). Test is not read.

Usage (from the repository root):

    .venv\\Scripts\\python.exe experiments/transformers/presence/compare_models.py
"""

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).resolve().parents[1])]

import data  # noqa: E402
from evaluation.learned_presence.final_evaluate_learned_presence import _frozen_model  # noqa: E402
from liversira_ai.nlp.metrics import format_ci, paired_comparison, presence_report  # noqa: E402
from liversira_ai.nlp.parser import parse_liver_report  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = Path("results/nlp/transformers/presence")
LR_VAL_PATH = Path("results/nlp/predictions/word_logistic_regression_validation_predictions.jsonl")
PARSER_VAL_PATH = Path("results/nlp/predictions/baseline_v1_1_validation_inferences.jsonl")
METRICS_PATH = RESULTS_ROOT / "comparison.json"
REPORT_PATH = RESULTS_ROOT / "comparison.md"
RESAMPLES, SEED = 2000, 42
CONFIDENCE_LEVELS = (0.5, 0.9, 0.99)
POINT_METRICS = ("accuracy", "precision", "recall", "specificity", "f1")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _point(truth: list[bool], pred: list[bool]) -> dict[str, float | None]:
    t, p = np.array(truth), np.array(pred)
    tp, fp = int((p & t).sum()), int((p & ~t).sum())
    tn, fn = int((~p & ~t).sum()), int((~p & t).sum())

    def r(a: int, b: int) -> float | None:
        return a / b if b else None

    return {"accuracy": r(tp + tn, len(t)), "precision": r(tp, tp + fp), "recall": r(tp, tp + fn),
            "specificity": r(tn, tn + fp), "f1": r(2 * tp, 2 * tp + fp + fn), "fp": fp, "fn": fn}


def _calibration(truth: list[bool], prob: list[float]) -> dict[str, float]:
    y, p = np.array(truth, dtype=float), np.array(prob)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"brier": float(np.mean((p - y) ** 2)), "ece_10_bins": float(ece)}


def _selective(truth: list[bool], prob: list[float]) -> dict[str, dict[str, float]]:
    """Answer only when max(p, 1-p) >= level; report coverage and errors among answered."""
    t, p = np.array(truth), np.array(prob)
    confidence = np.maximum(p, 1 - p)
    out = {}
    for level in CONFIDENCE_LEVELS:
        answered = confidence >= level
        errors = int(((p[answered] >= 0.5) != t[answered]).sum())
        out[str(level)] = {"coverage": float(answered.mean()), "errors": errors}
    return out


def _fit_lr(train: list[dict[str, Any]]):
    model = _frozen_model()
    model.fit([e["text"] for e in train], [int(e["label"]) for e in train])
    return model


def _verify_lr(model, val: list[dict[str, Any]]) -> dict[str, Any]:
    saved = {r["id"]: r for r in _jsonl(LR_VAL_PATH)}
    probs = model.predict_proba([e["text"] for e in val])[:, 1]
    diff = max(abs(p - saved[e["id"]]["score_positive"]) for e, p in zip(val, probs))
    mismatches = sum((p >= 0.5) != saved[e["id"]]["prediction"] for e, p in zip(val, probs))
    if mismatches or diff > 1e-9:
        raise RuntimeError(f"Train-only LR refit does not reproduce saved Validation predictions "
                           f"({mismatches} mismatches, max |dp| = {diff:.2e})")
    return {"reference": LR_VAL_PATH.as_posix(), "records": len(val), "prediction_mismatches": 0,
            "max_abs_probability_difference": float(diff), "passed": True}


def _mean_sd(values: list[float | None]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "sd": None}
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def run() -> dict[str, Any]:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    ids = data.load_split_ids(["train", "val"])
    train, val = data.load_examples(ids["train"]), data.load_examples(ids["val"])
    challenge = data.load_challenge()
    scored = [r for r in challenge if r["scored"]]
    val_truth = [e["label"] for e in val]
    audit = {r["id"]: r.get("audit_group") for r in _jsonl(LR_VAL_PATH)}

    # Probabilities / decisions per system: {"val": {id: p}, "challenge": {cid: p}}
    lr = _fit_lr(train)
    lr_check = _verify_lr(lr, val)
    probs: dict[str, dict[str, dict[str, float]]] = {
        "lr": {
            "val": dict(zip((e["id"] for e in val), lr.predict_proba([e["text"] for e in val])[:, 1])),
            "challenge": dict(zip((r["challenge_id"] for r in challenge),
                                  lr.predict_proba([r["text"] for r in challenge])[:, 1])),
        }
    }
    saved_parser = {r["id"]: r["liver_lesions"].get("has_lesion") for r in _jsonl(PARSER_VAL_PATH)}
    parser_val = {e["id"]: parse_liver_report(e["text"]).presence for e in val}
    parser_check = sum(parser_val[i] != saved_parser[i] for i in parser_val)
    parser_challenge = {r["challenge_id"]: parse_liver_report(r["text"]).presence for r in challenge}

    runs: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for model in config["models"]:
        for seed in config["seeds"]:
            run_dir = RESULTS_ROOT / model / f"seed{seed}"
            if not (run_dir / "run_manifest.json").exists():
                raise FileNotFoundError(f"missing run {run_dir}; run run_all.py first")
            runs.setdefault(model, {})[seed] = {
                "val": {r["id"]: r["probability"] for r in _jsonl(run_dir / "val_predictions.jsonl")},
                "challenge": {r["challenge_id"]: r["probability"] for r in _jsonl(run_dir / "challenge_predictions.jsonl")},
            }

    options = {"n_resamples": RESAMPLES, "seed": SEED}
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "reference": "source-derived labels (not a clinical gold standard)",
        "test_policy": "Test was not read.",
        "method": {"interval": "percentile bootstrap over studies", "n_resamples": RESAMPLES, "seed": SEED,
                   "threshold": config["threshold"], "seeds": config["seeds"]},
        "lr_refit_check": lr_check,
        "parser_reproduces_saved_v1_1_validation": {"mismatches": parser_check},
        "validation": {}, "audit_groups": {}, "challenge": {}, "selective": {},
    }

    # ---- Validation
    def val_block(pred: list[bool | None], prob: list[float] | None) -> dict[str, Any]:
        block = {"report": presence_report(val_truth, pred, **options),
                 "point": _point(val_truth, [bool(p) for p in pred])}
        if prob is not None:
            block["calibration"] = _calibration(val_truth, prob)
        return block

    lr_val = [probs["lr"]["val"][e["id"]] for e in val]
    lr_pred = [p >= 0.5 for p in lr_val]
    result["validation"]["lr"] = val_block(lr_pred, lr_val)
    result["validation"]["parser"] = val_block([parser_val[e["id"]] for e in val], None)
    for model, seeds in runs.items():
        per_seed = {}
        for seed, p in seeds.items():
            prob = [p["val"][e["id"]] for e in val]
            pred = [x >= config["threshold"] for x in prob]
            per_seed[seed] = val_block(pred, prob)
            per_seed[seed]["vs_lr"] = paired_comparison(val_truth, pred, lr_pred, **options)
            result["selective"].setdefault(model, {})[seed] = _selective(val_truth, prob)
        summary = {m: _mean_sd([s["point"][m] for s in per_seed.values()]) for m in POINT_METRICS + ("fp", "fn")}
        summary.update({m: _mean_sd([s["calibration"][m] for s in per_seed.values()]) for m in ("brier", "ece_10_bins")})
        result["validation"][model] = {"per_seed": per_seed, "summary": summary}

    # ---- Validation by audit group (accuracy)
    def group_accuracy(pred_by_id: dict[str, bool]) -> dict[str, dict[str, float]]:
        out = {}
        for group in sorted({g for g in audit.values() if g}):
            members = [e for e in val if audit.get(e["id"]) == group]
            out[group] = {"n": len(members),
                          "accuracy": float(np.mean([pred_by_id[e["id"]] == e["label"] for e in members]))}
        return out

    result["audit_groups"]["lr"] = group_accuracy({i: p >= 0.5 for i, p in probs["lr"]["val"].items()})
    result["audit_groups"]["parser"] = group_accuracy({i: bool(p) for i, p in parser_val.items()})
    for model, seeds in runs.items():
        per_seed = [group_accuracy({i: p >= config["threshold"] for i, p in s["val"].items()}) for s in seeds.values()]
        result["audit_groups"][model] = {
            g: {"n": per_seed[0][g]["n"], **_mean_sd([s[g]["accuracy"] for s in per_seed])} for g in per_seed[0]
        }

    # ---- Challenge Set
    categories = sorted({r["category"] for r in challenge})

    def challenge_block(decision: dict[str, bool]) -> dict[str, Any]:
        out = {}
        for c in categories:
            items = [r for r in challenge if r["category"] == c]
            positive_rate = float(np.mean([decision[r["challenge_id"]] for r in items]))
            accuracy = (float(np.mean([decision[r["challenge_id"]] == r["label"] for r in items]))
                        if items[0]["scored"] else None)
            out[c] = {"n": len(items), "accuracy": accuracy, "positive_rate": positive_rate}
        out["all_scored"] = {"n": len(scored), "accuracy": float(np.mean(
            [decision[r["challenge_id"]] == r["label"] for r in scored]))}
        return out

    lr_challenge = {k: v >= 0.5 for k, v in probs["lr"]["challenge"].items()}
    result["challenge"]["lr"] = challenge_block(lr_challenge)
    result["challenge"]["parser"] = challenge_block({k: bool(v) for k, v in parser_challenge.items()})
    result["challenge"]["parser"]["abstained"] = sum(v is None for v in parser_challenge.values())
    result["challenge"]["parser"]["abstained_by_category"] = {
        c: sum(parser_challenge[r["challenge_id"]] is None for r in challenge if r["category"] == c)
        for c in categories
    }
    scored_truth = [bool(r["label"]) for r in scored]
    for model, seeds in runs.items():
        per_seed = {}
        for seed, s in seeds.items():
            decision = {k: v >= config["threshold"] for k, v in s["challenge"].items()}
            per_seed[seed] = challenge_block(decision)
            per_seed[seed]["vs_lr"] = paired_comparison(
                scored_truth, [decision[r["challenge_id"]] for r in scored],
                [lr_challenge[r["challenge_id"]] for r in scored], **options)
        summary = {c: _mean_sd([per_seed[s][c]["accuracy"] if per_seed[s][c]["accuracy"] is not None
                                else per_seed[s][c]["positive_rate"] for s in per_seed])
                   for c in categories + ["all_scored"]}
        result["challenge"][model] = {"per_seed": per_seed, "summary": summary}
    result["git_commit"] = data.git_commit()
    result["inputs_sha256"] = {p.as_posix(): data.sha256_file(p) for p in
                               (data.MASTER_PATH, data.GROUND_TRUTH_PATH, data.SPLITS_PATH, data.CHALLENGE_PATH,
                                LR_VAL_PATH, PARSER_VAL_PATH)}
    return result


def _ms(item: dict[str, float | None], digits: int = 3) -> str:
    if item["mean"] is None:
        return "—"
    return f"{item['mean']:.{digits}f} ± {item['sd']:.{digits}f}"


def render(result: dict[str, Any]) -> str:
    v, models = result["validation"], [k for k in result["validation"] if k not in ("lr", "parser")]
    seeds = result["method"]["seeds"]
    lines = [
        "# Presence: transformers vs TF-IDF + LR vs parser v1.1",
        "",
        f"Validation ({v['lr']['report']['studies']} studies) and the Challenge Set, against source-derived "
        f"labels. Test was not read. Transformers: mean ± SD over seeds {seeds}. LR is the frozen "
        "configuration re-fitted on Train only (it reproduces the saved Validation predictions exactly). "
        "Parser abstentions count as negative.",
        "",
        "## Validation",
        "",
        "| System | Accuracy | Precision | Recall | Specificity | F1 | FP | FN | Brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("lr", "parser"):
        r, pt = v[name]["report"]["abstain_as_negative"], v[name]["point"]
        brier = f"{v[name]['calibration']['brier']:.4f}" if "calibration" in v[name] else "—"
        lines.append(f"| {name} | {format_ci(r['accuracy'])} | {format_ci(r['precision'])} | {format_ci(r['recall'])} | "
                     f"{format_ci(r['specificity'])} | {format_ci(r['f1'])} | {pt['fp']} | {pt['fn']} | {brier} |")
    for name in models:
        s = v[name]["summary"]
        lines.append(f"| {name} | {_ms(s['accuracy'])} | {_ms(s['precision'])} | {_ms(s['recall'])} | "
                     f"{_ms(s['specificity'])} | {_ms(s['f1'])} | {_ms(s['fp'], 1)} | {_ms(s['fn'], 1)} | "
                     f"{_ms(s['brier'], 4)} |")
    lines += ["", "Each transformer seed vs LR on Validation (exact McNemar):", "",
              "| System | " + " | ".join(f"seed {s}: model right / LR right, p" for s in seeds) + " |",
              "|---|" + "---|" * len(seeds)]
    for name in models:
        cells = []
        for seed in seeds:
            c = v[name]["per_seed"][seed]["vs_lr"]
            cells.append(f"{c['first_correct_second_wrong']} / {c['second_correct_first_wrong']}, p = {c['mcnemar_exact_p']:.3g}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    groups = result["audit_groups"]
    group_names = list(groups["lr"])
    lines += ["", "## Validation accuracy by audit group", "",
              "| System | " + " | ".join(f"{g} (n={groups['lr'][g]['n']})" for g in group_names) + " |",
              "|---|" + "---|" * len(group_names)]
    for name in ("lr", "parser"):
        lines.append(f"| {name} | " + " | ".join(f"{groups[name][g]['accuracy']:.3f}" for g in group_names) + " |")
    for name in models:
        lines.append(f"| {name} | " + " | ".join(_ms(groups[name][g]) for g in group_names) + " |")

    ch = result["challenge"]
    categories = [c for c in ch["lr"] if c != "all_scored"]
    lines += ["", "## Challenge Set", "",
              "Accuracy per category; for `hedged` (no correct answer) the share called positive.", "",
              "| System | " + " | ".join(f"{c} (n={ch['lr'][c]['n']})" for c in categories) +
              f" | All scored (n={ch['lr']['all_scored']['n']}) |",
              "|---|" + "---|" * (len(categories) + 1)]

    def cell(block: dict[str, Any], c: str) -> str:
        value = block[c]["accuracy"] if block[c]["accuracy"] is not None else block[c]["positive_rate"]
        return f"{value:.3f}"

    for name in ("lr", "parser"):
        lines.append(f"| {name} | " + " | ".join(cell(ch[name], c) for c in categories) +
                     f" | {ch[name]['all_scored']['accuracy']:.3f} |")
    for name in models:
        s = ch[name]["summary"]
        lines.append(f"| {name} | " + " | ".join(_ms(s[c]) for c in categories) + f" | {_ms(s['all_scored'])} |")
    by_category = ", ".join(f"{c} {n}" for c, n in ch["parser"]["abstained_by_category"].items())
    lines += ["", f"Parser abstained on {ch['parser']['abstained']} Challenge items, counted as negative "
              f"({by_category}). Its accuracy on the negative-label categories (negation, liver_removed, "
              "other_organ_lesion) therefore includes abstentions, not only explicit negatives.", "",
              "Each transformer seed vs LR on the scored Challenge items (exact McNemar):", "",
              "| System | " + " | ".join(f"seed {s}: model right / LR right, p" for s in seeds) + " |",
              "|---|" + "---|" * len(seeds)]
    for name in models:
        cells = []
        for seed in seeds:
            c = ch[name]["per_seed"][seed]["vs_lr"]
            cells.append(f"{c['first_correct_second_wrong']} / {c['second_correct_first_wrong']}, p = {c['mcnemar_exact_p']:.3g}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Selective prediction on Validation", "",
              "Answer only when max(p, 1 - p) reaches the level; coverage and errors among answered, mean over seeds.", "",
              "| System | " + " | ".join(f"≥ {lvl}: coverage / errors" for lvl in CONFIDENCE_LEVELS) + " |",
              "|---|" + "---|" * len(CONFIDENCE_LEVELS)]
    for name in models:
        sel = result["selective"][name]
        cells = []
        for lvl in CONFIDENCE_LEVELS:
            cov = statistics.fmean(sel[s][str(lvl)]["coverage"] for s in sel)
            err = statistics.fmean(sel[s][str(lvl)]["errors"] for s in sel)
            cells.append(f"{cov:.3f} / {err:.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Notes",
              "- Validation was used to pick each run's epoch, so Validation numbers are slightly optimistic "
              "for the transformers; the Challenge Set was not used for any choice.",
              "- Labels are source-derived. Clinical accuracy needs the human gold standard.",
              "- No system is selected here; selection and the single Test run belong to Phase 8.", ""]
    return "\n".join(lines)


def main() -> None:
    result = run()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
