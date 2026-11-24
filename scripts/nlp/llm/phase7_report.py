"""Phase 7 Step 7: the LLM evaluation report (no API calls).

Reads the frozen-prompt evaluation predictions of the two models kept by Phase 7
Amendment 3 (Nemotron 3.5 Lightning on NVIDIA, Gemini 3.5 Flash-Lite) and puts them
beside the frozen Phase 8 candidates on the same items:

- Validation fields (all 220 lesion-positive Validation reports, verifiable fields only): C1 parser v1.1,
  C2 parser v1.2, C3 first model (6.2), H = C2 + C3, from results/nlp/phase8/validation_rescored.json.
- Validation presence (220 positives + 180 sampled negatives): abstention and errors; P1 frozen LR beside
  it only in the scoring table, because the LLM is not a presence candidate.
- Extraction Challenge v3 (260 subsample items) and Presence Challenge v1 (210): Challenge-A diagnosis only.
  Every system in a table is scored on the same items. Gemini ran 174 of the 260 v3 items, so it gets
  its own table on those items.
- repeat agreement, evidence rejections (was the rejected value right or wrong?), LLM measures and
  transport errors.

Nothing here chooses a Phase 8 system. Test is not read.

Writes results/nlp/llm/phase7_report.json and docs/nlp/phase7_report.md.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from liversira_ai.nlp import llm_extraction as le
from liversira_ai.nlp.baseline import extract_liver_lesion_features, extract_liver_lesion_features_v1_2
from liversira_ai.nlp.extraction_targets import count_class
from liversira_ai.nlp.metrics import wilson_interval
from scripts.nlp.llm import eval_sets
from scripts.nlp.llm import run_llm_extraction as runner
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.challenge_a_diagnostics import hybrid

ROOT = Path("results/nlp/llm")
OUT_JSON = ROOT / "phase7_report.json"
OUT_MD = Path("docs/nlp/phase7_report.md")
MODELS = {  # name -> (provider, model folder, label)
    "nemotron": ("nvidia", "nvidia-nemotron-3.5-lightning-30b-a3b", "Nemotron 3.5 Lightning (NVIDIA, open weights)"),
    "flash_lite": ("gemini", "gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite (Google, closed)"),
}
NOT_RUN = {("flash_lite", "presence_challenge_v1"): "not run: daily quota (Amendment 3)",
           ("flash_lite", "extraction_challenge_v3"): "rest not run: daily quota ended (Amendment 3)",
           ("flash_lite", "repeat"): "not run: daily quota ended (Amendment 3)"}
SETS = ("validation", "extraction_challenge_v3", "presence_challenge_v1", "repeat")
FIELDS = ("count", "segments", "attenuation", "diameter_cm")
LLM_TO_TARGET = {"lesion_count": "count", "location_segments": "segments", "attenuation": "attenuation",
                 "reported_dimensions_cm": "diameter_cm"}
LLM_TO_V3 = {"lesion_count": "lesion_count", "location_segments": "location_segments", "attenuation": "attenuation",
             "reported_dimensions_cm": "derived_max_diameter_cm"}
BASELINES = {"C1 parser v1.1": "C1 parser v1.1", "C2 parser v1.2": "C2 parser v1.2",
             "C3 first model (6.2)": "C3 first model (6.2, unmasked)", "H = C2 + C3": "H = C2 + C3 (6.2)"}
PARSER_V3 = Path("results/nlp/transformers/extraction/parser_challenge_v3_predictions.jsonl")
C3_RUNS = [Path(f"results/nlp/transformers/extraction/runs/clinicalbert/seed{s}/challenge_v3_predictions.jsonl")
           for s in (13, 42, 87)]
LR_VALIDATION = Path("results/nlp/predictions/learned_presence_validation_predictions.jsonl")
UNLOGGED = ROOT / "transport_errors_unlogged.json"
PACING_CHANGE = ("24 November 2026, 12:53 UTC, during the Nemotron runs: at 40 requests per minute NVIDIA rejected most "
                 "requests with HTTP 429 (281 failed attempts, 51 skipped items), so `config.json` was set to 10 requests per "
                 "minute and 8 SDK retries (from the default 4); after it, 161 attempts still got 429 and 4 items were skipped, "
                 "all finished on a rerun. "
                 "This paces calls only, as the freeze allows (`prompt_freeze.json` note); outputs and settings are unchanged.")


def rows_of(model: str, set_name: str) -> list[dict[str, Any]]:
    provider, folder, _ = MODELS[model]
    path = ROOT / provider / folder / f"{set_name}_v2" / "predictions.jsonl"
    return runner.read_jsonl(path) if path.exists() else []


def fmt(w: dict[str, Any] | None) -> str:
    if not w or not w.get("n"):
        return "—"
    if "sd" in w:
        return f"{w['estimate']:.3f} ± {w['sd']:.3f} [{w['ci_low']:.3f}, {w['ci_high']:.3f}] (n={w['n']})"
    return f"{w['estimate']:.3f} [{w['ci_low']:.3f}, {w['ci_high']:.3f}] (n={w['n']})"


def mean_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± SD over seeds, with the envelope of the seeds' Wilson intervals."""
    values = [r["estimate"] for r in runs]
    return {"estimate": statistics.mean(values), "sd": statistics.pstdev(values), "n": runs[0]["n"],
            "ci_low": min(r["ci_low"] for r in runs), "ci_high": max(r["ci_high"] for r in runs)}


def short(w: dict[str, Any], row_n: int | None = None) -> str:
    if not w.get("n"):
        return "—"
    text = f"{w['estimate']:.2f}" + (f" ± {w['sd']:.2f}" if "sd" in w else "")
    return text + (f" (n={w['n']})" if row_n is not None and w["n"] != row_n else "")


# ── Subsample ───────────────────────────────────────────────────────────────
def subsample() -> dict[str, Any]:
    sub = json.loads(eval_sets.SUBSAMPLE.read_text(encoding="utf-8"))
    validation = eval_sets._read(eval_sets.VALIDATION)
    positives = sum(r["targets"]["count"] != "0" for r in validation)
    return {"seed": sub.get("seed"), "validation_reports": len(validation),
            "validation_positive_all": positives, "validation_negative_all": len(validation) - positives,
            "validation_positive_used": len(sub["validation"]["positive"]),
            "validation_negative_used": len(sub["validation"]["negative"]),
            "presence_challenge_v1": {k: len(v) for k, v in sub["presence_challenge_v1"].items()},
            "extraction_challenge_v3": {k: len(v) for k, v in sub["extraction_challenge_v3"].items()}}


# ── Validation ──────────────────────────────────────────────────────────────
def validation() -> dict[str, Any]:
    items = {i["id"]: i for i in eval_sets.load_items("validation")}
    out: dict[str, Any] = {"fields": {}, "presence": {}, "presence_scoring": {}, "interval_overlap_with_C2": {}}
    for model in MODELS:
        rows = [r for r in rows_of(model, "validation") if r["id"] in items]
        if rows:
            out["fields"][model] = runner.summarize(rows, items)["fields_lesion_positive"]
    systems = json.loads(Path("results/nlp/phase8/validation_rescored.json").read_text(encoding="utf-8"))["systems"]
    for label, key in BASELINES.items():
        runs = systems[key]["corrected"]
        out["fields"][label] = {f: (runs[0][f] if len(runs) == 1 else mean_runs([r[f] for r in runs])) for f in FIELDS}
    c2 = out["fields"]["C2 parser v1.2"]
    for model in MODELS:
        if model in out["fields"]:
            out["interval_overlap_with_C2"][model] = {
                f: {"llm_errors": out["fields"][model][f]["n"] - out["fields"][model][f]["correct"],
                    "c2_errors": c2[f]["n"] - c2[f]["correct"],
                    "intervals_overlap": out["fields"][model][f]["ci_high"] >= c2[f]["ci_low"]
                    and c2[f]["ci_high"] >= out["fields"][model][f]["ci_low"]} for f in FIELDS}

    truth = {i: items[i]["targets"]["count"] != "0" for i in items}
    lr = {r["id"]: (None if r.get("abstained") else r["prediction"]) for r in runner.read_jsonl(LR_VALIDATION) if r["id"] in items}
    answers = {m: {r["id"]: r["record"]["liver_lesions"]["has_lesion"] for r in rows_of(m, "validation") if r["id"] in items}
               for m in MODELS}
    for system, got in [*answers.items(), ("P1 frozen LR", lr)]:
        if not got:
            continue
        pos = [i for i in got if truth[i]]
        neg = [i for i in got if not truth[i]]
        out["presence"][system] = {"positives": len(pos), "negatives": len(neg),
                                   "unclear_on_positives": sum(got[i] is None for i in pos),
                                   "unclear_on_negatives": sum(got[i] is None for i in neg),
                                   "wrong": sum(got[i] is not None and got[i] != truth[i] for i in got)}
        out["presence_scoring"][system] = {
            "sensitivity_strict": wilson_interval(sum(got[i] is True for i in pos), len(pos)),
            "specificity_strict": wilson_interval(sum(got[i] is False for i in neg), len(neg)),
            "specificity_abstain_as_negative": wilson_interval(sum(got[i] is not True for i in neg), len(neg))}
    return out


# ── Extraction Challenge v3 ─────────────────────────────────────────────────
def challenge_v3() -> dict[str, Any]:
    items = {i["id"]: i for i in eval_sets.load_items("extraction_challenge_v3")}
    records = [{"challenge_id": i, "category": it["category"], "subtype": it["subtype"], "targets": it["targets_fields"],
                "expected": it["expected"]} for i, it in items.items()]
    v12 = {}
    for i, it in items.items():
        f = extract_liver_lesion_features_v1_2(it["text"])  # already liver-only text
        v12[i] = {k: f[k] for k in ("has_lesion", *ev.FIELDS)}
    outcomes: dict[str, list[dict[str, dict[str, str]]]] = {
        "C1 parser v1.1": [{r["challenge_id"]: r["outcomes"] for r in runner.read_jsonl(PARSER_V3) if r["challenge_id"] in items}],
        "C2 parser v1.2": [{r["challenge_id"]: r["outcomes"] for r in ev.score(records, v12)}],
    }
    c3_runs = [{r["challenge_id"]: r for r in runner.read_jsonl(p) if r["challenge_id"] in items} for p in C3_RUNS]
    outcomes["C3 first model (6.2)"] = [{i: r["outcomes"] for i, r in run.items()} for run in c3_runs]
    outcomes["H = C2 + C3"] = [{r["challenge_id"]: r["outcomes"] for r in
                                ev.score(records, {i: hybrid(v12[i], run[i]["predicted"]) for i in items})} for run in c3_runs]
    llm = {m: {r["id"]: {f: ev.outcome(f, items[r["id"]]["expected"][f], eval_sets.parser_fields(r["record"])[f]) for f in ev.FIELDS}
               for r in rows_of(m, "extraction_challenge_v3") if r["id"] in items} for m in MODELS}

    def table(systems: list[str], restrict_to: str | None) -> list[dict[str, Any]]:
        rows = []
        for sub in sorted({(it["category"], it["subtype"]) for it in items.values()}):
            members = [i for i, it in items.items() if (it["category"], it["subtype"]) == sub]
            if restrict_to:
                members = [m for m in members if m in llm[restrict_to]]
                if not members:
                    continue
            for field in items[members[0]]["targets_fields"]:
                cells = {}
                for s in systems:
                    runs = [llm[s]] if s in MODELS else outcomes[s]
                    per_run = [wilson_interval(sum(run[m][field] == "correct" for m in members if m in run),
                                               sum(m in run for m in members)) for run in runs]
                    cells[s] = per_run[0] if len(per_run) == 1 else mean_runs(per_run)
                rows.append({"category": sub[0], "subtype": sub[1], "field": field, "n": len(members), "cells": cells})
        return rows

    baselines = list(outcomes)
    done = {f"{items[i]['category']}/{items[i]['subtype']}" for i in llm["flash_lite"]}
    return {"items": len(items), "items_done": {m: len(llm[m]) for m in MODELS},
            "all_items": table(["nemotron", *baselines], None),
            "flash_lite_items": table(["flash_lite", "nemotron", *baselines], "flash_lite") if llm["flash_lite"] else [],
            "subtypes_without_flash_lite": sorted({f"{it['category']}/{it['subtype']}" for it in items.values()} - done)}


# ── Presence Challenge v1 ───────────────────────────────────────────────────
def presence_challenge() -> dict[str, Any]:
    items = {i["id"]: i for i in eval_sets.load_items("presence_challenge_v1")}
    answers: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        got = {r["id"]: r["record"]["liver_lesions"]["has_lesion"] for r in rows_of(model, "presence_challenge_v1") if r["id"] in items}
        if got:
            answers[model] = got
    answers["C1 parser v1.1"] = {i: extract_liver_lesion_features(it["text"])["has_lesion"] for i, it in items.items()}
    answers["C2 parser v1.2"] = {i: extract_liver_lesion_features_v1_2(it["text"])["has_lesion"] for i, it in items.items()}
    out: dict[str, Any] = {}
    for category in sorted({it["category"] for it in items.values()}):
        members = [i for i, it in items.items() if it["category"] == category]
        scored = items[members[0]]["scored"]
        labels = {"lesion": sum(items[m]["label"] is True for m in members),
                  "no lesion": sum(items[m]["label"] is False for m in members),
                  "none (hedged)": sum(items[m]["label"] is None for m in members)}
        cell = {}
        for system, got in answers.items():
            done = [m for m in members if m in got]
            entry = {"n": len(done), "yes": sum(got[m] is True for m in done), "no": sum(got[m] is False for m in done),
                     "unclear": sum(got[m] is None for m in done)}
            if scored:
                entry["correct"] = sum(got[m] == items[m]["label"] for m in done)
                entry["wrong"] = sum(got[m] is not None and got[m] != items[m]["label"] for m in done)
                entry["accuracy_unclear_not_correct"] = wilson_interval(entry["correct"], len(done))
            cell[system] = entry
        out[category] = {"scored": scored, "n": len(members), "labels": labels, "systems": cell}
    return out


# ── Evidence rejections: was the rejected value right? ──────────────────────
def rejection_outcomes() -> dict[str, Any]:
    """For every field the evidence check rejected, score the model's raw value against the label.

    Description only: the verifier is frozen and is not changed. "not scored" = no label for that field on
    that item (a lesion-free report, an unverifiable field, or a field the v3 subtype does not target)."""
    sets = {s: {i["id"]: i for i in eval_sets.load_items(s)} for s in ("validation", "extraction_challenge_v3",
                                                                        "presence_challenge_v1")}
    sets["repeat"] = sets["validation"]
    out: dict[str, Any] = {}
    for model in MODELS:
        counts = {f: {"value right": 0, "value wrong": 0, "not scored": 0} for f in le.FIELDS}
        for set_name, items in sets.items():
            for r in rows_of(model, set_name):
                if not r["rejected"] or r["id"] not in items:
                    continue
                item, text = items[r["id"]], items[r["id"]]["text"]
                output, _ = le._parse(r["calls"][-1]["text"])
                checked = le.verify(output, text)
                raw_values = {**checked.values, **{f: getattr(output, f).value for f in r["rejected"]}}
                raw = le.to_inference_record(r["id"], text, le.Verified(raw_values, checked.spans, [], 0), {}).model_dump()
                for f in r["rejected"]:
                    counts[f][judge(set_name, item, f, raw, output)] += 1
        out[model] = counts
    return out


def judge(set_name: str, item: dict[str, Any], field: str, raw: dict[str, Any], output: le.LLMOutput) -> str:
    presence = raw["liver_lesions"]["has_lesion"]
    if set_name in ("validation", "repeat"):
        positive = item["targets"]["count"] != "0"
        if field == "has_lesion":
            return "not scored" if presence is None else ("value right" if presence == positive else "value wrong")
        target = LLM_TO_TARGET[field]
        if positive and target in runner.unverifiable_fields(item):
            return "not scored"
        # The model's raw value, straight from its output: the record drops values when presence is "unclear",
        # and a lesion-free report has targets too (count "0", the others empty).
        raw_value = getattr(output, field).value
        value = (None if raw_value is None else
                 {"count": lambda v: count_class(v), "segments": lambda v: sorted(v) or None,
                  "attenuation": lambda v: v, "diameter_cm": lambda v: max(v) if v else None}[target](raw_value))
        ok = runner.field_correct(target, value, item["targets"][target])
        return "value right" if ok else "value wrong"
    if set_name == "presence_challenge_v1":
        if field != "has_lesion" or not item["scored"] or presence is None:
            return "not scored"
        return "value right" if presence == item["label"] else "value wrong"
    target = LLM_TO_V3.get(field)
    if target is None or target not in item["targets_fields"]:
        return "not scored"
    raw_value = getattr(output, field).value
    if field == "reported_dimensions_cm" and raw_value is not None:
        raw_value = max(raw_value) if raw_value else None
    result = ev.outcome(target, item["expected"][target], raw_value)
    return "value right" if result == "correct" else "value wrong"


# ── Repeat, LLM measures, transport ─────────────────────────────────────────
def repeat() -> dict[str, Any]:
    out = {}
    for model in MODELS:
        rows = rows_of(model, "repeat")
        first = {r["id"]: r for r in rows_of(model, "validation")}
        if rows:
            out[model] = eval_sets.summarize_repeat(rows, first)
    return out


def llm_measures() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in MODELS:
        rows = [r for s in SETS for r in rows_of(model, s)]
        if not rows:
            continue
        m = runner.summarize_llm(rows)
        m["items"] = len(rows)
        m["duplicate_quote_items"] = sum(r["duplicate_quotes"] > 0 for r in rows)
        m["cost_usd"] = round(sum(r["cost_usd"] for r in rows), 6)
        out[model] = m
    return out


def transport_errors() -> dict[str, Any]:
    """Per model: skipped items from the runs before the log (counts only), and every logged event since."""
    unlogged = json.loads(UNLOGGED.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for model, (provider, folder, _) in MODELS.items():
        before = sum(n for run in unlogged["runs"] for n in run["skipped_items"].get(provider, {}).values())
        events: dict[str, int] = {}
        skipped = 0
        for s in SETS:
            path = ROOT / provider / folder / f"{s}_v2" / "transport_log.jsonl"
            for e in (runner.read_jsonl(path) if path.exists() else []):
                if e["final"]:
                    skipped += 1
                else:
                    events[e["event"]] = events.get(e["event"], 0) + 1
        out[model] = {"skipped_items_before_log": before, "logged_failed_attempts": events, "logged_skipped_items": skipped}
    return out


def coverage() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in MODELS:
        out[model] = {}
        for s in SETS:
            total = len(eval_sets.load_items(s))
            done = len({r["id"] for r in rows_of(model, s)})
            out[model][s] = {"done": done, "total": total, "note": NOT_RUN.get((model, s), "" if done >= total else "incomplete")}
    return out


def dev_reference() -> dict[str, Any]:
    path = ROOT / "groq" / "openai-gpt-oss-120b" / "dev_v2" / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    return {"model": "openai/gpt-oss-120b (Groq)", "items": m["items"], "fields": m["fields_lesion_positive"],
            "note": "Development set (Train), prompt v2, 49 of 100 items. Dropped from the evaluation for the rate limit "
                    "only (Phase 7 Amendment 3). A reference, not an evaluation result."}


# ── Report ──────────────────────────────────────────────────────────────────
def write_md(r: dict[str, Any]) -> str:
    label = {m: MODELS[m][2] for m in MODELS}
    name = lambda k: label.get(k, k)  # noqa: E731
    s = r["subsample"]
    L = ["# Phase 7 report: LLM extraction (frozen prompt v2)", "",
         "Two API models, kept by Phase 7 Amendment 3, run once on the fixed subsample (`results/nlp/llm/subsample_v1.json`, "
         f"seed {s['seed']}) with the frozen prompt v2 (hash `337a2921…`). They read the liver-only text, like every "
         "extraction candidate. Test was not read. API models are references: they are not deployable (plan §2), "
         "so nothing here enters the Phase 8 routing decision.", "",
         "## What the subsample is", "",
         f"- **Validation, lesion-positive: all {s['validation_positive_used']} of {s['validation_positive_all']}** "
         "lesion-positive Validation reports. The field numbers are therefore computed on the whole lesion-positive "
         "population, with the same n as the frozen candidates (`results/nlp/phase8/validation_rescored.md`).",
         f"- **Validation, lesion-free: {s['validation_negative_used']} sampled of {s['validation_negative_all']}.** "
         "Every presence number on lesion-free reports (abstention share, specificity) comes from this sample only.",
         f"- **Extraction Challenge v3:** {sum(s['extraction_challenge_v3'].values())} items, 20 per subtype. "
         f"**Presence Challenge v1:** {sum(s['presence_challenge_v1'].values())} items, 30 per category.",
         "- **Challenge-A, diagnosis only.** Both Challenge Sets were seen during development (Phase 6). By the Phase 8 "
         "plan their numbers explain behaviour and never enter a decision; a gap they show is future work, not a result.",
         "", "Accuracy cells: estimate [Wilson 95% interval] (n). C3 and H: mean ± SD over seeds 13 / 42 / 87, "
         "[envelope of the three seeds' Wilson intervals].", "",
         "## Coverage", "", "| Model | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for m, sets in r["coverage"].items():
        L.append(f"| {name(m)} | " + " | ".join(f"{c['done']}/{c['total']}" + (f" ({c['note']})" if c["note"] else "")
                                               for c in sets.values()) + " |")

    v = r["validation"]
    L += ["", "## Validation: fields on lesion-positive reports (verifiable fields only)", "",
          "| System | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for system, cells in v["fields"].items():
        L.append(f"| {name(system)} | " + " | ".join(fmt(cells[f]) for f in FIELDS) + " |")
    L += ["", "**Against C2 parser v1.2** (errors, and whether the Wilson intervals overlap):", "",
          "| Model | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS)]
    for m, cells in v["interval_overlap_with_C2"].items():
        L.append(f"| {name(m)} | " + " | ".join(
            f"{c['llm_errors']} vs {c['c2_errors']} errors, " + ("overlap" if c["intervals_overlap"] else "**no overlap**")
            for c in cells.values()) + " |")
    L += ["", "Where the intervals overlap, the difference is one to a few reports and is within noise at this n. "
          "Where they do not overlap, the difference is likely real.", ""]

    L += ["## Validation: presence", "",
          "The prompt tells the model to answer \"unclear\" when the text does not state presence; parser v1.1/v1.2 follow "
          "the same rule. An \"unclear\" is an abstention, not an error.", "",
          "| System | positives: unclear | lesion-free (sample): unclear | wrong answers |", "|---|---|---|---|"]
    for system, c in v["presence"].items():
        L.append(f"| {name(system)} | {c['unclear_on_positives']} of {c['positives']} | "
                 f"{c['unclear_on_negatives']} of {c['negatives']} ({c['unclear_on_negatives'] / c['negatives']:.0%}) | {c['wrong']} |")
    L += ["", "**Scoring views (abstain-as-negative and strict).** The LLM is not a presence candidate (P1 LR and "
          "P2 ClinicalBERT are). LR is shown for scale only: it always answers. Specificity uses the 180-report sample.", "",
          "| System | sensitivity | specificity, abstain-as-negative (\"unclear\" = no lesion) | specificity, strict (\"unclear\" = miss) |",
          "|---|---|---|---|"]
    for system, c in v["presence_scoring"].items():
        L.append(f"| {name(system)} | {fmt(c['sensitivity_strict'])} | {fmt(c['specificity_abstain_as_negative'])} | "
                 f"{fmt(c['specificity_strict'])} |")

    c3 = r["challenge_v3"]
    base = ["C1 parser v1.1", "C2 parser v1.2", "C3 first model (6.2)", "H = C2 + C3"]
    L += ["", "## Extraction Challenge v3 (Challenge-A, diagnosis only)", "",
          f"Every system in a row is scored on the same items. **Gemini Flash-Lite ran {c3['items_done']['flash_lite']} of "
          f"{c3['items']}** (its daily quota ended), so it is compared only in the second table, on its own items. "
          f"It ran no item of: {', '.join(c3['subtypes_without_flash_lite'])}. **Whether a model tells a liver span, "
          "node or vessel size from a lesion was answered by Nemotron alone.**", "",
          f"### All {c3['items']} items", "",
          "| Category | Subtype | Field | n | " + " | ".join(name(x) for x in ["nemotron", *base]) + " |",
          "|---|---|---|---|" + "---|" * (len(base) + 1)]
    for row in c3["all_items"]:
        L.append(f"| {row['category']} | {row['subtype']} | {row['field']} | {row['n']} | "
                 + " | ".join(short(row["cells"][x], row["n"]) for x in ["nemotron", *base]) + " |")
    if c3["flash_lite_items"]:
        L += ["", f"### The {c3['items_done']['flash_lite']} items Gemini Flash-Lite ran (every system on these items)", "",
              "| Category | Subtype | Field | n | " + " | ".join(name(x) for x in ["flash_lite", "nemotron", *base]) + " |",
              "|---|---|---|---|" + "---|" * (len(base) + 2)]
        for row in c3["flash_lite_items"]:
            L.append(f"| {row['category']} | {row['subtype']} | {row['field']} | {row['n']} | "
                     + " | ".join(short(row["cells"][x], row["n"]) for x in ["flash_lite", "nemotron", *base]) + " |")
    L += ["", "Reading v3: a subtype holds 20 items (fewer for Gemini), so one subtype says how a system handles one "
          "pattern, not that it is better overall. Gemini's 1.00 against H's 0.05 on `order/ascending` segments is such a "
          "diagnostic. Two H weaknesses show up here beyond the known unquantified-count one:", "",
          "- **`order/*` (largest lesion not the first one):** parser v1.2 answers with the first lesion's values "
          "instead of leaving the field empty, so H keeps them and C3 is never asked.",
          "- **`count/listed_sentences`:** the parser and H miss counts given as one lesion per sentence.", "",
          "Both are recorded for Phase 8 as Challenge-A observations (Phase 8 plan, Amendment 4), with no change to H or the parser."]

    L += ["", "## Presence Challenge v1 (Challenge-A, diagnosis only)", "",
          "Gemini Flash-Lite: not run (daily quota, Amendment 3). **\"unclear\" is never counted as correct:** accuracy = "
          "correct ÷ n, so a system that abstains on every item scores 0 with 0 wrong answers (as all three do on "
          "`liver_removed`). Hedged items have no correct answer; the share answered \"unclear\" is the measure.", "",
          "| Category | labels | System | yes / no / unclear | correct | wrong | accuracy (unclear not correct) |",
          "|---|---|---|---|---|---|---|"]
    for category, c in r["presence_challenge"].items():
        labels = ", ".join(f"{k} {n}" for k, n in c["labels"].items() if n)
        for system, e in c["systems"].items():
            if c["scored"]:
                L.append(f"| {category} | {labels} | {name(system)} | {e['yes']} / {e['no']} / {e['unclear']} | "
                         f"{e['correct']} | {e['wrong']} | {fmt(e['accuracy_unclear_not_correct'])} |")
            else:
                L.append(f"| {category} | {labels} | {name(system)} | {e['yes']} / {e['no']} / {e['unclear']} | — | — | "
                         f"hedged: {e['unclear']} of {e['n']} unclear |")

    L += ["", "## Repeat: run-to-run agreement with the Validation run", "",
          "Gemini Flash-Lite: not run (daily quota ended, Amendment 3), so its stability is not measured.", "",
          "| Model | pairs | " + " | ".join(FIELDS) + " | presence | all identical |", "|---|---|" + "---|" * (len(FIELDS) + 2)]
    for m, c in r["repeat"].items():
        L.append(f"| {name(m)} | {c['pairs']} | " + " | ".join(fmt(c["field_agreement"][f]) for f in FIELDS)
                 + f" | {fmt(c['presence_agreement'])} | {fmt(c['all_fields_identical'])} |")

    L += ["", "## Evidence rejections: was the rejected value right?", "",
          "A value is rejected when its quote is not in the text or does not show it; the field is then left empty. "
          "Here each rejected raw value is scored against the label, over every evaluation set. This is a description: "
          "the verifier is frozen and is not changed. \"value right\" = the check was too strict; \"value wrong\" = the "
          "check stopped an error; \"not scored\" = no label for that field on that item (an unverifiable Validation field, "
          "a v3 field the subtype does not target, or any field but presence on the Presence Challenge). On Validation, "
          "lesion-free reports count too: their labels are count 0 and no other value.", "",
          "| Model | Field | rejections | value right | value wrong | not scored |", "|---|---|---|---|---|---|"]
    for m, fields in r["rejection_outcomes"].items():
        for f, c in fields.items():
            total = sum(c.values())
            if total:
                L.append(f"| {name(m)} | {f} | {total} | {c['value right']} | {c['value wrong']} | {c['not scored']} |")
    L += ["", "Why Nemotron's count rejections are mostly right values: on a lesion-free report it often answers presence "
          "\"unclear\" with a count of 0 and no quote. The check accepts a 0 without a quote only after a \"no\", so it "
          "rejects a right value; the field is empty either way on such a report. Its segment rejections are the "
          "opposite: the check mostly stopped wrong values."]

    L += ["", "## LLM measures (all evaluation items)", "",
          "Latency is per call as the client saw it, so it includes the SDK's waits between retries.", "",
          "| Model | items | invalid | refusal | evidence rejections (per field) | duplicate-quote items | tokens in / out | latency p50 / p95 (ms) | cost |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, c in r["llm_measures"].items():
        rej = ", ".join(f"{k} {n}" for k, n in c["evidence_rejections"].items())
        lat = c["latency_ms"] or {"p50": "—", "p95": "—"}
        L.append(f"| {name(m)} | {c['items']} | {c['invalid_output_rate']:.4f} | {c['refusal_rate']:.4f} | {rej} | "
                 f"{c['duplicate_quote_items']} | {c['tokens']['input_tokens']:,} / {c['tokens']['output_tokens']:,} | "
                 f"{lat['p50']} / {lat['p95']} | ${c['cost_usd']:.2f} (free tier) |")

    L += ["", "## Transport errors (timeouts, connection errors, HTTP 408/409/429/5xx)", "",
          "The SDK retries them. An item that fails every retry is skipped (no row) and called again by the next run, so "
          "none is counted as invalid output or as an abstention.", "",
          "- **Before the log (counts only).** Until `scripts/nlp/llm/run_logged.py` existed, only the number of skipped "
          "items per run was kept (`results/nlp/llm/transport_errors_unlogged.json`): no error type, no item, and no "
          "failed attempt that a later retry recovered.",
          "- **Since the log.** Every failed attempt and every skip is in `<set folder>/transport_log.jsonl`.",
          f"- **Pacing change.** {PACING_CHANGE}", "",
          "| Model | skipped items, before the log (count only) | failed attempts, logged | skipped items, logged |",
          "|---|---|---|---|"]
    for m, c in r["transport_errors"].items():
        attempts = ", ".join(f"{k} {n}" for k, n in sorted(c["logged_failed_attempts"].items())) or "0"
        L.append(f"| {name(m)} | {c['skipped_items_before_log']} | {attempts} | {c['logged_skipped_items']} |")

    t = r["transport_errors"]["nemotron"]["logged_failed_attempts"]
    L += ["", "## Limitations", "",
          "- **Gemini Flash-Lite is partial:** v3 174 of 260 (no non-lesion subtype), no Presence Challenge, no repeat. "
          "Its quota of 500 ended after about 360 successful calls; where the rest went is not known (retried attempts "
          "were not logged in that run).",
          f"- **NVIDIA throttling:** {t.get('http_429', 0)} Nemotron attempts were rejected with HTTP 429 after the log "
          "started; all items finished after the pacing change.",
          "- **Presence on lesion-free reports** rests on a sample of 180 of 1,169.",
          "- **v3 and Presence Challenge v1 were seen during development** (Challenge-A): diagnosis only.",
          "- **Labels are source-derived** until the Gold Set is finished; no clinical accuracy is claimed.",
          "- **Data:** only the liver-only text of public, de-identified AbdomenAtlas 3.0 reports was sent. Gemini's free "
          "tier may use inputs (accepted by the team, Amendment 1). Result files hold liver-sentence excerpts under "
          "CC BY-NC-SA 4.0 (README, Licence and data notice).",
          "- **gpt-oss-120b** was dropped for the rate limit only; its development result is below as a reference."]

    d = r["dev_reference"]
    L += ["", "## Reference only: gpt-oss-120b on the development set", "", d["note"], "",
          "| Field | " + " | ".join(FIELDS) + " |", "|---|" + "---|" * len(FIELDS),
          "| gpt-oss-120b | " + " | ".join(fmt(d["fields"][f]) for f in FIELDS) + " |", ""]
    return "\n".join(L)


def main() -> None:
    report = {"subsample": subsample(), "coverage": coverage(), "validation": validation(), "challenge_v3": challenge_v3(),
              "presence_challenge": presence_challenge(), "repeat": repeat(), "rejection_outcomes": rejection_outcomes(),
              "llm_measures": llm_measures(), "transport_errors": transport_errors(), "dev_reference": dev_reference(),
              "pacing_change": PACING_CHANGE, "test_policy": "Test was not read."}
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    OUT_MD.write_text(write_md(report), encoding="utf-8", newline="\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
