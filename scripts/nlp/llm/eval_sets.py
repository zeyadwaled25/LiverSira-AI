"""Phase 7 evaluation sets (fixed subsample, Amendment 1) and their scoring.

Every set is read through the fixed subsample (results/nlp/llm/subsample_v1.json).
The Challenge Sets hold full reports; like every extraction candidate, the LLM reads
their liver-only text. Test is never read.

- ``validation``: extraction targets from the source; fields scored on lesion-positive
  reports, verifiable fields only (Phase 8 plan, Amendment 3); presence against the source.
- ``presence_challenge_v1``: presence per category; ``hedged`` has no correct answer, so the
  share of answers is reported.
- ``extraction_challenge_v3``: the fields each subtype targets, against the by-construction values.
- ``repeat``: 25 lesion-positive and 25 lesion-free Validation reports, called again without the
  cache; agreement with the ``validation`` run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from liversira_ai.nlp.liver_text import extract_liver_text
from liversira_ai.nlp.metrics import wilson_interval
from scripts.nlp.transformers.extraction import evaluate_parser as ev

SUBSAMPLE = Path("results/nlp/llm/subsample_v1.json")
VALIDATION = Path("data/processed/nlp/extraction/extraction_val_v1.jsonl")
PRESENCE = Path("data/processed/nlp/challenge/presence_challenge_v1.jsonl")
V3 = Path("data/processed/nlp/challenge/extraction_challenge_v3.jsonl")
EVAL_SETS = ("validation", "presence_challenge_v1", "extraction_challenge_v3", "repeat")
REPEAT_PER_CLASS = 25


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _subsample() -> dict[str, Any]:
    return json.loads(SUBSAMPLE.read_text(encoding="utf-8"))


def repeat_ids() -> list[str]:
    sub = _subsample()["validation"]
    return sorted(sub["positive"])[:REPEAT_PER_CLASS] + sorted(sub["negative"])[:REPEAT_PER_CLASS]


def load_items(set_name: str) -> list[dict[str, Any]]:
    """Items as {id, text, category, subtype, ...labels}; text is what the model reads."""
    sub = _subsample()
    if set_name in ("validation", "repeat"):
        wanted = set(repeat_ids()) if set_name == "repeat" else set(sub["validation"]["positive"] + sub["validation"]["negative"])
        return sorted((r for r in _read(VALIDATION) if r["id"] in wanted), key=lambda r: r["id"])
    if set_name == "presence_challenge_v1":
        wanted = {i for ids in sub["presence_challenge_v1"].values() for i in ids}
        return [{"id": r["challenge_id"], "text": extract_liver_text(r["text"]).text, "category": r["category"],
                 "subtype": r["category"], "label": r["label"], "scored": r["scored"]}
                for r in _read(PRESENCE) if r["challenge_id"] in wanted]
    if set_name == "extraction_challenge_v3":
        wanted = {i for ids in sub["extraction_challenge_v3"].values() for i in ids}
        return [{"id": r["challenge_id"], "text": extract_liver_text(r["text"]).text, "category": r["category"],
                 "subtype": r["subtype"], "targets_fields": r["targets"], "expected": r["expected"]}
                for r in _read(V3) if r["challenge_id"] in wanted]
    raise KeyError(set_name)


def parser_fields(record: dict[str, Any]) -> dict[str, Any]:
    """An InferenceRecord in the field form the Challenge Sets use."""
    lesions = record["liver_lesions"]
    largest = lesions["largest_lesion"] or {}
    return {"has_lesion": lesions["has_lesion"], "lesion_count": lesions["lesion_count"],
            "location_segments": largest.get("location_segments"),
            "derived_max_diameter_cm": largest.get("derived_max_diameter_cm"), "attenuation": largest.get("attenuation")}


def summarize_presence_challenge(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for category in sorted({items[r["id"]]["category"] for r in rows}):
        group = [r for r in rows if items[r["id"]]["category"] == category]
        answers = [r["record"]["liver_lesions"]["has_lesion"] for r in group]
        entry = {"n": len(group), "yes": answers.count(True), "no": answers.count(False), "unclear": answers.count(None)}
        if items[group[0]["id"]]["scored"]:
            correct = sum(a == items[r["id"]]["label"] for a, r in zip(answers, group))
            entry["accuracy"] = wilson_interval(correct, len(group))
        else:
            entry["note"] = "no correct answer (hedged); the share answered 'unclear' is the measure"
        out[category] = entry
    return out


def summarize_v3(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted({(items[r["id"]]["category"], items[r["id"]]["subtype"]) for r in rows}):
        group = [r for r in rows if (items[r["id"]]["category"], items[r["id"]]["subtype"]) == key]
        cell = {}
        for field in items[group[0]["id"]]["targets_fields"]:
            outcomes = [ev.outcome(field, items[r["id"]]["expected"][field], _count_cap(field, parser_fields(r["record"])[field]))
                        for r in group]
            cell[field] = {**wilson_interval(outcomes.count("correct"), len(outcomes)),
                           "abstained": outcomes.count("abstained"), "wrong": outcomes.count("wrong")}
        out["/".join(key)] = cell
    return out


def _count_cap(field: str, value: Any) -> Any:
    # The Challenge Sets write counts as integers; "10+" is never a v3 target.
    return value


def summarize_repeat(rows: list[dict[str, Any]], first_run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(first_run[r["id"]], r) for r in rows if r["id"] in first_run]
    fields = ("count", "segments", "attenuation", "diameter_cm")
    agreement = {f: wilson_interval(sum(a["predicted"][f] == b["predicted"][f] for a, b in pairs), len(pairs)) for f in fields}
    presence = sum(a["record"]["liver_lesions"]["has_lesion"] == b["record"]["liver_lesions"]["has_lesion"] for a, b in pairs)
    whole = sum(a["predicted"] == b["predicted"] and a["record"]["liver_lesions"]["has_lesion"]
                == b["record"]["liver_lesions"]["has_lesion"] for a, b in pairs)
    return {"pairs": len(pairs), "field_agreement": agreement, "presence_agreement": wilson_interval(presence, len(pairs)),
            "all_fields_identical": wilson_interval(whole, len(pairs))}
