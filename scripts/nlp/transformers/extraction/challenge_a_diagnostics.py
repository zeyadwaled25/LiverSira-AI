"""Per-subtype accuracy with Wilson 95% intervals for every frozen candidate on the Challenge-A sets.

Challenge-A = the 6.2 Challenge Sets seen during development (v1 distractor
items, v2_liver, v3). This is diagnosis only: none of it enters the Phase 8
decision (docs/nlp/phase8_analysis_plan.md). Run after the freeze; it checks the
freeze first. Test and Challenge-B are not read.

Writes results/nlp/phase8/challenge_a_diagnostics.{md,json}.

Its "hybrid H" column is C5 (``hybrid.hybrid_h``, same rule). H2 (C6, Amendment 1) is
deliberately not run on Challenge-A: the amendment allowed one attempt, on Train-aug only.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from liversira_ai.nlp.baseline import extract_liver_lesion_features_v1_2
from liversira_ai.nlp.liver_text import extract_liver_text
from liversira_ai.nlp.metrics import wilson_interval
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.freeze_candidates import check
from scripts.nlp.transformers.extraction.train import read_jsonl

OUT = Path("results/nlp/phase8")
PARSER_DIR = Path("results/nlp/transformers/extraction")
RUNS = {"C3": Path("results/nlp/transformers/extraction/runs/clinicalbert"),
        "C4": Path("results/nlp/transformers/extraction/runs_lesion/clinicalbert")}
SEEDS = (13, 42, 87)
NAMES = {"C1": "parser v1.1", "C2": "parser v1.2", "C3": "first model", "C4": "per-lesion model", "C5": "hybrid H"}
SETS = {  # name -> (challenge file version, apply liver-only filter, parser v1.1 predictions, run file, category filter)
    "v1_distractor": (cs.VERSION, True, "parser_challenge_liver_only_predictions.jsonl",
                      "challenge_v1_distractor_predictions.jsonl", "distractor"),
    "v2_liver": (cs.LIVER_VERSION, False, "parser_challenge_v2_liver_predictions.jsonl",
                 "challenge_v2_liver_predictions.jsonl", None),
    "v3": ("extraction_challenge_v3", True, "parser_challenge_v3_predictions.jsonl",
           "challenge_v3_predictions.jsonl", None),
}


def hybrid(parser: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Analysis plan §4: parser v1.2 value unless empty, then the first model's value."""
    if parser["has_lesion"] is False:
        return {"has_lesion": False, "lesion_count": 0, "location_segments": None,
                "derived_max_diameter_cm": None, "attenuation": None}
    return {"has_lesion": True, **{f: parser[f] if parser[f] is not None else model.get(f) for f in ev.FIELDS}}


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {r["challenge_id"]: r for r in read_jsonl(path)}


def main() -> None:
    changed = [p for p in check() if not p.endswith("model.safetensors")] if Path(
        "results/nlp/phase8/candidates_freeze.json").exists() else ["no freeze"]
    if changed:
        raise SystemExit(f"frozen files changed: {changed}")

    result: dict[str, Any] = {}
    for name, (version, filtered, parser_file, run_file, category) in SETS.items():
        records = [r for r in read_jsonl(cs.challenge_path(version)) if category is None or r["category"] == category]
        ids = [r["challenge_id"] for r in records]
        v12 = {}
        for r in records:
            text = extract_liver_text(r["text"]).text if filtered else r["text"]
            features = extract_liver_lesion_features_v1_2(text)
            v12[r["challenge_id"]] = {k: features[k] for k in ("has_lesion", *ev.FIELDS)}
        rows: dict[str, list[dict[str, dict[str, Any]]]] = {
            "C1": [by_id(PARSER_DIR / parser_file)],
            "C2": [{row["challenge_id"]: row for row in ev.score(records, v12)}],
        }
        for cid, root in RUNS.items():
            rows[cid] = [by_id(root / f"seed{s}" / run_file) for s in SEEDS]
        rows["C5"] = [
            {row["challenge_id"]: row for row in ev.score(
                records, {i: hybrid(v12[i], first[i]["predicted"]) for i in ids})}
            for first in rows["C3"]
        ]
        table = []
        for sub in sorted({(r["category"], r["subtype"]) for r in records}):
            members = [r for r in records if (r["category"], r["subtype"]) == sub]
            for field in members[0]["targets"]:
                cells = {}
                for cid, runs in rows.items():
                    per_run = [wilson_interval(sum(run[m["challenge_id"]]["outcomes"][field] == "correct"
                                                   for m in members), len(members)) for run in runs]
                    cells[cid] = {"mean": round(float(np.mean([w["estimate"] for w in per_run])), 4),
                                  "ci_low": min(w["ci_low"] for w in per_run),
                                  "ci_high": max(w["ci_high"] for w in per_run), "runs": per_run}
                table.append({"category": sub[0], "subtype": sub[1], "field": field, "n": len(members),
                              "targeted": len(members[0]["targets"]) < len(ev.FIELDS), "cells": cells})
        result[name] = table

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "challenge_a_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Challenge-A diagnostics (not part of the Phase 8 decision)",
        "",
        "Per-subtype accuracy of every frozen candidate on the 6.2 Challenge Sets that were seen during development: "
        "the v1 distractor items, `v2_liver` and `v3` (liver-only filter applied where the set needs it). These numbers "
        "explain behaviour; they do not choose anything. The decision uses Challenge-B and Test once, in Phase 8 "
        "(`docs/nlp/phase8_analysis_plan.md`).",
        "",
        "Cells: accuracy [Wilson 95% interval]. Model-based candidates (first model, per-lesion, hybrid H) show the mean "
        "over seeds 13 / 42 / 87 and the envelope of the three seeds' Wilson intervals. Only targeted fields are shown "
        "(control rows: all fields); the JSON has every run.",
        "",
    ]
    for name, table in result.items():
        lines += [f"## {name}", "",
                  "| Category | Subtype | Field | n | " + " | ".join(NAMES[c] for c in NAMES) + " |",
                  "|---|---|---|---|" + "---|" * len(NAMES)]
        for row in table:
            if row["targeted"] or row["category"] == "control":
                cells = " | ".join(f"{row['cells'][c]['mean']:.2f} [{row['cells'][c]['ci_low']:.2f}, "
                                   f"{row['cells'][c]['ci_high']:.2f}]" for c in NAMES)
                lines.append(f"| {row['category']} | {row['subtype']} | {row['field']} | {row['n']} | {cells} |")
        lines.append("")
    (OUT / "challenge_a_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'challenge_a_diagnostics.md'}")


if __name__ == "__main__":
    main()
