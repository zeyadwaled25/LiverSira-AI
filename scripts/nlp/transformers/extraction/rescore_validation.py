"""Validation accuracy of every extraction candidate under the old and the corrected scoring rule.

Old rule (6.2): a null target must be matched by a null prediction.
Corrected rule (Phase 8 plan, Amendment 3): on a real report with a lesion, a null
target means "source value not found in the text", so that field is not scored.

Validation is development data; Test, Challenge-B and the Challenge-A sets are not
read. Writes results/nlp/phase8/validation_rescored.{md,json}.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from liversira_ai.nlp.baseline import extract_liver_lesion_features, extract_liver_lesion_features_v1_2
from liversira_ai.nlp.extraction_targets import unverifiable_fields
from liversira_ai.nlp.metrics import wilson_interval
from scripts.nlp.transformers.extraction.compare import parser_targets
from scripts.nlp.transformers.extraction.train import TARGET_FIELDS, field_correct, read_jsonl

VAL = Path("data/processed/nlp/extraction/extraction_val_v1.jsonl")
OUT = Path("results/nlp/phase8")
SEEDS = (13, 42, 87)
RUNS = {
    "C3 first model (6.2, unmasked)": Path("results/nlp/transformers/extraction/runs/clinicalbert"),
    "C3 first model v2 (masked, Amendment 3)": Path("results/nlp/transformers/extraction/runs_v2/clinicalbert"),
    "C4 per-lesion model": Path("results/nlp/transformers/extraction/runs_lesion/clinicalbert"),
}


def hybrid(parser: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Hybrid H in target form: parser v1.2 unless empty, then the model."""
    if parser["count"] == "0":
        return dict(parser)
    return {f: parser[f] if parser[f] is not None else model[f] for f in TARGET_FIELDS}


def score(items: list[dict[str, Any]], preds: list[dict[str, Any]], corrected: bool) -> dict[str, Any]:
    out = {}
    for field in TARGET_FIELDS:
        idx = [i for i, it in enumerate(items)
               if it["targets"]["count"] != "0" and not (corrected and field in unverifiable_fields(it))]
        k = sum(field_correct(field, preds[i][field], items[i]["targets"][field]) for i in idx)
        out[field] = wilson_interval(k, len(idx))
    return out


def main() -> None:
    items = read_jsonl(VAL)
    systems: dict[str, list[list[dict[str, Any]]]] = {
        "C1 parser v1.1": [[parser_targets(extract_liver_lesion_features(i["text"])) for i in items]],
        "C2 parser v1.2": [[parser_targets(extract_liver_lesion_features_v1_2(i["text"])) for i in items]],
    }
    for name, root in RUNS.items():
        if all((root / f"seed{s}" / "val_predictions.jsonl").exists() for s in SEEDS):
            runs = []
            for s in SEEDS:
                by_id = {r["id"]: r["predicted"] for r in read_jsonl(root / f"seed{s}" / "val_predictions.jsonl")}
                runs.append([by_id[i["id"]] for i in items])
            systems[name] = runs
    for name in [n for n in systems if n.startswith("C3")]:
        label = "H = C2 + " + ("C3 v2" if "v2" in name else "C3 (6.2)")
        systems[label] = [[hybrid(p, m) for p, m in zip(systems["C2 parser v1.2"][0], run)] for run in systems[name]]

    result: dict[str, Any] = {"validation_items": len(items),
                              "lesion_positive": sum(i["targets"]["count"] != "0" for i in items),
                              "unverifiable_fields": {f: sum(f in unverifiable_fields(i) for i in items)
                                                      for f in TARGET_FIELDS},
                              "systems": {}}
    for name, runs in systems.items():
        result["systems"][name] = {
            rule: [score(items, run, rule == "corrected") for run in runs] for rule in ("old", "corrected")
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_rescored.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def cell(scores: list[dict[str, Any]], field: str) -> str:
        values = [s[field]["estimate"] for s in scores]
        n = scores[0][field]["n"]
        if len(values) == 1:
            s = scores[0][field]
            return f"{s['estimate']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] (n={n})"
        return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f} (n={n})"

    lines = [
        "# Validation re-scored: old rule vs verifiable fields only (Amendment 3)",
        "",
        f"Lesion-positive Validation reports: {result['lesion_positive']}. Unverifiable fields (null target on a real "
        f"report with a lesion: the source value is not found in the text): "
        + ", ".join(f"{f} {k}" for f, k in result["unverifiable_fields"].items()) + ".",
        "",
        "Old rule: a null target must be matched by a null prediction, so a system that reads a value the text does "
        "state is marked wrong whenever the source label disagrees with the text. Corrected rule: those fields are "
        "not scored. Parsers: one run, Wilson 95% interval. Models and hybrids: mean ± SD over seeds 13 / 42 / 87.",
        "",
    ]
    for rule, title in (("old", "Old rule (6.2)"), ("corrected", "Corrected rule (verifiable fields only)")):
        lines += [f"## {title}", "", "| System | " + " | ".join(TARGET_FIELDS) + " |", "|---|" + "---|" * len(TARGET_FIELDS)]
        for name, rules in result["systems"].items():
            lines.append(f"| {name} | " + " | ".join(cell(rules[rule], f) for f in TARGET_FIELDS) + " |")
        lines.append("")
    (OUT / "validation_rescored.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'validation_rescored.md'}")


if __name__ == "__main__":
    main()
