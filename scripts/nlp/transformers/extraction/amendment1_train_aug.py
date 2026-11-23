"""Amendment 1 to the Phase 8 plan: field status in parser v1.2 and hybrid H2, checked on Train-aug only.

One attempt, no iteration. Reads only the 6.2 Train-aug data
(extraction_train_v1, extraction_train_lesion_v1). Challenge Sets, Validation
and Test are not read. Checks:

1. Parser v1.2 values are unchanged by the status (compared with the frozen
   parser_v1_2.py from the original freeze commit).
2. A Train-aug item with an unquantified count on which H is wrong and H2 right
   (the evidence the amendment asks for), per first-model seed.
3. H vs H2 on every Train-aug item where they can differ (a field the parser
   leaves empty with status not_mentioned), per seed: a paired comparison.
   Elsewhere H and H2 are identical by construction.

Writes results/nlp/phase8/amendment1_train_aug.{md,json}.
"""

import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from liversira_ai.common.provenance import sha256_file
from liversira_ai.nlp.baseline import extract_liver_lesion_features, extract_liver_lesion_features_v1_2
from liversira_ai.nlp.extraction_model import ExtractionModel, encode_item, to_parser_fields
from liversira_ai.nlp.hybrid import FIELDS, hybrid_h, hybrid_h2
from liversira_ai.nlp.transformer_models import CHECKPOINTS, load_tokenizer
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.train import CONFIG_PATH, MODELS_ROOT, RESULTS_ROOT, predict, read_jsonl

ORIGINAL_FREEZE_COMMIT = "9b088a8"
DATA = Path("data/processed/nlp/extraction")
SETS = ("extraction_train_v1", "extraction_train_lesion_v1")
OUT = Path("results/nlp/phase8")
SCRATCH = Path("models/_amendment1_tmp")
TARGET_TO_FIELD = {"count": "lesion_count", "segments": "location_segments", "diameter_cm": "derived_max_diameter_cm",
                   "attenuation": "attenuation"}


def expected(targets: dict[str, Any]) -> dict[str, Any]:
    """Train-aug targets in parser-field form (count class "10+" kept as a string for comparison)."""
    count = targets["count"]
    return {"lesion_count": None if count is None else (count if count == "10+" else int(count)),
            "location_segments": targets["segments"], "derived_max_diameter_cm": targets["diameter_cm"],
            "attenuation": targets["attenuation"]}


def correct(field: str, exp: Any, pred: Any) -> bool:
    if field == "lesion_count" and isinstance(pred, int) and pred >= 10:
        pred = "10+"
    return ev.outcome(field, exp, pred) == "correct"


def frozen_parser():
    """parser_v1_2.py as frozen in the original freeze commit."""
    source = subprocess.run(["git", "show", f"{ORIGINAL_FREEZE_COMMIT}:src/liversira_ai/nlp/parser_v1_2.py"],
                            capture_output=True, text=True, check=True, encoding="utf-8").stdout
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / "parser_v1_2_frozen.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("parser_v1_2_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["parser_v1_2_frozen"] = module
    spec.loader.exec_module(module)
    return module


def model_predictions(texts: dict[str, str], seed: int, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from safetensors.torch import load_file

    model_key = config["models"][0]
    manifest = json.loads((RESULTS_ROOT / model_key / f"seed{seed}" / "run_manifest.json").read_text(encoding="utf-8"))
    weight_file = MODELS_ROOT / model_key / f"seed{seed}" / "model.safetensors"
    if sha256_file(weight_file) != manifest["weights"]["sha256"]:
        raise SystemExit(f"{weight_file} does not match its run manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    checkpoint = CHECKPOINTS[model_key]
    model = ExtractionModel.from_checkpoint(checkpoint.repo_id, checkpoint.revision)
    model.load_state_dict(load_file(str(weight_file)))
    model.to(device)
    tokenizer = load_tokenizer(model_key)
    empty = {"count": None, "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}
    ids = list(texts)
    encoded = [encode_item(tokenizer, texts[i], empty, config["max_length"]) for i in ids]
    decoded, _ = predict(model, encoded, [texts[i] for i in ids], config["eval_batch_size"], device, dtype,
                         tokenizer.pad_token_id)
    del model
    torch.cuda.empty_cache()
    return {i: to_parser_fields(d) for i, d in zip(ids, decoded)}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    old = frozen_parser()
    items = {(name, row["id"]): row for name in SETS for row in read_jsonl(DATA / f"{name}.jsonl")}

    changed_values, status_counts, affected = 0, Counter(), {}
    features = {}
    for key, item in items.items():
        new = extract_liver_lesion_features_v1_2(item["text"])
        before = extract_liver_lesion_features(item["text"], parse=old.parse_liver_report)
        if any(new[f] != before[f] for f in ("has_lesion", *FIELDS)):
            changed_values += 1
        features[key] = new
        if new["field_status"]:
            for f in FIELDS:
                status_counts[(f, new["field_status"][f])] += 1
        if new["has_lesion"] is True and any(
            new[f] is None and new["field_status"][f] == "not_mentioned" for f in FIELDS
        ):
            affected[key] = item
    if changed_values:
        raise SystemExit(f"parser v1.2 values changed on {changed_values} items; the status must be additive")

    texts = {f"{k[0]}|{k[1]}": item["text"] for k, item in affected.items()}
    per_seed = {}
    evidence = []
    for seed in config["seeds"]:
        preds = model_predictions(texts, seed, config)
        table: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])  # n, H correct, H2 correct
        for key, item in affected.items():
            model = preds[f"{key[0]}|{key[1]}"]
            exp = expected(item["targets"])
            h, h2 = hybrid_h(features[key], model), hybrid_h2(features[key], model)
            family = item["category"] if item["category"] in ("real", "ocr") else f"{item['category']}/{item['subtype']}"
            for f in FIELDS:
                if features[key][f] is None and features[key]["field_status"][f] == "not_mentioned":
                    row = table[(family, f)]
                    row[0] += 1
                    row[1] += correct(f, exp[f], h[f])
                    row[2] += correct(f, exp[f], h2[f])
            if (features[key]["field_status"]["lesion_count"] == "not_mentioned"
                    and not correct("lesion_count", exp["lesion_count"], h["lesion_count"])
                    and correct("lesion_count", exp["lesion_count"], h2["lesion_count"])):
                evidence.append({"seed": seed, "set": key[0], "id": key[1], "family": family,
                                 "text": item.get("inserted_text") or item["text"][-300:],
                                 "target_count": item["targets"]["count"], "model_count": model["lesion_count"]})
        per_seed[seed] = [{"family": fam, "field": f, "n": n, "h_correct": a, "h2_correct": b}
                          for (fam, f), (n, a, b) in sorted(table.items())]

    totals = {seed: {"n": sum(r["n"] for r in rows), "h_correct": sum(r["h_correct"] for r in rows),
                     "h2_correct": sum(r["h2_correct"] for r in rows)} for seed, rows in per_seed.items()}
    result = {
        "amendment": 1,
        "data_policy": "Train-aug only; one attempt; Challenge Sets, Validation and Test not read.",
        "parser_values_changed": changed_values,
        "status_counts": {f"{f}:{s}": k for (f, s), k in sorted(status_counts.items())},
        "affected_items": len(affected),
        "per_seed": per_seed,
        "totals": totals,
        "evidence": evidence,
        "justified_by_train_aug": bool(evidence),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "amendment1_train_aug.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    first_by_seed = {s: next((e for e in evidence if e["seed"] == s), None) for s in config["seeds"]}
    lines = [
        "# Amendment 1: field status and hybrid H2 on Train-aug",
        "",
        "One attempt on 6.2 Train-aug only (`extraction_train_v1`, `extraction_train_lesion_v1`). Challenge Sets, "
        "Validation and Test were not read. H2 differs from H only where parser v1.2 leaves a field empty with "
        "status `not_mentioned`: H takes the first model's value there, H2 keeps it empty.",
        "",
        f"- Parser v1.2 values changed by adding the status: **{changed_values}** items (must be 0).",
        f"- Train-aug items where H and H2 can differ: **{len(affected)}** (of {len(items)}).",
        "",
        "## Evidence: unquantified count, H wrong, H2 right",
        "",
    ]
    if evidence:
        lines += [f"{len(evidence)} (item, seed) cases. First per seed:", "",
                  "| Seed | Item | Target count | Model count | Text |", "|---|---|---|---|---|"]
        for seed, e in first_by_seed.items():
            lines.append(f"| {seed} | `{e['id']}` | {e['target_count']} | {e['model_count']} | {e['text'][:160]} |"
                         if e else f"| {seed} | none | | | |")
    else:
        lines += ["None. On Train-aug the first model leaves unquantified counts empty, so H is never wrong there. "
                  "The weakness was found in the Challenge-A diagnostics (Phase 8 plan, Amendment 1)."]
    lines += ["", "## H vs H2 where they differ (paired, per seed)", "",
              "| Seed | Fields compared | H correct | H2 correct |", "|---|---|---|---|",
              *[f"| {s} | {t['n']} | {t['h_correct']} | {t['h2_correct']} |" for s, t in totals.items()], "",
              "### By family and field (seed " + str(config["seeds"][0]) + ")", "",
              "| Family | Field | n | H correct | H2 correct |", "|---|---|---|---|---|",
              *[f"| {r['family']} | {r['field']} | {r['n']} | {r['h_correct']} | {r['h2_correct']} |"
                for r in per_seed[config["seeds"][0]]], "",
              "## Parser v1.2 field status on Train-aug", "",
              "| Field | Status | Fields |", "|---|---|---|",
              *[f"| {f} | {s} | {k} |" for (f, s), k in sorted(status_counts.items())], ""]
    (OUT / "amendment1_train_aug.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'amendment1_train_aug.md'}; evidence: {len(evidence)}; totals: {totals}")


if __name__ == "__main__":
    main()
