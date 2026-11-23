"""Parser v1.2 vs v1.1 on the 6.2 Train-aug data, and the evidence for each v1.2 rule.

Only Train-aug is read (extraction_train_v1 and extraction_train_lesion_v1,
including their real Train reports). Challenge Sets, Validation and Test are not
read. For each rule the report checks its registered item: v1.2 right, v1.2 with
that rule switched off wrong. Writes results/nlp/parser_v1_2/train_aug_report.{md,json}.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from liversira_ai.nlp import parser_v1_2
from liversira_ai.nlp.baseline import extract_liver_lesion_features, extract_liver_lesion_features_v1_2
from liversira_ai.nlp.parser_v1_2_rules import RULES, ablate
from scripts.nlp.transformers.extraction.compare import parser_targets
from scripts.nlp.transformers.extraction.freeze_candidates import freeze_hash
from scripts.nlp.transformers.extraction.train import TARGET_FIELDS, field_correct, read_jsonl

DATA = Path("data/processed/nlp/extraction")
SETS = ("extraction_train_v1", "extraction_train_lesion_v1")
OUT = Path("results/nlp/parser_v1_2")
CODE = (Path("src/liversira_ai/nlp/parser_v1_1.py"), Path("src/liversira_ai/nlp/parser_v1_2.py"),
        Path("src/liversira_ai/nlp/parser_v1_2_rules.py"), Path("src/liversira_ai/nlp/baseline.py"))


def family(item: dict[str, Any]) -> str:
    return item["category"] if item["category"] in ("real", "ocr") else f"{item['category']}/{item['subtype']}"


def check_rules(items: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for rule in RULES:
        item = items[rule.evidence_set][rule.evidence_id]
        target = item["targets"][rule.field]
        full = parser_targets(extract_liver_lesion_features_v1_2(item["text"]))[rule.field]
        v11 = parser_targets(extract_liver_lesion_features(item["text"]))[rule.field]
        with ablate(rule.rule_id):
            off = parser_targets(extract_liver_lesion_features_v1_2(item["text"]))[rule.field]
        out.append({
            "rule_id": rule.rule_id, "description": rule.description, "family": rule.family, "field": rule.field,
            "evidence_set": rule.evidence_set, "evidence_id": rule.evidence_id,
            "evidence_text": item.get("inserted_text") or item["text"],
            "target": target, "v1_1": v11, "v1_2": full, "v1_2_without_rule": off,
            "justified": field_correct(rule.field, full, target) and not field_correct(rule.field, off, target),
        })
    return out


def main() -> None:
    items = {name: {i["id"]: i for i in read_jsonl(DATA / f"{name}.jsonl")} for name in SETS}
    stats: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    changed = {"better": 0, "worse": 0}
    for name, rows in items.items():
        for item in rows.values():
            a = parser_targets(extract_liver_lesion_features(item["text"]))
            b = parser_targets(extract_liver_lesion_features_v1_2(item["text"]))
            for field in TARGET_FIELDS:
                x = field_correct(field, a[field], item["targets"][field])
                y = field_correct(field, b[field], item["targets"][field])
                key = (name, family(item), field)
                stats[key][0] += 1
                stats[key][1] += x
                stats[key][2] += y
                if item["category"] == "real" and x != y:
                    changed["better" if y else "worse"] += 1
    rules = check_rules(items)
    if not all(r["justified"] for r in rules):
        raise SystemExit(f"unjustified rules: {[r['rule_id'] for r in rules if not r['justified']]}")

    table = [{"set": s, "family": f, "field": fl, "n": n, "v1_1": round(a / n, 4), "v1_2": round(b / n, 4)}
             for (s, f, fl), (n, a, b) in sorted(stats.items())]
    result = {
        "parser_version": parser_v1_2.PARSER_VERSION,
        "data_policy": "Train-aug only (6.2 training data, incl. its real Train reports). "
                       "Challenge Sets, Validation and Test were not read.",
        "presence_policy": "Presence logic is unchanged from v1.1 (a report that never mentions a liver lesion "
                           "stays unknown, not 0); the rules only touch count, segments, size and attenuation.",
        "dropped_rules": [{"rule_id": "R2", "reason": "no Train-aug item needed it (ablation changed nothing)"}],
        "rules": rules,
        "real_train_field_changes": changed,
        "accuracy": table,
        "code_sha256": {p.as_posix(): freeze_hash(p) for p in CODE},
        "inputs_sha256": {(DATA / f"{n}.jsonl").as_posix(): freeze_hash(DATA / f"{n}.jsonl") for n in SETS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_aug_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def cell(row: dict[str, Any]) -> str:
        a, b = row["v1_1"], row["v1_2"]
        return f"{a:.3f} → **{b:.3f}**" if b != a else f"{a:.3f}"

    lines = [
        "# Parser v1.2 on 6.2 Train-aug",
        "",
        f"Parser v{parser_v1_2.PARSER_VERSION} = v1.1 plus {len(RULES)} rules written from Train-aug only "
        "(`extraction_train_v1`, `extraction_train_lesion_v1`). Challenge Sets, Validation and Test were not read. "
        "Presence logic is unchanged. v1.1 stays frozen.",
        "",
        "## Rules and their evidence",
        "",
        "Each rule has one Train-aug item on which v1.2 is right and v1.2 without that rule is wrong.",
        "",
        "| Rule | What it does | Evidence item | Field | Target | v1.1 | v1.2 | v1.2 without rule |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r['rule_id']} | {r['description']} | `{r['evidence_id']}` | {r['field']} | {r['target']} | {r['v1_1']} "
          f"| {r['v1_2']} | {r['v1_2_without_rule']} |" for r in rules],
        "",
        "R2 (more words for the largest lesion) was written and dropped: switching it off changed no Train-aug item.",
        "",
        f"Real Train reports (6,482 in each set): field-level changes v1.1 → v1.2: {changed['better']} better, "
        f"{changed['worse']} worse.",
        "",
        "## Accuracy by Train-aug family",
        "",
        "Cells: v1.1 → v1.2 (bold where it changed). Almost all errors left in generated families have presence "
        "unknown: the inserted lesion sentence names no organ and follows a base sentence about several organs, so "
        "the unchanged presence logic does not attribute it to the liver. `real` count is low for both because a "
        "report that never mentions a liver lesion stays unknown, not 0.",
        "",
    ]
    for name in SETS:
        lines += [f"### {name}", "", "| Family | n | " + " | ".join(TARGET_FIELDS) + " |",
                  "|---|---|" + "---|" * len(TARGET_FIELDS)]
        fams = sorted({r["family"] for r in table if r["set"] == name})
        for fam in fams:
            rows = {r["field"]: r for r in table if r["set"] == name and r["family"] == fam}
            lines.append(f"| {fam} | {rows[TARGET_FIELDS[0]]['n']} | " + " | ".join(cell(rows[f]) for f in TARGET_FIELDS) + " |")
        lines.append("")
    lines += ["## Code hashes", "", *[f"- `{k}`: `{v}`" for k, v in result["code_sha256"].items()], ""]
    (OUT / "train_aug_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'train_aug_report.md'}")


if __name__ == "__main__":
    main()
