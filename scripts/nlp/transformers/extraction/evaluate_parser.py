"""Score parser v1.1 on the extraction Challenge Set: the gate for Phase 6.2.

Uses the frozen v1.1 configuration (``baseline.extract_liver_lesion_features``).
A field goes on to 6.2 modelling only if the parser falls below GATE_ACCURACY
on at least one subtype that targets it. The threshold was set before the
parser was run on this set.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

GATE_ACCURACY = 0.90
DIAMETER_TOLERANCE_CM = 0.05  # values are written with one decimal; mm convert exactly
FIELDS = ("lesion_count", "location_segments", "derived_max_diameter_cm", "attenuation")
OUT_DIR = Path("results/nlp/transformers/extraction")
EXAMPLES_PER_SUBTYPE = 2


def outcome(field: str, expected: Any, predicted: Any) -> str:
    """'correct', 'abstained' (no value where one is stated) or 'wrong'."""
    if field == "location_segments" and predicted is not None:
        predicted = sorted(predicted)
    if expected is None:
        return "correct" if predicted is None else "wrong"
    if predicted is None:
        return "abstained"
    if field == "derived_max_diameter_cm":
        return "correct" if abs(float(predicted) - float(expected)) <= DIAMETER_TOLERANCE_CM else "wrong"
    return "correct" if predicted == expected else "wrong"


def score(records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        pred = predictions[r["challenge_id"]]
        rows.append(
            {
                "challenge_id": r["challenge_id"],
                "category": r["category"],
                "subtype": r["subtype"],
                "targets": r["targets"],
                "presence": pred.get("has_lesion"),
                "outcomes": {f: outcome(f, r["expected"][f], pred.get(f)) for f in FIELDS},
            }
        )
    return rows


def _accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    from liversira_ai.nlp.metrics import bootstrap

    outcomes = [row["outcomes"][field] for row in rows]
    correct = np.array([o == "correct" for o in outcomes], dtype=float)
    result = bootstrap(lambda c: c.mean(axis=1), correct)
    result["abstained"] = outcomes.count("abstained")
    result["wrong"] = outcomes.count("wrong")
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["category"], row["subtype"]), []).append(row)

    subtypes = []
    for (category, subtype), members in groups.items():
        all_correct = np.array([all(o == "correct" for o in m["outcomes"].values()) for m in members], dtype=float)
        subtypes.append(
            {
                "category": category,
                "subtype": subtype,
                "n": len(members),
                "targets": members[0]["targets"],
                "presence_detected": sum(m["presence"] is True for m in members),
                "fields": {f: _accuracy(members, f) for f in FIELDS},
                "all_fields_correct": float(all_correct.mean()),
            }
        )

    gate = {}
    for field in FIELDS:
        targeted = [s for s in subtypes if field in s["targets"] and s["category"] not in ("control",)]
        failing = [
            f"{s['category']}/{s['subtype']}"
            for s in targeted
            if s["fields"][field]["estimate"] is not None and s["fields"][field]["estimate"] < GATE_ACCURACY
        ]
        pooled_rows = [
            row for row in rows
            if field in row["targets"] and row["category"] != "control"
        ]
        gate[field] = {
            "pooled_targeted_accuracy": _accuracy(pooled_rows, field),
            "control_accuracy": _accuracy([r for r in rows if r["category"] == "control"], field),
            "failing_subtypes": failing,
            "needs_model": bool(failing),
        }
    return {"subtypes": subtypes, "gate": gate}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _short(value: Any) -> str:
    return "null" if value is None else json.dumps(value)


def render(summary: dict[str, Any], records: list[dict[str, Any]], predictions: dict[str, dict[str, Any]],
           rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    from liversira_ai.nlp.metrics import format_ci

    lines = [
        "# Parser v1.1 on the extraction Challenge Set",
        "",
        f"`{manifest['version']}`: {manifest['records']} generated items on lesion-free Validation reports; "
        "Test was not read. Expected values are set by construction, not by source labels. "
        "Parser: frozen v1.1 configuration (`baseline.extract_liver_lesion_features`). "
        f"Accuracy counts abstentions as errors; diameter is correct within {DIAMETER_TOLERANCE_CM} cm. "
        "95% bootstrap intervals over items (2000 resamples, seed 42).",
        "",
        "## Gate for 6.2",
        "",
        f"Rule, fixed before the run: a field goes to 6.2 modelling if the parser's accuracy is below "
        f"{GATE_ACCURACY:.2f} on at least one subtype that targets it.",
        "",
        "| Field | Control | Pooled over targeting subtypes | Subtypes below the gate | Needs a model |",
        "|---|---|---|---|---|",
    ]
    for field, g in summary["gate"].items():
        failing = ", ".join(f"`{s}`" for s in g["failing_subtypes"]) or "none"
        lines.append(
            f"| {field} | {format_ci(g['control_accuracy'])} | {format_ci(g['pooled_targeted_accuracy'])} "
            f"| {failing} | {'**yes**' if g['needs_model'] else 'no'} |"
        )

    lines += [
        "",
        "## Accuracy by subtype",
        "",
        "Bold = a field the subtype targets. Cells: accuracy (abstained / wrong).",
        "",
        "| Category | Subtype | n | Presence found | " + " | ".join(FIELDS) + " | All fields |",
        "|---|---|---|---|" + "---|" * len(FIELDS) + "---|",
    ]
    for s in summary["subtypes"]:
        cells = []
        for field in FIELDS:
            f = s["fields"][field]
            cell = f"{_pct(f['estimate'])} ({f['abstained']}/{f['wrong']})"
            cells.append(f"**{cell}**" if field in s["targets"] and len(s["targets"]) < len(FIELDS) else cell)
        lines.append(
            f"| {s['category']} | {s['subtype']} | {s['n']} | {s['presence_detected']} | "
            + " | ".join(cells) + f" | {_pct(s['all_fields_correct'])} |"
        )

    lines += ["", "## Examples of failures", "",
              "Generated lesion text only (no dataset text). Up to "
              f"{EXAMPLES_PER_SUBTYPE} per subtype, for the fields the subtype targets.", ""]
    by_id = {r["challenge_id"]: r for r in records}
    shown: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["category"], row["subtype"])
        bad = [f for f in row["targets"] if row["outcomes"][f] != "correct"]
        if not bad or shown.get(key, 0) >= EXAMPLES_PER_SUBTYPE:
            continue
        shown[key] = shown.get(key, 0) + 1
        r, p = by_id[row["challenge_id"]], predictions[row["challenge_id"]]
        diffs = "; ".join(f"{f}: expected {_short(r['expected'][f])}, got {_short(p.get(f))}" for f in bad)
        lines += [f"- `{key[0]}/{key[1]}`: {diffs}", f"  > {r['inserted_text']}"]
    if not shown:
        lines.append("None.")

    lines += [
        "",
        "## Notes",
        "- The items are synthetic in their lesion description; the surrounding report is real Validation text. "
        "A failure here shows a phrasing the parser does not handle, not its error rate on the dataset.",
        "- `unquantified` and `lobe_only` expect `null`: the right output is to abstain.",
        "- The Challenge Set must not be used to tune parser rules; doing so would turn it into a development set.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    from liversira_ai.common.provenance import git_commit, sha256_file
    from liversira_ai.nlp.baseline import BASELINE_VERSION, extract_liver_lesion_features
    from liversira_ai.nlp.parser_v1_1 import PARSER_VERSION
    from scripts.nlp.transformers.extraction.build_challenge_set import CHALLENGE_PATH, OUT_DIR as CS_DIR, VERSION

    manifest = json.loads((CS_DIR / f"{VERSION}_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(CHALLENGE_PATH) != manifest["output_sha256"]:
        raise SystemExit("Challenge Set file does not match its manifest hash; rebuild it.")
    with CHALLENGE_PATH.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    predictions = {}
    for r in records:
        features = extract_liver_lesion_features(r["text"])
        predictions[r["challenge_id"]] = {k: features[k] for k in ("has_lesion", *FIELDS)}
    rows = score(records, predictions)
    summary = summarize(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "parser_challenge_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "predicted": predictions[row["challenge_id"]]}) + "\n")
    metrics = {
        "schema_version": "1.0",
        "system": BASELINE_VERSION,
        "parser_version": PARSER_VERSION,
        "challenge_set": VERSION,
        "challenge_set_sha256": manifest["output_sha256"],
        "gate_accuracy": GATE_ACCURACY,
        "diameter_tolerance_cm": DIAMETER_TOLERANCE_CM,
        "git_commit": git_commit(),
        **summary,
    }
    (OUT_DIR / "parser_challenge.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = render(summary, records, predictions, rows, manifest)
    (OUT_DIR / "parser_challenge.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
