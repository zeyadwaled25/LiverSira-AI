"""Check and freeze Challenge-B. It never runs, imports or loads any candidate (parser or model).

Stage 1, items (before the author opens the Train-aug taxonomy):

    python -m scripts.nlp.transformers.extraction.validate_challenge_b items --author "<initials>" \
        --seen-train-aug no --seen-parser-rules no --seen-challenge-sets no

checks challenge_b_items.jsonl against the brief (docs/nlp/challenge_b_brief.md)
and writes its hash to results/nlp/phase8/challenge_b_freeze.json.

Stage 2, coverage (after stage 1, with the taxonomy open):

    python -m scripts.nlp.transformers.extraction.validate_challenge_b coverage

checks challenge_b_coverage.jsonl (each item -> a Train-aug family id or "new"),
checks the items file is unchanged, and adds the coverage hash.

    python -m scripts.nlp.transformers.extraction.validate_challenge_b check

verifies both hashes (the Phase 8 run calls ``check_frozen()`` first).
Only item ids and labels are printed; item text is never printed.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

FOLDER = Path("data/processed/nlp/challenge/challenge_b")
ITEMS = FOLDER / "challenge_b_items.jsonl"
COVERAGE = FOLDER / "challenge_b_coverage.jsonl"
FREEZE = Path("results/nlp/phase8/challenge_b_freeze.json")
SPLITS = Path("data/processed/nlp/splits.json")
TAXONOMY = Path("docs/nlp/extraction_train_aug_taxonomy.json")

FIELDS = ("count", "segments", "attenuation", "diameter_cm")
COUNT_CLASSES = tuple(str(n) for n in range(10)) + ("10+",)
ATTENUATION = ("hypoattenuating", "isoattenuating", "hyperattenuating")
CATEGORIES = {"clean": 60, "negative": 60, "targeted": 280}
TOTAL = sum(CATEGORIES.values())
MIN_PER_FIELD = 80  # targeted items testing each field
MIN_PER_SUBTYPE = 10
KEYS = {"id", "text", "category", "subtype", "targets", "labels", "base_id"}
OPTIONAL = {"notes"}
ID_RE = re.compile(r"^B\d{4}$")
SUBTYPE_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def sha256(path: Path) -> str:
    """SHA-256 of the file after CRLF->LF normalization (all files here are text)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {n}: not a JSON object")
            continue
        rows.append(row)
    return rows, errors


def _label_errors(rid: str, labels: Any) -> list[str]:
    if not isinstance(labels, dict) or set(labels) != set(FIELDS):
        return [f"{rid}: labels must have exactly the keys {list(FIELDS)}"]
    out = []
    count, segments, attenuation, diameter = (labels[f] for f in FIELDS)
    if count is not None and count not in COUNT_CLASSES:
        out.append(f"{rid}: count must be one of {list(COUNT_CLASSES)} or null, got {count!r}")
    if segments is not None:
        if (not isinstance(segments, list) or not segments or segments != sorted(set(segments))
                or not all(isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= 8 for s in segments)):
            out.append(f"{rid}: segments must be a sorted list of distinct integers 1-8, or null")
    if attenuation is not None and attenuation not in ATTENUATION:
        out.append(f"{rid}: attenuation must be one of {list(ATTENUATION)} or null")
    if diameter is not None:
        if isinstance(diameter, bool) or not isinstance(diameter, (int, float)) or not 0.1 <= diameter <= 30:
            out.append(f"{rid}: diameter_cm must be a number from 0.1 to 30, or null")
        elif round(float(diameter), 1) != float(diameter):
            out.append(f"{rid}: diameter_cm must have at most one decimal")
    if count == "0" and any(labels[f] is not None for f in FIELDS[1:]):
        out.append(f"{rid}: count 0 means the other fields must be null")
    return out


def validate_items(rows: list[dict[str, Any]], test_ids: set[str], known_ids: set[str]) -> tuple[list[str], list[str]]:
    """Errors (block the freeze) and warnings (printed only)."""
    errors, warnings = [], []
    ids = [r.get("id") for r in rows]
    for rid, k in Counter(ids).items():
        if k > 1:
            errors.append(f"{rid}: id used {k} times")
    texts = Counter(r.get("text") for r in rows)
    if any(k > 1 for k in texts.values()):
        errors.append(f"{sum(k - 1 for k in texts.values() if k > 1)} items repeat another item's text exactly")

    for row in rows:
        rid = str(row.get("id"))
        if set(row) - OPTIONAL != KEYS:
            errors.append(f"{rid}: keys must be {sorted(KEYS)} (+ optional {sorted(OPTIONAL)}), got {sorted(row)}")
            continue
        if not ID_RE.match(rid):
            errors.append(f"{rid}: id must look like B0001")
        if not isinstance(row["text"], str) or not row["text"].strip() or len(row["text"]) > 5000:
            errors.append(f"{rid}: text must be a non-empty string of at most 5000 characters")
        category, targets, labels = row["category"], row["targets"], row["labels"]
        if category not in CATEGORIES:
            errors.append(f"{rid}: category must be one of {list(CATEGORIES)}")
            continue
        if not isinstance(row["subtype"], str) or not SUBTYPE_RE.match(row["subtype"]):
            errors.append(f"{rid}: subtype must be lower_snake_case (2-41 characters)")
        if not isinstance(targets, list) or not targets or not set(targets) <= set(FIELDS) or len(set(targets)) != len(targets):
            errors.append(f"{rid}: targets must be a non-empty list of distinct fields from {list(FIELDS)}")
            continue
        errors += _label_errors(rid, labels)
        if not isinstance(labels, dict) or set(labels) != set(FIELDS):
            continue
        if category == "clean":
            if sorted(targets) != sorted(FIELDS):
                errors.append(f"{rid}: a clean item targets all four fields")
            if any(labels[f] is None for f in FIELDS) or labels["count"] == "0":
                errors.append(f"{rid}: a clean item states all four values (count at least 1)")
        if category == "negative":
            if targets != ["count"] or labels["count"] != "0":
                errors.append(f"{rid}: a negative item targets [\"count\"] with count \"0\"")
        if category == "targeted" and labels["count"] == "0":
            errors.append(f"{rid}: a targeted item describes at least one lesion (use category negative)")
        base = row["base_id"]
        if base is not None:
            if not isinstance(base, str) or base in test_ids:
                errors.append(f"{rid}: base_id is a Test study; Test reports may not be used")
            elif base not in known_ids:
                errors.append(f"{rid}: base_id {base!r} is not a Train or Validation study id")
        diameter = labels["diameter_cm"]
        if diameter is not None and isinstance(row["text"], str):
            numbers = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", row["text"])}
            if not any(abs(x - diameter) < 1e-9 or abs(x / 10 - diameter) < 1e-9 for x in numbers):
                warnings.append(f"{rid}: diameter_cm {diameter} (or {diameter * 10:g} mm) does not appear in the text")

    if errors:
        return errors, warnings
    by_category = Counter(r["category"] for r in rows)
    for category, n in CATEGORIES.items():
        if by_category[category] != n:
            errors.append(f"{category}: {by_category[category]} items, the brief asks for exactly {n}")
    targeted = [r for r in rows if r["category"] == "targeted"]
    for field in FIELDS:
        k = sum(field in r["targets"] for r in targeted)
        if k < MIN_PER_FIELD:
            errors.append(f"{field}: tested by {k} targeted items, the brief asks for at least {MIN_PER_FIELD}")
    for (category, subtype), k in Counter((r["category"], r["subtype"]) for r in rows).items():
        if k < MIN_PER_SUBTYPE:
            errors.append(f"{category}/{subtype}: {k} items, at least {MIN_PER_SUBTYPE} per subtype (merge small ones)")
    return errors, warnings


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(rows),
        "by_category": dict(Counter(r["category"] for r in rows)),
        "by_subtype": {f"{c}/{s}": k for (c, s), k in sorted(Counter((r["category"], r["subtype"]) for r in rows).items())},
        "targeted_items_per_field": {f: sum(f in r["targets"] for r in rows if r["category"] == "targeted") for f in FIELDS},
        "count_labels": dict(sorted(Counter(str(r["labels"]["count"]) for r in rows).items())),
        "unstated_targets": {f: sum(f in r["targets"] and r["labels"][f] is None for r in rows) for f in FIELDS},
        "items_with_base_id": sum(r["base_id"] is not None for r in rows),
    }


def _report(errors: list[str], warnings: list[str]) -> None:
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    if errors:
        sys.exit(f"{len(errors)} errors; nothing was frozen")


def stage_items(args: argparse.Namespace) -> None:
    if FREEZE.exists():
        sys.exit(f"{FREEZE} exists; the items are already frozen")
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    rows, errors = read_jsonl(ITEMS)
    more, warnings = validate_items(rows, set(splits["test"]), set(splits["train"]) | set(splits["val"]))
    _report(errors + more, warnings)
    record = {
        "schema_version": "1.0",
        "items_file": ITEMS.as_posix(),
        "items_sha256": sha256(ITEMS),
        "items_frozen_on": date.today().isoformat(),
        "author": args.author,
        "author_declaration": {
            "seen_train_aug": args.seen_train_aug == "yes",
            "seen_parser_rules": args.seen_parser_rules == "yes",
            "seen_challenge_sets_v1_v3": args.seen_challenge_sets == "yes",
            "taxonomy_opened_before_this_freeze": False,
        },
        "validator_sha256": sha256(Path(__file__)),
        "summary": summary(rows),
        "coverage": None,
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Items frozen: {record['items_sha256']} -> {FREEZE}. Now open the taxonomy and write {COVERAGE.name}.")


def validate_coverage(rows: list[dict[str, Any]], item_ids: set[str], families: set[str]) -> list[str]:
    errors = []
    seen = Counter(r.get("id") for r in rows)
    for row in rows:
        if set(row) != {"id", "coverage"}:
            errors.append(f"{row.get('id')}: keys must be ['coverage', 'id']")
        elif row["coverage"] != "new" and row["coverage"] not in families:
            errors.append(f"{row['id']}: coverage must be a taxonomy family id or \"new\", got {row['coverage']!r}")
    if set(seen) != item_ids:
        errors.append(f"coverage ids differ from item ids: {len(item_ids - set(seen))} missing, "
                      f"{len(set(seen) - item_ids)} unknown")
    errors += [f"{rid}: listed {k} times" for rid, k in seen.items() if k > 1]
    return errors


def stage_coverage(_: argparse.Namespace) -> None:
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record["coverage"] is not None:
        sys.exit("coverage is already frozen")
    if sha256(ITEMS) != record["items_sha256"]:
        sys.exit("the items file changed after it was frozen; Challenge-B is void")
    items, _ = read_jsonl(ITEMS)
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    rows, errors = read_jsonl(COVERAGE)
    errors += validate_coverage(rows, {r["id"] for r in items}, set(taxonomy["families"]))
    _report(errors, [])
    by_item = {r["id"]: r for r in items}
    record["coverage"] = {
        "file": COVERAGE.as_posix(),
        "sha256": sha256(COVERAGE),
        "frozen_on": date.today().isoformat(),
        "taxonomy_sha256": sha256(TAXONOMY),
        "new": sum(r["coverage"] == "new" for r in rows),
        "train_aug_family": sum(r["coverage"] != "new" for r in rows),
        "new_by_category": dict(Counter(by_item[r["id"]]["category"] for r in rows if r["coverage"] == "new")),
    }
    FREEZE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Coverage frozen: {record['coverage']['sha256']}. Challenge-B is complete; do not edit either file.")


def check_frozen() -> list[str]:
    """Problems with the frozen Challenge-B (empty list = ready for the Phase 8 run)."""
    if not FREEZE.exists():
        return ["Challenge-B is not frozen"]
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    problems = []
    if not ITEMS.exists() or sha256(ITEMS) != record["items_sha256"]:
        problems.append("items file missing or changed")
    if record["coverage"] is None:
        problems.append("coverage not frozen")
    elif not COVERAGE.exists() or sha256(COVERAGE) != record["coverage"]["sha256"]:
        problems.append("coverage file missing or changed")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    items = sub.add_parser("items")
    items.add_argument("--author", required=True)
    for flag in ("--seen-train-aug", "--seen-parser-rules", "--seen-challenge-sets"):
        items.add_argument(flag, required=True, choices=("yes", "no"))
    sub.add_parser("coverage")
    sub.add_parser("check")
    args = parser.parse_args()
    if args.stage == "items":
        stage_items(args)
    elif args.stage == "coverage":
        stage_coverage(args)
    else:
        problems = check_frozen()
        print("Challenge-B frozen and unchanged" if not problems else "; ".join(problems))
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
