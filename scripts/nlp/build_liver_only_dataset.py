"""Write a liver-only copy of liver_nlp_clean.csv and check what the filter keeps.

Output: data/processed/nlp/liver_nlp_liver_only.csv, with the same columns as
liver_nlp_clean.csv and ``narrative_clean`` reduced to its liver-related text
(``liversira_ai.nlp.liver_text``). Two columns are added: ``narrative_full_clean``
(the unfiltered text) and ``liver_spans`` (JSON offsets of the kept text in it).

Checks, on Train and Validation only (Test is not read):
- retention: a label value written in the full report is still in the liver text;
- leakage: kept clauses that still name another organ and a size in cm or mm;
- parser v1.1 on the full vs the liver-only text, against source labels (Validation).
"""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from liversira_ai.nlp.baseline import extract_liver_lesion_features
from liversira_ai.nlp.data.raw_csv import ATTENUATION, COUNT, DIAMETER, ID, LOCATION
from liversira_ai.nlp.liver_text import LIVER_TEXT_VERSION, extract_liver_text, organs_in
from liversira_ai.nlp.parser_v1_1 import NUMBER_WORDS, PARSER_VERSION

INPUT_PATH = Path("data/processed/nlp/liver_nlp_clean.csv")
SPLITS_PATH = Path("data/processed/nlp/dataset_splits.csv")
OUTPUT_PATH = Path("data/processed/nlp/liver_nlp_liver_only.csv")
REPORT_DIR = Path("results/nlp/liver_text")
CHECK_SPLITS = ("train", "val")
EXAMPLES = 8

_SIZE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*x\s*\d+(?:\.\d+)?)*\s*(cm|mm)\b", re.I)
_SEGMENTS = re.compile(r"segments?\s+((?:[1-8](?:\s*(?:/|,|-|and|,\s*and)\s*)?)+)", re.I)
_LESION_WORD = r"(?:\w+\s+){0,3}?(?:liver\s+|hepatic\s+)?(?:lesions|masses|nodules|tumors)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sizes_cm(text: str) -> set[float]:
    """Every dimension of every written size, in cm."""
    out: set[float] = set()
    for match in _SIZE.finditer(text):
        factor = 10 if match.group(2).lower() == "mm" else 1
        out |= {round(float(d) / factor, 1) for d in re.findall(r"\d+(?:\.\d+)?", match.group(0))}
    return out


def segments(text: str) -> set[int]:
    return {int(d) for group in _SEGMENTS.findall(text) for d in re.findall(r"[1-8]", group)}


def count_stated(text: str, count: int) -> bool:
    """The count is written before a lesion word, in a sentence about no other organ."""
    words = [str(count)] + [w for w, n in NUMBER_WORDS.items() if n == count and len(w) > 2]
    pattern = re.compile(rf"\b(?:{'|'.join(words)})\s+{_LESION_WORD}\b", re.I)
    return any(
        pattern.search(sentence) and "other" not in organs_in(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", text)
    )


def retention(rows: pd.DataFrame) -> dict:
    """For positives: label values stated in the full text and kept in the liver text."""
    checks = {
        "diameter": lambda r, t: float(r[DIAMETER]) in sizes_cm(t),
        "segments": lambda r, t: {int(s) for s in str(r[LOCATION]).split(",")} <= segments(t),
        "attenuation": lambda r, t: str(r[ATTENUATION]).lower() in t.lower(),
        "count": lambda r, t: count_stated(t, int(r[COUNT])),
    }
    result = {}
    for name, check in checks.items():
        stated = lost = 0
        lost_ids = []
        for _, r in rows.iterrows():
            if name == "diameter" and r[DIAMETER] in ("UNKNOWN", "NOT_APPLICABLE"):
                continue
            if name == "segments" and not str(r[LOCATION])[0].isdigit():
                continue
            if name == "count" and int(r[COUNT]) < 2:
                continue
            if check(r, r["narrative_full_clean"]):
                stated += 1
                if not check(r, r["narrative_clean"]):
                    lost += 1
                    lost_ids.append(r[ID])
        result[name] = {"stated_in_full": stated, "lost_in_liver": lost, "lost_ids": lost_ids[:EXAMPLES]}
    return result


def leakage(rows: pd.DataFrame) -> dict:
    """Kept spans that still name another organ, with and without a size."""
    other = other_size = 0
    examples = []
    for _, r in rows.iterrows():
        full = r["narrative_full_clean"]
        for start, end in json.loads(r["liver_spans"]):
            span = full[start:end]
            if "other" in organs_in(span):
                other += 1
                if _SIZE.search(span):
                    other_size += 1
                    if len(examples) < EXAMPLES:
                        examples.append({"id": r[ID], "span": span})
    return {"spans_naming_other_organ": other, "of_which_with_size": other_size, "examples": examples}


def _label(r: pd.Series) -> dict:
    positive = int(r[COUNT]) > 0
    return {
        "has_lesion": positive,
        "lesion_count": int(r[COUNT]),
        "segments": sorted(int(s) for s in str(r[LOCATION]).split(",")) if str(r[LOCATION])[0].isdigit() else None,
        "diameter": float(r[DIAMETER]) if positive and r[DIAMETER] != "UNKNOWN" else None,
        "attenuation": r[ATTENUATION] if positive and r[ATTENUATION] != "UNKNOWN" else None,
    }


def _score(pred: dict, label: dict) -> dict:
    """Per field: None if not scored, else 'correct' / 'abstained' / 'wrong'."""

    def judge(value, expected, equal) -> str:
        return "abstained" if value is None else ("correct" if equal(value, expected) else "wrong")

    out = {"presence": judge(pred["has_lesion"], label["has_lesion"], lambda a, b: a == b)}
    positive = label["has_lesion"]
    out["count"] = judge(pred["lesion_count"], label["lesion_count"], lambda a, b: a == b) if positive else None
    out["segments"] = (
        judge(pred["location_segments"], label["segments"], lambda a, b: sorted(a) == b)
        if positive and label["segments"]
        else None
    )
    out["diameter"] = (
        judge(pred["derived_max_diameter_cm"], label["diameter"], lambda a, b: abs(a - b) <= 0.5)
        if positive and label["diameter"] is not None
        else None
    )
    out["attenuation"] = (
        judge(pred["attenuation"], label["attenuation"], lambda a, b: a == b)
        if positive and label["attenuation"]
        else None
    )
    return out


def parser_comparison(rows: pd.DataFrame) -> dict:
    fields = ("presence", "count", "segments", "diameter", "attenuation")
    tallies = {v: {f: {"correct": 0, "abstained": 0, "wrong": 0} for f in fields} for v in ("full", "liver")}
    changed = {f: {"fixed": 0, "broken": 0} for f in fields}
    changed_examples = []
    for _, r in rows.iterrows():
        label = _label(r)
        scores = {
            "full": _score(extract_liver_lesion_features(r["narrative_full_clean"]), label),
            "liver": _score(extract_liver_lesion_features(r["narrative_clean"]), label),
        }
        for f in fields:
            a, b = scores["full"][f], scores["liver"][f]
            if a is None:
                continue
            tallies["full"][f][a] += 1
            tallies["liver"][f][b] += 1
            if a != "correct" and b == "correct":
                changed[f]["fixed"] += 1
            elif a == "correct" and b != "correct":
                changed[f]["broken"] += 1
                if len(changed_examples) < EXAMPLES:
                    changed_examples.append({"id": r[ID], "field": f, "full": a, "liver": b})
    return {"tallies": tallies, "changed": changed, "broken_examples": changed_examples}


def write_report(summary: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "liver_only_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Liver-only report text",
        "",
        f"`{OUTPUT_PATH.as_posix()}` (gitignored) is `{INPUT_PATH.name}` with `narrative_clean` reduced to the "
        f"liver-related text by `liversira_ai.nlp.liver_text` v{summary['liver_text_version']}. "
        "Kept text is verbatim; `liver_spans` gives its offsets in `narrative_full_clean`. "
        f"Checks use {' + '.join(CHECK_SPLITS)} ({summary['checked_studies']} studies); Test was not read.",
        "",
        "## Output",
        "",
        f"- Studies: {summary['studies']}; empty liver text: {summary['empty']['total']} "
        f"({summary['empty']['positive']} lesion-positive).",
        f"- Characters kept: {summary['chars_kept_share']:.1%} of the full text.",
        f"- Input SHA-256 `{summary['input_sha256'][:16]}…`, output SHA-256 `{summary['output_sha256'][:16]}…`.",
        "",
        "## Retention (lesion-positive studies)",
        "",
        "A value counts as stated if it is written in the full report; lost if it is then missing from the liver text.",
        "",
        "| Field | Stated in full text | Lost in liver text |",
        "|---|---|---|",
    ]
    for name, r in summary["retention"].items():
        lines.append(f"| {name} | {r['stated_in_full']} | {r['lost_in_liver']} |")
    lk = summary["leakage"]
    lines += [
        "",
        "## Other organs left in the liver text",
        "",
        f"{lk['spans_naming_other_organ']} kept spans still name another organ (mostly organ lists such as "
        f"\"The spleen, liver, pancreas, and kidneys are normal\"); {lk['of_which_with_size']} of them also hold a "
        "size in cm or mm.",
        "",
    ]
    lines += [f"- `{e['id']}`: {e['span']}" for e in lk["examples"]]
    pc = summary["parser_comparison"]
    lines += [
        "",
        f"## Parser v{PARSER_VERSION} on full vs liver-only text (Validation)",
        "",
        "Against source labels. Cells: correct / abstained / wrong. Count, segments, diameter (within 0.5 cm) "
        "and attenuation are scored on lesion-positive studies where the label is known.",
        "",
        "| Field | Full text | Liver-only text | Fixed | Broken |",
        "|---|---|---|---|---|",
    ]
    for f, full in pc["tallies"]["full"].items():
        liver = pc["tallies"]["liver"][f]
        fmt = lambda t: f"{t['correct']} / {t['abstained']} / {t['wrong']}"  # noqa: E731
        lines.append(f"| {f} | {fmt(full)} | {fmt(liver)} | {pc['changed'][f]['fixed']} | {pc['changed'][f]['broken']} |")
    if pc["broken_examples"]:
        lines += ["", "Broken by the filter:", ""]
        lines += [f"- `{e['id']}` {e['field']}: {e['full']} → {e['liver']}" for e in pc["broken_examples"]]
    lines += [
        "",
        "## Notes",
        "",
        "- The filter is rule-based: a clause that never names an organ inherits the organ of the clause before it.",
        "- Organ lists that include the liver are kept whole, so other organ names remain in them, sometimes with "
        "their volumes or HU values (\"volumes of 1500 cc and 150 cc, respectively\"); no lesion size is left.",
        "- Kept spans are joined by spaces, and a span without final punctuation gets a closing \".\"; "
        "offsets cover the source text only.",
        "",
    ]
    (REPORT_DIR / "liver_only_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    df[COUNT] = df[COUNT].astype(int)
    results = [extract_liver_text(t) for t in df["narrative_clean"]]

    columns = list(df.columns)
    df["narrative_full_clean"] = df["narrative_clean"]
    df["narrative_clean"] = [r.text for r in results]
    df["liver_spans"] = [json.dumps(r.spans) for r in results]
    df = df[columns + ["narrative_full_clean", "liver_spans"]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    splits = pd.read_csv(SPLITS_PATH).set_index("id")["split"]
    check = df[df[ID].map(splits).isin(CHECK_SPLITS)]
    val = df[df[ID].map(splits).eq("val")]
    empty = df["narrative_clean"].eq("")
    summary = {
        "liver_text_version": LIVER_TEXT_VERSION,
        "parser_version": PARSER_VERSION,
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "studies": len(df),
        "checked_studies": len(check),
        "empty": {"total": int(empty.sum()), "positive": int((empty & df[COUNT].gt(0)).sum())},
        "chars_kept_share": df["narrative_clean"].str.len().sum() / df["narrative_full_clean"].str.len().sum(),
        "retention": retention(check[check[COUNT].gt(0)]),
        "leakage": leakage(check),
        "parser_comparison": parser_comparison(val),
    }
    write_report(summary)
    print(f"Wrote {OUTPUT_PATH} ({len(df):,} studies) and {REPORT_DIR / 'liver_only_report.md'}")


if __name__ == "__main__":
    main()
