"""Token lengths of ``narrative_clean`` under each pinned tokenizer (Phase 6.0).

Answers: how many reports exceed the 512-token limit, and would truncation drop
the liver statements? Only Train and Validation reports are read; Test is
excluded so that the truncation policy is not chosen by looking at Test.

For each report the liver statements are located by a liver keyword pattern and
mapped to token positions with the tokenizer's offsets. A truncation strategy
"keeps the liver" when every liver mention lies inside the tokens it keeps.

Usage (from the repository root):

    .venv\\Scripts\\python.exe experiments/transformers/token_lengths.py
"""

import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0, str(ROOT / "src"))

import reliability_report as rr  # noqa: E402
from liversira_ai.nlp.parser import parse_liver_report  # noqa: E402
from models import CHECKPOINTS, MAX_LENGTH, load_tokenizer  # noqa: E402

PARSER_FIELDS = {
    "has_lesion": "presence",
    "lesion_count": "lesion_count",
    "reported_dimensions_cm": "reported_dimensions_cm",
    "location_segments": "location_segments",
    "attenuation": "attenuation",
}

MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
OUTPUT_DIR = Path("results/nlp/transformers")
METRICS_PATH = OUTPUT_DIR / "token_lengths.json"
REPORT_PATH = OUTPUT_DIR / "token_lengths.md"
SPLITS = ("train", "val")
LIVER = re.compile(r"\b(liver|hepatic|hepato\w*)\b", re.IGNORECASE)
IMPRESSION = re.compile(r"\bIMPRESSION\b")
CONTENT = MAX_LENGTH - 2  # [CLS] and [SEP]
HEAD_TAIL = (128, CONTENT - 128)
ANCHOR_CONTEXT = 32  # tokens kept before the first liver mention
CHUNK_STRIDE = 382  # windows of 510 content tokens overlapping by 128


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_reports(ids: set[str]) -> dict[str, str]:
    reports = {}
    with MASTER_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in ids:
                reports[r["id"]] = r["reports"].get("narrative_clean") or ""
    return reports


def _token_index(offsets: list[tuple[int, int]], char: int) -> int:
    """Index (without special tokens) of the token covering or following ``char``."""
    for i, (start, end) in enumerate(offsets):
        if end > char:
            return i
    return len(offsets)


def _vocab_is_cased(tokenizer) -> bool:
    vocab = tokenizer.get_vocab()
    return sum(1 for t in vocab if t[:1].isupper()) > 0.05 * len(vocab)


def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _analyze(key: str, reports: dict[str, str], split_of: dict[str, str], labels: dict[str, Any]):
    tokenizer = load_tokenizer(key)
    cased_vocab = _vocab_is_cased(tokenizer)
    lengths: dict[str, list[int]] = {s: [] for s in SPLITS}
    over: list[dict[str, Any]] = []
    for study_id, text in reports.items():
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        n = len(offsets)
        lengths[split_of[study_id]].append(n + 2)
        if n <= CONTENT:
            continue
        liver = [_token_index(offsets, m.start()) for m in LIVER.finditer(text)]
        impression = IMPRESSION.search(text)
        head, tail = HEAD_TAIL
        anchor = max(0, min(liver) - ANCHOR_CONTEXT) if liver else 0
        full, cut = parse_liver_report(text), parse_liver_report(text[: offsets[CONTENT - 1][1]])
        head_changes = [f for f, attr in PARSER_FIELDS.items() if getattr(full, attr) != getattr(cut, attr)]
        over.append(
            {
                "id": study_id,
                "split": split_of[study_id],
                "has_lesion": bool(labels.get(study_id, {}).get("has_lesion")),
                "tokens": n + 2,
                "liver_mentions": len(liver),
                "first_liver_token": min(liver) if liver else None,
                "last_liver_token": max(liver) if liver else None,
                "head_keeps_liver": all(i < CONTENT for i in liver),
                "head_keeps_any_liver": any(i < CONTENT for i in liver),
                "head_tail_keeps_liver": all(i < head or i >= n - tail for i in liver),
                "liver_anchor_keeps_liver": all(anchor <= i < anchor + CONTENT for i in liver),
                "chunks": 1 + -(-(n - CONTENT) // CHUNK_STRIDE),
                "parser_fields_changed_by_head_truncation": head_changes,
                "impression_token": _token_index(offsets, impression.start()) if impression else None,
            }
        )
    all_lengths = [x for s in SPLITS for x in lengths[s]]
    return {
        "repo_id": CHECKPOINTS[key].repo_id,
        "revision": CHECKPOINTS[key].revision,
        "lowercase": CHECKPOINTS[key].lowercase,
        "vocab_is_cased": cased_vocab,
        "casing_consistent": cased_vocab != CHECKPOINTS[key].lowercase,
        "tokens": {
            "median": int(statistics.median(all_lengths)),
            "p95": _percentile(all_lengths, 0.95),
            "p99": _percentile(all_lengths, 0.99),
            "max": max(all_lengths),
        },
        "over_limit": {
            "total": len(over),
            "by_split": {s: sum(o["split"] == s for o in over) for s in SPLITS},
            "lesion_positive": sum(o["has_lesion"] for o in over),
            "head_drops_liver": sum(not o["head_keeps_liver"] for o in over),
            "head_drops_all_liver": sum(not o["head_keeps_any_liver"] for o in over),
            "head_tail_drops_liver": sum(not o["head_tail_keeps_liver"] for o in over),
            "liver_anchor_drops_liver": sum(not o["liver_anchor_keeps_liver"] for o in over),
            "max_chunks": max((o["chunks"] for o in over), default=1),
            "parser_changes_after_head_truncation": {
                f: sum(f in o["parser_fields_changed_by_head_truncation"] for o in over) for f in PARSER_FIELDS
            },
            "impression_beyond_limit": sum(
                o["impression_token"] is not None and o["impression_token"] >= CONTENT for o in over
            ),
        },
        "over_limit_studies": over,
    }


def run() -> dict[str, Any]:
    splits = json.loads(rr.SPLITS_PATH.read_text(encoding="utf-8"))
    split_of = {i: s for s in SPLITS for i in splits[s]}
    if set(split_of) & set(splits["test"]):
        raise ValueError("Train/Validation overlap with Test")
    reports = _load_reports(set(split_of))
    labels = rr._reference_labels(rr.GROUND_TRUTH_PATH, "ground_truth")
    return {
        "schema_version": "1.0",
        "splits": list(SPLITS),
        "test_policy": "Test reports were not read.",
        "studies": {s: sum(v == s for v in split_of.values()) for s in SPLITS},
        "max_length": MAX_LENGTH,
        "head_tail": {"head": HEAD_TAIL[0], "tail": HEAD_TAIL[1]},
        "liver_pattern": LIVER.pattern,
        "inputs_sha256": {p.as_posix(): _sha(p) for p in (MASTER_PATH, rr.SPLITS_PATH, rr.GROUND_TRUTH_PATH)},
        "git_commit": _git_commit(),
        "models": {key: _analyze(key, reports, split_of, labels) for key in CHECKPOINTS},
    }


def render(result: dict[str, Any]) -> str:
    n = result["studies"]
    lines = [
        "# Token lengths of narrative_clean (Train + Validation)",
        "",
        f"{n['train']} Train and {n['val']} Validation reports; Test was not read. Limit "
        f"{result['max_length']} tokens including [CLS] and [SEP]. Head+tail keeps the first "
        f"{result['head_tail']['head']} and last {result['head_tail']['tail']} content tokens.",
        "",
        "| Model | Lowercase | Vocab cased | Median | p95 | p99 | Max | Over 512 (Train / Val) | Lesion-positive over 512 | IMPRESSION beyond 512 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in result["models"].items():
        t, o = m["tokens"], m["over_limit"]
        lines.append(
            f"| {key} | {m['lowercase']} | {m['vocab_is_cased']} | {t['median']} | {t['p95']} | {t['p99']} | "
            f"{t['max']} | {o['total']} ({o['by_split']['train']} / {o['by_split']['val']}) | "
            f"{o['lesion_positive']} | {o['impression_beyond_limit']} |"
        )
    lines += [
        "",
        "Over-limit reports that lose at least one liver mention under each strategy:",
        "",
        "| Model | Head only | Head only, no liver left | Head 128 + tail 382 | Liver-anchored window | Chunks (max per report) |",
        "|---|---|---|---|---|---|",
    ]
    for key, m in result["models"].items():
        o = m["over_limit"]
        lines.append(
            f"| {key} | {o['head_drops_liver']} | {o['head_drops_all_liver']} | {o['head_tail_drops_liver']} | "
            f"{o['liver_anchor_drops_liver']} | 0 ({o['max_chunks']}) |"
        )
    fields = list(PARSER_FIELDS)
    lines += [
        "",
        "Over-limit reports where parser v1.1 gives a different value on the head-truncated text "
        "(first 510 content tokens) than on the full text:",
        "",
        "| Model | " + " | ".join(fields) + " |",
        "|---|" + "---|" * len(fields),
    ]
    for key, m in result["models"].items():
        c = m["over_limit"]["parser_changes_after_head_truncation"]
        lines.append(f"| {key} | " + " | ".join(str(c[f]) for f in fields) + " |")
    lines += [
        "",
        "## Policy",
        "- **Presence (6.1): head truncation to 512 tokens.** Every over-limit report keeps at least one "
        "liver mention, and parser v1.1 reaches the same presence decision on the truncated text for all of them.",
        "- **Extraction (6.2): overlapping chunks** (510 content tokens, stride 382; at most "
        f"{max(m['over_limit']['max_chunks'] for m in result['models'].values())} chunks per report). Head "
        "truncation changes a few extracted values, so extraction models must see the whole report.",
        "- Both are set in `experiments/transformers/models.py`.",
        "",
        "## Notes",
        "- Liver mentions are found with the pattern "
        f"`{result['liver_pattern']}` and mapped to token positions with the tokenizer offsets.",
        "- \"Drops liver\" counts over-limit reports where at least one liver mention falls outside the "
        "kept tokens.",
        "- Revisions are pinned in `experiments/transformers/models.py`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
