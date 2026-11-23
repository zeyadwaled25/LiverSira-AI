"""Fix the evaluation subsample for the free-tier LLM runs (Phase 7 plan, Amendment 1).

Written once, before any LLM call, from IDs and labels only. Writes
results/nlp/llm/subsample_v1.json (IDs, counts, input hashes; no report text).

- Validation: every lesion-positive report (the fields need them) plus a seeded
  sample of lesion-free reports.
- Presence Challenge v1: a seeded sample per category.
- Extraction Challenge v3: a seeded sample per subtype.

Test is not read. Refuses to overwrite an existing subsample.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from liversira_ai.common.provenance import sha256_file

SEED = 2027
NEGATIVES = 180
PER_PRESENCE_CATEGORY = 30
PER_V3_SUBTYPE = 20
SOURCES = {
    "validation": Path("data/processed/nlp/extraction/extraction_val_v1.jsonl"),
    "presence_challenge_v1": Path("data/processed/nlp/challenge/presence_challenge_v1.jsonl"),
    "extraction_challenge_v3": Path("data/processed/nlp/challenge/extraction_challenge_v3.jsonl"),
}
OUT = Path("results/nlp/llm/subsample_v1.json")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_group(rows: list[dict], key, k: int, rng: random.Random, id_key: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row[id_key])
    return {g: sorted(rng.sample(sorted(ids), min(k, len(ids)))) for g, ids in sorted(groups.items())}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the subsample is fixed")
    rng = random.Random(SEED)
    val = read(SOURCES["validation"])
    positives = sorted(r["id"] for r in val if r["targets"]["count"] != "0")
    negatives = sorted(rng.sample(sorted(r["id"] for r in val if r["targets"]["count"] == "0"), NEGATIVES))
    presence = per_group(read(SOURCES["presence_challenge_v1"]), lambda r: r["category"], PER_PRESENCE_CATEGORY,
                         rng, "challenge_id")
    v3 = per_group(read(SOURCES["extraction_challenge_v3"]), lambda r: f"{r['category']}/{r['subtype']}",
                   PER_V3_SUBTYPE, rng, "challenge_id")
    record = {
        "version": "subsample_v1",
        "seed": SEED,
        "rule": f"Validation: all lesion-positive + {NEGATIVES} lesion-free; presence challenge: "
                f"{PER_PRESENCE_CATEGORY} per category; v3: {PER_V3_SUBTYPE} per subtype",
        "validation": {"positive": positives, "negative": negatives},
        "presence_challenge_v1": presence,
        "extraction_challenge_v3": v3,
        "counts": {
            "validation": len(positives) + len(negatives),
            "presence_challenge_v1": sum(map(len, presence.values())),
            "extraction_challenge_v3": sum(map(len, v3.values())),
        },
        "inputs_sha256": {name: sha256_file(path) for name, path in SOURCES.items()},
        "test_policy": "Test was not read.",
    }
    record["counts"]["total"] = sum(record["counts"].values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}: {record['counts']}")


if __name__ == "__main__":
    main()
