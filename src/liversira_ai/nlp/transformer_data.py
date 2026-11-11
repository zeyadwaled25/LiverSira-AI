"""Shared data access for the Phase 6 transformer experiments.

Reads the frozen study-level split, ``narrative_clean`` and the source-derived
``has_lesion`` label. Callers name the splits they need; Test is refused unless
explicitly allowed, so an experiment cannot read it by accident.
"""

import json
from pathlib import Path
from typing import Any, Iterable

MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
GROUND_TRUTH_PATH = Path("data/processed/nlp/ground_truth.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
CHALLENGE_PATH = Path("data/processed/nlp/challenge/presence_challenge_v1.jsonl")


def load_split_ids(splits: Iterable[str], allow_test: bool = False) -> dict[str, list[str]]:
    wanted = list(splits)
    if "test" in wanted and not allow_test:
        raise ValueError("Test is reserved for the single final evaluation (Phase 8)")
    frozen = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    return {name: list(frozen[name]) for name in wanted}


def load_examples(ids: Iterable[str]) -> list[dict[str, Any]]:
    """Return ``{"id", "text", "label"}`` for the given IDs, in the given order.

    Studies without a source label are skipped (the same label policy as the
    classical model: all non-empty source labels are kept).
    """
    order = list(ids)
    wanted = set(order)
    texts: dict[str, str] = {}
    with MASTER_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["id"] in wanted:
                texts[record["id"]] = record["reports"].get("narrative_clean") or ""
    labels: dict[str, bool] = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            label = record["ground_truth"].get("has_lesion")
            if record["id"] in wanted and label is not None:
                labels[record["id"]] = bool(label)
    return [{"id": i, "text": texts[i], "label": labels[i]} for i in order if i in labels and i in texts]


def load_challenge() -> list[dict[str, Any]]:
    with CHALLENGE_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
