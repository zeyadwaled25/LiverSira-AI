"""Frozen Phase 8 files (code, configs, plan) must not change; weights are checked by --check."""

import json
from pathlib import Path

import pytest

from liversira_ai.common.provenance import sha256_file
from scripts.nlp.transformers.extraction.freeze_candidates import FREEZE_PATH


def test_frozen_text_files_are_unchanged():
    if not FREEZE_PATH.exists():
        pytest.skip("no freeze")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    small = {p: h for p, h in frozen["sha256"].items() if not p.endswith("model.safetensors") and Path(p).exists()}
    changed = [p for p, h in small.items() if sha256_file(Path(p)) != h]
    assert not changed, f"frozen for Phase 8; change only by an amendment to the analysis plan: {changed}"


def test_every_candidate_in_the_plan_is_frozen():
    if not FREEZE_PATH.exists():
        pytest.skip("no freeze")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert set(frozen["candidates"]) == {"C1", "C2", "C3", "C4", "C5"}
    assert all(len(c.get("seeds", {"x": 1})) in (1, 3) for c in frozen["candidates"].values())
