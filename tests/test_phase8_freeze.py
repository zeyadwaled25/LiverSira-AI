"""Frozen Phase 8 files (code, configs, plan) must not change; weights are checked by --check."""

import json
from pathlib import Path

import pytest

from scripts.nlp.transformers.extraction.freeze_candidates import FREEZE_PATH, freeze_hash


def test_frozen_text_files_are_unchanged():
    if not FREEZE_PATH.exists():
        pytest.skip("no freeze")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    small = {p: h for p, h in frozen["sha256"].items() if not p.endswith("model.safetensors") and Path(p).exists()}
    changed = [p for p, h in small.items() if freeze_hash(Path(p)) != h]
    assert not changed, f"frozen for Phase 8; change only by an amendment to the analysis plan: {changed}"


def test_every_candidate_in_the_plan_is_frozen():
    if not FREEZE_PATH.exists():
        pytest.skip("no freeze")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    assert set(frozen["candidates"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert all(len(c.get("seeds", {"x": 1})) in (1, 3) for c in frozen["candidates"].values())


def test_freeze_hash_ignores_line_endings_for_text_only(tmp_path):
    lf, crlf = tmp_path / "a.py", tmp_path / "b.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert freeze_hash(lf) == freeze_hash(crlf)
    raw_lf, raw_crlf = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    raw_lf.write_bytes(b"\x00\n")
    raw_crlf.write_bytes(b"\x00\r\n")
    assert freeze_hash(raw_lf) != freeze_hash(raw_crlf)
