"""Shared helpers for learned presence experiments."""

import csv
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(
    path: Path, selected_ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Load JSONL records, optionally restricting them to selected IDs."""
    with path.open(encoding="utf-8") as handle:
        return {
            record["id"]: record
            for record in map(json.loads, handle)
            if selected_ids is None or record["id"] in selected_ids
        }


def load_audit(path: Path, selected_ids: set[str]) -> dict[str, str]:
    """Load audit statuses for a selected set of IDs."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["id"]: row["overall_status"]
            for row in csv.DictReader(handle)
            if row.get("id") in selected_ids
        }


def snippet(text: str) -> str:
    """Keep examples non-identifying and remove report-specific numbers."""
    redacted = re.sub(r"BDMAP[_-]?\d+", "[ID]", text, flags=re.I)
    redacted = re.sub(r"\b\d+(?:\.\d+)?\b", "[NUM]", redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted[:240]


def error_category(text: str, audit_status: str) -> str:
    """Classify a source-label comparison error for reporting."""
    lowered = text.lower()
    if re.search(
        r"cannot be excluded|cannot exclude|cannot rule out|possible|"
        r"questionable|indeterminate|suspicious|suspected",
        lowered,
    ):
        return "ambiguous_report"
    if re.search(r"\bno\b|\bwithout\b|\bnegative for\b|\bfree of\b", lowered):
        return "negation_related"
    if audit_status == "MISMATCH":
        return "conflicting_source_label"
    return "other"
