"""Shared helpers for the learned presence experiments and the frozen model."""

import csv
import json
import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

SEED = 42


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


def classification_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    report = classification_report(
        y_true,
        y_pred,
        labels=[False, True],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "evaluated": len(y_true),
        "coverage": 1.0 if y_true else 0.0,
        "confusion_matrix": {
            "labels": [False, True],
            "rows_true": {
                "false": {
                    "predicted_false": int(matrix[0, 0]),
                    "predicted_true": int(matrix[0, 1]),
                },
                "true": {
                    "predicted_false": int(matrix[1, 0]),
                    "predicted_true": int(matrix[1, 1]),
                },
            },
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                metric: float(report[label][metric])
                for metric in ("precision", "recall", "f1-score", "support")
            }
            for label in ("negative", "positive")
        },
    }


def frozen_model() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
