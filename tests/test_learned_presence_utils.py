from pathlib import Path

from experiments.learned_presence.final_evaluate_learned_presence import (
    _metrics_with_specificity,
)
from experiments.learned_presence.train_learned_presence import _metrics
from experiments.learned_presence.utils import error_category, load_audit, load_jsonl


def test_load_jsonl_preserves_optional_all_records_and_selected_records(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"id": "train-1", "value": 1}\n{"id": "val-1", "value": 2}\n',
        encoding="utf-8",
    )

    all_records = load_jsonl(path)
    selected_records = load_jsonl(path, {"val-1"})

    assert set(all_records) == {"train-1", "val-1"}
    assert selected_records == {"val-1": {"id": "val-1", "value": 2}}


def test_load_audit_preserves_selected_statuses(tmp_path: Path):
    path = tmp_path / "audit.csv"
    path.write_text(
        "id,overall_status\ntrain-1,MATCH\nval-1,UNCERTAIN\n",
        encoding="utf-8",
    )

    assert load_audit(path, {"val-1"}) == {"val-1": "UNCERTAIN"}


def test_error_category_keeps_validation_and_test_error_semantics():
    assert error_category("Possible lesion.", "MISMATCH") == "ambiguous_report"
    assert error_category("No focal lesion.", "MATCH") == "negation_related"
    assert error_category("Lesion is present.", "MISMATCH") == "conflicting_source_label"
    assert error_category("Lesion is present.", "MATCH") == "other"


def test_validation_and_test_metric_wrappers_keep_distinct_semantics():
    validation = _metrics([False, True], [False, True])
    final_test = _metrics_with_specificity([False, True], [False, True])

    assert validation["coverage"] == 1.0
    assert "specificity" not in validation
    assert final_test["coverage"] == 1.0
    assert final_test["specificity"] == 1.0
