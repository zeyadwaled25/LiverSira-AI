from experiments.learned_presence.train_learned_presence import _metrics
from experiments.learned_presence.utils import snippet


def test_learned_presence_metrics_include_balanced_views():
    result = _metrics([False, False, True, True], [False, True, True, False])

    assert result["confusion_matrix"]["rows_true"] == {
        "false": {"predicted_false": 1, "predicted_true": 1},
        "true": {"predicted_false": 1, "predicted_true": 1},
    }
    assert result["accuracy"] == 0.5
    assert result["macro_f1"] == 0.5
    assert set(result["per_class"]) == {"negative", "positive"}


def test_error_snippet_redacts_ids_and_numeric_values():
    redacted = snippet("BDMAP_00012345: lesion measures 2.5 cm.")

    assert "BDMAP" not in redacted
    assert "12345" not in redacted
    assert "2.5" not in redacted
    assert "[ID]" in redacted
    assert "[NUM]" in redacted
