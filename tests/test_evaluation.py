from scripts.nlp.evaluate_baseline import evaluate_split


def _record(record_id, has_lesion, count, largest=None, missing=None):
    return {
        "id": record_id,
        "ground_truth": {
            "has_lesion": has_lesion,
            "lesion_count": count,
            "largest_lesion": largest,
        },
    }


def _inference(record_id, has_lesion, count, largest=None, missing=None):
    return {
        "id": record_id,
        "liver_lesions": {
            "has_lesion": has_lesion,
            "lesion_count": count,
            "largest_lesion": largest,
        },
        "evidence_spans": [],
        "missing_or_unclear": missing or [],
    }


def test_presence_metrics_include_specificity_and_confusion_matrix():
    gt = {
        "n": _record("n", False, 0),
        "p": _record("p", True, 1, {"location_segments": [2], "source_diameter_cm": 2.0, "attenuation": "hypoattenuating"}),
    }
    inf = {
        "n": _inference("n", False, 0),
        "p": _inference("p", True, 1, {"location_segments": [2], "derived_max_diameter_cm": 2.0, "attenuation": "hypoattenuating"}),
    }

    result = evaluate_split(gt, inf, ["n", "p"], "test")

    assert result["presence"]["confusion_matrix"]["rows_true"] == {
        "false": {"predicted_false": 1, "predicted_true": 0},
        "true": {"predicted_false": 0, "predicted_true": 1},
    }
    assert result["presence"]["specificity"] == 1.0
    assert result["presence"]["f1"] == 1.0


def test_abstentions_are_excluded_and_reported_not_scored_as_zero():
    gt = {
        "p": _record("p", True, 2, {"location_segments": [4], "source_diameter_cm": 3.0, "attenuation": "hypoattenuating"}),
    }
    inf = {
        "p": _inference(
            "p",
            None,
            None,
            None,
            [{"field": "liver_lesions.has_lesion", "reason": "ambiguous"}],
        ),
    }

    result = evaluate_split(gt, inf, ["p"], "test")

    assert result["presence"]["evaluated"] == 0
    assert result["presence"]["accuracy"] is None
    assert result["per_field"]["has_lesion"]["exclusion_reasons"] == {
        "prediction_uncertain": 1
    }
    assert result["per_field"]["lesion_count"]["exclusion_reasons"] == {
        "prediction_missing": 1
    }


def test_diameter_uses_source_diameter_against_derived_maximum():
    gt = {
        "p": _record("p", True, 1, {"location_segments": [1], "source_diameter_cm": 4.0, "attenuation": "hyperattenuating"}),
    }
    inf = {
        "p": _inference("p", True, 1, {"location_segments": [1], "derived_max_diameter_cm": 3.5, "attenuation": "hyperattenuating"}),
    }

    result = evaluate_split(gt, inf, ["p"], "test")

    assert result["diameter"]["mae_cm"] == 0.5
    assert result["diameter"]["tolerance_accuracy"] == 1.0
