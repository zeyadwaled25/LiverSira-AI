"""Tests for bootstrap confidence intervals and coverage-adjusted metrics."""

import pytest

from liversira_ai.nlp.metrics import (
    field_report,
    format_ci,
    paired_comparison,
    presence_report,
)

# 4 positives, 6 negatives. Predictions: 2 TP, 1 abstained positive, 1 FN,
# 4 TN, 1 FP, 1 abstained negative.
TRUTH = [True, True, True, True, False, False, False, False, False, False]
PRED = [True, True, None, False, False, False, False, False, True, None]


def test_presence_views_count_abstentions_explicitly():
    report = presence_report(TRUTH, PRED, n_resamples=200)

    assert report["answered"] == 8 and report["abstained"] == 2
    assert report["coverage"]["coverage"]["estimate"] == pytest.approx(0.8)
    # Selective: TP=2, FP=1, TN=4, FN=1.
    assert report["selective"]["recall"]["estimate"] == pytest.approx(2 / 3)
    assert report["selective"]["specificity"]["estimate"] == pytest.approx(4 / 5)
    assert report["selective"]["f1"]["estimate"] == pytest.approx(4 / 6)
    # End to end: abstentions are errors.
    assert report["end_to_end"]["accuracy"]["estimate"] == pytest.approx(6 / 10)
    assert report["end_to_end"]["recall"]["estimate"] == pytest.approx(2 / 4)
    assert report["end_to_end"]["specificity"]["estimate"] == pytest.approx(4 / 6)
    # Abstain as negative: the abstained negative becomes a TN.
    assert report["abstain_as_negative"]["specificity"]["estimate"] == pytest.approx(5 / 6)
    assert report["abstain_as_negative"]["accuracy"]["estimate"] == pytest.approx(7 / 10)
    assert report["abstain_as_negative"]["f1"]["estimate"] == pytest.approx(4 / 7)


def test_confidence_interval_brackets_estimate_and_is_reproducible():
    truth = [i % 3 == 0 for i in range(300)]
    pred = [(i % 3 == 0) != (i % 17 == 0) for i in range(300)]
    first = presence_report(truth, pred, n_resamples=500, seed=7)
    second = presence_report(truth, pred, n_resamples=500, seed=7)

    f1 = first["selective"]["f1"]
    assert f1["ci_low"] <= f1["estimate"] <= f1["ci_high"]
    assert f1["ci_low"] < f1["ci_high"]
    assert first == second


def test_field_report_separates_coverage_from_mean():
    report = field_report([1.0, 0.0, None, 1.0], n_resamples=200)

    assert report["eligible"] == 4 and report["evaluated"] == 3
    assert report["coverage"]["estimate"] == pytest.approx(0.75)
    assert report["mean"]["estimate"] == pytest.approx(2 / 3)


def test_empty_inputs_return_no_estimate():
    report = field_report([], n_resamples=50)
    assert report["mean"]["estimate"] is None
    assert format_ci(report["mean"]) == "—"


def test_paired_comparison_counts_discordant_studies():
    truth = [True, True, False, False, True, False]
    first = [True, True, False, False, True, True]
    second = [True, False, False, True, False, True]
    result = paired_comparison(truth, first, second, n_resamples=200)

    assert result["first_correct_second_wrong"] == 3
    assert result["second_correct_first_wrong"] == 0
    assert result["mcnemar_exact_p"] == pytest.approx(0.25)
    assert result["accuracy_difference"]["estimate"] == pytest.approx(3 / 6)


def test_format_ci():
    assert format_ci({"estimate": 0.5, "ci_low": 0.4, "ci_high": 0.6}) == "0.500 [0.400, 0.600]"
