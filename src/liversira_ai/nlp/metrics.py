"""Reliability metrics: study-level bootstrap CIs and coverage-adjusted views.

All confidence intervals use a non-parametric percentile bootstrap over
studies (records are resampled with replacement), with a fixed seed so every
report is reproducible. Abstentions (``None`` predictions) are never dropped
silently: each presence report states coverage and three views of quality.

* ``selective``: metrics on answered studies only (what earlier reports showed).
* ``end_to_end``: every study counts; an abstention is treated as an error.
* ``abstain_as_negative``: abstentions are read as "no lesion", the closed-world
  policy a full abdominal report would imply.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE = 0.95

ABSTAIN = -1

StatFn = Callable[..., np.ndarray]


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value) + 0.0  # normalise -0.0 to 0.0


def bootstrap(
    stat: StatFn,
    *arrays: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Point estimate and percentile CI for ``stat`` over aligned record arrays.

    ``stat`` receives 2-D arrays shaped ``(replicates, records)`` and returns one
    value per replicate (``nan`` when undefined, e.g. no positives sampled).
    """
    n = len(arrays[0]) if arrays else 0
    if n == 0:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "valid_resamples": 0,
        }
    estimate = float(stat(*(array[None, :] for array in arrays))[0])
    indices = _resample_indices(n, n_resamples, seed)
    replicates = stat(*(array[indices] for array in arrays))
    valid = replicates[np.isfinite(replicates)]
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = (
        np.percentile(valid, [tail, 100.0 - tail]) if valid.size else (np.nan, np.nan)
    )
    return {
        "estimate": _round(estimate),
        "ci_low": _round(low),
        "ci_high": _round(high),
        "n": n,
        "valid_resamples": int(valid.size),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1), np.nan)


def _counts(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, ...]:
    """TP, FP, TN, FN, abstained-positive, abstained-negative per replicate."""
    positive = truth == 1
    negative = ~positive
    tp = np.sum(positive & (pred == 1), axis=-1)
    fp = np.sum(negative & (pred == 1), axis=-1)
    tn = np.sum(negative & (pred == 0), axis=-1)
    fn = np.sum(positive & (pred == 0), axis=-1)
    ap = np.sum(positive & (pred == ABSTAIN), axis=-1)
    an = np.sum(negative & (pred == ABSTAIN), axis=-1)
    return tp, fp, tn, fn, ap, an


def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _coverage(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.mean(p != ABSTAIN, axis=-1)


def _selective_accuracy(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    tp, fp, tn, fn, _, _ = _counts(t, p)
    return _ratio(tp + tn, tp + fp + tn + fn)


def _precision(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Abstentions never count as positive calls, so precision is the same
    # under the selective and abstain-as-negative views.
    tp, fp, _, _, _, _ = _counts(t, p)
    return _ratio(tp, tp + fp)


def _selective_recall(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    tp, _, _, fn, _, _ = _counts(t, p)
    return _ratio(tp, tp + fn)


def _selective_specificity(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    _, fp, tn, _, _, _ = _counts(t, p)
    return _ratio(tn, tn + fp)


def _selective_f1(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    tp, fp, _, fn, _, _ = _counts(t, p)
    return _f1(tp, fp, fn)


def _end_to_end_accuracy(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.mean(p == t, axis=-1)


def _end_to_end_recall(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Abstained positives are misses. This equals recall under the
    # abstain-as-negative policy.
    tp, _, _, fn, ap, _ = _counts(t, p)
    return _ratio(tp, tp + fn + ap)


def _end_to_end_specificity(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    _, fp, tn, _, _, an = _counts(t, p)
    return _ratio(tn, tn + fp + an)


def _negative_accuracy(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.mean(np.where(p == ABSTAIN, 0, p) == t, axis=-1)


def _negative_specificity(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    _, fp, tn, _, _, an = _counts(t, p)
    return _ratio(tn + an, tn + an + fp)


def _negative_f1(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    tp, fp, _, fn, ap, _ = _counts(t, p)
    return _f1(tp, fp, fn + ap)


PRESENCE_STATS: dict[str, dict[str, StatFn]] = {
    "coverage": {"coverage": _coverage},
    "selective": {
        "accuracy": _selective_accuracy,
        "precision": _precision,
        "recall": _selective_recall,
        "specificity": _selective_specificity,
        "f1": _selective_f1,
    },
    "end_to_end": {
        "accuracy": _end_to_end_accuracy,
        "recall": _end_to_end_recall,
        "specificity": _end_to_end_specificity,
    },
    "abstain_as_negative": {
        "accuracy": _negative_accuracy,
        "precision": _precision,
        "recall": _end_to_end_recall,
        "specificity": _negative_specificity,
        "f1": _negative_f1,
    },
}


def encode_presence(predictions: Sequence[bool | None]) -> np.ndarray:
    """Encode predictions as 1 / 0 / ``ABSTAIN``."""
    return np.array(
        [ABSTAIN if value is None else int(bool(value)) for value in predictions],
        dtype=np.int8,
    )


def presence_report(
    truth: Sequence[bool],
    predictions: Sequence[bool | None],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Coverage plus selective, end-to-end and abstain-as-negative metrics with CIs."""
    t = np.array([int(bool(value)) for value in truth], dtype=np.int8)
    p = encode_presence(predictions)
    if len(t) != len(p):
        raise ValueError("truth and predictions must have the same length")
    report: dict[str, Any] = {
        "studies": int(len(t)),
        "answered": int(np.sum(p != ABSTAIN)),
        "abstained": int(np.sum(p == ABSTAIN)),
        "positives": int(np.sum(t == 1)),
    }
    for view, stats in PRESENCE_STATS.items():
        report[view] = {
            name: bootstrap(fn, t, p, n_resamples=n_resamples, seed=seed)
            for name, fn in stats.items()
        }
    return report


def field_report(
    values: Sequence[float | None],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Coverage and mean of a per-study score over eligible studies.

    ``values`` holds one entry per eligible study: the score (1/0 for exact
    match, an absolute error, an IoU) or ``None`` when no prediction was made.
    The mean is computed over predicted studies only and is reported next to
    its coverage, so missing predictions never inflate it silently.
    """
    array = np.array([np.nan if value is None else float(value) for value in values])

    def coverage(x: np.ndarray) -> np.ndarray:
        return np.mean(np.isfinite(x), axis=-1)

    def mean(x: np.ndarray) -> np.ndarray:
        # nanmean warns on all-missing replicates; those become nan and are
        # excluded from the percentile interval.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(x, axis=-1)

    return {
        "eligible": int(len(array)),
        "evaluated": int(np.sum(np.isfinite(array))),
        "coverage": bootstrap(coverage, array, n_resamples=n_resamples, seed=seed),
        "mean": bootstrap(mean, array, n_resamples=n_resamples, seed=seed),
    }


def paired_comparison(
    truth: Sequence[bool],
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare two binary predictors on the same studies.

    Returns the bootstrap CI of the F1 and accuracy differences (first minus
    second) and an exact McNemar test on the discordant studies.
    """
    t = np.array([int(bool(v)) for v in truth], dtype=np.int8)
    a = np.array([int(bool(v)) for v in first], dtype=np.int8)
    b = np.array([int(bool(v)) for v in second], dtype=np.int8)

    def f1_diff(t_: np.ndarray, a_: np.ndarray, b_: np.ndarray) -> np.ndarray:
        ca, cb = _counts(t_, a_), _counts(t_, b_)
        return _f1(ca[0], ca[1], ca[3]) - _f1(cb[0], cb[1], cb[3])

    def accuracy_diff(t_: np.ndarray, a_: np.ndarray, b_: np.ndarray) -> np.ndarray:
        return np.mean(a_ == t_, axis=-1) - np.mean(b_ == t_, axis=-1)

    first_only = int(np.sum((a == t) & (b != t)))
    second_only = int(np.sum((a != t) & (b == t)))
    discordant = first_only + second_only
    p_value = (
        float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "studies": int(len(t)),
        "f1_difference": bootstrap(f1_diff, t, a, b, n_resamples=n_resamples, seed=seed),
        "accuracy_difference": bootstrap(
            accuracy_diff, t, a, b, n_resamples=n_resamples, seed=seed
        ),
        "first_correct_second_wrong": first_only,
        "second_correct_first_wrong": second_only,
        "mcnemar_exact_p": p_value,
    }


def format_ci(item: dict[str, Any], digits: int = 3) -> str:
    """Render ``estimate [low, high]`` for reports."""
    if item.get("estimate") is None:
        return "—"
    text = f"{item['estimate']:.{digits}f}"
    if item.get("ci_low") is not None and item.get("ci_high") is not None:
        text += f" [{item['ci_low']:.{digits}f}, {item['ci_high']:.{digits}f}]"
    return text
