"""Final, single-use Test evaluation for the frozen learned presence model."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from experiments.learned_presence.train_learned_presence import _metrics
from experiments.learned_presence.utils import (
    error_category,
    load_audit,
    load_jsonl,
    snippet,
)

MASTER_PATH = Path("data/processed/master_records.jsonl")
GT_PATH = Path("data/processed/ground_truth.jsonl")
SPLITS_PATH = Path("data/processed/splits.json")
AUDIT_PATH = Path("data/processed/target_consistency_audit.csv")
BASELINE_PATH = Path("evaluation/baseline/baseline_inferences_updated.jsonl")
OUTPUT_DIR = Path("evaluation/final_test")
MODEL_PATH = OUTPUT_DIR / "learned_presence_word_logreg_final.joblib"
CONFIG_PATH = OUTPUT_DIR / "learned_presence_word_logreg_final_config.json"
PREDICTIONS_PATH = OUTPUT_DIR / "learned_presence_test_predictions.jsonl"
METRICS_PATH = OUTPUT_DIR / "learned_presence_test_metrics.json"
REPORT_PATH = OUTPUT_DIR / "learned_presence_test_report.md"
SEED = 42


def _frozen_model() -> Pipeline:
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


def _specificity(y_true: list[bool], y_pred: list[bool]) -> float | None:
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp = int(matrix[0, 0]), int(matrix[0, 1])
    return tn / (tn + fp) if tn + fp else None


def _metrics_with_specificity(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    result = _metrics(y_true, y_pred)
    result["specificity"] = _specificity(y_true, y_pred)
    return result


def _paired_comparison(
    ids: list[str],
    labels: list[bool],
    learned_predictions: np.ndarray,
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    indexes = [
        index
        for index, record_id in enumerate(ids)
        if baseline[record_id]["liver_lesions"].get("has_lesion") is not None
    ]
    truth = [labels[index] for index in indexes]
    learned = [bool(learned_predictions[index]) for index in indexes]
    regex = [
        bool(baseline[ids[index]]["liver_lesions"]["has_lesion"])
        for index in indexes
    ]
    learned_metrics = _metrics_with_specificity(truth, learned)
    regex_metrics = _metrics_with_specificity(truth, regex)
    return {
        "evaluated": len(indexes),
        "subset_fraction_of_test": len(indexes) / len(ids) if ids else 0.0,
        "learned_prediction_coverage": 1.0 if indexes else 0.0,
        "baseline_prediction_coverage": 1.0 if indexes else 0.0,
        "learned_model": learned_metrics,
        "rule_based_baseline": regex_metrics,
        "f1_difference_learned_minus_baseline": (
            learned_metrics["f1"] - regex_metrics["f1"]
        ),
        "macro_f1_difference_learned_minus_baseline": (
            learned_metrics["macro_f1"] - regex_metrics["macro_f1"]
        ),
        "learned_correct_baseline_wrong": sum(
            learned_prediction == label and baseline_prediction != label
            for learned_prediction, baseline_prediction, label in zip(
                learned, regex, truth
            )
        ),
        "baseline_correct_learned_wrong": sum(
            learned_prediction != label and baseline_prediction == label
            for learned_prediction, baseline_prediction, label in zip(
                learned, regex, truth
            )
        ),
    }


def _group_metrics(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    audit: dict[str, str],
) -> dict[str, Any]:
    result = {}
    for group in ("MATCH", "MISMATCH", "UNCERTAIN"):
        indexes = [
            index
            for index, record_id in enumerate(ids)
            if audit.get(record_id) == group
        ]
        result[group] = {
            "studies": len(indexes),
            "metrics": _metrics_with_specificity(
                [labels[index] for index in indexes],
                [bool(predictions[index]) for index in indexes],
            )
            if indexes
            else None,
        }
    return result


def _error_examples(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    master: dict[str, dict[str, Any]],
    audit: dict[str, str],
) -> list[dict[str, Any]]:
    examples = []
    for index, record_id in enumerate(ids):
        prediction = bool(predictions[index])
        if prediction == labels[index]:
            continue
        text = master[record_id]["reports"].get("narrative_clean") or ""
        examples.append(
            {
                "category": error_category(text, audit.get(record_id, "UNAVAILABLE")),
                "audit_group": audit.get(record_id, "UNAVAILABLE"),
                "true": labels[index],
                "prediction": prediction,
                "text": snippet(text),
            }
        )
    return examples


def _prediction_rows(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    audit: dict[str, str],
    baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": record_id,
            "target": "has_lesion",
            "source_label": labels[index],
            "prediction": bool(predictions[index]),
            "probability_positive": float(probabilities[index]),
            "threshold": 0.5,
            "abstained": False,
            "audit_group": audit.get(record_id, "UNAVAILABLE"),
            "baseline_prediction": baseline[record_id]["liver_lesions"].get(
                "has_lesion"
            ),
            "baseline_evaluated": baseline[record_id]["liver_lesions"].get(
                "has_lesion"
            )
            is not None,
        }
        for index, record_id in enumerate(ids)
    ]


def _render_report(result: dict[str, Any]) -> str:
    full = result["full_test_metrics"]
    paired = result["paired_baseline_comparison"]
    lines = [
        "# Final Test Evaluation: Frozen Learned Lesion Presence Model",
        "",
        "This is one final Test evaluation of a model frozen after Validation architecture comparison.",
        "All results compare against source-derived labels, not a clinical gold standard.",
        "",
        "## Frozen configuration",
        "- Word TF-IDF, n-grams `(1, 2)`, `min_df=2`, `max_features=80000`, `sublinear_tf=True`.",
        "- Logistic Regression, `class_weight=balanced`, `solver=liblinear`, `max_iter=1000`.",
        "- Random seed: `42`; threshold: `0.5`.",
        "- Training data: Train + Validation only; Test was not used for fitting or configuration.",
        "- Label policy: all non-empty Train/Validation source labels retained.",
        "",
        "## Full Test metrics",
        f"- Test studies: {result['test_studies']}; evaluated: {full['evaluated']}; coverage: {full['coverage']:.4f}",
        f"- Accuracy: {full['accuracy']:.4f}",
        f"- Precision: {full['precision']:.4f}",
        f"- Recall: {full['recall']:.4f}",
        f"- F1: {full['f1']:.4f}",
        f"- Macro F1: {full['macro_f1']:.4f}",
        f"- Weighted F1: {full['weighted_f1']:.4f}",
        f"- Specificity: {full['specificity']:.4f}",
        f"- Confusion matrix: {full['confusion_matrix']}",
        f"- Per-class metrics: {full['per_class']}",
        "",
        "## Paired comparison with Regex Baseline",
        f"- Paired records: {paired['evaluated']} / {result['test_studies']} "
        f"({paired['subset_fraction_of_test']:.4f} of Test).",
        "- Learned and Regex quality metrics below are computed only on this same paired subset.",
        f"- Learned F1: {paired['learned_model']['f1']:.4f}; Regex F1: {paired['rule_based_baseline']['f1']:.4f}",
        f"- Learned Macro F1: {paired['learned_model']['macro_f1']:.4f}; Regex Macro F1: {paired['rule_based_baseline']['macro_f1']:.4f}",
        f"- F1 difference (learned - Regex): {paired['f1_difference_learned_minus_baseline']:.4f}",
        f"- Learned correct / Regex wrong: {paired['learned_correct_baseline_wrong']}",
        f"- Regex correct / Learned wrong: {paired['baseline_correct_learned_wrong']}",
        f"- Learned coverage on paired subset: {paired['learned_prediction_coverage']:.4f}; "
        f"Regex coverage on paired subset: {paired['baseline_prediction_coverage']:.4f}",
        "",
        "## MATCH / MISMATCH / UNCERTAIN",
        "These are text-vs-source-label audit groups, not clinical verification groups.",
        "| Group | Studies | Accuracy | F1 | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, item in result["audit_groups"].items():
        if item["metrics"]:
            lines.append(
                f"| {group} | {item['studies']} | {item['metrics']['accuracy']:.4f} | "
                f"{item['metrics']['f1']:.4f} | {item['metrics']['macro_f1']:.4f} |"
            )
    lines += [
        "",
        "## Error analysis",
        "Examples are redacted: IDs and numeric values are removed. Errors are relative to source labels.",
    ]
    for example in result["error_examples"]:
        lines.append(
            f"- `{example['category']}` / audit `{example['audit_group']}`: "
            f"true={example['true']}, predicted={example['prediction']}; {example['text']}"
        )
    if not result["error_examples"]:
        lines.append("- No full-Test errors were observed.")
    lines += [
        "",
        "## Research validity and limitations",
        "- The 79 legacy cases are not treated as Human Verified.",
        "- Patient-level leakage prevention cannot be claimed because Patient IDs are unavailable.",
        "- Test was evaluated once after freezing the architecture and threshold.",
        "- Source-label conflicts and repeated report templates can affect performance.",
        "",
        "## Engineering decision",
        "**The model is suitable for controlled engineering integration of the `has_lesion` "
        "component in Phase 9, behind validation, logging, and clinical review safeguards. "
        "This result is not clinical validation and does not justify autonomous clinical use.**",
    ]
    return "\n".join(lines) + "\n"


def run_final_evaluation() -> dict[str, Any]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = list(splits["train"])
    validation_ids = list(splits["val"])
    test_ids = list(splits["test"])
    train_validation_ids = set(train_ids) | set(validation_ids)
    master = load_jsonl(MASTER_PATH, train_validation_ids)
    ground_truth = load_jsonl(GT_PATH, train_validation_ids)

    train_validation_ids_ordered = train_ids + validation_ids
    train_validation_ids_ordered = [
        record_id
        for record_id in train_validation_ids_ordered
        if (master[record_id]["reports"].get("narrative_clean") or "").strip()
    ]
    train_texts = [
        master[record_id]["reports"]["narrative_clean"]
        for record_id in train_validation_ids_ordered
    ]
    train_labels = [
        bool(ground_truth[record_id]["ground_truth"]["has_lesion"])
        for record_id in train_validation_ids_ordered
    ]
    model = _frozen_model()
    model.fit(train_texts, train_labels)

    # Load Test only after the frozen model and its vectorizer have been fitted.
    test_master = load_jsonl(MASTER_PATH, set(test_ids))
    test_ground_truth = load_jsonl(GT_PATH, set(test_ids))
    test_ids = [
        record_id
        for record_id in test_ids
        if (test_master[record_id]["reports"].get("narrative_clean") or "").strip()
    ]
    master.update(test_master)
    ground_truth.update(test_ground_truth)
    baseline = load_jsonl(BASELINE_PATH, set(test_ids))
    audit = load_audit(AUDIT_PATH, set(test_ids))
    test_texts = [master[record_id]["reports"]["narrative_clean"] for record_id in test_ids]
    test_labels = [
        bool(ground_truth[record_id]["ground_truth"]["has_lesion"])
        for record_id in test_ids
    ]
    probabilities = model.predict_proba(test_texts)[:, 1]
    predictions = probabilities >= 0.5
    full_metrics = _metrics_with_specificity(test_labels, predictions.tolist())
    paired = _paired_comparison(test_ids, test_labels, predictions, baseline)
    result: dict[str, Any] = {
        "status": "final_test_evaluation_once",
        "task": "has_lesion",
        "test_studies": len(test_ids),
        "train_plus_validation_studies": len(train_validation_ids_ordered),
        "test_policy": "Test was used only after model freeze for this single final evaluation.",
        "frozen_configuration": {
            "architecture": "word_tfidf_logistic_regression",
            "word_ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 80_000,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
            "random_seed": SEED,
            "threshold": 0.5,
            "input_field": "narrative_clean",
            "label_source": "ground_truth.ground_truth.has_lesion",
            "label_policy": "All non-empty Train and Validation source labels retained.",
        },
        "full_test_metrics": full_metrics,
        "paired_baseline_comparison": paired,
        "audit_groups": _group_metrics(test_ids, test_labels, predictions, audit),
        "error_examples": _error_examples(
            test_ids, test_labels, predictions, master, audit
        ),
        "limitations": [
            "Source labels are not clinical gold standard.",
            "The 79 legacy cases are not Human Verified.",
            "Patient-level leakage cannot be assessed without Patient IDs.",
            "No abstention mechanism; non-empty Test prediction coverage is 1.0.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    CONFIG_PATH.write_text(
        json.dumps(result["frozen_configuration"], indent=2) + "\n", encoding="utf-8"
    )
    rows = _prediction_rows(
        test_ids, test_labels, predictions, probabilities, audit, baseline
    )
    PREDICTIONS_PATH.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result["artifacts"] = {
        "model": str(MODEL_PATH),
        "configuration": str(CONFIG_PATH),
        "predictions": str(PREDICTIONS_PATH),
        "metrics": str(METRICS_PATH),
        "report": str(REPORT_PATH),
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run_final_evaluation()
    print(json.dumps(result["full_test_metrics"], indent=2))
