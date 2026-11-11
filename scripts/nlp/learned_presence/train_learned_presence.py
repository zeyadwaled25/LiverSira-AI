"""Train and evaluate the first learned model for lesion presence.

This experiment uses only ``narrative_clean`` as input and only train/validation
IDs.  Source labels are used as provisional labels; audit groups are reported
for analysis and are not silently filtered from training.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from liversira_ai.nlp.learned_presence import (
    classification_metrics as _metrics,
    error_category,
    load_audit,
    load_jsonl,
    snippet,
)

MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
GT_PATH = Path("data/processed/nlp/ground_truth.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
AUDIT_PATH = Path("data/processed/nlp/target_consistency_audit.csv")
BASELINE_PATH = Path("results/nlp/predictions/baseline_inferences_updated.jsonl")
ARTIFACT_DIR = Path("results/nlp/model_artifacts")
MODEL_PATH = ARTIFACT_DIR / "learned_presence_tfidf_logreg.joblib"
CONFIG_PATH = ARTIFACT_DIR / "learned_presence_config.json"
PREDICTIONS_PATH = Path("results/nlp/predictions/learned_presence_validation_predictions.jsonl")
METRICS_PATH = Path("results/nlp/learned_presence_validation/learned_presence_validation_metrics.json")
REPORT_PATH = Path("results/nlp/learned_presence_validation/learned_presence_validation_report.md")
SEED = 42


def _write_predictions(
    ids: list[str],
    master: dict[str, dict[str, Any]],
    truth: list[bool],
    predicted: list[bool],
    probabilities: np.ndarray,
    baseline: dict[str, dict[str, Any]],
    audit: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, record_id in enumerate(ids):
        baseline_value = baseline[record_id]["liver_lesions"].get("has_lesion")
        records.append(
            {
                "id": record_id,
                "target": "has_lesion",
                "source_label": truth[index],
                "prediction": predicted[index],
                "probability_positive": float(probabilities[index]),
                "abstained": False,
                "audit_group": audit.get(record_id, "UNAVAILABLE"),
                "baseline_prediction": baseline_value,
                "baseline_evaluated": baseline_value is not None,
            }
        )
    PREDICTIONS_PATH.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return records


def _compare_baseline(
    records: list[dict[str, Any]],
    master: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common = [record for record in records if record["baseline_evaluated"]]
    learned_true = [record["source_label"] for record in common]
    learned_pred = [record["prediction"] for record in common]
    baseline_pred = [record["baseline_prediction"] for record in common]
    result: dict[str, Any] = {
        "common_coverage": len(common) / len(records) if records else 0.0,
        "common_evaluated": len(common),
        "learned_model": _metrics(learned_true, learned_pred),
        "rule_based_baseline": _metrics(learned_true, baseline_pred),
        "learned_correct_baseline_wrong": 0,
        "baseline_correct_learned_wrong": 0,
    }
    examples: list[dict[str, Any]] = []
    for record in common:
        learned_correct = record["prediction"] == record["source_label"]
        baseline_correct = record["baseline_prediction"] == record["source_label"]
        if learned_correct and not baseline_correct:
            result["learned_correct_baseline_wrong"] += 1
            winner = "learned_model"
        elif baseline_correct and not learned_correct:
            result["baseline_correct_learned_wrong"] += 1
            winner = "rule_based_baseline"
        else:
            continue
        examples.append(
            {
                "winner": winner,
                "audit_group": record["audit_group"],
                "true": record["source_label"],
                "learned_prediction": record["prediction"],
                "baseline_prediction": record["baseline_prediction"],
                "category": error_category(
                    master[record["id"]]["reports"].get("narrative_clean") or "",
                    record["audit_group"],
                ),
                "text": snippet(
                    master[record["id"]]["reports"].get("narrative_clean") or ""
                ),
            }
        )
    result["examples"] = examples[:40]
    return result


def _group_metrics(
    records: list[dict[str, Any]], group: str
) -> dict[str, Any]:
    subset = [record for record in records if record["audit_group"] == group]
    return {
        "studies": len(subset),
        "metrics": _metrics(
            [record["source_label"] for record in subset],
            [record["prediction"] for record in subset],
        )
        if subset
        else None,
    }


def train_experiment() -> dict[str, Any]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    selected_ids = set(splits["train"]) | set(splits["val"])
    master = load_jsonl(MASTER_PATH, selected_ids)
    ground_truth = load_jsonl(GT_PATH, selected_ids)
    baseline = load_jsonl(BASELINE_PATH, set(splits["val"]))
    audit = load_audit(AUDIT_PATH, set(splits["val"]))
    train_ids = splits["train"]
    validation_ids = splits["val"]

    train_ids = [
        record_id
        for record_id in train_ids
        if (master[record_id].get("reports", {}).get("narrative_clean") or "").strip()
    ]
    validation_ids = [
        record_id
        for record_id in validation_ids
        if (master[record_id].get("reports", {}).get("narrative_clean") or "").strip()
    ]
    train_texts = [
        master[record_id]["reports"]["narrative_clean"] for record_id in train_ids
    ]
    train_labels = [
        bool(ground_truth[record_id]["ground_truth"]["has_lesion"])
        for record_id in train_ids
    ]
    validation_texts = [
        master[record_id]["reports"]["narrative_clean"] for record_id in validation_ids
    ]
    validation_labels = [
        bool(ground_truth[record_id]["ground_truth"]["has_lesion"])
        for record_id in validation_ids
    ]

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    model = Pipeline(
        [
            ("features", features),
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
    model.fit(train_texts, train_labels)
    probabilities = model.predict_proba(validation_texts)[:, 1]
    predictions = probabilities >= 0.5
    records = _write_predictions(
        validation_ids,
        master,
        validation_labels,
        predictions.tolist(),
        probabilities,
        baseline,
        audit,
    )
    validation_metrics = _metrics(validation_labels, predictions.tolist())
    comparison = _compare_baseline(records, master)
    group_metrics = {
        group: _group_metrics(records, group)
        for group in ("MATCH", "MISMATCH", "UNCERTAIN")
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    config = {
        "task": "has_lesion",
        "input_field": "narrative_clean",
        "label_source": "ground_truth.ground_truth.has_lesion",
        "label_policy": "Use all non-empty train labels; source labels are provisional, and audit groups are retained for analysis.",
        "audit_policy": "MATCH/MISMATCH/UNCERTAIN are not used as automatic inclusion filters.",
        "missing_policy": "Exclude only empty narrative_clean from model fitting/evaluation.",
        "test_policy": "No Test records are loaded, fitted, tuned, or evaluated.",
        "architecture": "word+character TF-IDF FeatureUnion followed by class-weighted LogisticRegression",
        "hyperparameters": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "min_df": 2,
            "max_features_per_view": 80000,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "liblinear",
            "threshold": 0.5,
        },
        "random_seed": SEED,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "train_label_counts": dict(
            Counter("positive" if label else "negative" for label in train_labels)
        ),
        "validation_label_counts": dict(
            Counter("positive" if label else "negative" for label in validation_labels)
        ),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    result = {
        "task": "has_lesion",
        "status": "validation_only_experiment",
        "configuration": config,
        "validation_metrics": validation_metrics,
        "comparison_on_common_eligible_records": comparison,
        "audit_group_metrics": group_metrics,
        "limitations": [
            "Labels are source-derived and not clinical gold standard.",
            "The 79 legacy cases are not treated as human verified.",
            "Validation only; test set was not used.",
            "No independent evidence annotations exist.",
        ],
        "artifacts": {
            "model": str(MODEL_PATH),
            "config": str(CONFIG_PATH),
            "validation_predictions": str(PREDICTIONS_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    return result


def _render_report(result: dict[str, Any]) -> str:
    metrics = result["validation_metrics"]
    comparison = result["comparison_on_common_eligible_records"]
    lines = [
        "# First Learned NLP Experiment: Lesion Presence",
        "",
        "This is a validation-only experiment against source-derived labels, not clinical validation.",
        "The 79 legacy cases are not treated as human verified, and no Test records were used.",
        "",
        "## Selection",
        "Lesion presence was selected because it has complete binary labels and sufficient "
        "positive/negative examples. Count is imbalanced and location/diameter/attenuation "
        "are sparse or require extraction/regression, so they are deferred.",
        "",
        "## Architecture and policy",
        "- Word (1-2 gram) and character (3-5 gram) TF-IDF features.",
        "- Class-weighted Logistic Regression, seed 42, threshold 0.5.",
        "- Input: `narrative_clean` only; no structured report or ground-truth fields.",
        "- All non-empty train labels were retained. MATCH/MISMATCH/UNCERTAIN were not automatic filters.",
        "- Empty text was excluded; no Test records were loaded or used.",
        "",
        "## Validation metrics",
        f"- Coverage: {metrics['coverage']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        f"- Per class: {metrics['per_class']}",
        "",
        "## Baseline comparison",
        f"Comparison uses the {comparison['common_evaluated']} validation records where the baseline emitted a prediction "
        f"(common coverage {comparison['common_coverage']:.4f}).",
        f"- Learned F1: {comparison['learned_model']['f1']:.4f}; "
        f"baseline F1: {comparison['rule_based_baseline']['f1']:.4f}",
        f"- Learned macro F1: {comparison['learned_model']['macro_f1']:.4f}; "
        f"baseline macro F1: {comparison['rule_based_baseline']['macro_f1']:.4f}",
        f"- Learned correct / baseline wrong: {comparison['learned_correct_baseline_wrong']}",
        f"- Baseline correct / learned wrong: {comparison['baseline_correct_learned_wrong']}",
        "",
        "## Audit-group analysis",
        "| Group | Studies | Accuracy | F1 |",
        "|---|---:|---:|---:|",
    ]
    for group, item in result["audit_group_metrics"].items():
        if item["metrics"]:
            lines.append(
                f"| {group} | {item['studies']} | {item['metrics']['accuracy']:.4f} | "
                f"{item['metrics']['f1']:.4f} |"
            )
    lines += [
        "",
        "## Error analysis",
        "Examples below are redacted: IDs and numeric values are removed. "
        "They are source-label comparison errors, not clinical error adjudications.",
        "",
    ]
    for example in comparison["examples"][:20]:
        lines.append(
            f"- `{example['winner']}` / `{example['category']}` / "
            f"audit `{example['audit_group']}`: {example['text']}"
        )
    if not comparison["examples"]:
        lines.append("- No disagreement examples were found on the common subset.")
    lines += [
        "",
        "## Limitations and decision",
        "- Source labels contain 214 positive-label MISMATCH cases and 456 positive-label UNCERTAIN cases in the full dataset.",
        "- No threshold or architecture tuning was performed on Test.",
        "- This model has no abstention mechanism; its validation coverage is 1.0 on non-empty reports.",
        "- It is suitable for the next controlled experiment only if the validation comparison is reviewed; it is not a clinical model or gold-standard benchmark.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    result = train_experiment()
    print(json.dumps(result["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
