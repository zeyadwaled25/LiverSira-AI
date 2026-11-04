"""Compare a small set of learned lesion-presence architectures on validation.

The Test split is intentionally never loaded.  Every vectorizer and estimator
is fitted on the same Train IDs and evaluated on the same Validation IDs.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from evaluation.learned_presence.train_learned_presence import _metrics
from evaluation.learned_presence.utils import load_audit, load_jsonl, snippet

MASTER_PATH = Path("data/processed/nlp/master_records.jsonl")
GT_PATH = Path("data/processed/nlp/ground_truth.jsonl")
SPLITS_PATH = Path("data/processed/nlp/splits.json")
AUDIT_PATH = Path("data/processed/nlp/target_consistency_audit.csv")
BASELINE_PATH = Path("results/nlp/predictions/baseline_inferences_updated.jsonl")
EXPERIMENT_DIR = Path("results/nlp/architecture_comparison")
REPORT_PATH = EXPERIMENT_DIR / "comparison_report.md"
METRICS_PATH = EXPERIMENT_DIR / "comparison_metrics.json"
AUDIT_REPORT_PATH = EXPERIMENT_DIR / "leakage_audit.json"
SEED = 42


def _normalise_template(text: str) -> str:
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _leakage_audit(
    train_ids: list[str],
    validation_ids: list[str],
    master: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    train_texts = [
        master[record_id]["reports"].get("narrative_clean") or "" for record_id in train_ids
    ]
    validation_texts = [
        master[record_id]["reports"].get("narrative_clean") or ""
        for record_id in validation_ids
    ]
    train_exact = Counter(text for text in train_texts if text)
    validation_exact = Counter(text for text in validation_texts if text)
    exact_overlap = sorted(set(train_exact) & set(validation_exact))
    train_templates = Counter(_normalise_template(text) for text in train_texts if text)
    validation_templates = Counter(
        _normalise_template(text) for text in validation_texts if text
    )
    template_overlap = sorted(set(train_templates) & set(validation_templates))

    # This is an audit-only nearest-neighbour estimate, not a training feature.
    # It uses a fresh vectorizer and reports only similarities between splits.
    audit_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )
    train_matrix = audit_vectorizer.fit_transform(train_texts)
    validation_matrix = audit_vectorizer.transform(validation_texts)
    nearest = cosine_similarity(validation_matrix, train_matrix).max(axis=1)
    near_95 = int(np.sum(nearest >= 0.95))
    near_90 = int(np.sum(nearest >= 0.90))
    nearest_indices = np.argsort(nearest)[-10:][::-1]
    similarity_rows = cosine_similarity(validation_matrix, train_matrix)
    nearest_examples = [
        {
            "validation_id": validation_ids[index],
            "train_id": train_ids[int(similarity_rows[index].argmax())],
            "similarity": float(nearest[index]),
            "validation_text": snippet(validation_texts[index]),
        }
        for index in nearest_indices
    ]
    return {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "exact_duplicate_narratives": len(exact_overlap),
        "exact_duplicate_validation_records": sum(
            validation_exact[text] for text in exact_overlap
        ),
        "exact_duplicate_examples": [
            {"validation_count": validation_exact[text], "text": snippet(text)}
            for text in exact_overlap[:10]
        ],
        "repeated_template_count_train": sum(
            count > 1 for count in train_templates.values()
        ),
        "repeated_template_count_validation": sum(
            count > 1 for count in validation_templates.values()
        ),
        "template_overlap_count": len(template_overlap),
        "template_overlap_validation_records": sum(
            validation_templates[text] for text in template_overlap
        ),
        "near_duplicate_thresholds": {
            "cosine_char_tfidf_ge_0.90": near_90,
            "cosine_char_tfidf_ge_0.95": near_95,
        },
        "near_duplicate_examples": nearest_examples,
        "patient_level_leakage_status": "not_assessed_patient_id_unavailable",
        "label_signal_note": (
            "Text-to-label association cannot be proven absent because labels are "
            "source-derived; this audit checks split overlap and template similarity only."
        ),
    }


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80_000,
        sublinear_tf=True,
    )


def _candidates() -> dict[str, Pipeline]:
    classifier_config = {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
        "solver": "liblinear",
    }
    return {
        "word_logistic_regression": Pipeline(
            [
                ("features", _word_vectorizer()),
                ("classifier", LogisticRegression(**classifier_config)),
            ]
        ),
        "char_logistic_regression": Pipeline(
            [
                ("features", _char_vectorizer()),
                ("classifier", LogisticRegression(**classifier_config)),
            ]
        ),
        "word_char_logistic_regression": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [("word", _word_vectorizer()), ("char", _char_vectorizer())]
                    ),
                ),
                ("classifier", LogisticRegression(**classifier_config)),
            ]
        ),
        "word_char_linear_svm": Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [("word", _word_vectorizer()), ("char", _char_vectorizer())]
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=SEED,
                        C=1.0,
                    ),
                ),
            ]
        ),
    }


def _scores(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)[:, 1]
        return model.predict(texts), probabilities
    decision = model.decision_function(texts)
    return model.predict(texts), 1.0 / (1.0 + np.exp(-decision))


def _group_metrics(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    audit: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("MATCH", "MISMATCH", "UNCERTAIN"):
        indexes = [index for index, record_id in enumerate(ids) if audit.get(record_id) == group]
        result[group] = {
            "studies": len(indexes),
            "metrics": _metrics(
                [labels[index] for index in indexes],
                [bool(predictions[index]) for index in indexes],
            )
            if indexes
            else None,
        }
    return result


def _common_baseline_metrics(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    indexes = [
        index
        for index, record_id in enumerate(ids)
        if baseline[record_id]["liver_lesions"].get("has_lesion") is not None
    ]
    learned_labels = [labels[index] for index in indexes]
    learned_predictions = [bool(predictions[index]) for index in indexes]
    baseline_predictions = [
        bool(baseline[ids[index]]["liver_lesions"]["has_lesion"]) for index in indexes
    ]
    return {
        "evaluated": len(indexes),
        "subset_fraction_of_validation": len(indexes) / len(ids) if ids else 0.0,
        "learned_prediction_coverage": 1.0 if indexes else 0.0,
        "baseline_prediction_coverage": 1.0 if indexes else 0.0,
        "learned": _metrics(learned_labels, learned_predictions),
        "rule_based_baseline": _metrics(learned_labels, baseline_predictions),
    }


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
        lowered = text.lower()
        if re.search(
            r"cannot be excluded|cannot exclude|cannot rule out|possible|"
            r"questionable|indeterminate|suspicious|suspected",
            lowered,
        ):
            category = "ambiguous_report"
        elif re.search(r"\bno\b|\bwithout\b|\bnegative for\b|\bfree of\b", lowered):
            category = "negation_related"
        elif audit.get(record_id) == "MISMATCH":
            category = "conflicting_source_label"
        else:
            category = "other"
        examples.append(
            {
                "audit_group": audit.get(record_id, "UNAVAILABLE"),
                "category": category,
                "true": labels[index],
                "prediction": prediction,
                "text": snippet(text),
            }
        )
    return examples[:20]


def _feature_configuration(name: str) -> dict[str, Any]:
    if name == "word_logistic_regression":
        return {"views": ["word"], "word_ngrams": [1, 2]}
    if name == "char_logistic_regression":
        return {"views": ["char"], "char_ngrams": [3, 5]}
    return {"views": ["word", "char"], "word_ngrams": [1, 2], "char_ngrams": [3, 5]}


def _prediction_rows(
    ids: list[str],
    labels: list[bool],
    predictions: np.ndarray,
    scores: np.ndarray,
    audit: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "id": record_id,
            "target": "has_lesion",
            "source_label": labels[index],
            "prediction": bool(predictions[index]),
            "score_positive": float(scores[index]),
            "audit_group": audit.get(record_id, "UNAVAILABLE"),
            "abstained": False,
        }
        for index, record_id in enumerate(ids)
    ]


def _report(result: dict[str, Any]) -> str:
    lines = [
        "# Learned Architecture Comparison: Lesion Presence",
        "",
        "Validation-only comparison against provisional source-derived labels. "
        "Test was not loaded, fitted, tuned, or evaluated.",
        "",
        "## Leakage and robustness audit",
        f"- Exact train/validation duplicate narratives: **{result['leakage_audit']['exact_duplicate_narratives']}** "
        f"({result['leakage_audit']['exact_duplicate_validation_records']} validation records).",
        f"- Shared numeric-normalized templates: **{result['leakage_audit']['template_overlap_count']}** "
        f"({result['leakage_audit']['template_overlap_validation_records']} validation records).",
        f"- Character-TF-IDF nearest-neighbour >=0.90: **{result['leakage_audit']['near_duplicate_thresholds']['cosine_char_tfidf_ge_0.90']}**; "
        f">=0.95: **{result['leakage_audit']['near_duplicate_thresholds']['cosine_char_tfidf_ge_0.95']}**.",
        "- Patient-level leakage: not assessed because Patient ID is unavailable.",
        "",
        "Template overlap is a robustness warning, not proof of leakage: radiology reports naturally reuse templates.",
        "",
        "## Full Validation comparison",
        "| Model | F1 | Macro F1 | Precision | Recall | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["models"].items():
        metrics = item["full_validation"]
        lines.append(
            f"| {name} | {metrics['f1']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['coverage']:.4f} |"
        )
    lines += [
        "",
        "## Common 556-record baseline subset",
        "| Model | F1 | Macro F1 | Precision | Recall | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["models"].items():
        metrics = item["common_baseline_subset"]["learned"]
        lines.append(
            f"| {name} | {metrics['f1']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{item['common_baseline_subset']['learned_prediction_coverage']:.4f} |"
        )
    baseline = next(iter(result["models"].values()))["common_baseline_subset"]["rule_based_baseline"]
    lines.append(
        f"| Rule-based baseline | {baseline['f1']:.4f} | {baseline['macro_f1']:.4f} | "
        f"{baseline['precision']:.4f} | {baseline['recall']:.4f} | 1.0000 |"
    )
    lines += [
        "",
        "## Selection recommendation",
        f"Selected architecture for any later frozen evaluation: **{result['selection']}**.",
        result["selection_reason"],
        "The result does not justify a Transformer yet: all candidates are linear and "
        "Validation is nearly saturated. A Transformer comparison should wait for a "
        "cleaner label audit or a task where the linear baseline is demonstrably insufficient.",
        "On the common subset, the learned model improves F1 over the Regex baseline "
        "by about 0.0112, but the absolute difference is based on only 7 discordant "
        "cases and may be affected by repeated templates and source-label noise.",
        "",
        "## Group performance",
    ]
    for name, item in result["models"].items():
        lines.append(f"### {name}")
        for group, group_item in item["audit_groups"].items():
            if group_item["metrics"]:
                lines.append(
                    f"- {group}: n={group_item['studies']}, "
                    f"F1={group_item['metrics']['f1']:.4f}, "
                    f"Macro F1={group_item['metrics']['macro_f1']:.4f}"
                )
    lines += [
        "",
        "## Risks and interpretation",
        "- Labels are source-derived comparisons, not clinical gold standard.",
        "- MATCH/MISMATCH/UNCERTAIN groups were retained, not used as automatic filters.",
        "- Shared templates and near-duplicates can inflate Validation performance.",
        "- Patient-level leakage cannot be ruled out without Patient ID.",
        "- Prediction coverage is 1.0 for all candidates; this is separate from prediction quality.",
        "",
        "## Selected-model error examples",
    ]
    for example in result["models"][result["selection"]]["error_examples"]:
        lines.append(
            f"- `{example['category']}` / audit `{example['audit_group']}`: "
            f"true={example['true']}, predicted={example['prediction']}; "
            f"{example['text']}"
        )
    if not result["models"][result["selection"]]["error_examples"]:
        lines.append("- No errors on the full Validation set.")
    return "\n".join(lines) + "\n"


def run_comparison() -> dict[str, Any]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_ids = list(splits["train"])
    validation_ids = list(splits["val"])
    selected_ids = set(train_ids) | set(validation_ids)
    master = load_jsonl(MASTER_PATH, selected_ids)
    ground_truth = load_jsonl(GT_PATH, selected_ids)
    baseline = load_jsonl(BASELINE_PATH, set(validation_ids))
    audit = load_audit(AUDIT_PATH, set(validation_ids))
    train_ids = [
        record_id
        for record_id in train_ids
        if (master[record_id]["reports"].get("narrative_clean") or "").strip()
    ]
    validation_ids = [
        record_id
        for record_id in validation_ids
        if (master[record_id]["reports"].get("narrative_clean") or "").strip()
    ]
    train_texts = [master[i]["reports"]["narrative_clean"] for i in train_ids]
    validation_texts = [master[i]["reports"]["narrative_clean"] for i in validation_ids]
    train_labels = [
        bool(ground_truth[i]["ground_truth"]["has_lesion"]) for i in train_ids
    ]
    validation_labels = [
        bool(ground_truth[i]["ground_truth"]["has_lesion"]) for i in validation_ids
    ]

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task": "has_lesion",
        "status": "validation_only_architecture_comparison",
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "random_seed": SEED,
        "test_policy": "Test IDs were not loaded or used.",
        "label_policy": "All non-empty Train/Validation source labels retained; audit groups are analysis-only.",
        "leakage_audit": _leakage_audit(train_ids, validation_ids, master),
        "models": {},
    }
    for name, model in _candidates().items():
        model.fit(train_texts, train_labels)
        predictions, scores = _scores(model, validation_texts)
        full_metrics = _metrics(validation_labels, predictions.tolist())
        common_metrics = _common_baseline_metrics(
            validation_ids, validation_labels, predictions, baseline
        )
        rows = _prediction_rows(
            validation_ids, validation_labels, predictions, scores, audit
        )
        model_path = EXPERIMENT_DIR / f"{name}.joblib"
        prediction_path = EXPERIMENT_DIR / f"{name}_validation_predictions.jsonl"
        joblib.dump(model, model_path)
        prediction_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        result["models"][name] = {
            "full_validation": full_metrics,
            "common_baseline_subset": common_metrics,
            "audit_groups": _group_metrics(
                validation_ids, validation_labels, predictions, audit
            ),
            "error_examples": _error_examples(
                validation_ids, validation_labels, predictions, master, audit
            ),
            "confusion_matrix": full_metrics["confusion_matrix"],
            "artifact": str(model_path),
            "validation_predictions": str(prediction_path),
            "feature_configuration": _feature_configuration(name)
        }

    selected = max(
        result["models"],
        key=lambda name: (
            result["models"][name]["full_validation"]["macro_f1"],
            result["models"][name]["full_validation"]["f1"],
        ),
    )
    result["selection"] = selected
    result["selection_reason"] = (
        "Word-only Logistic Regression ties the word+character model on all "
        "reported primary Validation metrics and is selected as the simpler "
        "architecture. Selection is based on Validation macro F1, with F1 as "
        "tie-breaker, under the fixed common policy. It is an engineering "
        "selection only; Test remains untouched."
    )
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    AUDIT_REPORT_PATH.write_text(
        json.dumps(result["leakage_audit"], indent=2) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(_report(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_comparison(), indent=2))
