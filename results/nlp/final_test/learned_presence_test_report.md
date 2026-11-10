# Final Test Evaluation: Frozen Learned Lesion Presence Model

This is one final Test evaluation of a model frozen after Validation architecture comparison.
All results compare against source-derived labels, not a clinical gold standard.

## Frozen configuration
- Word TF-IDF, n-grams `(1, 2)`, `min_df=2`, `max_features=80000`, `sublinear_tf=True`.
- Logistic Regression, `class_weight=balanced`, `solver=liblinear`, `max_iter=1000`.
- Random seed: `42`; threshold: `0.5`.
- Training data: Train + Validation only; Test was not used for fitting or configuration.
- Label policy: all non-empty Train/Validation source labels retained.

## Full Test metrics
- Test studies: 1390; evaluated: 1390; coverage: 1.0000
- Accuracy: 0.9993
- Precision: 1.0000
- Recall: 0.9955
- F1: 0.9977
- Macro F1: 0.9987
- Weighted F1: 0.9993
- Specificity: 1.0000
- Confusion matrix: {'labels': [False, True], 'rows_true': {'false': {'predicted_false': 1169, 'predicted_true': 0}, 'true': {'predicted_false': 1, 'predicted_true': 220}}}
- Per-class metrics: {'negative': {'precision': 0.9991452991452991, 'recall': 1.0, 'f1-score': 0.9995724668661822, 'support': 1169.0}, 'positive': {'precision': 1.0, 'recall': 0.995475113122172, 'f1-score': 0.9977324263038548, 'support': 221.0}}

## Paired comparison with Regex Baseline
- Paired records: 525 / 1390 (0.3777 of Test).
- Learned and Regex quality metrics below are computed only on this same paired subset.
- Learned F1: 0.9977; Regex F1: 0.9888
- Learned Macro F1: 0.9980; Regex Macro F1: 0.9903
- F1 difference (learned - Regex): 0.0089
- Learned correct / Regex wrong: 5
- Regex correct / Learned wrong: 1
- Learned coverage on paired subset: 1.0000; Regex coverage on paired subset: 1.0000

## MATCH / MISMATCH / UNCERTAIN
These are text-vs-source-label audit groups, not clinical verification groups.
| Group | Studies | Accuracy | F1 | Macro F1 |
|---|---:|---:|---:|---:|
| MATCH | 420 | 1.0000 | 1.0000 | 1.0000 |
| MISMATCH | 37 | 1.0000 | 1.0000 | 1.0000 |
| UNCERTAIN | 933 | 0.9989 | 0.9926 | 0.9960 |

## Error analysis
Examples are redacted: IDs and numeric values are removed. Errors are relative to source labels.
- `other` / audit `UNCERTAIN`: true=True, predicted=False; The patient has an enlarged spleen, measuring [NUM] cc in volume, with a mean HU value of [NUM] +/- [NUM]. The liver is of normal size, with a volume of [NUM] cc and a mean HU value of [NUM] +/- [NUM]. A liver lesion is identified in hepati

## Research validity and limitations
- The 79 legacy cases are not treated as Human Verified.
- Patient-level leakage prevention cannot be claimed because Patient IDs are unavailable.
- Test was evaluated once after freezing the architecture and threshold.
- Source-label conflicts and repeated report templates can affect performance.

## Engineering decision
**The model is suitable for controlled engineering integration of the `has_lesion` component in Phase 9, behind validation, logging, and clinical review safeguards. This result is not clinical validation and does not justify autonomous clinical use.**
