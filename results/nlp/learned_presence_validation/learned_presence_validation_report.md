# First Learned NLP Experiment: Lesion Presence

This is a validation-only experiment against source-derived labels, not clinical validation.
The 79 legacy cases are not treated as human verified, and no Test records were used.

## Selection
Lesion presence was selected because it has complete binary labels and sufficient positive/negative examples. Count is imbalanced and location/diameter/attenuation are sparse or require extraction/regression, so they are deferred.

## Architecture and policy
- Word (1-2 gram) and character (3-5 gram) TF-IDF features.
- Class-weighted Logistic Regression, seed 42, threshold 0.5.
- Input: `narrative_clean` only; no structured report or ground-truth fields.
- All non-empty train labels were retained. MATCH/MISMATCH/UNCERTAIN were not automatic filters.
- Empty text was excluded; no Test records were loaded or used.

## Validation metrics
- Coverage: 1.0000
- Accuracy: 0.9993
- Precision: 1.0000
- Recall: 0.9955
- F1: 0.9977
- Macro F1: 0.9986
- Weighted F1: 0.9993
- Confusion matrix: {'labels': [False, True], 'rows_true': {'false': {'predicted_false': 1169, 'predicted_true': 0}, 'true': {'predicted_false': 1, 'predicted_true': 219}}}
- Per class: {'negative': {'precision': 0.9991452991452991, 'recall': 1.0, 'f1-score': 0.9995724668661822, 'support': 1169.0}, 'positive': {'precision': 1.0, 'recall': 0.9954545454545455, 'f1-score': 0.9977220956719818, 'support': 220.0}}

## Baseline comparison
Comparison uses the 556 validation records where the baseline emitted a prediction (common coverage 0.4003).
- Learned F1: 0.9977; baseline F1: 0.9865
- Learned macro F1: 0.9981; baseline macro F1: 0.9888
- Learned correct / baseline wrong: 6
- Baseline correct / learned wrong: 1

## Audit-group analysis
| Group | Studies | Accuracy | F1 |
|---|---:|---:|---:|
| MATCH | 438 | 0.9977 | 0.9953 |
| MISMATCH | 37 | 1.0000 | 1.0000 |
| UNCERTAIN | 914 | 1.0000 | 1.0000 |

## Error analysis
Examples below are redacted: IDs and numeric values are removed. They are source-label comparison errors, not clinical error adjudications.

- `learned_model` / `conflicting_source_label` / audit `MISMATCH`: The patient's spleen is of normal size, measuring [NUM] cc in volume, with a mean Hounsfield Unit (HU) value of [NUM] +/- [NUM]. The liver is also of normal size, with a volume of [NUM] cc and a mean HU value of [NUM] +/- [NUM]. The pancrea
- `rule_based_baseline` / `other` / audit `MATCH`: The patient underwent a non-contrast CT scan of the abdomen. The spleen is normal in size, measuring [NUM] cc in volume, with a mean Hounsfield Unit (HU) value of [NUM] +/- [NUM]. The liver is also normal in size, with a volume of [NUM] cc,
- `learned_model` / `ambiguous_report` / audit `MISMATCH`: The spleen, liver, pancreas, and kidneys are within normal limits in terms of size and enhancement. The spleen measures [NUM] cc in volume, the liver measures [NUM] cc, the pancreas measures [NUM] cc, and the combined kidney volume is [NUM]
- `learned_model` / `conflicting_source_label` / audit `MISMATCH`: The patient has a hypoattenuating mass in the head of the pancreas, measuring [NUM] x [NUM] cm (image [NUM]). The mass has a volume of [NUM] cc and is in contact with the duodenum. The tumor does not involve the SMA, aorta, portal vein and 
- `learned_model` / `conflicting_source_label` / audit `MISMATCH`: The spleen, liver, pancreas, and kidneys are all within normal limits in terms of size and appearance. The spleen measures [NUM] cc in volume, the liver measures [NUM] cc, the pancreas measures [NUM] cc, and the combined kidney volume is [N
- `learned_model` / `negation_related` / audit `MISMATCH`: The spleen, liver, pancreas, and kidneys are all within normal limits in terms of size and enhancement pattern. The spleen measures [NUM] cc in volume, the liver measures [NUM] cc in volume, the pancreas measures [NUM] cc in volume, and the
- `learned_model` / `conflicting_source_label` / audit `MISMATCH`: The patient has two hypoattenuating pancreatic masses. The larger mass, located in the body of the pancreas, measures [NUM] x [NUM] cm (image [NUM]) and has a volume of [NUM] cc. The smaller mass, located in the tail of the pancreas, measur

## Limitations and decision
- Source labels contain 214 positive-label MISMATCH cases and 456 positive-label UNCERTAIN cases in the full dataset.
- No threshold or architecture tuning was performed on Test.
- This model has no abstention mechanism; its validation coverage is 1.0 on non-empty reports.
- It is suitable for the next controlled experiment only if the validation comparison is reviewed; it is not a clinical model or gold-standard benchmark.
