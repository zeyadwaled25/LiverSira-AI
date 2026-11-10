# Learned Architecture Comparison: Lesion Presence

Validation-only comparison against provisional source-derived labels. Test was not loaded, fitted, tuned, or evaluated.

## Leakage and robustness audit
- Exact train/validation duplicate narratives: **0** (0 validation records).
- Shared numeric-normalized templates: **298** (468 validation records).
- Character-TF-IDF nearest-neighbour >=0.90: **1**; >=0.95: **0**.
- Patient-level leakage: not assessed because Patient ID is unavailable.

Template overlap is a robustness warning, not proof of leakage: radiology reports naturally reuse templates.

## Full Validation comparison
| Model | F1 | Macro F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|
| word_logistic_regression | 0.9977 | 0.9986 | 1.0000 | 0.9955 | 1.0000 |
| char_logistic_regression | 0.9955 | 0.9973 | 0.9955 | 0.9955 | 1.0000 |
| word_char_logistic_regression | 0.9977 | 0.9986 | 1.0000 | 0.9955 | 1.0000 |
| word_char_linear_svm | 0.9954 | 0.9973 | 1.0000 | 0.9909 | 1.0000 |

## Common 556-record baseline subset
| Model | F1 | Macro F1 | Precision | Recall | Coverage |
|---|---:|---:|---:|---:|---:|
| word_logistic_regression | 0.9977 | 0.9981 | 1.0000 | 0.9955 | 1.0000 |
| char_logistic_regression | 0.9977 | 0.9981 | 1.0000 | 0.9955 | 1.0000 |
| word_char_logistic_regression | 0.9977 | 0.9981 | 1.0000 | 0.9955 | 1.0000 |
| word_char_linear_svm | 0.9954 | 0.9962 | 1.0000 | 0.9909 | 1.0000 |
| Rule-based baseline | 0.9865 | 0.9888 | 0.9735 | 1.0000 | 1.0000 |

## Selection recommendation
Selected architecture for any later frozen evaluation: **word_logistic_regression**.
Word-only Logistic Regression ties the word+character model on all reported primary Validation metrics and is selected as the simpler architecture. Selection is based on Validation macro F1, with F1 as tie-breaker, under the fixed common policy. It is an engineering selection only; Test remains untouched.
The result does not justify a Transformer yet: all candidates are linear and Validation is nearly saturated. A Transformer comparison should wait for a cleaner label audit or a task where the linear baseline is demonstrably insufficient.
On the common subset, the learned model improves F1 over the Regex baseline by about 0.0112, but the absolute difference is based on only 7 discordant cases and may be affected by repeated templates and source-label noise.

## Group performance
### word_logistic_regression
- MATCH: n=438, F1=0.9953, Macro F1=0.9969
- MISMATCH: n=37, F1=1.0000, Macro F1=1.0000
- UNCERTAIN: n=914, F1=1.0000, Macro F1=1.0000
### char_logistic_regression
- MATCH: n=438, F1=0.9953, Macro F1=0.9969
- MISMATCH: n=37, F1=1.0000, Macro F1=1.0000
- UNCERTAIN: n=914, F1=0.9939, Macro F1=0.9966
### word_char_logistic_regression
- MATCH: n=438, F1=0.9953, Macro F1=0.9969
- MISMATCH: n=37, F1=1.0000, Macro F1=1.0000
- UNCERTAIN: n=914, F1=1.0000, Macro F1=1.0000
### word_char_linear_svm
- MATCH: n=438, F1=0.9907, Macro F1=0.9938
- MISMATCH: n=37, F1=1.0000, Macro F1=1.0000
- UNCERTAIN: n=914, F1=1.0000, Macro F1=1.0000

## Risks and interpretation
- Labels are source-derived comparisons, not clinical gold standard.
- MATCH/MISMATCH/UNCERTAIN groups were retained, not used as automatic filters.
- Shared templates and near-duplicates can inflate Validation performance.
- Patient-level leakage cannot be ruled out without Patient ID.
- Prediction coverage is 1.0 for all candidates; this is separate from prediction quality.

## Selected-model error examples
- `other` / audit `MATCH`: true=True, predicted=False; The patient underwent a non-contrast CT scan of the abdomen. The spleen is normal in size, measuring [NUM] cc in volume, with a mean Hounsfield Unit (HU) value of [NUM] +/- [NUM]. The liver is also normal in size, with a volume of [NUM] cc,
