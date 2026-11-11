# Presence: transformers vs TF-IDF + LR vs parser v1.1

Validation (1389 studies) and the Challenge Set, against source-derived labels. Test was not read. Transformers: mean ± SD over seeds [13, 42, 87]. LR is the frozen configuration re-fitted on Train only (it reproduces the saved Validation predictions exactly). Parser abstentions count as negative.

## Validation

| System | Accuracy | Precision | Recall | Specificity | F1 | FP | FN | Brier |
|---|---|---|---|---|---|---|---|---|
| lr | 0.999 [0.998, 1.000] | 1.000 [1.000, 1.000] | 0.995 [0.985, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.993, 1.000] | 0 | 1 | 0.0062 |
| parser | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0 | 0 | — |
| bert | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.003 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.0 ± 0.0 | 0.3 ± 0.6 | 0.0002 ± 0.0004 |
| biobert | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0000 ± 0.0000 |
| clinicalbert | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0000 ± 0.0000 |

Each transformer seed vs LR on Validation (exact McNemar):

| System | seed 13: model right / LR right, p | seed 42: model right / LR right, p | seed 87: model right / LR right, p |
|---|---|---|---|
| bert | 1 / 0, p = 1 | 1 / 0, p = 1 | 1 / 1, p = 1 |
| biobert | 1 / 0, p = 1 | 1 / 0, p = 1 | 1 / 0, p = 1 |
| clinicalbert | 1 / 0, p = 1 | 1 / 0, p = 1 | 1 / 0, p = 1 |

## Validation accuracy by audit group

| System | MATCH (n=438) | MISMATCH (n=37) | UNCERTAIN (n=914) |
|---|---|---|---|
| lr | 0.998 | 1.000 | 1.000 |
| parser | 1.000 | 1.000 | 1.000 |
| bert | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.001 |
| biobert | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| clinicalbert | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Challenge Set

Accuracy per category; for `hedged` (no correct answer) the share called positive.

| System | hedged (n=150) | liver_last (n=150) | liver_removed (n=150) | negation (n=150) | ocr_noise (n=150) | other_organ_lesion (n=150) | paraphrase (n=150) | All scored (n=900) |
|---|---|---|---|---|---|---|---|---|
| lr | 0.093 | 1.000 | 0.973 | 0.973 | 0.993 | 1.000 | 0.973 | 0.986 |
| parser | 0.013 | 0.993 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 |
| bert | 0.076 ± 0.067 | 0.998 ± 0.004 | 1.000 ± 0.000 | 0.962 ± 0.034 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.993 ± 0.006 |
| biobert | 0.089 ± 0.090 | 0.973 ± 0.041 | 1.000 ± 0.000 | 0.967 ± 0.037 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.008 |
| clinicalbert | 0.036 ± 0.050 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.984 ± 0.017 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.003 |

Parser abstained on 555 Challenge items, counted as negative (hedged 148, liver_last 53, liver_removed 150, negation 2, ocr_noise 49, other_organ_lesion 139, paraphrase 14). Its accuracy on the negative-label categories (negation, liver_removed, other_organ_lesion) therefore includes abstentions, not only explicit negatives.

Each transformer seed vs LR on the scored Challenge items (exact McNemar):

| System | seed 13: model right / LR right, p | seed 42: model right / LR right, p | seed 87: model right / LR right, p |
|---|---|---|---|
| bert | 10 / 4, p = 0.18 | 13 / 0, p = 0.000244 | 12 / 10, p = 0.832 |
| biobert | 13 / 1, p = 0.00183 | 10 / 8, p = 0.815 | 11 / 13, p = 0.839 |
| clinicalbert | 13 / 0, p = 0.000244 | 10 / 2, p = 0.0386 | 13 / 2, p = 0.00739 |

## Selective prediction on Validation

Answer only when max(p, 1 - p) reaches the level; coverage and errors among answered, mean over seeds.

| System | ≥ 0.5: coverage / errors | ≥ 0.9: coverage / errors | ≥ 0.99: coverage / errors |
|---|---|---|---|
| bert | 1.000 / 0.3 | 1.000 / 0.3 | 1.000 / 0.3 |
| biobert | 1.000 / 0.0 | 1.000 / 0.0 | 1.000 / 0.0 |
| clinicalbert | 1.000 / 0.0 | 1.000 / 0.0 | 1.000 / 0.0 |

## Notes
- Validation was used to pick each run's epoch, so Validation numbers are slightly optimistic for the transformers; the Challenge Set was not used for any choice.
- Labels are source-derived. Clinical accuracy needs the human gold standard.
- No system is selected here; selection and the single Test run belong to Phase 8.
