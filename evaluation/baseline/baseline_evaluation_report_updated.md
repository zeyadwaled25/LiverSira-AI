# Updated NLP Baseline Evaluation

These are rule-based comparisons against source-derived labels. `MATCH`, `MISMATCH`, and `UNCERTAIN` come from the text-vs-source-label consistency audit; they are not clinical gold-standard groups.

No new model was trained. Existing schemas, source data, and split IDs were not modified.

## ID and split integrity
- Leakage-free: **True**
- Ground-truth IDs: 9,262; inference IDs: 9,262

## Validation/Test comparison
| Split | Paired | Presence F1 | Count exact | Location IoU | Diameter MAE (cm) | Attenuation accuracy |
|---|---:|---:|---:|---:|---:|---:|
| validation | 1389 | 0.9865 | 0.7364 | 0.9890 | 0.3738 | 0.9628 |
| test | 1390 | 0.9888 | 0.8054 | 1.0000 | 0.2756 | 0.9954 |

## Per-field eligibility and exclusions
| Split | Field | Eligible | Evaluated | Excluded | Not applicable | Reasons |
|---|---|---:|---:|---:|---:|---|
| validation | has_lesion | 1389 | 556 | 833 | 0 | {'prediction_uncertain': 833} |
| validation | lesion_count | 220 | 220 | 0 | 1169 | — |
| validation | location_segments | 220 | 209 | 11 | 1169 | {'ground_truth_missing': 5, 'prediction_missing': 6} |
| validation | diameter_cm | 220 | 183 | 37 | 1169 | {'prediction_missing': 37} |
| validation | attenuation | 220 | 215 | 5 | 1169 | {'prediction_missing': 5} |
| test | has_lesion | 1390 | 525 | 865 | 0 | {'prediction_uncertain': 865} |
| test | lesion_count | 221 | 221 | 0 | 1169 | — |
| test | location_segments | 221 | 213 | 8 | 1169 | {'ground_truth_missing': 4, 'prediction_missing': 4} |
| test | diameter_cm | 221 | 193 | 28 | 1169 | {'prediction_missing': 28} |
| test | attenuation | 221 | 219 | 2 | 1169 | {'prediction_missing': 2} |

## Target/prediction definitions
- Lesion count: `ground_truth.lesion_count` vs `liver_lesions.lesion_count`.
- Location: `ground_truth.largest_lesion.location_segments` vs `liver_lesions.largest_lesion.location_segments`.
- Diameter: `ground_truth.largest_lesion.source_diameter_cm` vs `liver_lesions.largest_lesion.derived_max_diameter_cm` (maximum reported dimension).
- Evidence coverage is reported, but evidence accuracy is not claimed because there are no independent evidence annotations.

## Comparison-audit groups
| Split | Audit group | Studies | Presence F1 | Diameter MAE (cm) |
|---|---|---:|---:|---:|
| validation | MATCH | 1271 | 1.0000 | 0.0000 |
| validation | MISMATCH | 38 | 0.9143 | 2.2065 |
| validation | UNCERTAIN | 80 | 1.0000 | 0.0000 |
| test | MATCH | 1284 | 1.0000 | 0.0000 |
| test | MISMATCH | 38 | 0.9275 | 1.6625 |
| test | UNCERTAIN | 68 | 1.0000 | 0.0000 |

## Readiness
**Not ready as a clinical validation benchmark.** It is usable as a transparent source-label reference only after documenting the reported exclusions, abstentions, and unresolved source-label inconsistencies. The 79 reviewed-sample records are not treated as clinical verification because the repository does not establish documented human-review provenance for these outputs.
