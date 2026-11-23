# Extraction model vs parser v1.1

Model: clinicalbert, seeds [13, 42, 87], trained on `extraction_train_v1` (Train only), epoch chosen by Validation loss. Parser: frozen v1.1. Both read the liver-only text. Test was not read. The decision rule was fixed in `config.json` before the first run.

## Decision per field

Adds value = for every seed, (1) higher pooled accuracy than the parser on the `extraction_challenge_v2_liver` subtypes that target the field, with exact McNemar p < 0.05, and (2) lesion-positive Validation accuracy at most 0.01 below the parser's.

| Field | Challenge: parser | Challenge: model (per seed) | McNemar p (per seed) | Val positive: parser | Val positive: model (per seed) | Adds value |
|---|---|---|---|---|---|---|
| count | 0.912 | 0.955 / 0.935 / 0.928 | 0.00016 / 0.065 / 0.22 | 0.886 | 0.991 / 0.995 / 0.995 | no |
| segments | 0.818 | 0.961 / 0.976 / 0.946 | 4.8e-19 / 1.9e-25 / 1.8e-14 | 0.918 | 0.968 / 0.973 / 0.973 | **yes** |
| attenuation | 0.802 | 0.956 / 0.949 / 0.949 | 2.5e-21 / 5.7e-22 / 2.3e-18 | 0.973 | 0.995 / 0.995 / 0.995 | **yes** |
| diameter_cm | 0.909 | 0.994 / 0.990 / 0.991 | 1.5e-16 / 3e-14 / 5.8e-15 | 1.000 | 1.000 / 1.000 / 1.000 | **yes** |

Validation: 220 lesion-positive studies; a field is correct when it equals the target, and `null` (unstated) must match `null`.

## Challenge v2 by subtype

Cells: parser → model (mean ± SD over seeds). Bold = a field the subtype targets.

| Category | Subtype | n | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|---|---|
| attenuation_format | comparative | 50 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **0.000 → 0.907 ± 0.162** | 1.000 → 1.000 ± 0.000 |
| attenuation_format | dense | 50 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 |
| attenuation_format | low_high | 50 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **0.160 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 |
| attenuation_format | separate_sentence | 50 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 |
| control | canonical | 200 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| count_format | digits | 40 | **1.000 → 0.983 ± 0.014** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| count_format | enumerated | 40 | **0.300 → 0.508 ± 0.058** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| count_format | solitary | 40 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| count_format | total_of | 40 | **1.000 → 0.842 ± 0.170** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| count_format | unquantified | 40 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| largest_order | last_implicit | 67 | 1.000 → 1.000 ± 0.000 | **1.000 → 0.905 ± 0.048** | **1.000 → 0.876 ± 0.062** | **1.000 → 0.960 ± 0.023** |
| largest_order | last_keyword | 67 | 1.000 → 1.000 ± 0.000 | **1.000 → 0.716 ± 0.122** | **1.000 → 1.000 ± 0.000** | **1.000 → 0.945 ± 0.009** |
| largest_order | middle_implicit | 66 | 1.000 → 1.000 ± 0.000 | **1.000 → 0.944 ± 0.032** | **1.000 → 0.687 ± 0.049** | **1.000 → 0.995 ± 0.009** |
| ocr_noise | canonical | 200 | 0.875 → 0.958 ± 0.003 | 0.725 → 0.998 ± 0.003 | 0.670 → 0.973 ± 0.003 | 0.865 → 1.000 ± 0.000 |
| paraphrase | multiple | 100 | 1.000 → 0.987 ± 0.023 | 0.600 → 0.993 ± 0.012 | 1.000 → 1.000 ± 0.000 | 0.540 → 1.000 ± 0.000 |
| paraphrase | single | 100 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | couinaud | 33 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | hyphen | 33 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | lobe_only | 33 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | ordinal | 33 | 1.000 → 1.000 ± 0.000 | **0.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | roman | 34 | 1.000 → 1.000 ± 0.000 | **1.000 → 0.971 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| segment_format | subsegment | 34 | 1.000 → 1.000 ± 0.000 | **0.471 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 |
| size_format | by | 34 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| size_format | centimeters | 33 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| size_format | mm | 34 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| size_format | single_dimension | 33 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| size_format | three_dimensions | 33 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| size_format | up_to | 33 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |

## Distractor check (`extraction_challenge_v1`, liver-only filter applied; not part of the decision)

| Category | Subtype | n | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|---|---|
| distractor | lymph_node | 66 | **1.000 → 0.995 ± 0.009** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 0.707 ± 0.097** |
| distractor | preceding_sentence | 67 | **1.000 → 0.995 ± 0.009** | 1.000 → 1.000 ± 0.000 | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |
| distractor | same_sentence | 67 | **1.000 → 1.000 ± 0.000** | **1.000 → 1.000 ± 0.000** | 1.000 → 1.000 ± 0.000 | **1.000 → 1.000 ± 0.000** |

## Runs

| Model | Seed | Best epoch | Best Val loss | Train seconds |
|---|---|---|---|---|
| clinicalbert | 13 | 4 | 0.0825 | 746 |
| clinicalbert | 42 | 4 | 0.0630 | 769 |
| clinicalbert | 87 | 4 | 0.0651 | 814 |

## Notes

- The Challenge Sets are synthetic in their lesion text; they show which phrasings each system handles, not error rates on real reports.
- Training items cover the same phenomena as the Challenge Sets with a separate lexicon (`extraction_train_v1.md`), so these scores test generalization across wording, not memorized templates.
- Counts of 10 or more are one class (`10+`) for the model.
