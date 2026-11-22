# Extraction training data (extraction_train_v1)

Train: every Train study (liver-only text) plus generated items; Validation: every Validation study, real only. Test was not read. Targets follow `liversira_ai.nlp.extraction_targets` v1.0.0: a source value is a target only when the text states it, else `null` (unstated). Text is in `data/processed/nlp/extraction/extraction_train_v1.jsonl` and `data/processed/nlp/extraction/extraction_val_v1.jsonl` (gitignored).

## Template disjointness from the Challenge Set

1416 masked 4-grams occur in the inserted text of `extraction_challenge_v1`, `extraction_challenge_v2_liver` and nowhere in the real Train reports. Generated training items containing any of them: **0**.

## Train

10482 items: 6482 real, 3500 generated (seed 1337, 500 per category, OCR on 500 generated and 500 real), from 2833 eligible lesion-free Train bases.

| Category / subtype | Items | Count stated | Segments stated | Attenuation stated | Diameter stated |
|---|---|---|---|---|---|
| real/real | 6482 | 6324 | 943 | 1021 | 1024 |
| ocr/real | 500 | 430 | 455 | 496 | 498 |
| count/present | 84 | 84 | 84 | 84 | 84 |
| count/in_total | 84 | 84 | 84 | 84 | 84 |
| count/listed | 83 | 83 | 83 | 83 | 83 |
| count/another | 83 | 83 | 83 | 83 | 83 |
| count/only_one | 83 | 83 | 83 | 83 | 83 |
| count/vague | 83 | 0 | 83 | 83 | 83 |
| size/single_mm | 100 | 100 | 100 | 100 | 100 |
| size/cm_by_cm | 100 | 100 | 100 | 100 | 100 |
| size/greatest_diameter | 100 | 100 | 100 | 100 | 100 |
| size/three_in_size | 100 | 100 | 100 | 100 | 100 |
| size/reaching | 100 | 100 | 100 | 100 | 100 |
| segment/roman | 84 | 84 | 84 | 84 | 84 |
| segment/subsegment | 84 | 84 | 84 | 84 | 84 |
| segment/couinaud | 83 | 83 | 83 | 83 | 83 |
| segment/ordinal | 83 | 83 | 83 | 83 | 83 |
| segment/range | 83 | 83 | 83 | 83 | 83 |
| segment/lobe | 83 | 83 | 0 | 83 | 83 |
| attenuation/appearing | 125 | 125 | 125 | 125 | 125 |
| attenuation/of_density | 125 | 125 | 125 | 125 | 125 |
| attenuation/relative | 125 | 125 | 125 | 125 | 125 |
| attenuation/separate | 125 | 125 | 125 | 125 | 125 |
| order/size_list | 167 | 167 | 167 | 167 | 167 |
| order/larger_of_two | 167 | 167 | 167 | 167 | 167 |
| order/there_is | 166 | 166 | 166 | 166 | 166 |
| paraphrase/single | 250 | 250 | 250 | 250 | 250 |
| paraphrase/multiple | 250 | 250 | 250 | 250 | 250 |
| ocr/count/present | 20 | 20 | 20 | 20 | 20 |
| ocr/count/in_total | 20 | 20 | 20 | 20 | 20 |
| ocr/count/listed | 20 | 20 | 20 | 20 | 20 |
| ocr/count/another | 20 | 20 | 20 | 20 | 20 |
| ocr/count/only_one | 20 | 20 | 20 | 20 | 20 |
| ocr/count/vague | 20 | 0 | 20 | 20 | 20 |
| ocr/size/single_mm | 19 | 19 | 19 | 19 | 19 |
| ocr/size/cm_by_cm | 19 | 19 | 19 | 19 | 19 |
| ocr/size/greatest_diameter | 19 | 19 | 19 | 19 | 19 |
| ocr/size/three_in_size | 19 | 19 | 19 | 19 | 19 |
| ocr/size/reaching | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/roman | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/subsegment | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/couinaud | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/ordinal | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/range | 19 | 19 | 19 | 19 | 19 |
| ocr/segment/lobe | 19 | 19 | 0 | 19 | 19 |
| ocr/attenuation/appearing | 19 | 19 | 19 | 19 | 19 |
| ocr/attenuation/of_density | 19 | 19 | 19 | 19 | 19 |
| ocr/attenuation/relative | 19 | 19 | 19 | 19 | 19 |
| ocr/attenuation/separate | 19 | 19 | 19 | 19 | 19 |
| ocr/order/size_list | 19 | 19 | 19 | 19 | 19 |
| ocr/order/larger_of_two | 19 | 19 | 19 | 19 | 19 |
| ocr/order/there_is | 19 | 19 | 19 | 19 | 19 |
| ocr/paraphrase/single | 19 | 19 | 19 | 19 | 19 |
| ocr/paraphrase/multiple | 19 | 19 | 19 | 19 | 19 |

Count classes (`None` = unstated): 0: 5453, 1: 2943, 10+: 54, 2: 799, 3: 416, 4: 296, 5: 96, 6: 52, 7: 42, None: 331

## Validation (real only)

1389 studies. Count classes: 0: 1169, 1: 117, 10+: 3, 2: 27, 3: 10, 4: 12, 5: 2, 6: 3, None: 46

## Notes

- Real targets: count is unstated when a multi-lesion report gives no number ("multiple lesions"); segments, attenuation and diameter are unstated when the text does not contain the source value.
- Counts of 10 or more share the class `10+`.
- Generated lesions use the Challenge Set's sampling of counts, sizes and segments; only the wording differs.
- Distractors are not generated: other-organ text is removed by the liver-only filter before the model runs.
