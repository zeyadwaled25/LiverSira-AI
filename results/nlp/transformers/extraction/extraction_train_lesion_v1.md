# Per-lesion training data (extraction_train_lesion_v1)

Every Train study (liver-only text) plus generated items, with every lesion size phrase, the written count, repeats and the largest lesion labelled (`liversira_ai.nlp.lesion_targets` v1.0.0). Validation: real only. Test was not read. Text is in `data/processed/nlp/extraction/extraction_train_lesion_v1.jsonl` and `data/processed/nlp/extraction/extraction_val_lesion_v1.jsonl` (gitignored).

## Checks

- Generated items sharing a Challenge-specific 4-gram with `extraction_challenge_v1`, `extraction_challenge_v2_liver`, `extraction_challenge_v3`: **0** (1631 such 4-grams).
- Validation: per-lesion labels decode to targets that differ from the source-derived targets in 0 field values; Train: 13 (the largest lesion's size is not written while another field is).

## Train

11482 items: 6482 real, 500 real with OCR noise, 4500 generated (seed 1338).

| Category / subtype | Items | Lesion spans | Repeats | none | number | enumerated | unstated |
|---|---|---|---|---|---|---|---|
| real/real | 6482 | 2441 | 786 | 5453 | 349 | 522 | 158 |
| ocr/real | 500 | 1208 | 394 | 0 | 170 | 249 | 81 |
| count/present | 84 | 84 | 0 | 0 | 84 | 0 | 0 |
| count/in_total | 84 | 84 | 0 | 0 | 84 | 0 | 0 |
| count/listed | 83 | 261 | 0 | 0 | 0 | 83 | 0 |
| count/another | 83 | 235 | 0 | 0 | 0 | 83 | 0 |
| count/only_one | 83 | 83 | 0 | 0 | 0 | 83 | 0 |
| count/vague | 83 | 83 | 0 | 0 | 0 | 0 | 83 |
| size/single_mm | 100 | 100 | 0 | 0 | 0 | 100 | 0 |
| size/cm_by_cm | 100 | 100 | 0 | 0 | 0 | 100 | 0 |
| size/greatest_diameter | 100 | 100 | 0 | 0 | 0 | 100 | 0 |
| size/three_in_size | 100 | 100 | 0 | 0 | 0 | 100 | 0 |
| size/reaching | 100 | 100 | 0 | 0 | 0 | 100 | 0 |
| segment/roman | 84 | 84 | 0 | 0 | 0 | 84 | 0 |
| segment/subsegment | 84 | 84 | 0 | 0 | 0 | 84 | 0 |
| segment/couinaud | 83 | 83 | 0 | 0 | 0 | 83 | 0 |
| segment/ordinal | 83 | 83 | 0 | 0 | 0 | 83 | 0 |
| segment/range | 83 | 83 | 0 | 0 | 0 | 83 | 0 |
| segment/lobe | 83 | 83 | 0 | 0 | 0 | 83 | 0 |
| attenuation/appearing | 125 | 125 | 0 | 0 | 0 | 125 | 0 |
| attenuation/of_density | 125 | 125 | 0 | 0 | 0 | 125 | 0 |
| attenuation/relative | 125 | 125 | 0 | 0 | 0 | 125 | 0 |
| attenuation/separate | 125 | 125 | 0 | 0 | 0 | 125 | 0 |
| order/size_list | 167 | 471 | 0 | 0 | 0 | 167 | 0 |
| order/larger_of_two | 167 | 334 | 0 | 0 | 0 | 167 | 0 |
| order/there_is | 166 | 448 | 0 | 0 | 0 | 166 | 0 |
| paraphrase/single | 250 | 250 | 0 | 0 | 0 | 250 | 0 |
| paraphrase/multiple | 250 | 250 | 0 | 0 | 250 | 0 | 0 |
| repeat/single | 250 | 500 | 250 | 0 | 0 | 250 | 0 |
| repeat/multiple | 250 | 982 | 250 | 0 | 250 | 0 | 0 |
| non_lesion/length | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| non_lesion/duct | 167 | 167 | 0 | 0 | 0 | 167 | 0 |
| non_lesion/node | 166 | 166 | 0 | 0 | 0 | 166 | 0 |
| ocr/count/present | 17 | 17 | 0 | 0 | 17 | 0 | 0 |
| ocr/count/in_total | 17 | 17 | 0 | 0 | 17 | 0 | 0 |
| ocr/count/listed | 17 | 61 | 0 | 0 | 0 | 17 | 0 |
| ocr/count/another | 17 | 49 | 0 | 0 | 0 | 17 | 0 |
| ocr/count/only_one | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/count/vague | 16 | 16 | 0 | 0 | 0 | 0 | 16 |
| ocr/size/single_mm | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/size/cm_by_cm | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/size/greatest_diameter | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/size/three_in_size | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/size/reaching | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/roman | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/subsegment | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/couinaud | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/ordinal | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/range | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/segment/lobe | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/attenuation/appearing | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/attenuation/of_density | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/attenuation/relative | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/attenuation/separate | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/order/size_list | 16 | 47 | 0 | 0 | 0 | 16 | 0 |
| ocr/order/larger_of_two | 16 | 32 | 0 | 0 | 0 | 16 | 0 |
| ocr/order/there_is | 16 | 44 | 0 | 0 | 0 | 16 | 0 |
| ocr/paraphrase/single | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/paraphrase/multiple | 16 | 16 | 0 | 0 | 15 | 0 | 1 |
| ocr/repeat/single | 16 | 32 | 16 | 0 | 0 | 16 | 0 |
| ocr/repeat/multiple | 16 | 57 | 16 | 0 | 16 | 0 | 0 |
| ocr/non_lesion/length | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/non_lesion/duct | 16 | 16 | 0 | 0 | 0 | 16 | 0 |
| ocr/non_lesion/node | 16 | 16 | 0 | 0 | 0 | 16 | 0 |

## Validation

1389 studies: none 1169, number 57, enumerated 117, unstated 46; 492 lesion spans, 156 repeats.

## Notes

- Real reports: sizes inside a range ("ranging from 0.6 to 6.3 cm") are not lesion sizes; the largest lesion is the source label's, which is chosen by volume and is not always the longest written dimension.
- Generated wording is that of `extraction_train_v1`, plus the `repeat` and `non_lesion` categories.
