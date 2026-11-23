# Extraction Challenge Set (extraction_challenge_v2_liver)

Built from Validation reports only (seed 42); Test was not read. Each item is the liver-only text of a lesion-free Validation report (`liver_nlp_liver_only.csv`, `liver_text` v1.0.1) (599 eligible) with a generated liver-lesion description inserted after its first liver sentence. Expected values are set by construction, not taken from source labels or a parser. The text is in `data/processed/nlp/challenge/extraction_challenge_v2_liver.jsonl` (gitignored); hashes, IDs and expected values are in the manifest.

Same base reports, seeds and generated descriptions as `extraction_challenge_v1`, so items pair with it by ID. Left out: `distractor`, because other-organ text is removed before the parser runs; that case stays in `extraction_challenge_v1`. 1597 of 1600 items equal the liver-only filter applied to the paired `extraction_challenge_v1` item.

| Category | Subtype | Items | Target fields |
|---|---|---|---|
| control | canonical | 200 | all |
| count_format | digits | 40 | lesion_count |
| count_format | total_of | 40 | lesion_count |
| count_format | enumerated | 40 | lesion_count |
| count_format | solitary | 40 | lesion_count |
| count_format | unquantified | 40 | lesion_count |
| size_format | mm | 34 | derived_max_diameter_cm |
| size_format | by | 34 | derived_max_diameter_cm |
| size_format | single_dimension | 33 | derived_max_diameter_cm |
| size_format | three_dimensions | 33 | derived_max_diameter_cm |
| size_format | centimeters | 33 | derived_max_diameter_cm |
| size_format | up_to | 33 | derived_max_diameter_cm |
| largest_order | last_keyword | 67 | location_segments, derived_max_diameter_cm, attenuation |
| largest_order | last_implicit | 67 | location_segments, derived_max_diameter_cm, attenuation |
| largest_order | middle_implicit | 66 | location_segments, derived_max_diameter_cm, attenuation |
| segment_format | roman | 34 | location_segments |
| segment_format | subsegment | 34 | location_segments |
| segment_format | couinaud | 33 | location_segments |
| segment_format | ordinal | 33 | location_segments |
| segment_format | hyphen | 33 | location_segments |
| segment_format | lobe_only | 33 | location_segments |
| attenuation_format | dense | 50 | attenuation |
| attenuation_format | low_high | 50 | attenuation |
| attenuation_format | comparative | 50 | attenuation |
| attenuation_format | separate_sentence | 50 | attenuation |
| ocr_noise | canonical | 200 | all |
| paraphrase | single | 100 | all |
| paraphrase | multiple | 100 | all |

1600 items over 582 base reports. Expected `null` means the text does not state the value (count for `unquantified`, segments for `lobe_only`), so the correct output is to abstain.

Category definitions are in the docstring of `scripts/nlp/transformers/extraction/build_challenge_set.py`.
