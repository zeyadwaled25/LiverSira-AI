# Extraction Challenge Set (extraction_challenge_v3)

Held-out test for the per-lesion model: lesion order, counting, repeated mentions and non-lesion measurements, in wording not used before. Built from Validation only (seed 2026); Test was not read. Each item is the liver-only text of a lesion-free Validation report with a generated description; expected values are set by construction. Text: `data/processed/nlp/challenge/extraction_challenge_v3.jsonl` (gitignored).

- Items whose inserted text the liver-only filter drops: 0.
- Challenge-specific masked 4-grams: 227; found in `extraction_train_v1` items: 0.

| Category | Subtype | Items | Target fields |
|---|---|---|---|
| control | canonical | 100 | all |
| order | ascending | 67 | all |
| order | smallest_first | 67 | location_segments, derived_max_diameter_cm, attenuation |
| order | middle | 66 | location_segments, derived_max_diameter_cm, attenuation |
| count | altogether | 50 | lesion_count |
| count | listed_sentences | 50 | lesion_count |
| count | isolated | 50 | lesion_count |
| count | unquantified | 50 | lesion_count |
| repeat | single | 100 | lesion_count, derived_max_diameter_cm |
| repeat | multiple | 100 | lesion_count, derived_max_diameter_cm |
| non_lesion | liver_span | 67 | lesion_count, derived_max_diameter_cm |
| non_lesion | vessel | 67 | lesion_count, derived_max_diameter_cm |
| non_lesion | node | 66 | lesion_count, derived_max_diameter_cm |

900 items over 511 base reports. Category definitions are in the docstring of `scripts/nlp/transformers/extraction/build_challenge_v3.py`.
