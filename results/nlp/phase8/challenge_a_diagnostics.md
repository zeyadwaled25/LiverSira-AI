# Challenge-A diagnostics (not part of the Phase 8 decision)

Per-subtype accuracy of every frozen candidate on the 6.2 Challenge Sets that were seen during development: the v1 distractor items, `v2_liver` and `v3` (liver-only filter applied where the set needs it). These numbers explain behaviour; they do not choose anything. The decision uses Challenge-B and Test once, in Phase 8 (`docs/nlp/phase8_analysis_plan.md`).

Cells: accuracy [Wilson 95% interval]. Model-based candidates (first model, per-lesion, hybrid H) show the mean over seeds 13 / 42 / 87 and the envelope of the three seeds' Wilson intervals. Only targeted fields are shown (control rows: all fields); the JSON has every run.

## v1_distractor

| Category | Subtype | Field | n | parser v1.1 | parser v1.2 | first model | per-lesion model | hybrid H |
|---|---|---|---|---|---|---|---|---|
| distractor | lymph_node | lesion_count | 66 | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] | 0.99 [0.92, 1.00] | 0.74 [0.63, 0.83] | 1.00 [0.94, 1.00] |
| distractor | lymph_node | derived_max_diameter_cm | 66 | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] | 0.71 [0.52, 0.89] | 0.16 [0.04, 0.31] | 1.00 [0.94, 1.00] |
| distractor | preceding_sentence | lesion_count | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.99 [0.92, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] |
| distractor | preceding_sentence | derived_max_diameter_cm | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] |
| distractor | same_sentence | lesion_count | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] |
| distractor | same_sentence | derived_max_diameter_cm | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] |
| distractor | same_sentence | location_segments | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] |

## v2_liver

| Category | Subtype | Field | n | parser v1.1 | parser v1.2 | first model | per-lesion model | hybrid H |
|---|---|---|---|---|---|---|---|---|
| attenuation_format | comparative | attenuation | 50 | 0.00 [0.00, 0.07] | 0.00 [0.00, 0.07] | 0.91 [0.58, 1.00] | 1.00 [0.93, 1.00] | 0.91 [0.58, 1.00] |
| attenuation_format | dense | attenuation | 50 | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] |
| attenuation_format | low_high | attenuation | 50 | 0.16 [0.08, 0.29] | 0.16 [0.08, 0.29] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] |
| attenuation_format | separate_sentence | attenuation | 50 | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] |
| control | canonical | lesion_count | 200 | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] |
| control | canonical | location_segments | 200 | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] |
| control | canonical | derived_max_diameter_cm | 200 | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] |
| control | canonical | attenuation | 200 | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] |
| count_format | digits | lesion_count | 40 | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 0.98 [0.87, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] |
| count_format | enumerated | lesion_count | 40 | 0.30 [0.18, 0.45] | 0.30 [0.18, 0.45] | 0.51 [0.33, 0.71] | 1.00 [0.91, 1.00] | 0.30 [0.18, 0.45] |
| count_format | solitary | lesion_count | 40 | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] |
| count_format | total_of | lesion_count | 40 | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 0.84 [0.50, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] |
| count_format | unquantified | lesion_count | 40 | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] | 1.00 [0.91, 1.00] |
| largest_order | last_implicit | location_segments | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.91 [0.75, 0.98] | 0.91 [0.73, 0.99] | 1.00 [0.95, 1.00] |
| largest_order | last_implicit | derived_max_diameter_cm | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.96 [0.86, 1.00] | 0.94 [0.78, 0.99] | 1.00 [0.95, 1.00] |
| largest_order | last_implicit | attenuation | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.88 [0.70, 0.97] | 0.89 [0.66, 1.00] | 1.00 [0.95, 1.00] |
| largest_order | last_keyword | location_segments | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.72 [0.46, 0.89] | 0.75 [0.49, 0.89] | 1.00 [0.95, 1.00] |
| largest_order | last_keyword | derived_max_diameter_cm | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.95 [0.86, 0.98] | 0.94 [0.84, 0.98] | 1.00 [0.95, 1.00] |
| largest_order | last_keyword | attenuation | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.99 [0.90, 1.00] | 1.00 [0.95, 1.00] |
| largest_order | middle_implicit | location_segments | 66 | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] | 0.94 [0.82, 0.99] | 0.98 [0.87, 1.00] | 1.00 [0.94, 1.00] |
| largest_order | middle_implicit | derived_max_diameter_cm | 66 | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] | 0.99 [0.92, 1.00] | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] |
| largest_order | middle_implicit | attenuation | 66 | 1.00 [0.94, 1.00] | 1.00 [0.94, 1.00] | 0.69 [0.53, 0.83] | 0.87 [0.55, 1.00] | 1.00 [0.94, 1.00] |
| segment_format | couinaud | location_segments | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| segment_format | hyphen | location_segments | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| segment_format | lobe_only | location_segments | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| segment_format | ordinal | location_segments | 33 | 0.00 [0.00, 0.10] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| segment_format | roman | location_segments | 34 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 0.97 [0.85, 0.99] | 0.97 [0.81, 1.00] | 1.00 [0.90, 1.00] |
| segment_format | subsegment | location_segments | 34 | 0.47 [0.31, 0.63] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | by | derived_max_diameter_cm | 34 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | centimeters | derived_max_diameter_cm | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | mm | derived_max_diameter_cm | 34 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | single_dimension | derived_max_diameter_cm | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | three_dimensions | derived_max_diameter_cm | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |
| size_format | up_to | derived_max_diameter_cm | 33 | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] | 1.00 [0.90, 1.00] |

## v3

| Category | Subtype | Field | n | parser v1.1 | parser v1.2 | first model | per-lesion model | hybrid H |
|---|---|---|---|---|---|---|---|---|
| control | canonical | lesion_count | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| control | canonical | location_segments | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| control | canonical | derived_max_diameter_cm | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| control | canonical | attenuation | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| count | altogether | lesion_count | 50 | 0.46 [0.33, 0.60] | 0.46 [0.33, 0.60] | 0.96 [0.76, 1.00] | 0.59 [0.35, 0.87] | 0.96 [0.76, 1.00] |
| count | isolated | lesion_count | 50 | 0.96 [0.87, 0.99] | 0.96 [0.87, 0.99] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] |
| count | listed_sentences | lesion_count | 50 | 0.00 [0.00, 0.07] | 0.00 [0.00, 0.07] | 0.01 [0.00, 0.13] | 1.00 [0.93, 1.00] | 0.01 [0.00, 0.10] |
| count | unquantified | lesion_count | 50 | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 0.43 [0.28, 0.61] | 1.00 [0.93, 1.00] | 0.43 [0.28, 0.61] |
| non_lesion | liver_span | lesion_count | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.48 [0.01, 0.87] | 1.00 [0.95, 1.00] |
| non_lesion | liver_span | derived_max_diameter_cm | 67 | 1.00 [0.95, 1.00] | 1.00 [0.95, 1.00] | 0.98 [0.86, 1.00] | 0.51 [0.03, 0.87] | 1.00 [0.95, 1.00] |
| non_lesion | node | lesion_count | 66 | 0.95 [0.87, 0.98] | 0.95 [0.87, 0.98] | 1.00 [0.94, 1.00] | 0.00 [0.00, 0.06] | 1.00 [0.94, 1.00] |
| non_lesion | node | derived_max_diameter_cm | 66 | 0.95 [0.87, 0.98] | 0.95 [0.87, 0.98] | 0.34 [0.19, 0.51] | 0.11 [0.04, 0.22] | 0.97 [0.90, 0.99] |
| non_lesion | vessel | lesion_count | 67 | 0.96 [0.88, 0.98] | 0.96 [0.88, 0.98] | 1.00 [0.95, 1.00] | 0.46 [0.31, 0.62] | 1.00 [0.95, 1.00] |
| non_lesion | vessel | derived_max_diameter_cm | 67 | 0.96 [0.88, 0.98] | 0.96 [0.88, 0.98] | 0.99 [0.90, 1.00] | 0.94 [0.84, 0.98] | 1.00 [0.95, 1.00] |
| order | middle | location_segments | 66 | 0.98 [0.92, 1.00] | 0.98 [0.92, 1.00] | 0.85 [0.63, 0.98] | 0.98 [0.92, 1.00] | 1.00 [0.94, 1.00] |
| order | middle | derived_max_diameter_cm | 66 | 0.98 [0.92, 1.00] | 0.98 [0.92, 1.00] | 0.98 [0.90, 1.00] | 0.99 [0.92, 1.00] | 1.00 [0.94, 1.00] |
| order | middle | attenuation | 66 | 0.98 [0.92, 1.00] | 0.98 [0.92, 1.00] | 0.82 [0.59, 0.97] | 0.99 [0.92, 1.00] | 1.00 [0.94, 1.00] |
| order | smallest_first | location_segments | 67 | 0.96 [0.88, 0.98] | 0.96 [0.88, 0.98] | 0.98 [0.90, 1.00] | 0.97 [0.86, 1.00] | 1.00 [0.95, 1.00] |
| order | smallest_first | derived_max_diameter_cm | 67 | 0.96 [0.88, 0.98] | 0.96 [0.88, 0.98] | 0.99 [0.92, 1.00] | 0.99 [0.90, 1.00] | 1.00 [0.95, 1.00] |
| order | smallest_first | attenuation | 67 | 0.96 [0.88, 0.98] | 0.96 [0.88, 0.98] | 0.98 [0.90, 1.00] | 0.99 [0.90, 1.00] | 0.99 [0.92, 1.00] |
| repeat | multiple | lesion_count | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| repeat | multiple | derived_max_diameter_cm | 100 | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| repeat | single | lesion_count | 100 | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
| repeat | single | derived_max_diameter_cm | 100 | 0.95 [0.89, 0.98] | 0.95 [0.89, 0.98] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] | 1.00 [0.96, 1.00] |
