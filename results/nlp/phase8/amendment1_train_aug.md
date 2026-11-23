# Amendment 1: field status and hybrid H2 on Train-aug

One attempt on 6.2 Train-aug only (`extraction_train_v1`, `extraction_train_lesion_v1`). Challenge Sets, Validation and Test were not read. H2 differs from H only where parser v1.2 leaves a field empty with status `not_mentioned`: H takes the first model's value there, H2 keeps it empty.

- Parser v1.2 values changed by adding the status: **0** items (must be 0).
- Train-aug items where H and H2 can differ: **871** (of 21964).

## Evidence: unquantified count, H wrong, H2 right

None. On Train-aug the first model leaves unquantified counts empty, so H is never wrong there. The weakness was found in the Challenge-A diagnostics (Phase 8 plan, Amendment 1).

## H vs H2 where they differ (paired, per seed)

| Seed | Fields compared | H correct | H2 correct |
|---|---|---|---|
| 13 | 938 | 934 | 613 |
| 42 | 938 | 935 | 613 |
| 87 | 938 | 934 | 613 |

### By family and field (seed 13)

| Family | Field | n | H correct | H2 correct |
|---|---|---|---|---|
| count/vague | lesion_count | 166 | 166 | 166 |
| ocr | attenuation | 100 | 100 | 1 |
| ocr | derived_max_diameter_cm | 31 | 30 | 1 |
| ocr | lesion_count | 132 | 130 | 94 |
| ocr | location_segments | 191 | 190 | 43 |
| real | attenuation | 6 | 6 | 0 |
| real | lesion_count | 126 | 126 | 126 |
| real | location_segments | 20 | 20 | 20 |
| repeat/multiple | attenuation | 4 | 4 | 0 |
| segment/lobe | location_segments | 162 | 162 | 162 |

## Parser v1.2 field status on Train-aug

| Field | Status | Fields |
|---|---|---|
| attenuation | not_mentioned | 4044 |
| attenuation | stated | 10391 |
| attenuation | unparsed | 7529 |
| derived_max_diameter_cm | not_mentioned | 3965 |
| derived_max_diameter_cm | stated | 10603 |
| derived_max_diameter_cm | unparsed | 7396 |
| lesion_count | not_mentioned | 424 |
| lesion_count | stated | 14119 |
| lesion_count | unparsed | 7421 |
| location_segments | not_mentioned | 4307 |
| location_segments | stated | 10182 |
| location_segments | unparsed | 7475 |
