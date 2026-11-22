# Parser v1.1 on extraction_challenge_v3

`extraction_challenge_v3`: 900 generated items on the liver-only text of lesion-free Validation reports; Test was not read. Expected values are set by construction, not by source labels. Parser: frozen v1.1 configuration (`baseline.extract_liver_lesion_features`). Accuracy counts abstentions as errors; diameter is correct within 0.05 cm. 95% bootstrap intervals over items (2000 resamples, seed 42).

## Gate for 6.2

Rule, fixed before the run: a field goes to 6.2 modelling if the parser's accuracy is below 0.90 on at least one subtype that targets it.

| Field | Control | Pooled over targeting subtypes | Subtypes below the gate | Needs a model |
|---|---|---|---|---|
| lesion_count | 1.000 [1.000, 1.000] | 0.865 [0.838, 0.891] | `count/altogether`, `count/listed_sentences` | **yes** |
| location_segments | 1.000 [1.000, 1.000] | 0.675 [0.605, 0.740] | `order/ascending` | **yes** |
| derived_max_diameter_cm | 1.000 [1.000, 1.000] | 0.863 [0.833, 0.890] | `order/ascending` | **yes** |
| attenuation | 1.000 [1.000, 1.000] | 0.980 [0.960, 0.995] | none | no |

## Accuracy by subtype

Bold = a field the subtype targets. Cells: accuracy (abstained / wrong).

| Category | Subtype | n | Presence found | lesion_count | location_segments | derived_max_diameter_cm | attenuation | All fields |
|---|---|---|---|---|---|---|---|---|
| control | canonical | 100 | 100 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| order | ascending | 67 | 67 | 1.000 (0/0) | 0.090 (0/61) | 0.000 (0/67) | 1.000 (0/0) | 0.000 |
| order | smallest_first | 67 | 64 | 0.000 (3/64) | **0.955 (3/0)** | **0.955 (3/0)** | **0.955 (3/0)** | 0.000 |
| order | middle | 66 | 65 | 0.000 (1/65) | **0.985 (1/0)** | **0.985 (1/0)** | **0.985 (1/0)** | 0.000 |
| count | altogether | 50 | 48 | **0.460 (27/0)** | 0.960 (2/0) | 0.960 (2/0) | 0.960 (2/0) | 0.460 |
| count | listed_sentences | 50 | 49 | **0.000 (1/49)** | 0.980 (1/0) | 0.980 (1/0) | 0.980 (1/0) | 0.000 |
| count | isolated | 50 | 48 | **0.960 (2/0)** | 0.260 (2/35) | 0.960 (2/0) | 0.960 (2/0) | 0.260 |
| count | unquantified | 50 | 50 | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| repeat | single | 100 | 95 | **0.950 (5/0)** | 0.950 (5/0) | **0.950 (5/0)** | 0.950 (5/0) | 0.950 |
| repeat | multiple | 100 | 100 | **1.000 (0/0)** | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| non_lesion | liver_span | 67 | 67 | **1.000 (0/0)** | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| non_lesion | vessel | 67 | 64 | **0.955 (3/0)** | 0.955 (3/0) | **0.955 (3/0)** | 0.955 (3/0) | 0.955 |
| non_lesion | node | 66 | 63 | **0.955 (3/0)** | 0.955 (3/0) | **0.955 (3/0)** | 0.955 (3/0) | 0.955 |

## Examples of failures

Generated lesion text only (no dataset text). Up to 2 per subtype, for the fields the subtype targets.

- `order/ascending`: location_segments: expected [3], got [2]; derived_max_diameter_cm: expected 5.4, got 1.5
  > Four hypoattenuating lesions are noted, measuring 1.5 cm for segment 2; 3.7 cm for segment 4; 4.7 cm for segment 4; and 5.4 cm for segment 3.
- `order/ascending`: location_segments: expected [3, 4], got [2]; derived_max_diameter_cm: expected 3.0, got 1.1
  > Three hyperattenuating lesions are noted, measuring 1.1 cm for segment 2; 2.2 cm for segment 5; and 3.0 cm for segments 3 and 4.
- `order/middle`: location_segments: expected [1, 5], got null; derived_max_diameter_cm: expected 5.9, got null; attenuation: expected "hypoattenuating", got null
  > Segment 1 holds an isoattenuating lesion measuring 1.6 x 1.3 cm, while segments 1 and 5 holds a hypoattenuating lesion measuring 5.9 x 5.1 cm, while segment 4 holds a hypoattenuating lesion measuring 5.0 x 3.9 cm, while segment 5 holds a hypoattenuating lesion measuring 4.6 x 2.5 cm.
- `order/smallest_first`: location_segments: expected [4, 8], got null; derived_max_diameter_cm: expected 2.5, got null; attenuation: expected "isoattenuating", got null
  > The smallest one, within segment 5, is 0.9 x 0.8 cm and hypoattenuating. Next, a hypoattenuating lesion measuring 1.5 x 0.8 cm is housed in segment 6. Finally, an isoattenuating lesion measuring 2.5 x 1.6 cm is housed in segments 4 and 8.
- `order/smallest_first`: location_segments: expected [1, 3], got null; derived_max_diameter_cm: expected 6.5, got null; attenuation: expected "hyperattenuating", got null
  > The smallest one, within segment 6, is 2.7 x 2.4 cm and hypoattenuating. Finally, a hyperattenuating lesion measuring 6.5 x 5.6 cm is housed in segments 1 and 3.
- `count/altogether`: lesion_count: expected 3, got null
  > The lesion count is 3, all hypoattenuating lesions with the largest of 8.7 x 5.9 cm in segments 2 and 3.
- `count/listed_sentences`: lesion_count: expected 2, got 1
  > A hypoattenuating lesion measuring 2.1 x 1.2 cm occupies segment 2. Separately, a 0.4 x 0.3 cm lesion occupies segment 4.
- `count/listed_sentences`: lesion_count: expected 3, got 1
  > A hypoattenuating lesion measuring 9.0 x 4.7 cm occupies segments 4 and 6. Separately, a 5.7 x 4.5 cm lesion occupies segment 8. Also seen is a 5.6 x 2.9 cm lesion occupies segment 6.
- `count/altogether`: lesion_count: expected 2, got null
  > The lesion count is 2, all hypoattenuating lesions with the largest of 6.9 x 3.7 cm in segment 3.
- `count/isolated`: lesion_count: expected 1, got null
  > An isolated hypoattenuating lesion occupies segment 5, 2.0 x 1.0 cm across.
- `count/isolated`: lesion_count: expected 1, got null
  > Just one hypoattenuating lesion is evident within segment 8, 2.3 x 1.3 cm across.
- `repeat/single`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 9.0, got null
  > A hyperattenuating lesion measuring 9.0 x 5.7 cm occupies segments 4 and 8. Summary: segments 4 and 8 hyperattenuating lesion, 9.0 x 5.7 cm, unchanged in description.
- `repeat/single`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 7.8, got null
  > A hypoattenuating lesion measuring 7.8 x 4.9 cm occupies segment 8. Summary: segment 8 hypoattenuating lesion, 7.8 x 4.9 cm, unchanged in description.
- `non_lesion/vessel`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 8.7, got null
  > The middle hepatic vein is 1.6 cm in calibre. A hypoattenuating lesion measuring 8.7 x 4.5 cm occupies segment 3.
- `non_lesion/node`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 9.0, got null
  > An isoattenuating lesion measuring 9.0 x 4.7 cm occupies segments 1 and 7. A 10.4 cm node lies adjacent to the liver.
- `non_lesion/node`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 4.9, got null
  > A hyperattenuating lesion measuring 4.9 x 3.5 cm occupies segment 7. A 7.2 cm node lies adjacent to the liver.
- `non_lesion/vessel`: lesion_count: expected 1, got null; derived_max_diameter_cm: expected 7.7, got null
  > Intrahepatic biliary ducts are dilated up to 1.2 cm. An isoattenuating lesion measuring 7.7 x 4.5 cm occupies segment 7.

## Notes
- The items are synthetic in their lesion description; the surrounding report is real Validation text. A failure here shows a phrasing the parser does not handle, not its error rate on the dataset.
- `unquantified` and `lobe_only` expect `null`: the right output is to abstain.
- The Challenge Set must not be used to tune parser rules; doing so would turn it into a development set.
