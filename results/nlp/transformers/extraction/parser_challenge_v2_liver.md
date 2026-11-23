# Parser v1.1 on extraction_challenge_v2_liver

`extraction_challenge_v2_liver`: 1600 generated items on the liver-only text of lesion-free Validation reports; Test was not read. Expected values are set by construction, not by source labels. Parser: frozen v1.1 configuration (`baseline.extract_liver_lesion_features`). Accuracy counts abstentions as errors; diameter is correct within 0.05 cm. 95% bootstrap intervals over items (2000 resamples, seed 42).

## Gate for 6.2

Rule, fixed before the run: a field goes to 6.2 modelling if the parser's accuracy is below 0.90 on at least one subtype that targets it.

| Field | Control | Pooled over targeting subtypes | Subtypes below the gate | Needs a model |
|---|---|---|---|---|
| lesion_count | 1.000 [1.000, 1.000] | 0.912 [0.888, 0.933] | `count_format/enumerated`, `ocr_noise/canonical` | **yes** |
| location_segments | 1.000 [1.000, 1.000] | 0.818 [0.790, 0.844] | `segment_format/subsegment`, `segment_format/ordinal`, `ocr_noise/canonical`, `paraphrase/multiple` | **yes** |
| derived_max_diameter_cm | 1.000 [1.000, 1.000] | 0.909 [0.889, 0.929] | `ocr_noise/canonical`, `paraphrase/multiple` | **yes** |
| attenuation | 1.000 [1.000, 1.000] | 0.802 [0.776, 0.829] | `attenuation_format/low_high`, `attenuation_format/comparative`, `ocr_noise/canonical` | **yes** |

## Accuracy by subtype

Bold = a field the subtype targets. Cells: accuracy (abstained / wrong).

| Category | Subtype | n | Presence found | lesion_count | location_segments | derived_max_diameter_cm | attenuation | All fields |
|---|---|---|---|---|---|---|---|---|
| control | canonical | 200 | 200 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| count_format | digits | 40 | 40 | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| count_format | total_of | 40 | 40 | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| count_format | enumerated | 40 | 40 | **0.300 (0/28)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 0.300 |
| count_format | solitary | 40 | 40 | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| count_format | unquantified | 40 | 40 | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| size_format | mm | 34 | 34 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| size_format | by | 34 | 34 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| size_format | single_dimension | 33 | 33 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| size_format | three_dimensions | 33 | 33 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| size_format | centimeters | 33 | 33 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| size_format | up_to | 33 | 33 | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 |
| largest_order | last_keyword | 67 | 67 | 1.000 (0/0) | **1.000 (0/0)** | **1.000 (0/0)** | **1.000 (0/0)** | 1.000 |
| largest_order | last_implicit | 67 | 67 | 1.000 (0/0) | **1.000 (0/0)** | **1.000 (0/0)** | **1.000 (0/0)** | 1.000 |
| largest_order | middle_implicit | 66 | 66 | 1.000 (0/0) | **1.000 (0/0)** | **1.000 (0/0)** | **1.000 (0/0)** | 1.000 |
| segment_format | roman | 34 | 34 | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| segment_format | subsegment | 34 | 34 | 1.000 (0/0) | **0.471 (18/0)** | 1.000 (0/0) | 1.000 (0/0) | 0.471 |
| segment_format | couinaud | 33 | 33 | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| segment_format | ordinal | 33 | 33 | 1.000 (0/0) | **0.000 (33/0)** | 1.000 (0/0) | 1.000 (0/0) | 0.000 |
| segment_format | hyphen | 33 | 33 | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| segment_format | lobe_only | 33 | 33 | 1.000 (0/0) | **1.000 (0/0)** | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| attenuation_format | dense | 50 | 50 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 |
| attenuation_format | low_high | 50 | 50 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | **0.160 (42/0)** | 0.160 |
| attenuation_format | comparative | 50 | 50 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | **0.000 (50/0)** | 0.000 |
| attenuation_format | separate_sentence | 50 | 50 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | **1.000 (0/0)** | 1.000 |
| ocr_noise | canonical | 200 | 194 | 0.875 (24/1) | 0.725 (38/17) | 0.865 (7/20) | 0.670 (66/0) | 0.505 |
| paraphrase | single | 100 | 100 | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 (0/0) | 1.000 |
| paraphrase | multiple | 100 | 100 | 1.000 (0/0) | 0.600 (0/40) | 0.540 (0/46) | 1.000 (0/0) | 0.540 |

## Compared with `extraction_challenge_v1` + filter (`parser_challenge_liver_only.md`)

Items pair with `extraction_challenge_v1` by ID; 1597 of 1600 are identical to the filtered `extraction_challenge_v1` item. The `extraction_challenge_v1` pooled values include the distractor subtypes, which this set leaves out.

| Field | Pooled, v1 + filter | Pooled, v2 | Needs a model, v1 + filter | Needs a model, v2 |
|---|---|---|---|---|
| lesion_count | 0.934 | 0.912 | yes | yes |
| location_segments | 0.832 | 0.818 | yes | yes |
| derived_max_diameter_cm | 0.927 | 0.909 | yes | yes |
| attenuation | 0.802 | 0.802 | yes | yes |

Accuracies that changed (bold = a field the subtype targets):

| Category | Subtype | Field | V1 + filter | V2 |
|---|---|---|---|---|
| — | — | — | — | — |

## Examples of failures

Generated lesion text only (no dataset text). Up to 2 per subtype, for the fields the subtype targets.

- `count_format/enumerated`: lesion_count: expected 4, got 2
  > A hypoattenuating liver lesion is seen in hepatic segment 2, measuring 8.2 x 4.3 cm. A second hypoattenuating lesion is seen in hepatic segment 5, measuring 3.5 x 2.6 cm. A third hypoattenuating lesion is seen in hepatic segment 7, measuring 3.3 x 2.2 cm. A fourth hypoattenuating lesion is seen in hepatic segment 4, measuring 2.8 x 2.3 cm.
- `count_format/enumerated`: lesion_count: expected 3, got 2
  > A hyperattenuating liver lesion is seen in hepatic segment 7, measuring 8.1 x 5.6 cm. A second hyperattenuating lesion is seen in hepatic segment 3, measuring 2.0 x 1.6 cm. A third hyperattenuating lesion is seen in hepatic segment 2, measuring 0.4 x 0.2 cm.
- `segment_format/ordinal`: location_segments: expected [4], got null
  > However, a hypoattenuating liver lesion is identified in the fourth hepatic segment, measuring 6.0 x 3.4 cm.
- `segment_format/ordinal`: location_segments: expected [1], got null
  > However, an isoattenuating liver lesion is identified in the first hepatic segment, measuring 2.7 x 2.1 cm.
- `segment_format/subsegment`: location_segments: expected [4], got null
  > However, a hypoattenuating liver lesion is identified in hepatic segment IVa, measuring 7.4 x 5.5 cm.
- `segment_format/subsegment`: location_segments: expected [4], got null
  > However, a hyperattenuating liver lesion is identified in hepatic segment IVa, measuring 5.6 x 4.1 cm.
- `attenuation_format/low_high`: attenuation: expected "hypoattenuating", got null
  > However, a low-attenuation liver lesion is identified in hepatic segments 5 and 6, measuring 8.6 x 6.8 cm.
- `attenuation_format/comparative`: attenuation: expected "hypoattenuating", got null
  > However, a liver lesion of lower attenuation than the surrounding liver parenchyma is identified in hepatic segment 3, measuring 2.0 x 1.4 cm.
- `attenuation_format/low_high`: attenuation: expected "hypoattenuating", got null
  > However, a low-density liver lesion is identified in hepatic segment 6, measuring 8.1 x 4.4 cm.
- `attenuation_format/comparative`: attenuation: expected "hypoattenuating", got null
  > However, a liver lesion of lower attenuation than the surrounding liver parenchyma is identified in hepatic segments 2 and 6, measuring 6.5 x 5.3 cm.
- `ocr_noise/canonical`: lesion_count: expected 6, got null; attenuation: expected "hypoattenuating", got null
  > The liver contains six hypoattenuating lcsions. The largest lesion, located in hepatic segment 6, measures 2.6 x 1.8 cm. The other lcsions are located in hcpatic semgents 6, 1, 8, 6, and 5, measuring 1.7 x 1.1 cm, 1.5 x 0.9 cm, 1.1 x 0.7 cm, 0.8 x 0.4 cm, and 0.4 x 0.3 cm, resectively.
- `ocr_noise/canonical`: location_segments: expected [6], got null
  > Howver, a hyperattenuating liver lesion is identified in hepatic segmnt 6, measuring 2.8 x 1.9 cm.
- `paraphrase/multiple`: location_segments: expected [4], got [3]; derived_max_diameter_cm: expected 5.3, got 3.7
  > Three hyperattenuating hepatic masses are seen. The biggest one lies in hepatic segment 4 and measures 5.3 x 3.2 cm. The other lesions are located in hepatic segments 3 and 3, measuring 3.7 x 2.0 cm and 3.3 x 2.9 cm, respectively.
- `paraphrase/multiple`: derived_max_diameter_cm: expected 4.9, got 3.9
  > Six hypoattenuating hepatic masses are seen. The biggest one lies in hepatic segment 1 and measures 4.9 x 3.0 cm. The other lesions are located in hepatic segments 1, 3, 2, 7, and 2, measuring 3.9 x 2.2 cm, 3.1 x 1.9 cm, 3.0 x 2.4 cm, 2.2 x 1.1 cm, and 0.4 x 0.3 cm, respectively.

## Notes
- The items are synthetic in their lesion description; the surrounding report is real Validation text. A failure here shows a phrasing the parser does not handle, not its error rate on the dataset.
- `unquantified` and `lobe_only` expect `null`: the right output is to abstain.
- The Challenge Set must not be used to tune parser rules; doing so would turn it into a development set.
