# Phase 7 report: LLM extraction (frozen prompt v2)

Two API models, kept by Phase 7 Amendment 3, run once on the fixed subsample (`results/nlp/llm/subsample_v1.json`, seed 2027) with the frozen prompt v2 (hash `337a2921…`). They read the liver-only text, like every extraction candidate. Test was not read. API models are references: they are not deployable (plan §2), so nothing here enters the Phase 8 routing decision.

## What the subsample is

- **Validation, lesion-positive: all 220 of 220** lesion-positive Validation reports. The field numbers are therefore computed on the whole lesion-positive population, with the same n as the frozen candidates (`results/nlp/phase8/validation_rescored.md`).
- **Validation, lesion-free: 180 sampled of 1169.** Every presence number on lesion-free reports (abstention share, specificity) comes from this sample only.
- **Extraction Challenge v3:** 260 items, 20 per subtype. **Presence Challenge v1:** 210 items, 30 per category.
- **Challenge-A, diagnosis only.** Both Challenge Sets were seen during development (Phase 6). By the Phase 8 plan their numbers explain behaviour and never enter a decision; a gap they show is future work, not a result.

Accuracy cells: estimate [Wilson 95% interval] (n). C3 and H: mean ± SD over seeds 13 / 42 / 87, [envelope of the three seeds' Wilson intervals].

## Coverage

| Model | validation | extraction_challenge_v3 | presence_challenge_v1 | repeat |
|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 400/400 | 260/260 | 210/210 | 50/50 |
| Gemini 3.5 Flash-Lite (Google, closed) | 400/400 | 174/260 (rest not run: daily quota ended (Amendment 3)) | 0/210 (not run: daily quota (Amendment 3)) | 0/50 (not run: daily quota ended (Amendment 3)) |

## Validation: fields on lesion-positive reports (verifiable fields only)

| System | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 0.983 [0.951, 0.994] (n=174) | 0.913 [0.865, 0.945] (n=196) | 0.977 [0.948, 0.990] (n=219) | 0.995 [0.975, 0.999] (n=220) |
| Gemini 3.5 Flash-Lite (Google, closed) | 0.994 [0.968, 0.999] (n=174) | 0.990 [0.964, 0.997] (n=196) | 0.977 [0.948, 0.990] (n=219) | 0.991 [0.967, 0.998] (n=220) |
| C1 parser v1.1 | 1.000 [0.978, 1.000] (n=174) | 0.990 [0.964, 0.997] (n=196) | 0.977 [0.948, 0.990] (n=219) | 1.000 [0.983, 1.000] (n=220) |
| C2 parser v1.2 | 1.000 [0.978, 1.000] (n=174) | 0.990 [0.964, 0.997] (n=196) | 0.977 [0.948, 0.990] (n=219) | 1.000 [0.983, 1.000] (n=220) |
| C3 first model (6.2) | 1.000 ± 0.000 [0.978, 1.000] (n=174) | 0.974 ± 0.000 [0.942, 0.989] (n=196) | 1.000 ± 0.000 [0.983, 1.000] (n=219) | 1.000 ± 0.000 [0.983, 1.000] (n=220) |
| H = C2 + C3 | 1.000 ± 0.000 [0.978, 1.000] (n=174) | 0.995 ± 0.000 [0.972, 0.999] (n=196) | 0.986 ± 0.000 [0.961, 0.995] (n=219) | 1.000 ± 0.000 [0.983, 1.000] (n=220) |

**Against C2 parser v1.2** (errors, and whether the Wilson intervals overlap):

| Model | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 3 vs 0 errors, overlap | 17 vs 2 errors, **no overlap** | 5 vs 5 errors, overlap | 1 vs 0 errors, overlap |
| Gemini 3.5 Flash-Lite (Google, closed) | 1 vs 0 errors, overlap | 2 vs 2 errors, overlap | 5 vs 5 errors, overlap | 2 vs 0 errors, overlap |

Where the intervals overlap, the difference is one to a few reports and is within noise at this n. Where they do not overlap, the difference is likely real.

## Validation: presence

The prompt tells the model to answer "unclear" when the text does not state presence; parser v1.1/v1.2 follow the same rule. An "unclear" is an abstention, not an error.

| System | positives: unclear | lesion-free (sample): unclear | wrong answers |
|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 0 of 220 | 136 of 180 (76%) | 0 |
| Gemini 3.5 Flash-Lite (Google, closed) | 0 of 220 | 73 of 180 (41%) | 0 |
| P1 frozen LR | 0 of 220 | 0 of 180 (0%) | 1 |

**Scoring views (abstain-as-negative and strict).** The LLM is not a presence candidate (P1 LR and P2 ClinicalBERT are). LR is shown for scale only: it always answers. Specificity uses the 180-report sample.

| System | sensitivity | specificity, abstain-as-negative ("unclear" = no lesion) | specificity, strict ("unclear" = miss) |
|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 1.000 [0.983, 1.000] (n=220) | 1.000 [0.979, 1.000] (n=180) | 0.244 [0.187, 0.312] (n=180) |
| Gemini 3.5 Flash-Lite (Google, closed) | 1.000 [0.983, 1.000] (n=220) | 1.000 [0.979, 1.000] (n=180) | 0.594 [0.521, 0.663] (n=180) |
| P1 frozen LR | 0.995 [0.975, 0.999] (n=220) | 1.000 [0.979, 1.000] (n=180) | 1.000 [0.979, 1.000] (n=180) |

## Extraction Challenge v3 (Challenge-A, diagnosis only)

Every system in a row is scored on the same items. **Gemini Flash-Lite ran 174 of 260** (its daily quota ended), so it is compared only in the second table, on its own items. It ran no item of: non_lesion/liver_span, non_lesion/node, non_lesion/vessel. **Whether a model tells a liver span, node or vessel size from a lesion was answered by Nemotron alone.**

### All 260 items

| Category | Subtype | Field | n | Nemotron 3.5 Lightning (NVIDIA, open weights) | C1 parser v1.1 | C2 parser v1.2 | C3 first model (6.2) | H = C2 + C3 |
|---|---|---|---|---|---|---|---|---|
| control | canonical | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | location_segments | 20 | 0.95 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | attenuation | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| count | altogether | lesion_count | 20 | 0.95 | 0.45 | 0.45 | 0.97 ± 0.05 | 0.97 ± 0.05 |
| count | isolated | lesion_count | 20 | 1.00 | 0.95 | 0.95 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| count | listed_sentences | lesion_count | 20 | 0.85 | 0.00 | 0.00 | 0.02 ± 0.02 | 0.00 ± 0.00 |
| count | unquantified | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 0.43 ± 0.02 | 0.43 ± 0.02 |
| non_lesion | liver_span | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| non_lesion | liver_span | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 0.97 ± 0.05 | 1.00 ± 0.00 |
| non_lesion | node | lesion_count | 20 | 0.95 | 0.95 | 0.95 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| non_lesion | node | derived_max_diameter_cm | 20 | 1.00 | 0.95 | 0.95 | 0.52 ± 0.06 | 1.00 ± 0.00 |
| non_lesion | vessel | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| non_lesion | vessel | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| order | ascending | location_segments | 20 | 0.15 | 0.05 | 0.05 | 0.87 ± 0.08 | 0.05 ± 0.00 |
| order | ascending | derived_max_diameter_cm | 20 | 0.70 | 0.00 | 0.00 | 0.62 ± 0.17 | 0.00 ± 0.00 |
| order | ascending | attenuation | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| order | ascending | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 0.90 ± 0.11 | 1.00 ± 0.00 |
| order | middle | location_segments | 20 | 0.00 | 1.00 | 1.00 | 0.75 ± 0.12 | 1.00 ± 0.00 |
| order | middle | derived_max_diameter_cm | 20 | 0.80 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| order | middle | attenuation | 20 | 0.70 | 1.00 | 1.00 | 0.90 ± 0.08 | 1.00 ± 0.00 |
| order | smallest_first | location_segments | 20 | 0.45 | 1.00 | 1.00 | 0.95 ± 0.00 | 1.00 ± 0.00 |
| order | smallest_first | derived_max_diameter_cm | 20 | 0.60 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| order | smallest_first | attenuation | 20 | 0.65 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| repeat | multiple | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | multiple | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | single | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | single | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

### The 174 items Gemini Flash-Lite ran (every system on these items)

| Category | Subtype | Field | n | Gemini 3.5 Flash-Lite (Google, closed) | Nemotron 3.5 Lightning (NVIDIA, open weights) | C1 parser v1.1 | C2 parser v1.2 | C3 first model (6.2) | H = C2 + C3 |
|---|---|---|---|---|---|---|---|---|---|
| control | canonical | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | location_segments | 20 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | derived_max_diameter_cm | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| control | canonical | attenuation | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| count | altogether | lesion_count | 20 | 1.00 | 0.95 | 0.45 | 0.45 | 0.97 ± 0.05 | 0.97 ± 0.05 |
| count | isolated | lesion_count | 20 | 0.95 | 1.00 | 0.95 | 0.95 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| count | listed_sentences | lesion_count | 20 | 0.85 | 0.85 | 0.00 | 0.00 | 0.02 ± 0.02 | 0.00 ± 0.00 |
| count | unquantified | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 ± 0.02 | 0.43 ± 0.02 |
| order | ascending | location_segments | 20 | 1.00 | 0.15 | 0.05 | 0.05 | 0.87 ± 0.08 | 0.05 ± 0.00 |
| order | ascending | derived_max_diameter_cm | 20 | 1.00 | 0.70 | 0.00 | 0.00 | 0.62 ± 0.17 | 0.00 ± 0.00 |
| order | ascending | attenuation | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| order | ascending | lesion_count | 20 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 ± 0.11 | 1.00 ± 0.00 |
| order | middle | location_segments | 19 | 1.00 | 0.00 | 1.00 | 1.00 | 0.75 ± 0.15 | 1.00 ± 0.00 |
| order | middle | derived_max_diameter_cm | 19 | 1.00 | 0.79 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| order | middle | attenuation | 19 | 1.00 | 0.68 | 1.00 | 1.00 | 0.89 ± 0.09 | 1.00 ± 0.00 |
| order | smallest_first | location_segments | 20 | 1.00 | 0.45 | 1.00 | 1.00 | 0.95 ± 0.00 | 1.00 ± 0.00 |
| order | smallest_first | derived_max_diameter_cm | 20 | 1.00 | 0.60 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| order | smallest_first | attenuation | 20 | 1.00 | 0.65 | 1.00 | 1.00 | 0.98 ± 0.02 | 1.00 ± 0.00 |
| repeat | multiple | lesion_count | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | multiple | derived_max_diameter_cm | 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | single | lesion_count | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| repeat | single | derived_max_diameter_cm | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

Reading v3: a subtype holds 20 items (fewer for Gemini), so one subtype says how a system handles one pattern, not that it is better overall. Gemini's 1.00 against H's 0.05 on `order/ascending` segments is such a diagnostic. Two H weaknesses show up here beyond the known unquantified-count one:

- **`order/*` (largest lesion not the first one):** parser v1.2 answers with the first lesion's values instead of leaving the field empty, so H keeps them and C3 is never asked.
- **`count/listed_sentences`:** the parser and H miss counts given as one lesion per sentence.

Both are recorded for Phase 8 as Challenge-A observations (Phase 8 plan, Amendment 4), with no change to H or the parser.

## Presence Challenge v1 (Challenge-A, diagnosis only)

Gemini Flash-Lite: not run (daily quota, Amendment 3). **"unclear" is never counted as correct:** accuracy = correct ÷ n, so a system that abstains on every item scores 0 with 0 wrong answers (as all three do on `liver_removed`). Hedged items have no correct answer; the share answered "unclear" is the measure.

| Category | labels | System | yes / no / unclear | correct | wrong | accuracy (unclear not correct) |
|---|---|---|---|---|---|---|
| hedged | none (hedged) 30 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 0 / 0 / 30 | — | — | hedged: 30 of 30 unclear |
| hedged | none (hedged) 30 | C1 parser v1.1 | 0 / 0 / 30 | — | — | hedged: 30 of 30 unclear |
| hedged | none (hedged) 30 | C2 parser v1.2 | 0 / 0 / 30 | — | — | hedged: 30 of 30 unclear |
| liver_last | lesion 15, no lesion 15 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 15 / 4 / 11 | 19 | 0 | 0.633 [0.455, 0.781] (n=30) |
| liver_last | lesion 15, no lesion 15 | C1 parser v1.1 | 15 / 6 / 9 | 21 | 0 | 0.700 [0.521, 0.833] (n=30) |
| liver_last | lesion 15, no lesion 15 | C2 parser v1.2 | 15 / 6 / 9 | 21 | 0 | 0.700 [0.521, 0.833] (n=30) |
| liver_removed | no lesion 30 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 0 / 0 / 30 | 0 | 0 | 0.000 [0.000, 0.114] (n=30) |
| liver_removed | no lesion 30 | C1 parser v1.1 | 0 / 0 / 30 | 0 | 0 | 0.000 [0.000, 0.114] (n=30) |
| liver_removed | no lesion 30 | C2 parser v1.2 | 0 / 0 / 30 | 0 | 0 | 0.000 [0.000, 0.114] (n=30) |
| negation | no lesion 30 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 0 / 30 / 0 | 30 | 0 | 1.000 [0.886, 1.000] (n=30) |
| negation | no lesion 30 | C1 parser v1.1 | 0 / 30 / 0 | 30 | 0 | 1.000 [0.886, 1.000] (n=30) |
| negation | no lesion 30 | C2 parser v1.2 | 0 / 30 / 0 | 30 | 0 | 1.000 [0.886, 1.000] (n=30) |
| ocr_noise | lesion 12, no lesion 18 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 12 / 5 / 13 | 17 | 0 | 0.567 [0.392, 0.726] (n=30) |
| ocr_noise | lesion 12, no lesion 18 | C1 parser v1.1 | 12 / 7 / 11 | 19 | 0 | 0.633 [0.455, 0.781] (n=30) |
| ocr_noise | lesion 12, no lesion 18 | C2 parser v1.2 | 12 / 7 / 11 | 19 | 0 | 0.633 [0.455, 0.781] (n=30) |
| other_organ_lesion | no lesion 30 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 0 / 1 / 29 | 1 | 0 | 0.033 [0.006, 0.167] (n=30) |
| other_organ_lesion | no lesion 30 | C1 parser v1.1 | 0 / 4 / 26 | 4 | 0 | 0.133 [0.053, 0.297] (n=30) |
| other_organ_lesion | no lesion 30 | C2 parser v1.2 | 0 / 4 / 26 | 4 | 0 | 0.133 [0.053, 0.297] (n=30) |
| paraphrase | lesion 11, no lesion 19 | Nemotron 3.5 Lightning (NVIDIA, open weights) | 11 / 12 / 7 | 23 | 0 | 0.767 [0.591, 0.882] (n=30) |
| paraphrase | lesion 11, no lesion 19 | C1 parser v1.1 | 11 / 15 / 4 | 26 | 0 | 0.867 [0.703, 0.947] (n=30) |
| paraphrase | lesion 11, no lesion 19 | C2 parser v1.2 | 11 / 15 / 4 | 26 | 0 | 0.867 [0.703, 0.947] (n=30) |

## Repeat: run-to-run agreement with the Validation run

Gemini Flash-Lite: not run (daily quota ended, Amendment 3), so its stability is not measured.

| Model | pairs | count | segments | attenuation | diameter_cm | presence | all identical |
|---|---|---|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 50 | 0.980 [0.895, 0.996] (n=50) | 0.940 [0.838, 0.979] (n=50) | 1.000 [0.929, 1.000] (n=50) | 0.980 [0.895, 0.996] (n=50) | 1.000 [0.929, 1.000] (n=50) | 0.940 [0.838, 0.979] (n=50) |

## Evidence rejections: was the rejected value right?

A value is rejected when its quote is not in the text or does not show it; the field is then left empty. Here each rejected raw value is scored against the label, over every evaluation set. This is a description: the verifier is frozen and is not changed. "value right" = the check was too strict; "value wrong" = the check stopped an error; "not scored" = no label for that field on that item (an unverifiable Validation field, a v3 field the subtype does not target, or any field but presence on the Presence Challenge). On Validation, lesion-free reports count too: their labels are count 0 and no other value.

| Model | Field | rejections | value right | value wrong | not scored |
|---|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | has_lesion | 11 | 0 | 0 | 11 |
| Nemotron 3.5 Lightning (NVIDIA, open weights) | lesion_count | 169 | 118 | 1 | 50 |
| Nemotron 3.5 Lightning (NVIDIA, open weights) | location_segments | 44 | 1 | 35 | 8 |
| Nemotron 3.5 Lightning (NVIDIA, open weights) | reported_dimensions_cm | 8 | 7 | 1 | 0 |
| Nemotron 3.5 Lightning (NVIDIA, open weights) | attenuation | 4 | 0 | 2 | 2 |
| Gemini 3.5 Flash-Lite (Google, closed) | has_lesion | 4 | 4 | 0 | 0 |
| Gemini 3.5 Flash-Lite (Google, closed) | lesion_count | 22 | 4 | 0 | 18 |

Why Nemotron's count rejections are mostly right values: on a lesion-free report it often answers presence "unclear" with a count of 0 and no quote. The check accepts a 0 without a quote only after a "no", so it rejects a right value; the field is empty either way on such a report. Its segment rejections are the opposite: the check mostly stopped wrong values.

## LLM measures (all evaluation items)

Latency is per call as the client saw it, so it includes the SDK's waits between retries.

| Model | items | invalid | refusal | evidence rejections (per field) | duplicate-quote items | tokens in / out | latency p50 / p95 (ms) | cost |
|---|---|---|---|---|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 920 | 0.0000 | 0.0000 | has_lesion 11, lesion_count 169, location_segments 44, reported_dimensions_cm 8, attenuation 4 | 263 | 1,413,665 / 118,727 | 833.6 / 6779.2 | $0.00 (free tier) |
| Gemini 3.5 Flash-Lite (Google, closed) | 574 | 0.0000 | 0.0000 | has_lesion 4, lesion_count 22, location_segments 0, reported_dimensions_cm 0, attenuation 0 | 219 | 841,226 / 89,713 | 1311.7 / 25064.7 | $0.00 (free tier) |

## Transport errors (timeouts, connection errors, HTTP 408/409/429/5xx)

The SDK retries them. An item that fails every retry is skipped (no row) and called again by the next run, so none is counted as invalid output or as an abstention.

- **Before the log (counts only).** Until `scripts/nlp/llm/run_logged.py` existed, only the number of skipped items per run was kept (`results/nlp/llm/transport_errors_unlogged.json`): no error type, no item, and no failed attempt that a later retry recovered.
- **Since the log.** Every failed attempt and every skip is in `<set folder>/transport_log.jsonl`.
- **Pacing change.** 24 November 2026, 12:53 UTC, during the Nemotron runs: at 40 requests per minute NVIDIA rejected most requests with HTTP 429 (281 failed attempts, 51 skipped items), so `config.json` was set to 10 requests per minute and 8 SDK retries (from the default 4); after it, 161 attempts still got 429 and 4 items were skipped, all finished on a rerun. This paces calls only, as the freeze allows (`prompt_freeze.json` note); outputs and settings are unchanged.

| Model | skipped items, before the log (count only) | failed attempts, logged | skipped items, logged |
|---|---|---|---|
| Nemotron 3.5 Lightning (NVIDIA, open weights) | 232 | http_429 442, http_502 1 | 55 |
| Gemini 3.5 Flash-Lite (Google, closed) | 321 | 0 | 0 |

## Limitations

- **Gemini Flash-Lite is partial:** v3 174 of 260 (no non-lesion subtype), no Presence Challenge, no repeat. Its quota of 500 ended after about 360 successful calls; where the rest went is not known (retried attempts were not logged in that run).
- **NVIDIA throttling:** 442 Nemotron attempts were rejected with HTTP 429 after the log started; all items finished after the pacing change.
- **Presence on lesion-free reports** rests on a sample of 180 of 1,169.
- **v3 and Presence Challenge v1 were seen during development** (Challenge-A): diagnosis only.
- **Labels are source-derived** until the Gold Set is finished; no clinical accuracy is claimed.
- **Data:** only the liver-only text of public, de-identified AbdomenAtlas 3.0 reports was sent. Gemini's free tier may use inputs (accepted by the team, Amendment 1). Result files hold liver-sentence excerpts under CC BY-NC-SA 4.0 (README, Licence and data notice).
- **gpt-oss-120b** was dropped for the rate limit only; its development result is below as a reference.

## Reference only: gpt-oss-120b on the development set

Development set (Train), prompt v2, 49 of 100 items. Dropped from the evaluation for the rate limit only (Phase 7 Amendment 3). A reference, not an evaluation result.

| Field | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|
| gpt-oss-120b | 1.000 [0.883, 1.000] (n=29) | 1.000 [0.886, 1.000] (n=30) | 1.000 [0.893, 1.000] (n=32) | 1.000 [0.893, 1.000] (n=32) |
