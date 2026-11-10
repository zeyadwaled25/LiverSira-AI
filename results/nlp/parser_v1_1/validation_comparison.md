# Parser v1.1 vs v1.0: Validation

rule-based-v1.1.0 / parser 1.1.0 against the frozen rule-based-v1.0.0 (saved predictions). Validation split only; Test reports were not parsed. 95% bootstrap intervals over studies (2000 resamples, seed 42).

## Against source-derived labels

| Metric | v1.0 | v1.1 |
|---|---|---|
| Presence coverage | 0.400 [0.374, 0.426] | 0.467 [0.441, 0.493] |
| Presence selective F1 | 0.987 [0.974, 0.996] | 1.000 [1.000, 1.000] |
| Presence selective precision | 0.973 [0.950, 0.991] | 1.000 [1.000, 1.000] |
| Presence end-to-end accuracy | 0.396 [0.369, 0.422] | 0.467 [0.441, 0.493] |
| Presence end-to-end specificity | 0.282 [0.257, 0.309] | 0.367 [0.340, 0.395] |
| Presence abstain-as-negative F1 | 0.987 [0.974, 0.996] | 1.000 [1.000, 1.000] |

Field extraction on reference-positive studies (coverage, then mean where predicted):

| Field | v1.0 coverage | v1.0 mean | v1.1 coverage | v1.1 mean |
|---|---|---|---|---|
| Count exact | 1.000 [1.000, 1.000] | 0.736 [0.677, 0.791] | 0.905 [0.864, 0.941] | 0.980 [0.959, 0.995] |
| Count abs error | 1.000 [1.000, 1.000] | 2.191 [1.464, 3.009] | 0.905 [0.864, 0.941] | 0.035 [0.005, 0.077] |
| Location exact | 0.972 [0.949, 0.991] | 0.986 [0.967, 1.000] | 0.981 [0.958, 0.995] | 0.995 [0.986, 1.000] |
| Diameter abs error (cm) | 0.832 [0.782, 0.882] | 0.374 [0.184, 0.599] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| Diameter within 0.5 cm | 0.832 [0.782, 0.882] | 0.896 [0.851, 0.940] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Attenuation exact | 0.977 [0.955, 0.995] | 0.963 [0.935, 0.986] | 0.991 [0.977, 1.000] | 0.982 [0.963, 0.995] |

Presence on the 555 studies both versions answered (v1.1 minus v1.0): F1 difference 0.0112 [0.0023, 0.0221]; v1.1 right / v1.0 wrong 5, v1.0 right / v1.1 wrong 0; exact McNemar p = 0.0625.

## Notes
- **These are development estimates.** v1.1 was built by inspecting Validation errors, so its Validation scores are optimistic. Unbiased estimates need the human gold standard or a single Test evaluation after v1.1 is frozen.
- v1.0 set an unstated count to 1; v1.1 leaves "multiple lesions" unquantified, so its count coverage is lower by design and its count accuracy is measured only where a count is stated.
- Remaining validation disagreements were reviewed: most are cases where the report text and the source label disagree (for example the text states a different attenuation for the largest lesion). They are not tuned away.
- The frozen audit, split and gold-set strata still use parser v1.0.
