# Parser v1.1 vs v1.0: Test (single frozen evaluation)

Parser 1.1.0 was frozen at commit `f16abea` before any Test report was parsed, and this evaluation runs once. Test studies: 1390. Reference: source-derived labels (not a clinical gold standard). 95% bootstrap intervals over studies (2000 resamples, seed 42).

## Against source-derived labels


| Metric | v1.0 | v1.1 |
|---|---|---|
| Presence coverage | 0.378 [0.352, 0.403] | 0.445 [0.419, 0.471] |
| Presence selective F1 | 0.989 [0.978, 0.998] | 1.000 [1.000, 1.000] |
| Presence selective precision | 0.978 [0.957, 0.996] | 1.000 [1.000, 1.000] |
| Presence end-to-end accuracy | 0.374 [0.348, 0.399] | 0.445 [0.419, 0.471] |
| Presence end-to-end specificity | 0.256 [0.231, 0.281] | 0.341 [0.313, 0.368] |
| Presence abstain-as-negative F1 | 0.989 [0.978, 0.998] | 0.998 [0.993, 1.000] |

Field extraction on reference-positive studies (coverage, then mean where predicted):

| Field | v1.0 coverage | v1.0 mean | v1.1 coverage | v1.1 mean |
|---|---|---|---|---|
| Count exact | 1.000 [1.000, 1.000] | 0.805 [0.751, 0.855] | 0.919 [0.882, 0.955] | 0.980 [0.960, 0.995] |
| Count abs error | 1.000 [1.000, 1.000] | 1.109 [0.774, 1.484] | 0.919 [0.882, 0.955] | 0.020 [0.005, 0.040] |
| Location exact | 0.982 [0.963, 0.995] | 1.000 [1.000, 1.000] | 0.982 [0.963, 0.995] | 0.991 [0.976, 1.000] |
| Diameter abs error (cm) | 0.873 [0.828, 0.919] | 0.276 [0.130, 0.453] | 0.995 [0.986, 1.000] | 0.026 [0.000, 0.064] |
| Diameter within 0.5 cm | 0.873 [0.828, 0.919] | 0.907 [0.864, 0.947] | 0.995 [0.986, 1.000] | 0.991 [0.977, 1.000] |
| Attenuation exact | 0.991 [0.977, 1.000] | 0.995 [0.986, 1.000] | 0.982 [0.964, 0.995] | 0.995 [0.986, 1.000] |

Presence on the 520 studies both versions answered (v1.1 minus v1.0): F1 difference 0.0023 [0.0000, 0.0071]; v1.1 right / v1.0 wrong 1, v1.0 right / v1.1 wrong 0; exact McNemar p = 1.

## Notes
- v1.1 was developed on Validation only; these Test numbers are its unbiased estimate against source labels. Clinical accuracy still needs the human gold standard.
- v1.1 leaves "multiple lesions" unquantified instead of setting 1, so count coverage is lower by design.
- Do not change the parser in response to these numbers and re-run on Test; judge later versions on Validation and the gold set.
