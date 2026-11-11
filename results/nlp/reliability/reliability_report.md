# NLP Reliability Report

95% percentile bootstrap intervals over studies (2000 resamples, seed 42). Saved predictions only; nothing was refitted.

Presence is reported in three views so abstentions are never hidden:
- **selective**: answered studies only (the view earlier reports used);
- **end to end**: every study counts and an abstention is an error;
- **abstain as negative**: an abstention is read as “no lesion”.

Learned model: the selected word TF-IDF + logistic regression. Validation rows come from the Train-only fit; Test rows come from the frozen Train + Validation fit.

## Reference: Source-derived labels (not a clinical gold standard)

### Validation (1389 studies)

| System | Metric | Estimate [95% CI] |
|---|---|---|
| Rule-based | Coverage | 0.400 [0.374, 0.426] |
| Rule-based | selective accuracy | 0.989 [0.980, 0.996] |
| Rule-based | selective precision | 0.973 [0.950, 0.991] |
| Rule-based | selective recall | 1.000 [1.000, 1.000] |
| Rule-based | selective specificity | 0.982 [0.967, 0.994] |
| Rule-based | selective f1 | 0.987 [0.974, 0.996] |
| Rule-based | end to end accuracy | 0.396 [0.369, 0.422] |
| Rule-based | end to end recall | 1.000 [1.000, 1.000] |
| Rule-based | end to end specificity | 0.282 [0.257, 0.309] |
| Rule-based | abstain as negative accuracy | 0.996 [0.992, 0.999] |
| Rule-based | abstain as negative precision | 0.973 [0.950, 0.991] |
| Rule-based | abstain as negative recall | 1.000 [1.000, 1.000] |
| Rule-based | abstain as negative specificity | 0.995 [0.991, 0.998] |
| Rule-based | abstain as negative f1 | 0.987 [0.974, 0.996] |
| Learned | Coverage | 1.000 [1.000, 1.000] |
| Learned | selective accuracy | 0.999 [0.998, 1.000] |
| Learned | selective precision | 1.000 [1.000, 1.000] |
| Learned | selective recall | 0.995 [0.985, 1.000] |
| Learned | selective specificity | 1.000 [1.000, 1.000] |
| Learned | selective f1 | 0.998 [0.993, 1.000] |
| Learned | end to end accuracy | 0.999 [0.998, 1.000] |
| Learned | end to end recall | 0.995 [0.985, 1.000] |
| Learned | end to end specificity | 1.000 [1.000, 1.000] |
| Learned | abstain as negative accuracy | 0.999 [0.998, 1.000] |
| Learned | abstain as negative precision | 1.000 [1.000, 1.000] |
| Learned | abstain as negative recall | 0.995 [0.985, 1.000] |
| Learned | abstain as negative specificity | 1.000 [1.000, 1.000] |
| Learned | abstain as negative f1 | 0.998 [0.993, 1.000] |

Paired comparison on the 556 studies the rule-based baseline answered (learned minus rule-based):
- F1 difference: 0.0112 [-0.0000, 0.0231]
- Accuracy difference: 0.0090 [0.0000, 0.0180]
- Discordant studies: learned right / rule-based wrong 6, rule-based right / learned wrong 1; exact McNemar p = 0.125

Rule-based field extraction on reference-positive studies:

| Field metric | Eligible | Coverage [95% CI] | Mean on predicted [95% CI] |
|---|---:|---|---|
| lesion count exact | 220 | 1.000 [1.000, 1.000] | 0.736 [0.677, 0.791] |
| lesion count abs error | 220 | 1.000 [1.000, 1.000] | 2.191 [1.464, 3.009] |
| location exact | 215 | 0.972 [0.949, 0.991] | 0.986 [0.967, 1.000] |
| location iou | 215 | 0.972 [0.949, 0.991] | 0.989 [0.975, 1.000] |
| diameter abs error cm | 220 | 0.832 [0.782, 0.882] | 0.374 [0.184, 0.599] |
| diameter within 0.5 cm | 220 | 0.832 [0.782, 0.882] | 0.896 [0.851, 0.940] |
| attenuation exact | 220 | 0.977 [0.955, 0.995] | 0.963 [0.935, 0.986] |

### Test (1390 studies)

| System | Metric | Estimate [95% CI] |
|---|---|---|
| Rule-based | Coverage | 0.378 [0.352, 0.403] |
| Rule-based | selective accuracy | 0.990 [0.982, 0.998] |
| Rule-based | selective precision | 0.978 [0.957, 0.996] |
| Rule-based | selective recall | 1.000 [1.000, 1.000] |
| Rule-based | selective specificity | 0.984 [0.968, 0.997] |
| Rule-based | selective f1 | 0.989 [0.978, 0.998] |
| Rule-based | end to end accuracy | 0.374 [0.348, 0.399] |
| Rule-based | end to end recall | 1.000 [1.000, 1.000] |
| Rule-based | end to end specificity | 0.256 [0.231, 0.281] |
| Rule-based | abstain as negative accuracy | 0.996 [0.993, 0.999] |
| Rule-based | abstain as negative precision | 0.978 [0.957, 0.996] |
| Rule-based | abstain as negative recall | 1.000 [1.000, 1.000] |
| Rule-based | abstain as negative specificity | 0.996 [0.991, 0.999] |
| Rule-based | abstain as negative f1 | 0.989 [0.978, 0.998] |
| Learned | Coverage | 1.000 [1.000, 1.000] |
| Learned | selective accuracy | 0.999 [0.998, 1.000] |
| Learned | selective precision | 1.000 [1.000, 1.000] |
| Learned | selective recall | 0.995 [0.986, 1.000] |
| Learned | selective specificity | 1.000 [1.000, 1.000] |
| Learned | selective f1 | 0.998 [0.993, 1.000] |
| Learned | end to end accuracy | 0.999 [0.998, 1.000] |
| Learned | end to end recall | 0.995 [0.986, 1.000] |
| Learned | end to end specificity | 1.000 [1.000, 1.000] |
| Learned | abstain as negative accuracy | 0.999 [0.998, 1.000] |
| Learned | abstain as negative precision | 1.000 [1.000, 1.000] |
| Learned | abstain as negative recall | 0.995 [0.986, 1.000] |
| Learned | abstain as negative specificity | 1.000 [1.000, 1.000] |
| Learned | abstain as negative f1 | 0.998 [0.993, 1.000] |

Paired comparison on the 525 studies the rule-based baseline answered (learned minus rule-based):
- F1 difference: 0.0089 [-0.0022, 0.0204]
- Accuracy difference: 0.0076 [-0.0019, 0.0171]
- Discordant studies: learned right / rule-based wrong 5, rule-based right / learned wrong 1; exact McNemar p = 0.219

Rule-based field extraction on reference-positive studies:

| Field metric | Eligible | Coverage [95% CI] | Mean on predicted [95% CI] |
|---|---:|---|---|
| lesion count exact | 221 | 1.000 [1.000, 1.000] | 0.805 [0.751, 0.855] |
| lesion count abs error | 221 | 1.000 [1.000, 1.000] | 1.109 [0.774, 1.484] |
| location exact | 217 | 0.982 [0.963, 0.995] | 1.000 [1.000, 1.000] |
| location iou | 217 | 0.982 [0.963, 0.995] | 1.000 [1.000, 1.000] |
| diameter abs error cm | 221 | 0.873 [0.828, 0.919] | 0.276 [0.130, 0.453] |
| diameter within 0.5 cm | 221 | 0.873 [0.828, 0.919] | 0.907 [0.864, 0.947] |
| attenuation exact | 221 | 0.991 [0.977, 1.000] | 0.995 [0.986, 1.000] |

## Reading these numbers
- Interval width reflects sampling uncertainty only; it does not cover label noise in the source-derived reference.
- Overlapping validation and test intervals mean an observed gap may be noise.
- Field means are computed only where a prediction exists; read them with coverage.
