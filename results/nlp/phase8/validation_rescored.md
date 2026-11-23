# Validation re-scored: old rule vs verifiable fields only (Amendment 3)

Lesion-positive Validation reports: 220. Unverifiable fields (null target on a real report with a lesion: the source value is not found in the text): count 46, segments 24, attenuation 1, diameter_cm 0.

Old rule: a null target must be matched by a null prediction, so a system that reads a value the text does state is marked wrong whenever the source label disagrees with the text. Corrected rule: those fields are not scored. Parsers: one run, Wilson 95% interval. Models and hybrids: mean ± SD over seeds 13 / 42 / 87.

## Old rule (6.2)

| System | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|
| C1 parser v1.1 | 0.886 [0.838, 0.922] (n=220) | 0.918 [0.874, 0.948] (n=220) | 0.973 [0.942, 0.987] (n=220) | 1.000 [0.983, 1.000] (n=220) |
| C2 parser v1.2 | 0.882 [0.832, 0.918] (n=220) | 0.918 [0.874, 0.948] (n=220) | 0.973 [0.942, 0.987] (n=220) | 1.000 [0.983, 1.000] (n=220) |
| C3 first model (6.2, unmasked) | 0.994 ± 0.003 (n=220) | 0.971 ± 0.003 (n=220) | 0.995 ± 0.000 (n=220) | 1.000 ± 0.000 (n=220) |
| C3 first model v2 (masked, Amendment 3) | 0.783 ± 0.009 (n=220) | 0.883 ± 0.005 (n=220) | 0.994 ± 0.003 (n=220) | 1.000 ± 0.000 (n=220) |
| C4 per-lesion model | 0.994 ± 0.003 (n=220) | 0.973 ± 0.000 (n=220) | 0.995 ± 0.000 (n=220) | 1.000 ± 0.000 (n=220) |
| H = C2 + C3 (6.2) | 0.882 ± 0.000 (n=220) | 0.920 ± 0.003 (n=220) | 0.982 ± 0.000 (n=220) | 1.000 ± 0.000 (n=220) |
| H = C2 + C3 v2 | 0.791 ± 0.000 (n=220) | 0.886 ± 0.000 (n=220) | 0.982 ± 0.000 (n=220) | 1.000 ± 0.000 (n=220) |

## Corrected rule (verifiable fields only)

| System | count | segments | attenuation | diameter_cm |
|---|---|---|---|---|
| C1 parser v1.1 | 1.000 [0.978, 1.000] (n=174) | 0.990 [0.964, 0.997] (n=196) | 0.977 [0.948, 0.990] (n=219) | 1.000 [0.983, 1.000] (n=220) |
| C2 parser v1.2 | 1.000 [0.978, 1.000] (n=174) | 0.990 [0.964, 0.997] (n=196) | 0.977 [0.948, 0.990] (n=219) | 1.000 [0.983, 1.000] (n=220) |
| C3 first model (6.2, unmasked) | 1.000 ± 0.000 (n=174) | 0.974 ± 0.000 (n=196) | 1.000 ± 0.000 (n=219) | 1.000 ± 0.000 (n=220) |
| C3 first model v2 (masked, Amendment 3) | 0.990 ± 0.012 (n=174) | 0.991 ± 0.006 (n=196) | 0.998 ± 0.003 (n=219) | 1.000 ± 0.000 (n=220) |
| C4 per-lesion model | 1.000 ± 0.000 (n=174) | 0.974 ± 0.000 (n=196) | 1.000 ± 0.000 (n=219) | 1.000 ± 0.000 (n=220) |
| H = C2 + C3 (6.2) | 1.000 ± 0.000 (n=174) | 0.995 ± 0.000 (n=196) | 0.986 ± 0.000 (n=219) | 1.000 ± 0.000 (n=220) |
| H = C2 + C3 v2 | 1.000 ± 0.000 (n=174) | 0.995 ± 0.000 (n=196) | 0.986 ± 0.000 (n=219) | 1.000 ± 0.000 (n=220) |
