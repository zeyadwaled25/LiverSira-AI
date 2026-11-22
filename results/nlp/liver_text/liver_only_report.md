# Liver-only report text

`data/processed/nlp/liver_nlp_liver_only.csv` (gitignored) is `liver_nlp_clean.csv` with `narrative_clean` reduced to the liver-related text by `liversira_ai.nlp.liver_text` v1.0.0. Kept text is verbatim; `liver_spans` gives its offsets in `narrative_full_clean`. Checks use train + val (7871 studies); Test was not read.

## Output

- Studies: 9261; empty liver text: 0 (0 lesion-positive).
- Characters kept: 26.2% of the full text.
- Input SHA-256 `7bc231b1c2e05044…`, output SHA-256 `420151a7bb3c635f…`.

## Retention (lesion-positive studies)

A value counts as stated if it is written in the full report; lost if it is then missing from the liver text.

| Field | Stated in full text | Lost in liver text |
|---|---|---|
| diameter | 1223 | 0 |
| segments | 1139 | 0 |
| attenuation | 1240 | 0 |
| count | 406 | 0 |

## Other organs left in the liver text

5167 kept spans still name another organ (mostly organ lists such as "The spleen, liver, pancreas, and kidneys are normal"); 0 of them also hold a size in cm or mm.


## Parser v1.1.0 on full vs liver-only text (Validation)

Against source labels. Cells: correct / abstained / wrong. Count, segments, diameter (within 0.5 cm) and attenuation are scored on lesion-positive studies where the label is known.

| Field | Full text | Liver-only text | Fixed | Broken |
|---|---|---|---|---|
| presence | 649 / 740 / 0 | 649 / 740 / 0 | 0 | 0 |
| count | 195 / 21 / 4 | 195 / 21 / 4 | 0 | 0 |
| segments | 210 / 4 / 1 | 210 / 4 / 1 | 0 | 0 |
| diameter | 220 / 0 / 0 | 220 / 0 / 0 | 0 | 0 |
| attenuation | 214 / 2 / 4 | 214 / 2 / 4 | 0 | 0 |

## Notes

- The filter is rule-based: a clause that never names an organ inherits the organ of the clause before it.
- Organ lists that include the liver are kept whole, so other organ names remain in them, sometimes with their volumes or HU values ("volumes of 1500 cc and 150 cc, respectively"); no lesion size is left.
- Joining kept spans inserts ". " after a span without final punctuation; offsets ignore these separators.
