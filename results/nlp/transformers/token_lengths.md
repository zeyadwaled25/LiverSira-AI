# Token lengths of narrative_clean (Train + Validation)

6482 Train and 1389 Validation reports; Test was not read. Limit 512 tokens including [CLS] and [SEP]. Head+tail keeps the first 128 and last 382 content tokens.

| Model | Lowercase | Vocab cased | Median | p95 | p99 | Max | Over 512 (Train / Val) | Lesion-positive over 512 | IMPRESSION beyond 512 |
|---|---|---|---|---|---|---|---|---|---|
| bert | True | False | 187 | 426 | 565 | 1220 | 150 (120 / 30) | 73 | 50 |
| biobert | False | True | 198 | 445 | 591 | 1289 | 191 (154 / 37) | 100 | 69 |
| clinicalbert | False | True | 198 | 445 | 591 | 1289 | 191 (154 / 37) | 100 | 69 |

Over-limit reports that lose at least one liver mention under each strategy:

| Model | Head only | Head only, no liver left | Head 128 + tail 382 | Liver-anchored window | Chunks (max per report) |
|---|---|---|---|---|---|
| bert | 41 | 0 | 50 | 34 | 0 (3) |
| biobert | 49 | 0 | 58 | 45 | 0 (4) |
| clinicalbert | 49 | 0 | 58 | 45 | 0 (4) |

Over-limit reports where parser v1.1 gives a different value on the head-truncated text (first 510 content tokens) than on the full text:

| Model | has_lesion | lesion_count | reported_dimensions_cm | location_segments | attenuation |
|---|---|---|---|---|---|
| bert | 0 | 1 | 1 | 4 | 5 |
| biobert | 0 | 1 | 1 | 4 | 5 |
| clinicalbert | 0 | 1 | 1 | 4 | 5 |

## Policy
- **Presence (6.1): head truncation to 512 tokens.** Every over-limit report keeps at least one liver mention, and parser v1.1 reaches the same presence decision on the truncated text for all of them.
- **Extraction (6.2): overlapping chunks** (510 content tokens, stride 382; at most 4 chunks per report). Head truncation changes a few extracted values, so extraction models must see the whole report.
- Both are set in `experiments/transformers/models.py`.

## Notes
- Liver mentions are found with the pattern `\b(liver|hepatic|hepato\w*)\b` and mapped to token positions with the tokenizer offsets.
- "Drops liver" counts over-limit reports where at least one liver mention falls outside the kept tokens.
- Revisions are pinned in `experiments/transformers/models.py`.
