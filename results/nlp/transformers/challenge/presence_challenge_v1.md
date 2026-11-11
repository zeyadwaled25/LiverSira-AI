# Presence Challenge Set (presence_challenge_v1)

Built from Validation reports only (seed 42); Test was not read. The text is in `data/processed/nlp/challenge/presence_challenge_v1.jsonl` (gitignored); hashes and IDs are in the manifest.

| Category | Expected label | Scored | Positive | Negative | Unscored |
|---|---|---|---|---|---|
| negation | false | True | 0 | 150 | 0 |
| other_organ_lesion | false | True | 0 | 150 | 0 |
| liver_removed | false | True | 0 | 150 | 0 |
| paraphrase | unchanged | True | 75 | 75 | 0 |
| ocr_noise | unchanged | True | 75 | 75 | 0 |
| liver_last | unchanged | True | 75 | 75 | 0 |
| hedged | none | False | 0 | 0 | 150 |

Category definitions are in the docstring of `experiments/transformers/presence/build_challenge_set.py`.
