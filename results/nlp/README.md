# NLP results

Every file here is written by a script; nothing is edited by hand. Each folder
below has a `.md` report to read first, and `.json` files holding the same
numbers in machine-readable form. `.jsonl` files hold one prediction per study.

All scores use source-derived labels. Clinical accuracy needs the human gold
standard.

## Where to start

| Question | Read |
|---|---|
| How good is the rule-based parser? | `parser_v1_1/test_evaluation.md` |
| How good is the learned (TF-IDF + LR) model? | `final_test/learned_presence_test_report.md` |
| How certain are those numbers? | `reliability/reliability_report.md` |
| Do transformers beat LR and the parser? | `transformers/presence/comparison.md` |
| Which extraction fields does the parser fail on (the 6.2 gate)? | `transformers/extraction/parser_challenge.md` |
| What does the liver-only text keep and drop? | `liver_text/liver_only_report.md` |
| Does the liver-only text change the 6.2 gate? | `transformers/extraction/parser_challenge_liver_only.md` |
| How does the parser do on the liver-only Challenge Set (v2)? | `transformers/extraction/parser_challenge_v2_liver.md` |
| What will the 6.2 extraction model train on? | `transformers/extraction/extraction_train_v1.md` |

## Folders

| Folder | What it holds | Written by |
|---|---|---|
| `baseline/` | Parser v1.0 evaluation on Validation and Test, its error analysis, and which studies are eligible for each field (`_updated` = after the audit correction) | `evaluate_baseline.py`, `error_analysis.py`, `create_benchmark_manifest.py` |
| `parser_v1_1/` | Parser v1.1 vs v1.0 on Validation, and the single Test run of v1.1 | `validate_parser.py`, `evaluate_parser_test.py` |
| `learned_presence_validation/`, `model_artifacts/` | First LR presence model on Validation, and its configuration | `learned_presence/train_learned_presence.py` |
| `architecture_comparison/` | Four classical models compared on Validation, plus a leakage check | `learned_presence/compare_learned_presence.py` |
| `final_test/` | The frozen LR model (`.joblib`, config, hash manifest) and its single Test run | `learned_presence/final_evaluate_learned_presence.py`, `export_frozen_model.py` |
| `predictions/` | Per-study predictions that the reports above are computed from | the scripts above |
| `reliability/` | Bootstrap 95% CIs and coverage-adjusted metrics for the parser and LR | `reliability_report.py` |
| `gold_set/` | IDs and hashes of the gold-set sample (no report text) | `gold_set.py sample` |
| `liver_text/` | Checks of the liver-only copy of the cleaned CSV: values kept, other organs left, parser v1.1 on full vs liver-only text | `build_liver_only_dataset.py` |
| `transformers/` | Token lengths, the presence and extraction Challenge Set manifests, the BERT / BioBERT / ClinicalBERT presence runs and comparison, and parser v1.1 on the extraction Challenge Set | `scripts/nlp/transformers/` |

`baseline_manifest.json` records the input and output hashes of
`predictions/baseline_inferences_updated.jsonl` (parser v1.0).

## Test

Test results exist for parser v1.0 (`baseline/`), parser v1.1
(`parser_v1_1/test_evaluation.*`) and the frozen LR model (`final_test/`).
None of them may be re-run.
The transformers have not been evaluated on Test.
