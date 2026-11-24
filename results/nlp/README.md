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
| Does the 6.2 extraction model beat the parser? | `transformers/extraction/runs/comparison.md` (its Validation column uses the old scoring rule; see the next row) |
| What are the Validation numbers when only verifiable fields are scored (Amendment 3)? | `phase8/validation_rescored.md` |
| What does the held-out Challenge Set v3 test, and how does the parser do? | `transformers/challenge/extraction_challenge_v3.md`, `transformers/extraction/parser_challenge_v3.md` |
| Does the per-lesion model fix order and counting? | `transformers/extraction/runs_lesion/comparison.md` |
| What does parser v1.2 add, and which Train-aug item justifies each rule? | `parser_v1_2/train_aug_report.md` |
| How do the API LLMs (Phase 7) do, beside the frozen candidates? | `docs/nlp/phase7_report.md` (numbers: `llm/phase7_report.json`) |
| What is frozen for the Phase 8 final evaluation? | `phase8/candidates_freeze.json` (plan: `docs/nlp/phase8_analysis_plan.md`) |
| How do all frozen candidates do per subtype on the seen Challenge Sets (diagnosis only)? | `phase8/challenge_a_diagnostics.md` |
| Did hybrid H2 (Amendment 1) beat H on Train-aug? | `phase8/amendment1_train_aug.md` |
| Where does 6.2 stand? | `docs/nlp/phase6_2_status.md` |

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
| `parser_v1_2/` | Parser v1.2 vs v1.1 on 6.2 Train-aug, and the ablation evidence for each v1.2 rule | `transformers/extraction/parser_v1_2_report.py` |
| `llm/` | Phase 7 LLM extraction: subsample, prompt versions and freeze, per-model predictions (`<provider>/<model>/<set>_<prompt>/`, with liver-sentence quotes as evidence), transport-error logs, and the Step 7 report numbers | `scripts/nlp/llm/` |
| `phase8/` | Hashes of every frozen Phase 8 candidate and the Challenge-A diagnostics (Challenge-B's freeze record goes here too) | `freeze_candidates.py`, `challenge_a_diagnostics.py`, `validate_challenge_b.py` |
| `transformers/` | Token lengths, the presence and extraction Challenge Set manifests, the BERT / BioBERT / ClinicalBERT presence runs and comparison, and parser v1.1 on the extraction Challenge Set | `scripts/nlp/transformers/` |

`baseline_manifest.json` records the input and output hashes of
`predictions/baseline_inferences_updated.jsonl` (parser v1.0).

## Test

Test results exist for parser v1.0 (`baseline/`), parser v1.1
(`parser_v1_1/test_evaluation.*`) and the frozen LR model (`final_test/`).
None of them may be re-run.
The transformers have not been evaluated on Test.
