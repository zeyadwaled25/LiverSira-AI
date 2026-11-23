# Phase 6.2 Extraction: status

23 November 2026. Development of 6.2 is finished and the candidates are frozen. The decision on which system extracts each field moves to the single Phase 8 evaluation (`docs/nlp/phase8_analysis_plan.md`). Every number below uses source-derived labels or labels set by construction; none of them is a clinical gold standard.

## The 6.2 plan, item by item

| Plan item | Status | Where |
|---|---|---|
| Gate fixed before running: a field whose parser accuracy is below 0.90 on any subtype goes to modelling | Done. All four fields failed it on v1, and again on the liver-only v2. | `results/nlp/transformers/extraction/parser_challenge*.md` |
| Challenge-A is seen, so it is used for diagnosis only | Done. v1, v2_liver and v3 are all treated as Challenge-A. v3 was written as held-out, but its results were seen, so it is now Challenge-A too. Their numbers do not enter the Phase 8 decision. | `results/nlp/phase8/challenge_a_diagnostics.md` |
| Check the Control error (1.8 became 7.3) | Done. It was a colon mass in the same report. The liver-only text (liver_text v1.0.1) removes it, and Control is 1.000 on it. | `results/nlp/liver_text/`, `parser_challenge_liver_only.md` |
| n and Wilson CI for every subtype | Done. `metrics.wilson_interval`. Per-subtype Wilson intervals for every frozen candidate on Challenge-A. Phase 8 reports them the same way. | `results/nlp/phase8/challenge_a_diagnostics.md` |
| Phrasing taxonomy | Done, Train-aug only. No families are reserved for Challenge-B: its author writes first, then marks each item as a Train-aug family or `new`. | `docs/nlp/extraction_train_aug_taxonomy.md` |
| Hypothesis written before running | Done for Phase 8: candidates, metrics, hybrid rule and decision rule are registered and frozen. The per-lesion model had its own registered rule. | `docs/nlp/phase8_analysis_plan.md`, `config_lesion.json` |
| Train-aug with a different generator, built on Train reports, gold spans from the generator | Done. Two sets (first model, per-lesion model). Separate lexicon; a masked 4-gram check against every Challenge Set found 0 overlaps. | `results/nlp/transformers/extraction/extraction_train*_v1.md` |
| Challenge-B written alone by a team member, without seeing Train-aug or the parser rules | **Waiting for the author.** Brief and validator are ready. | `docs/nlp/challenge_b_brief.md`, `validate_challenge_b.py` |
| Parser v1.2 tuned on Train-aug only | Done. 14 rules, each with the Train-aug item id that justifies it, checked by ablation. R2 was dropped because no item needed it. Presence logic is unchanged. Real Train reports: 2 fields better, 0 worse. | `results/nlp/parser_v1_2/train_aug_report.md` |
| Models: count, segments, attenuation, diameter | Done. First model: count classification, segments multi-label, attenuation classification, diameter span tagging. Per-lesion model: size mention detection, then counting. Three ClinicalBERT seeds each. | `runs/comparison.md`, `runs_lesion/comparison.md` |
| Evaluation once on Challenge-B, with Test and the Control | **Moved to Phase 8** (agreed), as one run for all candidates including the Phase 7 LLM. | `docs/nlp/phase8_analysis_plan.md` |
| Nothing tuned on Challenge-A, Challenge-B or Test | Held. Train-aug and Validation were the only development data. Test has not been read for extraction. | n/a |

## What is frozen

`results/nlp/phase8/candidates_freeze.json` (amendment 3) holds the hashes of 52 files. `freeze_candidates --check` and `tests/test_phase8_freeze.py` fail on any change.

| Id | Candidate |
|---|---|
| C1 | Parser v1.1 |
| C2 | Parser v1.2 |
| C3 | First extraction model (6.2), seeds 13 / 42 / 87 |
| C5 | Hybrid H: parser v1.2, with C3 filling empty fields |
| P1 | Presence: frozen TF-IDF + LR (default) |
| P2 | Presence: ClinicalBERT, seeds 13 / 42 / 87 |

C4 (per-lesion) and C6 (H2) were retired by Amendment 3 to keep the comparison simple. The plan, the taxonomy, the Challenge-B brief and validator, the liver-only filter and the scoring code are frozen with them. The candidate list closes when the Phase 8 run starts.

## Amendment 3: the reference for real reports was wrong

On a real report with a lesion, a null target meant "the source value was not found in the text", but it was scored as "the text does not state it". A system that read the text correctly was marked wrong whenever the source label disagreed with the text. For example, "a large mass ... six smaller lesions" states 7 lesions, but its count target was null.

| Validation, lesion-positive | count | segments | attenuation | diameter |
|---|---|---|---|---|
| Parser v1.2, old rule | 0.882 | 0.918 | 0.973 | 1.000 |
| Parser v1.2, verifiable fields only | **1.000** | **0.990** | 0.977 | 1.000 |
| First model (C3), verifiable fields only | 1.000 | 0.974 | 1.000 | 1.000 |
| Hybrid H, verifiable fields only | 1.000 | 0.995 | 0.986 | 1.000 |

- **What the 6.2 comparison got wrong.** The parser's 6.2 "weakness" on real Validation counts and segments came from the reference, not from the parser. Real reports are now scored on verifiable fields only, which applies to Validation and to Test in Phase 8 (`results/nlp/phase8/validation_rescored.md`).
- **The masked C3 retraining.** C3 was retrained with the loss of unverifiable fields masked. A condition was registered before its results (commit `77d1dc7`), and the model did not meet it: seed 87 scored 0.977 on count. So C3 stays the 6.2 model, and the masked model is reported only. H scores the same either way.

## What the diagnostics show (Challenge-A, not part of any decision)

- **Parser v1.2 fixes the phrasings that Train-aug contains, and only those.** Ordinal segments went from 0.00 to 1.00 and subsegments from 0.47 to 1.00. Phrasings that are not in Train-aug stay where v1.1 was: comparative attenuation 0.00, "low/high" attenuation 0.16, enumerated counts 0.30, "The lesion count is N" 0.46. This is the pattern the in-Train-aug vs `new` split in Phase 8 is designed to measure.
- **The models carry over to some phrasings that are not in Train-aug.** The first model gets comparative attenuation 0.91 and enumerated counts 0.51. They also lose some phrasings the parser handles: the first model drops to 0.43 on unquantified counts, and the per-lesion model counts lymph nodes and liver spans as lesions.
- **The frozen hybrid rule has a weakness.** Parser v1.2 leaves a field empty both when it cannot read it and when the report does not state it. H cannot tell these apart, so on unquantified counts ("Countless lesions"), where the parser is right (1.00), H takes the first model's wrong count (0.43). H also keeps the parser's wrong values, because it only fills empty fields: enumerated counts stay at 0.30. Changing H now would be informed by Challenge-A, so it needs your decision and must be recorded as an amendment. See below.

## Waiting on

1. **Challenge-B:** a team member writes it from the brief and freezes it in two steps.
2. **Gold Set:** annotation, agreement, clinician adjudication, finalize.
3. **Phase 7 (LLM extraction):** its candidates join the frozen list by amendment, before the Phase 8 run.
4. ~~Decision on H~~ **decided (Amendment 1).** H stays the hybrid. H2 lost on Train-aug (613 vs 934 of 938 fields) and was retired by Amendment 3. The H limitation on unquantified counts is written into the plan before Challenge-B.
5. ~~Presence plan, deployability, freeze timing~~ **done (Amendment 3).**
6. **Before Challenge-B is written:** push the branch, so the registration can be checked by others.
