# Phase 8 analysis plan: the single final evaluation of extraction and presence

Status: **registered**. This plan was written before Challenge-B existed and before any candidate was run on Test. It is frozen together with the candidates; `results/nlp/phase8/candidates_freeze.json` records the hash of this file. Any later change must be an amendment, appended at the end with its date and reason. Any amendment made after a result has been seen must be labelled post hoc, and the original analysis must be reported too.

## 1. What runs once

Phase 8 is one run: one script, executed once. It is the only code that reads Challenge-B predictions or Test reports for extraction. Nothing is tuned, selected or re-run after it. Phase 6.2's Challenge Sets (v1, v2_liver, v3) were seen during development. They are used for diagnosis only, and none of their numbers enter this decision.

## 2. Candidates (closed list)

Every candidate reads the liver-only text `extract_liver_text(text).text` (liver_text v1.0.1).

| Id | Candidate | Frozen as |
|---|---|---|
| C1 | Parser v1.1 | `parser_v1_1.py` (frozen since Phase 5) |
| C2 | Parser v1.2 | `parser_v1_2.py` + `parser_v1_2_rules.py`: 14 rules, each justified by a Train-aug item (`results/nlp/parser_v1_2/train_aug_report.md`) |
| C3 | First extraction model (6.2), ClinicalBERT, seeds 13 / 42 / 87 | `config.json`, `models/transformers/extraction/clinicalbert/seed*/model.safetensors` |
| C5 | Hybrid H = C2 + C3 (rule in §4), one run per C3 seed. **The hybrid.** | `hybrid.py` (`hybrid_h`) |

Retired by Amendment 3, to keep the comparison simple: C4 (per-lesion model; it failed its own registered rule) and C6 (hybrid H2; it lost to H on Train-aug). The masked retraining of C3 (Amendment 3) did not meet its condition and is reported only. Their results stay in the repository; they are not run in Phase 8.

Phase 7 (LLM extraction) candidates may join this list only by an amendment. The amendment must freeze their prompt, model id and settings the same way, and it must be made before the Phase 8 run starts (`freeze_candidates` refuses once `results/nlp/phase8/run_started.json` exists). Nobody who develops a candidate reads Challenge-B, so Challenge-B may be frozen before or after that amendment. After the run starts, the list is closed.

## 3. Evaluation sets

| Id | Set | Items | Reference |
|---|---|---|---|
| E1 | Challenge-B, targeted items | per the brief (`challenge_b_brief.md`) | labels written by the Challenge-B author |
| E2 | Challenge-B, `clean` and `negative` items (the control) | per the brief | same |
| E3 | Test, lesion-positive studies | studies whose source labels have a liver lesion | source-derived extraction targets (`targets_from_source`), as for Validation |
| E4 | Gold Set, Test half | the Test studies of the gold-set sample (`results/nlp/gold_set/gold_set_sample_manifest.json`) | finalized human annotation (`gold_standard.jsonl`) |

The Gold Set must be finalized before the run, as the roadmap requires. E4 is the only clinical (human) reference. It is reported per field with Wilson intervals, but it does not enter the decision rule, because it holds about 40 lesion-positive Test studies and is too small to separate candidates field by field. The Validation half of the gold set is not used here.

Challenge-B is used only once both of its hashes are recorded: the items hash, and the coverage hash (each item marked as a Train-aug family or `new`). Both are checked by `validate_challenge_b.py`. If the files do not match the recorded hashes, the run stops.

E3 scores verifiable fields only (§5, Amendment 3): a field whose source value is not found in the report text is left out, and the number left out is reported. Parser v1.1 (C1) is scored on it here under that scoring. Its earlier Test results (`results/nlp/parser_v1_1/test_evaluation.*`) stay as they are and are not replaced.

## 4. Hybrid rule H (fixed now)

For each study and each field:

- If parser v1.2 decides "no lesion" (presence False), H outputs count `0` and leaves the other fields empty.
- Otherwise, H takes parser v1.2's value for the field when it is not empty, and the first model's value (C3, same seed) when it is.
- Count classes, segment lists, attenuation labels and diameters pass through unchanged. There is no averaging and no thresholds.

H does not use the parser's `field_status` (added by Amendment 1, verified for count only). H2, which used it, was retired by Amendment 3.

Why H: parser v1.2 is deterministic and gives evidence spans for every value it outputs. The model fills only the fields the parser leaves empty. C3, not C4, is the fallback because C4 failed its registered rule in 6.2 (`runs_lesion/comparison.md`: it tags non-lesion sizes as lesions).

## 5. Metrics

Fields: `count` (classes `0`–`9`, `10+`, or unstated), `segments` (set of Couinaud segments 1–8, or unstated), `attenuation` (`hypoattenuating` / `isoattenuating` / `hyperattenuating`, or unstated), `diameter_cm` (largest lesion's longest dimension, in cm).

- **Correct.** A prediction must equal the target:
  - Segments are compared as sets.
  - A diameter must be within 0.05 cm.
  - An empty prediction is correct only when the target is unstated, and wrong otherwise.
  - **Real reports (E3, and Validation during development):** on a report with a lesion, a field whose source value is not found in the text has no verified target. It is not scored, and the count of such fields is reported (`extraction_targets.unverifiable_fields`, Amendment 3). Challenge-B and the Gold Set have human labels, so every field there is scored.
- **Per field.** Accuracy over the items that target the field, with a Wilson 95% interval and n. On E1 the pooled targeted accuracy is the primary metric.
- **Per subtype.** Accuracy with a Wilson 95% interval and n for every Challenge-B category × subtype, and for every coverage group (Train-aug family / `new`).
- **Seeds.** Model-based candidates (C3, C5) are reported per seed, plus the mean ± SD.
- **Paired tests.** On the items of each field, every candidate is compared with C2 using an exact McNemar test, per seed.
- **Descriptive only.** The following are reported but do not decide anything:
  - Accuracy on `new` items vs Train-aug-family items, per candidate.
  - E3 per field, beyond the guard in §6.
  - E4 (Gold Set) per field.
  - Abstention rate per field.

## 6. Decision rule (per field)

For each field, a candidate is **eligible** only if all of these hold:

1. **Control.** E2 accuracy ≥ 0.98 for every model seed.
2. **Clean data.** E3 accuracy ≥ (C1's E3 accuracy − 0.01) for every model seed.
3. **Deployable (Amendment 3).** It can run on patient reports under the Phase 9 data policy: it runs locally, or the project has a data agreement that allows the reports to be sent to its provider. A candidate that is not deployable is reported as a reference only.

Among eligible candidates, the one with the highest E1 accuracy wins (mean over seeds for model-based candidates). It replaces C2 only if all three of these hold:

- Its E1 accuracy is higher than C2's in every seed.
- The McNemar test against C2 on E1 has p < 0.05 in every seed.
- It is not C1, since C2 contains C1's rules.

Otherwise C2 is selected for that field. If C2 is not eligible either, the field is marked **not ready** for Phase 9.

The verdict is one system per field. It goes into Phase 9 as the per-field routing.

## 7. What may not change after the freeze

None of the following may change after the freeze:

- Candidate code and weights.
- liver_text.
- The hybrid rule.
- Metrics and tolerance.
- The decision rule.
- The candidate list, except through the Phase 7 amendment in §2.

If a bug is found in the scoring code, the fix is recorded as an amendment. If it is found after results were seen, both versions are reported.

## 8. Presence (Amendment 3)

Phase 6.1 left the presence decision to Phase 8; it is registered here the same way.

| Id | Candidate | Frozen as |
|---|---|---|
| P1 | TF-IDF + LR, the frozen Phase 5 model. **The default.** | `results/nlp/final_test/learned_presence_word_logreg_final.joblib` |
| P2 | ClinicalBERT presence model, seeds 13 / 42 / 87 | `scripts/nlp/transformers/presence/config.json`, `models/transformers/presence/clinicalbert/seed*/model.safetensors` |

ClinicalBERT is the only transformer candidate because every extraction model uses the same encoder, not because of its 6.1 Challenge result. Parser v1.1 presence is reported for reference: it abstains on reports that never mention a lesion, so it cannot serve as the presence system.

- **Sets.** Challenge-B (every item: `negative` = no lesion, the others = lesion), all 1,390 Test studies (source label), and the Gold Set Test half (descriptive).
- **Rule.** P2 replaces P1 only if, for every seed: its Challenge-B accuracy is higher with an exact McNemar p < 0.05, its Test accuracy is ≥ P1's − 0.01, and it is deployable (§6). Otherwise P1 stays.
- **Test.** P1's Test predictions already exist (Phase 5); they are re-scored, not re-run.

## 9. Known limitations (written before Challenge-B)

- **H and unquantified counts.** When a report states the count as unquantified ("Countless lesions", "A cluster of lesions"), parser v1.2 correctly leaves the count empty. H cannot tell this apart from a count the parser failed to read, so it takes the first model's count instead. On Challenge-A (v3 `count/unquantified`, 50 items) the parser scores 1.00 and H scores 0.43, which is the first model's own accuracy there. If Challenge-B contains such items, this weakness is expected to show in H's count accuracy. It is stated here before Challenge-B is written. H2 was the attempted fix; see Amendment 1.

## Amendments

### Amendment 1 (23 November 2026): parser field status and hybrid H2

**Reason.** The Challenge-A diagnostics (`results/nlp/phase8/challenge_a_diagnostics.md`) showed the weakness in §8. Because it was found on Challenge-A, the fix was allowed one attempt only, checked on Train-aug only. It was not to be repeated or tuned on Challenge-A.

**Change.**

- **Parser v1.2:** now outputs `field_status` for each field (§4). This is additive: on all 21,964 Train-aug items its values are identical to the originally frozen `parser_v1_2.py` (commit `9b088a8`).
- **`hybrid.py`:** holds H (unchanged) and H2.

**The single attempt** (`results/nlp/phase8/amendment1_train_aug.md`):

- **Evidence:** no Train-aug item with an unquantified count has H wrong and H2 right. On Train-aug, the first model already leaves those counts empty (`count/vague`: 166 / 166 for both H and H2). The weakness is known only from the Challenge-A diagnostics.
- **Result:** H and H2 can differ on 871 Train-aug items (938 fields). There H is right on 934 fields (935 for seed 42) and H2 on 613, the same for seeds 13, 42 and 87.
- **Cause:** H2 empties correct model values wherever the status cue misses. The segment and attenuation cues are searched only in affirmative lesion clauses, so they miss attenuation stated in a separate or relative sentence and segments stated outside such clauses. OCR damage breaks the cue words. The loss is in `ocr` (all four fields) and in `real` and `repeat/multiple` attenuation. On `count/vague`, `segment/lobe` and `real` count and segments, H and H2 are equal.
- **Stopped** as the one-attempt condition requires. The status cues were not revised.

**Decision.**

- **H stays the primary hybrid (C5).** H2 is added as C6: exploratory, appendix only, never eligible (§2, §6).
- **Field status is kept.** Its semantics are documented as verified for count only (§4).
- **Known limitation recorded** in §8.

**Timing.** This amendment was made before Challenge-B was written: no `results/nlp/phase8/challenge_b_freeze.json` exists, and no candidate has read Challenge-B or Test. The candidates were re-frozen once, after Amendment 2 (`candidates_freeze.json`, `amendment` = 2).

### Amendment 2 (23 November 2026): line endings in the freeze hashes (tooling only)

**Reason.** On Windows with `core.autocrlf`, git may check text files out with CRLF. A raw-byte hash would then change even though the content did not, and `freeze --check` would fail for no real reason.

**Change.**

- A new `.gitattributes` stores and checks out text files with LF. Weights and other binaries are marked `binary`.
- The freeze hashes text files after normalizing CRLF to LF, and hashes binaries as raw bytes (`freeze_candidates.freeze_hash`). The Challenge-B validator normalizes the same way.
- No analysis, candidate or rule changed.

**Note on the original freeze.** The freeze in commit `9b088a8` hashed raw bytes for every file. Its hashes are kept in git history, and the new freeze file records the old method under `original_freeze` and `supersedes`.

**Timing.** Like Amendment 1, this was made before Challenge-B was written.

### Amendment 3 (23 November 2026): verifiable targets, a masked C3, simpler candidates, presence, deployability

**Reason.** A check on Train only, before any new run, showed that the reference for real reports was wrong. `targets_from_source` sets a field to null when the source value is not found literally in the text, and the scoring then demanded a null prediction. A null therefore meant "not checked", not "not stated". Example: a report with "a large mass ... six smaller lesions" states 7 lesions, and the parser reads 7, but its target was null.

- On the 1,029 real Train reports with a lesion, 158 count targets and 86 segment targets were such nulls.
- On Validation (220 lesion-positive reports), the same held for 46 counts and 24 segments.
- This produced almost all of the parser's Validation count and segment "errors": count 0.886 under the old rule, 1.000 when only verifiable fields are scored (`results/nlp/phase8/validation_rescored.md`).
- The 6.2 first model was trained to output null on these reports.

The evidence is from Train and Validation only. Challenge-A, Challenge-B and Test were not used.

**Changes.**

1. **Scoring.** On real reports with a lesion, unverifiable fields (`unverifiable_fields`) are not scored. This applies to Validation and E3 (§3, §5).
2. **C3.** Retrained with the loss of unverifiable fields masked (`config_v2.json`). Data, model, seeds and every other setting are identical. Before its results were seen, this condition was fixed: C3 becomes the masked model if, for every seed, its Validation accuracy (verifiable fields, lesion-positive) on each field is at least the 6.2 model's − 0.01. Otherwise C3 stays the 6.2 model and the masked model is reported only. The outcome is recorded below.
3. **Simpler list.** C4 and C6 are retired (§2).
4. **Timing.** The candidate list closes when the Phase 8 run starts, not when Challenge-B is frozen (§2). The freeze tool enforces this.
5. **Deployability.** A new eligibility condition (§6).
6. **Presence.** Registered in §8.

**Timing.** Made before Challenge-B was written, and before any candidate read Test.

**Outcome of change 2 (recorded after the masked runs).** The condition was not met. On Validation (verifiable fields, lesion-positive), the masked model beside the 6.2 model was:

| Seed | count | segments | attenuation | diameter |
|---|---|---|---|---|
| 13 | 1.000 vs 1.000 | 0.995 vs 0.974 | 1.000 vs 1.000 | 1.000 vs 1.000 |
| 42 | 0.994 vs 1.000 | 0.995 vs 0.974 | 0.995 vs 1.000 | 1.000 vs 1.000 |
| 87 | **0.977** vs 1.000 | 0.985 vs 0.974 | 1.000 vs 1.000 | 1.000 vs 1.000 |

Seed 87 is below 0.990 on count: it reads "five liver lesions" as 4. Masking removes about 15% of the real count labels from training. As registered, **C3 stays the 6.2 model**. The masked model (`config_v2.json`, `runs_v2/`) is reported only, and it is not a candidate.

Hybrid H gives the same Validation accuracy with either model (count 1.000, segments 0.995, attenuation 0.986, diameter 1.000), because H takes the parser's value whenever the parser reads one. The decision was taken by the rule alone.
