# Phase 7 plan: LLM extraction (research stage, API)

Status: **registered before any API call.** Changes are appended as amendments at the end, with their date and reason.

> **Amendment 1 changes §2, §5, §6 and §9:** the first runs use three free API models (Groq, NVIDIA NIM, Gemini) on a fixed subsample. Claude stays configured for a later paid run. See the end of this file.

## 1. Question

Parser v1.1 is very precise when it answers, but its presence coverage on Test is 0.445, and it abstained on 555 Challenge items (148 of the 150 hedged ones). The LR and the transformers decide presence only, without evidence.

**Can an LLM cover every report and extract every field, close to the parser's accuracy, with real verbatim evidence and at a reasonable cost?**

Phase 7 produces the numbers. It does not choose the final system, and it does not read Test. Both belong to Phase 8.

## 2. Provider, models, settings

| Item | Choice | Why |
|---|---|---|
| Provider | Claude API (`anthropic` Python SDK) behind `LLMClient` | One interface: a local model can replace it later without changing the pipeline |
| Main model | `claude-opus-5-5` | The current most capable general model |
| Small model | `claude-haiku-5-5` | Answers whether size matters, at about 1/40 of the price |
| Effort | `low`, fixed | Short structured extraction. Not tuned |
| `temperature` | not sent | Current Claude models reject it. Run-to-run agreement is measured instead (§6) |
| Input | liver-only text (`liver_text` v1.0.1) | The same text every Phase 8 extraction candidate reads |
| Output | structured outputs (JSON schema), then Pydantic validation | Shape guaranteed by the provider, checked again by us |
| Prompt caching | system prompt and few-shot examples cached | Same on every call, so they are cheaper |
| Base URL | set explicitly to `https://api.anthropic.com` | Never inherits an `ANTHROPIC_BASE_URL` set for another tool |
| Credentials | from the environment (`ANTHROPIC_API_KEY` or `ant auth login`) | No key in code, config or logs |

Model IDs are fixed versions, not "latest" aliases. Every response also reports which model served it, and that is recorded per call.

**Data and deployment.** AbdomenAtlas 3.0 Mini is public and de-identified, so sending it to an API is acceptable for research. Real patient reports are different. Health data is sensitive personal data, and transferring personal data outside Egypt needs a licence or permit (Law 151/2020, as raised in the team's plan; to be checked by whoever owns the legal review). This is written into the report as a limitation of the research stage. Under the Phase 8 plan (§6, deployability), an API model is therefore a **reference**, not an eligible candidate, unless a data agreement exists. The provider-agnostic client keeps a local model open as the deployable path.

## 3. Pipeline (`liversira_ai.nlp.llm_extraction`)

1. **Output.** The model returns, for each of `has_lesion`, `lesion_count`, `location_segments`, `reported_dimensions_cm` and `attenuation`, a `value` and a verbatim `quote`. It never writes offsets.
2. **Validation.** An invalid output gets one retry, with the error message. If it fails again, every field is `None` with the reason `llm_invalid_output`. A refusal is recorded as `llm_refusal` and is not retried.
3. **Evidence.** A value is kept only if its quote is found verbatim in the input and the value is visible in the quote:
   - dimensions: the numbers, in cm or mm, are in the quote;
   - segments: the digits, Roman numerals or words are in the quote;
   - attenuation: a matching cue is in the quote;
   - count: the number or word is in the quote, or the quote holds the numbers it was counted from.

   Otherwise the value is dropped with `unsupported_by_evidence`.

   A quote that occurs more than once has identical text at every occurrence, so the first occurrence is used and the duplicate is counted in the report. This differs from the original proposal, which rejected such quotes; it was changed because rejecting would drop correct evidence whenever the impression repeats the findings.
4. **Record.** The `InferenceRecord` is built with offsets computed here. `check_evidence` asserts `text[start:end] == quote` for every span (the schema itself checks only `end > start`). `confidence` stays `None`, because self-reported confidence is not calibrated.
5. **Cache.** Results are cached on disk by `(model, prompt_hash, sha256(text))`, so a re-run costs nothing.
6. **Logging per call.** Each call records input, output and cache tokens, `latency_ms`, retries, the served model and the cost from the fixed price table in `scripts/nlp/llm/config.json`.

## 4. Prompt and few-shot examples

- **`prompt_hash`:** `sha256_text` over the system prompt, the few-shot examples and the JSON schema together. It is written in every manifest.
- **Labelling rules:** the prompt states the same rules as the Challenge-B brief (stated values only; unquantified counts are null; the largest lesion is the one the report calls largest, otherwise the longest dimension; non-lesion sizes are never lesion sizes; lobes are null).
- **Few-shot sources:** examples come from **Train only**. Their IDs are fixed in the config and a test asserts them against `splits.json`. The Challenge Sets are built from Validation, so they are excluded too.
- **Few-shot types:** a positive with one lesion, a positive with listed lesions, an explicit negation, a hedged report, and a report with measurements of other things next to a liver lesion.

## 5. Development protocol (before any evaluation run)

- **Data:** Train only, using about 200 reports drawn from real Train reports and Train-aug items.
- **Budget:** at most **3 prompt versions**. Every version is hashed and logged, the rejected ones included.
- **Choosing a version:** the version with the highest mean field accuracy on the development set is kept. Scoring uses only verifiable fields on real reports (Amendment 3 of the Phase 8 plan) and every field on Train-aug items. A tie goes to the version with fewer tokens.
- **Freeze:** the prompt hash is frozen before Validation is run.

## 6. Evaluation (frozen prompt, each set run once)

| Set | What it measures | Role |
|---|---|---|
| Validation (1,389) | presence (`presence_report`, source labels); fields on lesion-positive reports, verifiable fields only | main Phase 7 numbers, beside parser v1.1, v1.2, C3, H and LR |
| Presence Challenge v1 (1,050) | hedged reports (the share answered "unclear"), negation, other-organ lesions | diagnosis (Challenge-A) |
| Extraction Challenge v3 (900) | non-lesion sizes, order, counting | diagnosis (Challenge-A) |
| Repeat 100 Validation reports | run-to-run agreement per field | reliability |

**LLM-specific metrics:**

- invalid-output rate;
- evidence-rejection rate per field;
- duplicate-quote rate;
- refusal rate;
- the share of hedged items answered "unclear";
- cost in total and per 1,000 reports;
- p50 and p95 latency.

**Comparison.** The small model runs with the same frozen prompt.

## 7. Phase 8

The frozen LLM (prompt hash, model, settings) joins the Phase 8 candidate list by amendment, before the Phase 8 run starts. As an API model, it is a reference (§2).

A parser-then-LLM hybrid (the parser answers first; the LLM fills what the parser leaves empty) may be registered the same way. It inherits H's known limitation on unquantified counts.

## 8. Not in Phase 7

- Test.
- Choosing the final system.
- Clinical accuracy claims: the labels are source-derived until the Gold Set is finished.

## 9. Steps

| Step | What | API |
|---|---|---|
| 1 | Output schema, evidence check, record builder, tests (fake client) | no (done) |
| 2 | Client interface, cache, retry, cost and latency, tests | no (done) |
| 3 | Few-shot selection from Train, prompt v1 | no |
| 4 | Development run on Train (≤ 3 prompt versions) | yes |
| 5 | Freeze; Validation, diagnostics, repeat 100 | yes |
| 6 | Small model, same prompt | yes |
| 7 | Report; Phase 8 amendment | no |

Estimated cost:

| Model | Per call | Calls (≈4,000) | Total |
|---|---|---|---|
| `claude-opus-5-5` | ≈ $0.013 | Validation, diagnostics, repeat and development runs | ≈ $50–55 |
| `claude-haiku-5-5` | — | same | ≈ $1.5 |

The budget caps in the config stop a run if it is exceeded.

## Amendments

### Amendment 1 (23 November 2026): three free API models first, on a fixed subsample

**Reason.** The team chose to start with free API tiers instead of the paid Claude API. Free tiers have daily caps on requests and tokens. The full plan needs about 3,400 calls per model of about 4,000 tokens each, and that does not fit those caps in a reasonable time.

**Models.** Three models from different families, all called through one `OpenAICompatibleClient` (`openai` SDK, provider base URL, key read from an environment variable):

| Provider | Model | Key variable | Structured output |
|---|---|---|---|
| Groq | `openai/gpt-oss-120b` | `GROQ_API_KEY` | JSON schema, best effort |
| NVIDIA NIM | to confirm from the catalog at setup (a non-OpenAI family, e.g. Llama or Nemotron) | `NVIDIA_API_KEY` | JSON mode |
| Google Gemini | a Flash model, to confirm in AI Studio at setup | `GEMINI_API_KEY` | JSON schema |

In every case the output is still validated by Pydantic, with one retry. Model IDs and limits are confirmed against each provider's console before the first run and recorded in the config. No run uses a model whose ID is unconfirmed. `temperature` is not set for any provider.

**Data terms, accepted by the team on 23 November 2026.**

- **Gemini (free tier):** Google may use inputs and outputs to improve its products, and human reviewers may read them.
- **Groq:** zero data retention is available. A no-training clause is reported by a third party but was not verified on Groq's site.
- **NVIDIA (trial):** content is not stored or used after the session, with exceptions. Training is not addressed.
- **What is sent:** only the liver-only text of public, de-identified AbdomenAtlas reports is sent. The dataset licence (CC BY-NC-SA) is the team's responsibility. This is recorded as a limitation in the Phase 7 report.

**Pacing.** A client-side `RateLimiter` per provider enforces requests per minute and requests and tokens per UTC day, with its counts kept on disk. A run that reaches a daily cap stops and resumes the next day from the cache, so no call is repeated or lost.

**Prompt.** At most 3 short few-shot examples (instead of 5), to fit the token caps. The limit of 3 prompt versions, developed on Train only, is unchanged. The development set is 100 reports.

**Fixed subsample (`results/nlp/llm/subsample_v1.json`).** It was built from IDs and labels only (seed 2027), before any LLM call:

| Set | Rule | Items |
|---|---|---|
| Validation | every lesion-positive report, plus 180 lesion-free reports | 400 |
| Presence Challenge v1 | 30 per category | 210 |
| Extraction Challenge v3 | 20 per subtype | 260 |
| **Total** | | **870** |

The repeat-run subset is 50 Validation reports. Validation numbers from the subsample report their own n and Wilson intervals. They are compared with the other systems scored on the same IDs.

**Cost.** $0 for the free tiers. Tokens and latency are still logged per call.

**Phase 8.** Unchanged. API models are references, not eligible candidates.

### Amendment 2 (24 November 2026): development record and the freeze

**Models, as set up and changed during development (all before the freeze):**

| Provider | Model frozen | Changes on the way, and why |
|---|---|---|
| Groq | `openai/gpt-oss-120b` | none |
| NVIDIA NIM | `nvidia/nemotron-3.5-lightning-30b-a3b`, thinking off, JSON schema | Mistral Large 4 and GLM 5.3 Flash timed out on the free endpoint. Nemotron with thinking on used every token before answering. |
| Gemini | `gemini-3.5-flash-lite` | `gemini-3.8-flash` and `gemini-3.5-flash` returned 503 ("high demand"); `gemini-3.5-flash` finished 14 of 100 development items. `gemini-2.5-flash` allows 20 free requests per day. |

**Limits measured** (from headers and 429 messages):

- **Groq:** 8,000 tokens per minute; the requested `max_tokens` counts toward it, so it was set to 1,500. 200,000 tokens per day, which is about 90 calls.
- **Gemini 3.5 Flash-Lite:** completed 100 calls in one day with no 429.
- **NVIDIA:** trial credits. About 210 were used in development, and its balance is checked on build.nvidia.com.

**Prompt versions** (`results/nlp/llm/prompts.json`), on the 100-item Train development set (`dev_set_v1.json`):

- **v1:** Groq count 0.934, and all 5 of its evidence rejections were correct values. NVIDIA had 19 correct values among 29 rejections.
- **v2:** two quote rules in the prompt, and two general evidence-check changes in the pipeline:
  - a match that differs only in capital letters is accepted, and the stored evidence is the report's own characters;
  - for a count of 1, a singular lesion noun in the quote counts as evidence.

  Scores per model:
  - Groq: 1.000 on all four fields, over 49 items; it then reached its daily token cap.
  - NVIDIA: count 0.987, segments 0.897, attenuation 1.000, diameter 0.963.
  - Flash-Lite: count 0.987; segments, attenuation and diameter 1.000.

  No model gave a wrong presence answer. Every "unclear" was a report that never mentions a liver lesion.
- **v3:** not used.

**Freeze.** By the rule in §5, v2 was chosen and frozen (`results/nlp/llm/prompt_freeze.json`). The record holds:

- the prompt hash `337a2921…`;
- the frozen model and settings per provider;
- the hashes of the pipeline, few-shot, evaluation and subsample files.

The runner refuses an evaluation set if the prompt, a model, or any of these files differs from the freeze. Rate limits can still change, because they pace calls and do not change outputs.

**Output folders.** Results are stored per model (`results/nlp/llm/<provider>/<model>/<set>_<prompt>/`), and the runner refuses to mix models in one file. Before this was fixed, one development file held 23 `gemini-2.5-flash` rows and 1 `gemini-3.5-flash-lite` row; they were split by the model that served each row.

**Evaluation order:**

1. `validation`
2. `extraction_challenge_v3`
3. `presence_challenge_v1`
4. `repeat`

Each set runs once per model. Groq, at about 90 calls per day, needs about 10 days, and the runner resumes from the cache.

### Amendment 3 (24 November 2026): two models, finished in hours

**Reason.** The team decided the evaluation must finish within hours on free tiers, not over days. The first evaluation run (23-24 November) stopped on provider limits:

| Model | Items done of 920 | Limit hit |
|---|---|---|
| Groq `openai/gpt-oss-120b` | 0 | 200,000 tokens per day, about 90 calls, so about 10 days |
| Gemini `gemini-3.5-flash-lite` | 215 (Validation) | 500 requests per day per model on the free tier (429 `generate_content_free_tier_requests`, quota value 500) |
| NVIDIA `nemotron-3.5-lightning-30b-a3b` | 789 (Validation 400, Challenge v3 260, Presence 124, repeat 5) | temporary 429s and timeouts; a request on 24 November answered normally |

**No free host for gpt-oss-120b in hours** (checked 24 November):
- NVIDIA does not serve it (its model list has 32 models, none of them gpt-oss).
- Cerebras is a 30-day trial that needs a card, with 1M tokens per day, about 2 days.
- OpenRouter's free tier allows 50 requests per day, and SambaNova's 20.

**Decision (team, 24 November):**

1. **gpt-oss-120b is dropped**, for the rate limit only. No evaluation item ran on it. Its development result (prompt v2, 49 of 100 items, 1.000 on all four fields) is reported as a reference only, not as an evaluation result.
2. **No model is added.** Adding a model after the freeze, with the evaluation under way for the others, would need a justification. Replacements were tried on 1-3 Train items only and are not used:
   - `minimaxai/minimax-m3.1-flash`: about 30 s per call;
   - `google/gemma-4-31b-it` and `moonshotai/kimi-k3.1-pro`: timed out;
   - `nvidia/nemotron-3-ultra-550b-a55b`: noted in an earlier draft of this amendment, which was never run.

   Phase 7 reports two models: Nemotron (open weights, so it can later run locally) and Gemini Flash-Lite (closed, as a reference).
3. **Nemotron Lightning** finishes its 131 remaining items (Presence 86, repeat 45). Nothing changes for it.
4. **Gemini Flash-Lite** runs within one daily quota of 500 requests, in this order:
   - Validation (185 left);
   - Extraction Challenge v3 (260);
   - repeat (50);
   - then it stops.

   That is 495 calls. The repeat comes before Presence because it measures stability, and Presence is not an LLM role in Phase 8 (the presence candidates are P1 LR and P2 ClinicalBERT). **Presence Challenge v1 is not run for Gemini (quota).** If the quota ends earlier, for example because retries used it, whatever is missing is reported as not run, with its n, and is not run on a later day.
5. The frozen prompt, few-shot examples, pipeline, subsample and Nemotron/Gemini settings are unchanged, so no frozen file changes. `config.json` records Groq as dropped, the Gemini daily cap (rpd 500) and its scope.

**Results seen before this decision.** No evaluation metric was opened by the session or reported to the team before this amendment:
- The runner writes `metrics.json` by itself.
- To track progress, the session read only item counts and the status label of each prediction file (all rows "ok"). It did not read field values or compare them with labels.
- The decision was based only on the limits in the table above.

**Evaluation order per model:** Amendment 2's order. For Gemini, the order and scope in item 4.

**Addendum, after the runs (24 November, no result-driven change):**

- **Transport log.** `scripts/nlp/llm/run_logged.py` wraps the frozen runner without changing it. Every failed attempt (timeout, connection, HTTP 408/409/429/5xx) and every skipped item now goes to `<set folder>/transport_log.jsonl`. A skipped item writes no row and is retried by the next run, so it is never counted as `llm_invalid_output` or as an abstention. Skips from earlier runs are in `results/nlp/llm/transport_errors_unlogged.json`, as counts only: no error type, no item, and no failed attempt that a later retry recovered.
- **Nemotron pacing (24 November 2026, during the Nemotron runs).** At 40 rpm NVIDIA rejected most requests with HTTP 429 (281 failed attempts and 51 skipped items logged, 12:45-12:52 UTC). So `config.json` was set to rpm 10 and 8 SDK retries (from the default 4); after that, 161 attempts still got 429 and 4 items were skipped, all finished on a rerun. This only paces calls, as the freeze allows (`prompt_freeze.json` note); outputs and frozen settings are unchanged. Every item finished after it.
- **Gemini.** Its daily quota of 500 ran out after about 360 successful calls, so Challenge v3 stopped at 174 of 260 and repeat at 0 of 50. By item 4, these are reported as not run and are not run on a later day. Retried attempts were not logged in that run, so where the rest of the quota went is not known.
