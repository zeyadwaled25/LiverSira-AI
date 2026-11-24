"""Phase 7 LLM extraction: schema, evidence checks, record building, retry and cache, with a fake client (no network)."""

import json

import pytest

from liversira_ai.nlp import llm_extraction as le

TEXT = ("The liver is enlarged. There are multiple liver lesions. The largest lesion is located in hepatic "
        "segment 7, measuring 6.2 x 4.1 cm, and is hypoattenuating. Six smaller lesions are also present. "
        "The liver spans 21.9 cm.")


def answer(value, quote):
    return {"value": value, "quote": quote}


def output(**over):
    base = {
        "has_lesion": answer("yes", "There are multiple liver lesions"),
        "lesion_count": answer(7, "The largest lesion is located in hepatic segment 7, measuring 6.2 x 4.1 cm, and is "
                                  "hypoattenuating. Six smaller lesions are also present"),
        "location_segments": answer([7], "hepatic segment 7"),
        "reported_dimensions_cm": answer([6.2, 4.1], "6.2 x 4.1 cm"),
        "attenuation": answer("hypoattenuating", "is hypoattenuating"),
    }
    base.update(over)
    return base


class FakeClient:
    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, schema):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if reply == "REFUSE":
            return le.CallResult(text=None, model_served=self.model, stop_reason="refusal")
        return le.CallResult(text=reply if isinstance(reply, str) else json.dumps(reply), model_served=self.model,
                             stop_reason="end_turn", input_tokens=1000, output_tokens=200, cache_read_tokens=3000)


def test_schema_matches_the_pydantic_model():
    assert le.OUTPUT_SCHEMA["required"] == list(le.FIELDS) == list(le.LLMOutput.model_fields)
    assert le.OUTPUT_SCHEMA["additionalProperties"] is False
    for name in le.FIELDS:
        assert le.OUTPUT_SCHEMA["properties"][name]["required"] == ["value", "quote"]


def test_good_output_builds_a_record_with_exact_evidence():
    out = le.LLMOutput.model_validate(output())
    record = le.to_inference_record("S1", TEXT, le.verify(out, TEXT), {})
    lesions = record.liver_lesions
    assert lesions.has_lesion is True and lesions.lesion_count == 7
    assert lesions.largest_lesion.location_segments == [7]
    assert lesions.largest_lesion.derived_max_diameter_cm == 6.2
    assert lesions.largest_lesion.attenuation == "hypoattenuating"
    assert all(TEXT[s.start:s.end] == s.text for s in record.evidence_spans)
    assert record.input_reference.field == "liver_text" and record.confidence is None


def test_quote_not_in_text_or_value_not_in_quote_is_rejected():
    out = le.LLMOutput.model_validate(output(
        reported_dimensions_cm=answer([21.9], "6.2 x 4.1 cm"),  # value not in its quote
        location_segments=answer([7], "hepatic segment seven"),  # quote not in the text
    ))
    verified = le.verify(out, TEXT)
    assert set(verified.rejected) == {"reported_dimensions_cm", "location_segments"}
    record = le.to_inference_record("S1", TEXT, verified, {})
    reasons = {m.field: m.reason for m in record.missing_or_unclear}
    assert reasons["liver_lesions.largest_lesion.reported_dimensions_cm"] == "unsupported_by_evidence"
    assert record.liver_lesions.largest_lesion.reported_dimensions_cm is None


def test_millimetres_and_segment_words_count_as_visible():
    assert le.value_in_quote("reported_dimensions_cm", [1.2], "a 12 mm lesion")
    assert le.value_in_quote("location_segments", [4, 8], "segments IVa and eight")
    assert not le.value_in_quote("location_segments", [5], "segment 4.5 cm")
    assert le.value_in_quote("attenuation", "hypoattenuating", "darker than the liver")
    assert not le.value_in_quote("attenuation", "hyperattenuating", "hypodense")
    assert le.value_in_quote("lesion_count", 1, "a solitary lesion")
    assert not le.value_in_quote("lesion_count", 3, "multiple lesions")


def test_negative_and_unclear_reports():
    neg = le.LLMOutput.model_validate({
        "has_lesion": answer("no", "No focal liver lesion"), "lesion_count": answer(0, None),
        "location_segments": answer(None, None), "reported_dimensions_cm": answer(None, None),
        "attenuation": answer(None, None)})
    text = "No focal liver lesion is seen."
    record = le.to_inference_record("S2", text, le.verify(neg, text), {})
    assert record.liver_lesions.has_lesion is False and record.liver_lesions.lesion_count == 0
    unclear = neg.model_copy(update={"has_lesion": le.PresenceAnswer(value="unclear", quote="cannot be excluded"),
                                     "lesion_count": le.CountAnswer(value=None, quote=None)})
    text = "A small lesion cannot be excluded."
    record = le.to_inference_record("S3", text, le.verify(unclear, text), {})
    assert record.liver_lesions.has_lesion is None
    assert {m.reason for m in record.missing_or_unclear} == {"ambiguous"}


def test_one_retry_then_invalid_output():
    client = FakeClient(["not json", "still not json"])
    result = le.extract("S1", TEXT, client, [], "h")
    assert result.status == "invalid_output" and len(client.calls) == 2
    assert "not valid" in client.calls[1][-1]["content"]
    assert {m.reason for m in result.record.missing_or_unclear} == {"llm_invalid_output"}
    client = FakeClient(["oops", output()])
    assert le.extract("S1", TEXT, client, [], "h").status == "ok"


def test_refusal_is_recorded_not_retried():
    client = FakeClient(["REFUSE"])
    result = le.extract("S1", TEXT, client, [], "h")
    assert result.status == "refusal" and len(client.calls) == 1
    assert {m.reason for m in result.record.missing_or_unclear} == {"llm_refusal"}


def test_cache_makes_a_rerun_free(tmp_path):
    cache = le.Cache(tmp_path)
    first = le.extract("S1", TEXT, FakeClient([output()]), [], "h", cache)
    second = le.extract("S1", TEXT, FakeClient([]), [], "h", cache)  # would fail if it called the client
    assert not first.from_cache and second.from_cache
    assert second.record == first.record
    assert le.Cache.key("m", "h", TEXT) != le.Cache.key("m", "h2", TEXT) != le.Cache.key("m2", "h", TEXT)


def test_prompt_hash_covers_system_examples_and_schema():
    ex = [{"text": "t", "output": output()}]
    base = le.prompt_hash("s", ex, le.OUTPUT_SCHEMA)
    assert base != le.prompt_hash("s2", ex, le.OUTPUT_SCHEMA)
    assert base != le.prompt_hash("s", [], le.OUTPUT_SCHEMA)
    assert base != le.prompt_hash("s", ex, {**le.OUTPUT_SCHEMA, "x": 1})


def test_few_shot_examples_go_before_the_report():
    client = FakeClient([output()])
    le.extract("S1", TEXT, client, [{"text": "EX", "output": output()}], "h")
    roles = [m["role"] for m in client.calls[0]]
    assert roles == ["user", "assistant", "user"] and client.calls[0][-1]["content"].endswith(TEXT)


def test_cost_from_the_price_table():
    result = le.CallResult(text="{}", model_served="m", stop_reason="end_turn", input_tokens=1_000_000,
                           output_tokens=1_000_000, cache_read_tokens=1_000_000, cache_write_tokens=0)
    assert le.call_cost(result, {"input": 4.0, "output": 20.0, "cache_read": 0.2, "cache_write": 5.0}) == 24.2


def test_the_check_catches_a_span_that_does_not_match():
    out = le.LLMOutput.model_validate(output())
    record = le.to_inference_record("S1", TEXT, le.verify(out, TEXT), {})
    record.evidence_spans[0].start += 1
    with pytest.raises(ValueError):
        le.check_evidence(record, TEXT)


def test_config_few_shot_ids_are_train_only():
    from pathlib import Path

    config = json.loads(Path("scripts/nlp/llm/config.json").read_text(encoding="utf-8"))
    splits_path = Path("data/processed/nlp/splits.json")
    if not splits_path.exists():
        pytest.skip("splits.json not built")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    ids = config["few_shot"]["ids"]
    assert set(ids) <= set(splits["train"])
    assert not set(ids) & (set(splits["val"]) | set(splits["test"]))


# ── Amendment 1: OpenAI-compatible free tiers ──────────────────────────────
class _FakeOpenAI:
    """Stands in for the ``openai`` module: records the request and returns a canned completion."""

    last_kwargs = None

    class OpenAI:
        def __init__(self, base_url, api_key, timeout, max_retries):
            self.base_url = base_url
            self.chat = self
            self.completions = self
            self.with_raw_response = self

        def create(self, **kwargs):
            from types import SimpleNamespace as NS

            parsed = self._parsed(**kwargs)
            return NS(headers={"x-ratelimit-limit-requests": "1000", "date": "x"}, parse=lambda: parsed)

        def _parsed(self, **kwargs):
            from types import SimpleNamespace as NS

            _FakeOpenAI.last_kwargs = kwargs
            message = NS(content=json.dumps(output()), refusal=None)
            usage = NS(prompt_tokens=1200, completion_tokens=150, prompt_tokens_details=NS(cached_tokens=200))
            return NS(choices=[NS(message=message, finish_reason="stop")], usage=usage, model="served-model", id="r1")


def test_openai_compatible_client_request_and_usage(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAI)
    monkeypatch.setenv("TEST_KEY", "x")
    limiter = le.RateLimiter(tmp_path / "groq.json", rpm=None, rpd=10, tpd=None)
    client = le.OpenAICompatibleClient("m", "https://example.invalid/v1", "TEST_KEY", structured="json_schema",
                                       limiter=limiter)
    result = le.extract("S1", TEXT, client, [], "h")
    sent = _FakeOpenAI.last_kwargs
    assert sent["messages"][0]["role"] == "system" and sent["messages"][-1]["content"].endswith(TEXT)
    assert sent["response_format"]["type"] == "json_schema" and "temperature" not in sent
    assert result.status == "ok" and result.calls[0].cache_read_tokens == 200 and result.calls[0].input_tokens == 1000
    assert json.loads((tmp_path / "groq.json").read_text())["requests"] == 1
    assert client.last_limits == {"x-ratelimit-limit-requests": "1000"}


def test_missing_key_fails_clearly(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "openai", _FakeOpenAI)
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ABSENT_KEY"):
        le.OpenAICompatibleClient("m", "https://example.invalid/v1", "ABSENT_KEY")


def test_rate_limiter_paces_and_stops_at_daily_caps(tmp_path):
    now = [1_700_000_000.0]
    slept = []
    limiter = le.RateLimiter(tmp_path / "p.json", rpm=30, rpd=2, tpd=1000,
                             clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    limiter.wait()
    limiter.record(100)
    limiter.wait()  # 2 s gap at 30 rpm
    assert slept and abs(slept[0] - 2.0) < 1e-6
    limiter.record(100)
    with pytest.raises(le.DailyLimitReached):
        limiter.wait()  # 2 requests used
    now[0] += 86_400  # next UTC day resets the counts
    limiter.wait()
    tokens = le.RateLimiter(tmp_path / "t.json", rpm=None, rpd=None, tpd=150, clock=lambda: now[0])
    tokens.wait()
    tokens.record(200)
    with pytest.raises(le.DailyLimitReached):
        tokens.wait()


def test_subsample_is_fixed_and_never_from_test():
    from pathlib import Path

    path = Path("results/nlp/llm/subsample_v1.json")
    splits_path = Path("data/processed/nlp/splits.json")
    if not path.exists() or not splits_path.exists():
        pytest.skip("subsample or splits not built")
    sub = json.loads(path.read_text(encoding="utf-8"))
    test_ids = set(json.loads(splits_path.read_text(encoding="utf-8"))["test"])
    val_ids = sub["validation"]["positive"] + sub["validation"]["negative"]
    assert len(val_ids) == len(set(val_ids)) == sub["counts"]["validation"]
    assert not set(val_ids) & test_ids
    assert sub["counts"]["total"] == sum(v for k, v in sub["counts"].items() if k != "total")


def test_few_shot_examples_load_and_pass_their_own_checks():
    from pathlib import Path

    if not Path("data/processed/nlp/liver_nlp_liver_only.csv").exists():
        pytest.skip("liver-only CSV not built")
    from scripts.nlp.llm.few_shot import load_examples

    examples = load_examples()
    config = json.loads(Path("scripts/nlp/llm/config.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in examples] == config["few_shot"]["ids"]
    assert len(examples) <= config["few_shot"]["max_examples"]
    for e in examples:
        verified = le.verify(le.LLMOutput.model_validate(e["output"]), e["text"])
        assert verified.rejected == []


def test_runner_end_to_end_with_a_fake_client_and_resume(monkeypatch, tmp_path):
    from pathlib import Path

    if not Path("results/nlp/llm/dev_set_v1.json").exists() or not Path("data/processed/nlp/liver_nlp_liver_only.csv").exists():
        pytest.skip("dev set or data not built")
    from scripts.nlp.llm import run_llm_extraction as rl

    config = json.loads(Path("scripts/nlp/llm/config.json").read_text(encoding="utf-8"))
    config.update(results_root=str(tmp_path / "results"), cache_dir=str(tmp_path / "cache"), state_dir=str(tmp_path / "s"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(rl, "CONFIG", cfg)
    neg = {"has_lesion": answer("unclear", None), "lesion_count": answer(None, None),
           "location_segments": answer(None, None), "reported_dimensions_cm": answer(None, None),
           "attenuation": answer(None, None)}
    fake = FakeClient([neg] * 10)
    monkeypatch.setattr(rl, "make_client", lambda *a: fake)
    rl.run("groq", "dev", limit=3)
    out = tmp_path / "results" / "groq" / rl.model_slug(config["providers"]["groq"]["model"]) / f"dev_{rl.current_prompt()['version']}"
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 3
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["items"] == 3 and metrics["presence"]["abstained"] == 3
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["prompt_hash"] and manifest["items_done"] == 3
    rl.run("groq", "dev", limit=4)  # resumes: only the 4th item is new
    assert len(fake.calls) == 4
    monkeypatch.setattr(rl, "FREEZE", tmp_path / "no_freeze.json")
    with pytest.raises(SystemExit, match="frozen"):
        rl.run("groq", "validation", limit=1)  # evaluation sets wait for the frozen prompt
    monkeypatch.setattr(rl, "FREEZE", Path("results/nlp/llm/prompt_freeze.json"))
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    rows[0]["model"] = "another-model"
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(SystemExit, match="refusing to mix models"):
        rl.run("groq", "dev", limit=5)


# ── Prompt v2 and the evidence-check changes ──────────────────────────────
def test_v1_stays_reproducible_and_v2_differs():
    assert le.PROMPT_VERSION == "v2" and le.SYSTEM_PROMPT == le.SYSTEM_PROMPTS["v2"]
    assert le.SYSTEM_PROMPTS["v2"].startswith(le.SYSTEM_PROMPTS["v1"])
    assert le.prompt_hash(le.SYSTEM_PROMPTS["v1"], [], le.OUTPUT_SCHEMA) != le.prompt_hash(le.SYSTEM_PROMPTS["v2"], [], le.OUTPUT_SCHEMA)


def test_quote_matching_ignores_capitals_but_keeps_report_text():
    text = "A small mass. The larger of the two lies in segment 4."
    found = le.locate("the larger of the two", text)
    assert found and text[found.start:found.end] == "The larger of the two"
    out = le.LLMOutput.model_validate(output(lesion_count=answer(2, "the larger of the two"),
                                             has_lesion=answer("yes", "a small mass"),
                                             location_segments=answer(None, None),
                                             reported_dimensions_cm=answer(None, None), attenuation=answer(None, None)))
    record = le.to_inference_record("S9", text, le.verify(out, text), {})
    assert record.liver_lesions.lesion_count == 2
    assert all(text[s.start:s.end] == s.text for s in record.evidence_spans)


def test_singular_lesion_noun_is_evidence_for_one():
    assert le.value_in_quote("lesion_count", 1, "large hypoattenuating liver mass in segment 2")
    assert not le.value_in_quote("lesion_count", 1, "multiple hypoattenuating liver lesions")
    assert not le.value_in_quote("lesion_count", 1, "several masses")


def test_evaluation_sets_score_with_a_fake_client(monkeypatch, tmp_path):
    from pathlib import Path

    needed = ["results/nlp/llm/prompt_freeze.json", "data/processed/nlp/challenge/presence_challenge_v1.jsonl",
              "data/processed/nlp/extraction/extraction_val_v1.jsonl"]
    if not all(Path(n).exists() for n in needed):
        pytest.skip("freeze or data not built")
    from scripts.nlp.llm import run_llm_extraction as rl

    config = json.loads(Path("scripts/nlp/llm/config.json").read_text(encoding="utf-8"))
    config.update(results_root=str(tmp_path / "results"), cache_dir=str(tmp_path / "cache"), state_dir=str(tmp_path / "s"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(rl, "CONFIG", cfg)
    unclear = {"has_lesion": answer("unclear", None), "lesion_count": answer(None, None),
               "location_segments": answer(None, None), "reported_dimensions_cm": answer(None, None),
               "attenuation": answer(None, None)}
    fake = FakeClient([unclear] * 20)
    monkeypatch.setattr(rl, "make_client", lambda *a: fake)
    slug = rl.model_slug(config["providers"]["gemini"]["model"])
    rl.run("gemini", "presence_challenge_v1", limit=3)
    metrics = json.loads((tmp_path / "results" / "gemini" / slug / "presence_challenge_v1_v2" / "metrics.json").read_text())
    assert metrics["items"] == 3 and sum(c["unclear"] for c in metrics["by_category"].values()) == 3
    rl.run("gemini", "validation", limit=2)
    rl.run("gemini", "repeat", limit=2)  # the repeat run calls again (own cache)
    repeat = json.loads((tmp_path / "results" / "gemini" / slug / "repeat_v2" / "metrics.json").read_text())
    assert "field_agreement" in repeat
    config["providers"]["gemini"]["model"] = "some-other-model"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="not the frozen"):
        rl.run("gemini", "validation", limit=1)


# ── Transport errors (Phase 7 Amendment 3) ────────────────────────────────
def test_transport_failure_is_skipped_logged_and_retried_never_invalid(monkeypatch, tmp_path):
    """A timeout or 5xx after every SDK retry writes no row (not llm_invalid_output, not an abstention),
    is logged per attempt and as a final skip, and the item is called again on the next run."""
    import logging
    from pathlib import Path

    import httpx2 as httpx
    import openai

    if not Path("results/nlp/llm/dev_set_v1.json").exists() or not Path("data/processed/nlp/liver_nlp_liver_only.csv").exists():
        pytest.skip("dev set or data not built")
    from scripts.nlp.llm import run_llm_extraction as rl
    from scripts.nlp.llm import run_logged

    config = json.loads(Path("scripts/nlp/llm/config.json").read_text(encoding="utf-8"))
    config.update(results_root=str(tmp_path / "results"), cache_dir=str(tmp_path / "cache"), state_dir=str(tmp_path / "s"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(rl, "CONFIG", cfg)
    neg = {"has_lesion": answer("no", None), "lesion_count": answer(None, None), "location_segments": answer(None, None),
           "reported_dimensions_cm": answer(None, None), "attenuation": answer(None, None)}

    class Flaky(FakeClient):
        def __init__(self, replies, fail_first):
            super().__init__(replies)
            self.fail_first = fail_first

        def complete(self, system, messages, schema):
            if self.fail_first:
                self.fail_first -= 1
                sdk = logging.getLogger("openai._base_client")
                sdk.debug("Encountered an HTTP status error: %i", 503)
                sdk.debug("Encountered a timeout exception: %s", "ReadTimeout")
                raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
            return super().complete(system, messages, schema)

    fake = Flaky([neg] * 5, fail_first=1)
    monkeypatch.setattr(rl, "make_client", lambda *a: fake)
    run_logged.run("nvidia", "dev", limit=2)
    folder = run_logged.set_folder("nvidia", "dev")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "ok"  # the failed item has no row at all
    log = [json.loads(line) for line in (folder / "transport_log.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["http_503", "timeout", "skipped_APITimeoutError"]
    assert [e["final"] for e in log] == [False, False, True] and len({e["id"] for e in log}) == 1
    run_logged.run("nvidia", "dev", limit=2)  # the next run calls the skipped item again
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and {r["id"] for r in rows} >= {log[0]["id"]}
    assert all(r["status"] == "ok" for r in rows)
    assert le.extract.__module__ == "liversira_ai.nlp.llm_extraction"  # the wrapper restores the pipeline
