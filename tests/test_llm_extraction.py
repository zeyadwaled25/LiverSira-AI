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

        def create(self, **kwargs):
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
