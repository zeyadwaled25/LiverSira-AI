"""Phase 7: liver-finding extraction with an LLM, behind a provider-agnostic client.

The model never writes an ``InferenceRecord`` or character offsets. It returns a
small JSON object (``LLMOutput``): for each field a value and a verbatim quote
from the report. This module then

1. validates the output (Pydantic; one retry with the error, then
   ``llm_invalid_output``),
2. finds each quote in the input text and checks that the value is visible in it
   (``verify``); a value whose quote is missing or does not show it is dropped as
   ``unsupported_by_evidence``,
3. builds the ``InferenceRecord`` with offsets computed here, so
   ``text[start:end] == quote`` always holds.

The input is the liver-only text (``liver_text``), the same text every Phase 8
extraction candidate reads. ``confidence`` stays ``None``: self-reported
confidence is not calibrated.

Providers plug in through ``LLMClient``. ``OpenAICompatibleClient`` serves the
free tiers used first (Groq, NVIDIA NIM, Gemini; Phase 7 plan, Amendment 1),
``AnthropicClient`` the Claude API, and a local model can be added the same way.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from liversira_ai.common.provenance import sha256_text
from liversira_ai.nlp.liver_text import LIVER_TEXT_VERSION
from liversira_ai.nlp.schemas.inference_schema import (
    EvidenceSpan,
    InferenceRecord,
    InputReference,
    LargestLesionInference,
    LiverLesionsInference,
    MissingOrUnclearField,
)

PROMPT_VERSION = "v1"
FIELDS = ("has_lesion", "lesion_count", "location_segments", "reported_dimensions_cm", "attenuation")
TARGET = {  # InferenceRecord paths, as the parser writes them
    "has_lesion": "liver_lesions.has_lesion",
    "lesion_count": "liver_lesions.lesion_count",
    "location_segments": "liver_lesions.largest_lesion.location_segments",
    "reported_dimensions_cm": "liver_lesions.largest_lesion.reported_dimensions_cm",
    "attenuation": "liver_lesions.largest_lesion.attenuation",
}
ATTENUATIONS = ("hypoattenuating", "isoattenuating", "hyperattenuating")


# ── 1. What the model returns ───────────────────────────────────────────────
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PresenceAnswer(_Strict):
    value: Literal["yes", "no", "unclear"]
    quote: str | None


class CountAnswer(_Strict):
    value: int | None
    quote: str | None


class SegmentsAnswer(_Strict):
    value: list[int] | None
    quote: str | None


class DimensionsAnswer(_Strict):
    value: list[float] | None
    quote: str | None


class AttenuationAnswer(_Strict):
    value: Literal["hypoattenuating", "isoattenuating", "hyperattenuating"] | None
    quote: str | None


class LLMOutput(_Strict):
    has_lesion: PresenceAnswer
    lesion_count: CountAnswer
    location_segments: SegmentsAnswer
    reported_dimensions_cm: DimensionsAnswer
    attenuation: AttenuationAnswer


def _answer(value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"value": value_schema, "quote": {"type": ["string", "null"]}},
        "required": ["value", "quote"],
        "additionalProperties": False,
    }


# The JSON schema sent to the provider (structured outputs). Kept by hand so it is
# stable and readable; tests/test_llm_extraction.py checks it matches LLMOutput.
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "has_lesion": _answer({"type": "string", "enum": ["yes", "no", "unclear"]}),
        "lesion_count": _answer({"type": ["integer", "null"]}),
        "location_segments": _answer({"type": ["array", "null"], "items": {"type": "integer"}}),
        "reported_dimensions_cm": _answer({"type": ["array", "null"], "items": {"type": "number"}}),
        "attenuation": _answer({"type": ["string", "null"], "enum": [*ATTENUATIONS, None]}),
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}


# ── 2. The prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You extract findings about the liver from the liver part of a CT report. Read only what the report states; never infer from medical knowledge.

Return one JSON object with five fields. Each field has a "value" and a "quote". The quote is copied character for character from the report: the shortest phrase that shows the value. Do not paraphrase, fix typos, or join separate sentences. When the value is null, the quote is null.

- has_lesion: "yes" if the report states at least one liver lesion (lesion, mass, nodule, tumor, cyst, metastasis, focus); "no" if it states there is none or describes the liver as normal or unremarkable; "unclear" if it only suspects or hedges ("cannot be excluded", "possible", "likely", "suspicious for"). If it does not mention liver lesions at all, answer "unclear" with a null quote.
- lesion_count: the number of liver lesions when the report gives it: a number, "single" or "solitary", or lesions listed so they can be counted (a largest one plus "six smaller" is 7). Use null when the report says only "multiple", "several", "numerous" or similar. Use 0 when has_lesion is "no".
- location_segments: the Couinaud segments (1-8) of the largest liver lesion, as written (digits, Roman numerals, words). Subsegments 4a/4b are 4. A lobe name alone, the caudate lobe included, is null.
- reported_dimensions_cm: every dimension written for the largest liver lesion, in cm (divide millimetres by 10). Sizes of anything that is not a liver lesion (the liver itself, vessels, ducts, lymph nodes, other organs) are never lesion sizes. Ranges for a group of lesions ("from 0.6 to 3.1 cm") are not the largest lesion's size.
- attenuation: the attenuation of the largest liver lesion in any wording ("hypodense", "low attenuation" and "darker than the liver" are hypoattenuating). Null if not stated for the largest lesion.

The largest lesion is the one the report says is largest; if it does not say, the one with the longest stated dimension. When has_lesion is "no" or "unclear", every other value is null except lesion_count, which is 0 for "no"."""


def render_example(text: str, output: dict[str, Any]) -> list[dict[str, str]]:
    """One few-shot example as a user/assistant pair."""
    return [{"role": "user", "content": f"Report:\n{text}"},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}]


def prompt_hash(system: str, examples: list[dict[str, Any]], schema: dict[str, Any]) -> str:
    """sha256 over the system prompt, the few-shot examples and the JSON schema, together."""
    payload = json.dumps({"system": system, "examples": examples, "schema": schema}, sort_keys=True,
                         ensure_ascii=False)
    return sha256_text(payload)


# ── 3. Evidence ─────────────────────────────────────────────────────────────
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8}
_ATTENUATION_CUES = {
    "hypoattenuating": r"hypo|low\s+(?:density|attenuation)|darker|less\s+dense",
    "hyperattenuating": r"hyper|high\s+(?:density|attenuation)|brighter|denser|more\s+dense",
    "isoattenuating": r"\biso|equal\s+(?:density|attenuation)|as\s+dense\s+as|indistinguishable",
}


@dataclass
class Located:
    start: int
    end: int
    occurrences: int


def locate(quote: str | None, text: str) -> Located | None:
    """First verbatim occurrence of the quote. Several occurrences carry the same words, so
    the first is used and the number of occurrences is kept for the report."""
    if not quote or not quote.strip():
        return None
    start = text.find(quote)
    if start < 0:
        return None
    return Located(start, start + len(quote), text.count(quote))


def _numbers(quote: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(quote)}


def _segment_tokens(quote: str) -> set[int]:
    found = {int(d) for d in re.findall(r"(?<![\d.])([1-8])(?:[ab])?(?![\d.])", quote, re.I)}
    found |= {_ROMAN[t] for t in re.findall(r"\b(VIII|VII|VI|IV|V|III|II|I)(?:[ab])?\b", quote)}
    lowered = quote.lower()
    found |= {n for w, n in {**_WORDS, **_ORDINALS}.items() if 1 <= n <= 8 and re.search(rf"\b{w}\b", lowered)}
    return found


def value_in_quote(name: str, value: Any, quote: str) -> bool:
    """Is the value visible in its quote? (The quote itself must also be in the report.)"""
    lowered = quote.lower()
    if name == "has_lesion":
        return True  # presence is judged from the whole statement
    if name == "lesion_count":
        if value == 0:
            return True
        words = {w for w, n in _WORDS.items() if n == value}
        if value == 1:
            words |= {"a", "an", "single", "solitary", "one", "lone", "only"}
        if float(value) in _numbers(quote) or any(re.search(rf"\b{w}\b", lowered) for w in words):
            return True
        # counted from listed lesions ("a large mass ... six smaller lesions"): numbers must be written
        return value >= 2 and bool(_numbers(quote) or any(re.search(rf"\b{w}\b", lowered) for w in _WORDS))
    if name == "location_segments":
        return set(value) <= _segment_tokens(quote)
    if name == "reported_dimensions_cm":
        nums = _numbers(quote)
        return all(any(abs(v - n) < 1e-6 or abs(v * 10 - n) < 1e-6 for n in nums) for v in value)
    if name == "attenuation":
        return bool(re.search(_ATTENUATION_CUES[value], lowered))
    raise KeyError(name)


@dataclass
class Verified:
    values: dict[str, Any]
    spans: dict[str, Located]
    rejected: list[str] = field(default_factory=list)  # fields dropped as unsupported_by_evidence
    duplicate_quotes: int = 0


def verify(output: LLMOutput, text: str) -> Verified:
    """Keep a value only if its quote is in the text verbatim and shows the value."""
    values: dict[str, Any] = {}
    spans: dict[str, Located] = {}
    rejected: list[str] = []
    duplicates = 0
    for name in FIELDS:
        answer = getattr(output, name)
        value = answer.value
        if value is None or (name == "has_lesion" and value == "unclear" and not answer.quote):
            values[name] = None if name != "has_lesion" else "unclear"
            continue
        if name == "lesion_count" and value == 0 and output.has_lesion.value == "no":
            values[name] = 0  # follows from the negation; its evidence is the presence quote
            continue
        found = locate(answer.quote, text)
        if found is None or not value_in_quote(name, value, answer.quote):
            rejected.append(name)
            values[name] = None if name != "has_lesion" else "unclear"
            continue
        values[name] = value
        spans[name] = found
        duplicates += found.occurrences > 1
    return Verified(values, spans, rejected, duplicates)


def to_inference_record(study_id: str, text: str, verified: Verified | None, reasons: dict[str, str]) -> InferenceRecord:
    """Build the InferenceRecord. ``verified`` is None when the output was invalid or refused."""
    missing: list[MissingOrUnclearField] = []
    spans: list[EvidenceSpan] = []
    if verified is None:
        reason = reasons.get("_all", "llm_invalid_output")
        for name in FIELDS:
            missing.append(MissingOrUnclearField(field=TARGET[name], reason=reason))
        lesions = LiverLesionsInference(has_lesion=None, lesion_count=None, largest_lesion=None)
    else:
        v = verified.values
        presence = {"yes": True, "no": False}.get(v["has_lesion"])
        for name, found in verified.spans.items():
            span = EvidenceSpan(target_field=TARGET[name], text=text[found.start:found.end],
                                start=found.start, end=found.end)
            spans.append(span)
        for name in verified.rejected:
            missing.append(MissingOrUnclearField(field=TARGET[name], reason="unsupported_by_evidence"))
        largest = None
        if presence is True:
            dims = v["reported_dimensions_cm"]
            largest = LargestLesionInference(
                location_segments=sorted(set(v["location_segments"])) if v["location_segments"] else None,
                reported_dimensions_cm=[round(float(d), 4) for d in dims] if dims else None,
                attenuation=v["attenuation"],
            )
            for name in ("lesion_count", "location_segments", "reported_dimensions_cm", "attenuation"):
                if v[name] is None and name not in verified.rejected:
                    missing.append(MissingOrUnclearField(field=TARGET[name], reason="not_mentioned"))
        elif presence is None and "has_lesion" not in verified.rejected:
            missing.append(MissingOrUnclearField(field=TARGET["has_lesion"], reason="ambiguous"))
        lesions = LiverLesionsInference(
            has_lesion=presence,
            lesion_count=0 if presence is False else (v["lesion_count"] if presence else None),
            largest_lesion=largest,
        )
    record = InferenceRecord(
        id=study_id,
        input_reference=InputReference(field="liver_text", text_version=LIVER_TEXT_VERSION),
        liver_lesions=lesions,
        evidence_spans=spans,
        missing_or_unclear=missing,
        confidence=None,
    )
    check_evidence(record, text)
    return record


def check_evidence(record: InferenceRecord, text: str) -> None:
    """Every evidence span must be the exact text at its offsets (the schema checks only end > start)."""
    for span in record.evidence_spans:
        if text[span.start:span.end] != span.text:
            raise ValueError(f"{record.id}: evidence for {span.target_field} does not match the text at its offsets")


# ── 4. Calling a provider ───────────────────────────────────────────────────
@dataclass
class CallResult:
    """One provider call. ``text`` is the raw JSON text the model returned (None on refusal)."""
    text: str | None
    model_served: str
    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    request_id: str | None = None


class LLMClient(Protocol):
    """What the pipeline needs from a provider: one structured call."""
    model: str

    def complete(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> CallResult: ...


class AnthropicClient:
    """Claude API through the official ``anthropic`` SDK (imported on first use).

    ``base_url`` is passed explicitly so an ``ANTHROPIC_BASE_URL`` set for another tool
    in the environment is never picked up. Credentials come from the environment
    (``ANTHROPIC_API_KEY``, or an ``ant auth login`` profile); this code never stores a key.
    No ``temperature``: current models reject it; determinism is measured by a repeat run.
    """

    def __init__(self, model: str, effort: str, max_tokens: int, base_url: str = "https://api.anthropic.com",
                 timeout_s: float = 120.0, max_retries: int = 4):
        import anthropic  # noqa: PLC0415 - optional dependency, only needed for real calls

        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self._client = anthropic.Anthropic(base_url=base_url, timeout=timeout_s, max_retries=max_retries)

    def complete(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> CallResult:
        started = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt and the few-shot examples are the same on every call: cache them.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        latency = (time.perf_counter() - started) * 1000
        text = None
        if response.stop_reason != "refusal":
            text = next((b.text for b in response.content if b.type == "text"), None)
        usage = response.usage
        return CallResult(
            text=text,
            model_served=response.model,
            stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            latency_ms=round(latency, 1),
            request_id=getattr(response, "_request_id", None),
        )


class OpenAICompatibleClient:
    """Any provider with an OpenAI-compatible chat API (Groq, NVIDIA NIM, Gemini), through the ``openai`` SDK.

    ``structured`` picks how the JSON shape is requested, per provider:
    ``"json_schema"`` (the schema, non-strict: best effort), ``"json_object"`` (JSON mode) or
    ``"prompt"`` (instructions only). The output is validated by Pydantic in every case.
    The API key is read from the environment variable named in the config; it is never stored.
    """

    def __init__(self, model: str, base_url: str, api_key_env: str, structured: str = "json_schema",
                 max_tokens: int = 4000, extra: dict[str, Any] | None = None, timeout_s: float = 120.0,
                 max_retries: int = 4, limiter: "RateLimiter | None" = None):
        import os

        import openai  # noqa: PLC0415 - optional dependency, only needed for real calls

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set the environment variable {api_key_env} (the key is never stored by this code)")
        self.model, self.structured, self.max_tokens = model, structured, max_tokens
        self.extra = extra or {}
        self.limiter = limiter
        self._client = openai.OpenAI(base_url=base_url, api_key=key, timeout=timeout_s, max_retries=max_retries)

    def _response_format(self, schema: dict[str, Any]) -> dict[str, Any] | None:
        if self.structured == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": "liver_findings", "schema": schema, "strict": False}}
        if self.structured == "json_object":
            return {"type": "json_object"}
        return None

    def complete(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> CallResult:
        if self.limiter:
            self.limiter.wait()
        chat = [{"role": "system", "content": system}, *messages]
        kwargs: dict[str, Any] = {"model": self.model, "messages": chat, "max_tokens": self.max_tokens}
        fmt = self._response_format(schema)
        if fmt:
            kwargs["response_format"] = fmt
        if self.extra:
            kwargs["extra_body"] = self.extra
        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        result = CallResult(
            text=choice.message.content,
            model_served=response.model or self.model,
            stop_reason="refusal" if getattr(choice.message, "refusal", None) else (choice.finish_reason or ""),
            input_tokens=(usage.prompt_tokens - cached) if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_read_tokens=cached,
            latency_ms=round(latency, 1),
            request_id=getattr(response, "id", None),
        )
        if self.limiter:
            self.limiter.record(result.input_tokens + result.cache_read_tokens + result.output_tokens)
        return result


class DailyLimitReached(RuntimeError):
    """A free tier's daily request or token cap is reached; the run resumes from the cache later."""


class RateLimiter:
    """Client-side pacing for free tiers: requests per minute, and requests and tokens per UTC day.

    Daily counts are kept in a small JSON file so a run that stops and resumes the same day
    knows what it has used. The provider's own limits still apply; this only avoids hitting them.
    """

    def __init__(self, state_path: Path, rpm: int | None, rpd: int | None, tpd: int | None,
                 clock=time.time, sleep=time.sleep):
        self.state_path, self.rpm, self.rpd, self.tpd = state_path, rpm, rpd, tpd
        self.clock, self.sleep = clock, sleep
        self._last = 0.0

    def _day(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

    def _state(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        return state if state.get("day") == self._day() else {"day": self._day(), "requests": 0, "tokens": 0}

    def wait(self) -> None:
        state = self._state()
        if (self.rpd and state["requests"] >= self.rpd) or (self.tpd and state["tokens"] >= self.tpd):
            raise DailyLimitReached(f"daily cap reached for {self.state_path.stem} ({state})")
        if self.rpm:
            gap = 60.0 / self.rpm - (self.clock() - self._last)
            if gap > 0:
                self.sleep(gap)
        self._last = self.clock()

    def record(self, tokens: int) -> None:
        state = self._state()
        state["requests"] += 1
        state["tokens"] += tokens
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")


def call_cost(result: CallResult, prices: dict[str, float]) -> float:
    """USD from a fixed price table (per million tokens): input, output, cache_read, cache_write."""
    return round((result.input_tokens * prices["input"] + result.output_tokens * prices["output"]
                  + result.cache_read_tokens * prices["cache_read"]
                  + result.cache_write_tokens * prices["cache_write"]) / 1e6, 8)


# ── 5. One report, end to end ───────────────────────────────────────────────
class Cache:
    """Responses on disk, keyed by (model, prompt_hash, sha256(text)), so a re-run costs nothing."""

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def key(model: str, p_hash: str, text: str) -> str:
        return hashlib.sha256(f"{model}|{p_hash}|{sha256_text(text)}".encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.root / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class Extraction:
    record: InferenceRecord
    calls: list[CallResult]
    status: str  # "ok" | "invalid_output" | "refusal"
    rejected: list[str]
    duplicate_quotes: int
    from_cache: bool


def extract(study_id: str, text: str, client: LLMClient, examples: list[dict[str, Any]],
            p_hash: str, cache: Cache | None = None, system: str = SYSTEM_PROMPT) -> Extraction:
    """Ask the model once; on an invalid output, retry once with the error; then build the record."""
    key = Cache.key(client.model, p_hash, text)
    cached = cache.get(key) if cache else None
    if cached is not None:
        calls = [CallResult(**c) for c in cached["calls"]]
        from_cache = True
    else:
        messages = [m for e in examples for m in render_example(e["text"], e["output"])]
        messages.append({"role": "user", "content": f"Report:\n{text}"})
        calls = [client.complete(system, messages, OUTPUT_SCHEMA)]
        if calls[-1].stop_reason != "refusal" and _parse(calls[-1].text)[0] is None:
            error = _parse(calls[-1].text)[1]
            retry = messages + [{"role": "assistant", "content": calls[-1].text or ""},
                                {"role": "user", "content": f"That output was not valid: {error}. Return the JSON "
                                                            "object again, following the schema exactly."}]
            calls.append(client.complete(system, retry, OUTPUT_SCHEMA))
        if cache:
            cache.put(key, {"model": client.model, "prompt_hash": p_hash, "calls": [asdict(c) for c in calls]})
        from_cache = False

    final = calls[-1]
    if final.stop_reason == "refusal":
        record = to_inference_record(study_id, text, None, {"_all": "llm_refusal"})
        return Extraction(record, calls, "refusal", [], 0, from_cache)
    output, _ = _parse(final.text)
    if output is None:
        record = to_inference_record(study_id, text, None, {"_all": "llm_invalid_output"})
        return Extraction(record, calls, "invalid_output", [], 0, from_cache)
    verified = verify(output, text)
    record = to_inference_record(study_id, text, verified, {})
    return Extraction(record, calls, "ok", verified.rejected, verified.duplicate_quotes, from_cache)


def _parse(raw: str | None) -> tuple[LLMOutput | None, str | None]:
    if raw is None:
        return None, "empty response"
    try:
        return LLMOutput.model_validate_json(raw), None
    except ValidationError as exc:
        return None, exc.errors(include_url=False)[0]["msg"]
