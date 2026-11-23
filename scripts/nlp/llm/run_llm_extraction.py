"""Run one provider's LLM extraction on one set; resumable, so a free tier can finish over several days.

    python -m scripts.nlp.llm.run_llm_extraction --build-dev-set            # once, before any dev run
    python -m scripts.nlp.llm.run_llm_extraction --provider groq --set dev  # prompt version from prompts.json

Sets: ``dev`` (100 Train items: real reports and Train-aug, few-shot studies excluded), and,
only after the prompt is frozen (results/nlp/llm/prompt_freeze.json), the evaluation sets
``validation``, ``presence_challenge_v1``, ``extraction_challenge_v3`` and ``repeat``
(scripts/nlp/llm/eval_sets.py). ``repeat`` bypasses the cache and needs ``validation`` first.

Each item's prediction is appended to results/nlp/llm/<provider>/<model>/<set>_<prompt>/predictions.jsonl
as soon as it is done; a re-run skips finished items, and responses come from the cache.
A daily cap stops the run cleanly; run the same command again later. ``metrics.json`` and
``run_manifest.json`` are rewritten from all predictions at the end of every run.
Test is never read.
"""

import argparse
import json
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp import llm_extraction as le
from liversira_ai.nlp.extraction_targets import count_class, unverifiable_fields
from scripts.nlp.llm import eval_sets
from scripts.nlp.llm.few_shot import load_examples
from scripts.nlp.llm.list_models import CONFIG, api_key
from scripts.nlp.transformers.extraction.train import TARGET_FIELDS, field_correct

TRAIN_AUG = Path("data/processed/nlp/extraction/extraction_train_v1.jsonl")
DEV_SET = Path("results/nlp/llm/dev_set_v1.json")
PROMPTS = Path("results/nlp/llm/prompts.json")
FREEZE = Path("results/nlp/llm/prompt_freeze.json")
DEV_SEED = 2028


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ── The development set (Train only) ────────────────────────────────────────
def build_dev_set(few_shot_ids: set[str]) -> dict[str, Any]:
    """28 real lesion-positive + 20 real lesion-free Train reports, and 2 items per generated Train-aug subtype."""
    if DEV_SET.exists():
        raise SystemExit(f"{DEV_SET} exists; the development set is fixed")
    rng = random.Random(DEV_SEED)
    rows = [r for r in read_jsonl(TRAIN_AUG) if r["base_id"] not in few_shot_ids]
    real = [r for r in rows if r["category"] == "real"]
    pos = sorted(r["id"] for r in real if r["targets"]["count"] != "0")
    neg = sorted(r["id"] for r in real if r["targets"]["count"] == "0")
    generated: dict[str, list[str]] = {}
    for r in rows:
        if r["category"] not in ("real", "ocr"):
            generated.setdefault(f"{r['category']}/{r['subtype']}", []).append(r["id"])
    record = {
        "version": "dev_set_v1", "seed": DEV_SEED, "source": TRAIN_AUG.as_posix(),
        "source_sha256": sha256_file(TRAIN_AUG),
        "rule": "Train only; few-shot studies excluded; 28 real lesion-positive, 20 real lesion-free, 2 per generated subtype",
        "real_positive": sorted(rng.sample(pos, 28)),
        "real_negative": sorted(rng.sample(neg, 20)),
        "generated": {k: sorted(rng.sample(sorted(v), 2)) for k, v in sorted(generated.items())},
    }
    record["count"] = 48 + sum(map(len, record["generated"].values()))
    DEV_SET.parent.mkdir(parents=True, exist_ok=True)
    DEV_SET.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def dev_items() -> list[dict[str, Any]]:
    spec = json.loads(DEV_SET.read_text(encoding="utf-8"))
    wanted = set(spec["real_positive"]) | set(spec["real_negative"]) | {i for ids in spec["generated"].values() for i in ids}
    items = [r for r in read_jsonl(TRAIN_AUG) if r["id"] in wanted]
    if len(items) != spec["count"]:
        raise SystemExit("the development set no longer matches its source file")
    return items


# ── Scoring ─────────────────────────────────────────────────────────────────
def as_targets(record: dict[str, Any]) -> dict[str, Any]:
    """An InferenceRecord in extraction-target form (as parser_targets does for the parser)."""
    lesions = record["liver_lesions"]
    if lesions["has_lesion"] is False:
        return {"count": "0", "segments": None, "attenuation": None, "diameter_cm": None}
    if lesions["has_lesion"] is None:
        return {f: None for f in TARGET_FIELDS}
    largest = lesions["largest_lesion"] or {}
    return {"count": count_class(lesions["lesion_count"]),
            "segments": sorted(largest["location_segments"]) if largest.get("location_segments") else None,
            "attenuation": largest.get("attenuation"), "diameter_cm": largest.get("derived_max_diameter_cm")}


def summarize(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        scored = [r for r in rows if items[r["id"]]["targets"]["count"] != "0"
                  and field not in unverifiable_fields(items[r["id"]])]
        k = sum(field_correct(field, r["predicted"][field], items[r["id"]]["targets"][field]) for r in scored)
        fields[field] = le_wilson(k, len(scored))
    presence = [(items[r["id"]]["targets"]["count"] != "0", r["record"]["liver_lesions"]["has_lesion"]) for r in rows]
    calls = [c for r in rows for c in r["calls"]]
    latencies = sorted(c["latency_ms"] for c in calls if not r_from_cache(c))
    n = len(rows)
    return {
        "items": n,
        "fields_lesion_positive": fields,
        "presence": {"correct": sum(t == p for t, p in presence), "abstained": sum(p is None for _, p in presence),
                     "wrong": sum(p is not None and t != p for t, p in presence)},
        "invalid_output_rate": round(sum(r["status"] == "invalid_output" for r in rows) / n, 4) if n else None,
        "refusal_rate": round(sum(r["status"] == "refusal" for r in rows) / n, 4) if n else None,
        "retry_rate": round(sum(len(r["calls"]) > 1 for r in rows) / n, 4) if n else None,
        "evidence_rejections": {f: sum(f in r["rejected"] for r in rows) for f in le.FIELDS},
        "duplicate_quote_items": sum(r["duplicate_quotes"] > 0 for r in rows),
        "tokens": {k: sum(c[k] for c in calls) for k in ("input_tokens", "output_tokens", "cache_read_tokens")},
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95)} if latencies else None,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
    }


def le_wilson(k: int, n: int) -> dict[str, Any]:
    from liversira_ai.nlp.metrics import wilson_interval

    return wilson_interval(k, n)


def r_from_cache(call: dict[str, Any]) -> bool:
    return bool(call.get("from_cache"))


def _pct(values: list[float], q: int) -> float:
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100)[q - 1], 1)


# ── Running ─────────────────────────────────────────────────────────────────
def make_client(name: str, spec: dict[str, Any], state_dir: Path) -> le.LLMClient:
    key = api_key(spec["api_key_env"])
    if key:
        os.environ[spec["api_key_env"]] = key  # this process only
    limits = spec.get("limits", {})
    limiter = le.RateLimiter(state_dir / f"{name}.json", limits.get("rpm"), limits.get("rpd"), limits.get("tpd"))
    return le.OpenAICompatibleClient(spec["model"], spec["base_url"], spec["api_key_env"], structured=spec["structured"],
                                     max_tokens=spec["max_tokens"], extra=spec.get("extra"),
                                     max_retries=spec.get("max_retries", 4), limiter=limiter)


def model_slug(model: str) -> str:
    return model.replace("/", "-")


def current_prompt() -> dict[str, Any]:
    versions = json.loads(PROMPTS.read_text(encoding="utf-8"))["versions"]
    return versions[-1]


def check_freeze(p_hash: str) -> dict[str, Any]:
    """The evaluation sets run only on the frozen prompt and pipeline, unchanged since the freeze."""
    from scripts.nlp.transformers.extraction.freeze_candidates import freeze_hash

    if not FREEZE.exists():
        raise SystemExit("evaluation sets run only after the prompt is frozen (Phase 7 plan §5)")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["prompt_hash"] != p_hash:
        raise SystemExit("the prompt differs from the frozen one")
    changed = [f for f, h in frozen["files_sha256"].items() if freeze_hash(Path(f)) != h]
    if changed:
        raise SystemExit(f"frozen files changed since the freeze: {changed}")
    return frozen


def run(provider: str, set_name: str, limit: int | None) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    spec = config["providers"][provider]
    prompt = current_prompt()
    examples = [{"text": e["text"], "output": e["output"]} for e in load_examples()]
    p_hash = le.prompt_hash(le.SYSTEM_PROMPT, examples, le.OUTPUT_SCHEMA)
    if p_hash != prompt["hash"]:
        raise SystemExit(f"prompt hash {p_hash[:12]} differs from the logged {prompt['version']} ({prompt['hash'][:12]}); "
                         "log the new version in prompts.json first")
    if set_name == "dev":
        items = dev_items()
    elif set_name in eval_sets.EVAL_SETS:
        frozen = check_freeze(p_hash)
        if frozen["models"].get(provider) != spec["model"]:
            raise SystemExit(f"{provider}: model {spec['model']} is not the frozen {frozen['models'].get(provider)}")
        items = eval_sets.load_items(set_name)
    else:
        raise SystemExit(f"unknown set {set_name}")
    items = items[:limit] if limit else items
    by_id = {i["id"]: i for i in items}
    # One folder per model: a provider's model can change during development, and predictions
    # from two models must never share a file.
    out_dir = Path(config["results_root"]) / provider / model_slug(spec["model"]) / f"{set_name}_{prompt['version']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    existing = read_jsonl(pred_path) if pred_path.exists() else []
    others = {r.get("model") for r in existing} - {spec["model"]}
    if others:
        raise SystemExit(f"{pred_path} holds predictions from {others}, not {spec['model']}; refusing to mix models")
    done = {r["id"] for r in existing}
    # The repeat run must call the model again: it gets its own cache.
    cache = le.Cache(Path(config["cache_dir"]) / (f"{provider}_repeat" if set_name == "repeat" else provider))
    client = make_client(provider, spec, Path(config["state_dir"]))
    prices = config["prices_usd_per_million_tokens"]["free"]
    stopped, errors = None, 0
    with pred_path.open("a", encoding="utf-8", newline="\n") as handle:
        for item in items:
            if item["id"] in done:
                continue
            try:
                result = le.extract(item["id"], item["text"], client, examples, p_hash, cache)
            except le.DailyLimitReached as exc:
                stopped = str(exc)
                break
            except (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError) as exc:
                errors += 1
                print(f"[skip] {item['id']}: {type(exc).__name__} (will be retried on the next run)", flush=True)
                continue
            record = result.record.model_dump()
            calls = [{**c.__dict__, "from_cache": result.from_cache} for c in result.calls]
            row = {"id": item["id"], "model": spec["model"], "category": item.get("category"), "subtype": item.get("subtype"),
                   "status": result.status, "rejected": result.rejected, "duplicate_quotes": result.duplicate_quotes,
                   "predicted": as_targets(record), "record": record, "calls": calls,
                   "cost_usd": 0.0 if result.from_cache else sum(le.call_cost(c, prices) for c in result.calls)}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done.add(item["id"])
    rows = [r for r in read_jsonl(pred_path) if r["id"] in by_id]
    metrics = set_metrics(set_name, rows, by_id, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "provider": provider, "model": spec["model"], "set": set_name, "prompt_version": prompt["version"],
        "prompt_hash": p_hash, "structured": spec["structured"], "extra": spec.get("extra"),
        "items_done": len(rows), "items_total": len(items), "stopped_by_daily_cap": stopped, "api_errors_this_run": errors,
        "rate_limit_headers": getattr(client, "last_limits", {}), "dev_set": DEV_SET.as_posix(),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
        "test_policy": "Test was not read.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    state = "stopped at a daily cap" if stopped else ("finished" if len(rows) >= len(items) else "incomplete")
    print(f"{provider}/{set_name}: {len(rows)}/{len(items)} items, {state}; errors this run {errors}", flush=True)


def set_metrics(set_name: str, rows: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    if set_name in ("dev", "validation"):
        return summarize(rows, by_id)
    if set_name == "presence_challenge_v1":
        return {"items": len(rows), "by_category": eval_sets.summarize_presence_challenge(rows, by_id),
                **{k: v for k, v in summarize_llm(rows).items()}}
    if set_name == "extraction_challenge_v3":
        return {"items": len(rows), "by_subtype": eval_sets.summarize_v3(rows, by_id), **summarize_llm(rows)}
    if set_name == "repeat":
        first_path = out_dir.parent / out_dir.name.replace("repeat_", "validation_") / "predictions.jsonl"
        first = {r["id"]: r for r in read_jsonl(first_path)} if first_path.exists() else {}
        return {"items": len(rows), **eval_sets.summarize_repeat(rows, first), **summarize_llm(rows)}
    raise KeyError(set_name)


def summarize_llm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM-specific measures shared by every set."""
    n = len(rows)
    calls = [c for r in rows for c in r["calls"]]
    latencies = sorted(c["latency_ms"] for c in calls if not r_from_cache(c))
    return {
        "invalid_output_rate": round(sum(r["status"] == "invalid_output" for r in rows) / n, 4) if n else None,
        "refusal_rate": round(sum(r["status"] == "refusal" for r in rows) / n, 4) if n else None,
        "evidence_rejections": {f: sum(f in r["rejected"] for r in rows) for f in le.FIELDS},
        "tokens": {k: sum(c[k] for c in calls) for k in ("input_tokens", "output_tokens", "cache_read_tokens")},
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95)} if latencies else None,
    }


def rescore() -> None:
    """Recompute metrics.json for every dev folder from its predictions (no API calls)."""
    items = {i["id"]: i for i in dev_items()}
    for pred_path in sorted(Path(json.loads(CONFIG.read_text(encoding="utf-8"))["results_root"]).glob("*/*/dev_*/predictions.jsonl")):
        rows = read_jsonl(pred_path)
        (pred_path.parent / "metrics.json").write_text(json.dumps(summarize(rows, items), indent=2) + "\n",
                                                       encoding="utf-8")
        print(f"{pred_path.parent}: {len(rows)} items")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dev-set", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--set", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rescore", action="store_true", help="recompute dev metrics from predictions; no API calls")
    args = parser.parse_args()
    if args.rescore:
        rescore()
        return
    if args.build_dev_set:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        record = build_dev_set(set(config["few_shot"]["ids"]))
        print(f"Wrote {DEV_SET}: {record['count']} items")
        return
    run(args.provider, args.set, args.limit)


if __name__ == "__main__":
    main()
