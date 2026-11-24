"""One small call per provider: does the JSON shape work, and what limits does the provider report?

    python -m scripts.nlp.llm.smoke_test [--provider groq]

Uses a short synthetic report (no dataset text) with one lesion and one non-lesion
size. Runs the full pipeline (prompt, structured output, validation with one retry,
evidence check). Writes results/nlp/llm/smoke_test.json: status, values, tokens,
latency and rate-limit headers per provider. Keys are never printed or stored.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from liversira_ai.nlp import llm_extraction as le
from scripts.nlp.llm.list_models import CONFIG, api_key

REPORT = ("The liver is normal in size and spans 16.2 cm craniocaudally. A hypoattenuating lesion in hepatic "
          "segment 6 measures 2.4 x 1.8 cm. No other liver lesions are seen.")
EXPECTED = {"has_lesion": True, "lesion_count": 1, "location_segments": [6],
            "reported_dimensions_cm": [2.4, 1.8], "attenuation": "hypoattenuating"}
OUT = Path("results/nlp/llm/smoke_test.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name, spec in config["providers"].items():
        if spec.get("client") != "openai_compatible" or (args.provider and name != args.provider):
            continue
        key = api_key(spec["api_key_env"])
        if key:
            os.environ[spec["api_key_env"]] = key  # this process only
        try:
            client = le.OpenAICompatibleClient(spec["model"], spec["base_url"], spec["api_key_env"],
                                               structured=spec["structured"], max_tokens=spec["max_tokens"],
                                               extra=spec.get("extra"), max_retries=1)
            p_hash = le.prompt_hash(le.SYSTEM_PROMPT, [], le.OUTPUT_SCHEMA)
            result = le.extract("smoke", REPORT, client, [], p_hash)
        except Exception as exc:  # report and continue with the next provider
            results[name] = {"model": spec["model"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(f"{name}: ERROR {results[name]['error']}")
            continue
        lesions = result.record.liver_lesions
        largest = lesions.largest_lesion
        got = {
            "has_lesion": lesions.has_lesion,
            "lesion_count": lesions.lesion_count,
            "location_segments": largest.location_segments if largest else None,
            "reported_dimensions_cm": largest.reported_dimensions_cm if largest else None,
            "attenuation": largest.attenuation if largest else None,
        }
        results[name] = {
            "model": spec["model"],
            "structured": spec["structured"],
            "status": result.status,
            "calls": len(result.calls),
            "values": got,
            "matches_expected": {k: got[k] == v for k, v in EXPECTED.items()},
            "rejected_as_unsupported": result.rejected,
            "served_by": result.calls[-1].model_served,
            "finish_reason": result.calls[-1].stop_reason,
            "tokens": {"input": sum(c.input_tokens for c in result.calls),
                       "cached": sum(c.cache_read_tokens for c in result.calls),
                       "output": sum(c.output_tokens for c in result.calls)},
            "latency_ms": [c.latency_ms for c in result.calls],
            "rate_limit_headers": getattr(client, "last_limits", {}),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"{name}: {result.status}, {sum(results[name]['matches_expected'].values())}/5 fields as expected, "
              f"{results[name]['tokens']}, {results[name]['latency_ms']} ms")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"report": REPORT, "expected": EXPECTED, **results}, indent=2, default=str) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
