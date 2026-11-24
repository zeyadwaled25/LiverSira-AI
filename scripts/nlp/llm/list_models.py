"""List the models each configured OpenAI-compatible provider offers, and any rate-limit headers.

    python -m scripts.nlp.llm.list_models [--provider groq] [--filter llama]

Keys are read from the environment variables named in config.json. On Windows, a
key set at user level after this shell started is read from the user environment
(HKCU\\Environment). Keys are never printed or written anywhere. No completion is
requested: listing models uses no generation quota.
"""

import argparse
import json
import os
import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config.json"


def api_key(name: str) -> str | None:
    """The key from this process's environment, or (Windows) the user-level environment."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
            return winreg.QueryValueEx(handle, name)[0] or None
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider")
    parser.add_argument("--filter", default="", help="show only model IDs containing this text")
    args = parser.parse_args()

    import openai

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for name, spec in config["providers"].items():
        if spec.get("client") != "openai_compatible" or (args.provider and name != args.provider):
            continue
        key = api_key(spec["api_key_env"])
        print(f"\n== {name} ({spec['base_url']})")
        if not key:
            print(f"   {spec['api_key_env']} is not set")
            continue
        client = openai.OpenAI(base_url=spec["base_url"], api_key=key, max_retries=1, timeout=30)
        try:
            raw = client.models.with_raw_response.list()
        except openai.APIStatusError as exc:
            print(f"   error {exc.status_code}: {exc.message}")
            continue
        except openai.APIConnectionError as exc:
            print(f"   connection error: {exc}")
            continue
        limits = {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower()}
        ids = sorted(m.id for m in raw.parse().data if args.filter.lower() in m.id.lower())
        print(f"   {len(ids)} models" + (f"; rate-limit headers: {limits}" if limits else ""))
        for model_id in ids:
            print(f"   - {model_id}")


if __name__ == "__main__":
    main()
