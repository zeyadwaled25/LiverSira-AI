"""Run the frozen Phase 7 runner with a transport-error log (no frozen file changes).

Transport failures (timeouts, connection errors, HTTP 408/409/429/5xx) are retried by the
``openai`` SDK, up to ``max_retries`` per provider. If every retry fails, the runner skips the
item: no row is written, so a transport failure is never recorded as ``llm_invalid_output``
or as an abstention, and the next run calls the item again.

The runner keeps only the number of skipped items of its last run (``run_manifest.json``).
This wrapper adds one line per transport event to ``<set folder>/transport_log.jsonl``:

    {"time", "provider", "set", "id", "event", "final"}

- ``event``: ``timeout``, ``connection`` or ``http_<code>`` for each failed attempt (from the SDK's
  debug log), or ``skipped_<Exception>`` when the item is given up after every retry;
- ``final``: true only for the skipped line.

Usage: python -m scripts.nlp.llm.run_logged --provider nvidia --set repeat
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from liversira_ai.nlp import llm_extraction as le
from scripts.nlp.llm import run_llm_extraction as runner

TRANSPORT_ERRORS = (openai.APIStatusError, openai.APIConnectionError, openai.APITimeoutError)
HTTP_ERROR = re.compile(r"Encountered an HTTP status error: (\d+)")


class TransportLog(logging.Handler):
    def __init__(self, path: Path, provider: str, set_name: str):
        super().__init__(logging.DEBUG)
        self.path, self.base, self.item = path, {"provider": provider, "set": set_name}, None

    def write(self, event: str, final: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **self.base, "id": self.item,
               "event": event, "final": final}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Encountered a timeout exception"):
            self.write("timeout")
        elif message.startswith("Encountered exception"):
            self.write("connection")
        elif match := HTTP_ERROR.match(message):
            self.write(f"http_{match.group(1)}")


def set_folder(provider: str, set_name: str) -> Path:
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    model = config["providers"][provider]["model"]
    return Path(config["results_root"]) / provider / runner.model_slug(model) / f"{set_name}_{runner.current_prompt()['version']}"


def run(provider: str, set_name: str, limit: int | None = None) -> None:
    handler = TransportLog(set_folder(provider, set_name) / "transport_log.jsonl", provider, set_name)
    sdk_log = logging.getLogger("openai")
    level = sdk_log.level
    sdk_log.setLevel(logging.DEBUG)
    sdk_log.addHandler(handler)
    original = le.extract

    def extract(study_id: str, *args: Any, **kwargs: Any) -> le.Extraction:
        handler.item = study_id
        try:
            return original(study_id, *args, **kwargs)
        except TRANSPORT_ERRORS as exc:
            handler.write(f"skipped_{type(exc).__name__}", final=True)
            raise

    le.extract = extract
    try:
        runner.run(provider, set_name, limit)
    finally:
        le.extract = original
        sdk_log.removeHandler(handler)
        sdk_log.setLevel(level)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--set", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.provider, args.set, args.limit)


if __name__ == "__main__":
    main()
