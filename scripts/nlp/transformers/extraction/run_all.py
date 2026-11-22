"""Run every model x seed in config.json, one process per run; finished runs are skipped."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = Path("results/nlp/transformers/extraction/runs")


def main() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    failed = []
    for model in config["models"]:
        for seed in config["seeds"]:
            if (RESULTS_ROOT / model / f"seed{seed}" / "run_manifest.json").exists():
                print(f"[SKIP] {model} seed {seed}: finished", flush=True)
                continue
            print(f"[RUN] {model} seed {seed}", flush=True)
            started = time.time()
            code = subprocess.call(
                [sys.executable, "-m", "scripts.nlp.transformers.extraction.train", "--model", model, "--seed", str(seed)],
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
            )
            print(f"[{'DONE' if code == 0 else 'FAIL'}] {model} seed {seed} in {time.time() - started:.0f}s", flush=True)
            if code != 0:
                failed.append(f"{model}/seed{seed}")
    print(f"[END] failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
