"""Run every model x seed in ``config.json``, skipping runs that already finished.

Each run is a separate process, so GPU memory is released between runs and a
failure stops only that run. Re-running this script resumes where it stopped.

Usage (from the repository root):

    .venv\\Scripts\\python.exe scripts/nlp/transformers/presence/run_all.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = Path("results/nlp/transformers/presence")


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
            code = subprocess.call([sys.executable, str(HERE / "train.py"), "--model", model, "--seed", str(seed)])
            print(f"[{'DONE' if code == 0 else 'FAIL'}] {model} seed {seed} in {time.time() - started:.0f}s", flush=True)
            if code != 0:
                failed.append(f"{model}/seed{seed}")
    print(f"[END] failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
