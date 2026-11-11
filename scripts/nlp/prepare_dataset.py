"""Run the active NLP dataset preparation stages."""

import runpy
from pathlib import Path

from liversira_ai.nlp.data.audit import main as audit_main

STAGE_DIR = Path(__file__).resolve().parent


def run_stage(filename: str) -> None:
    runpy.run_path(str(STAGE_DIR / filename), run_name="__main__")


def main() -> None:
    run_stage("prepare_nlp_dataset.py")
    run_stage("generate_master_jsonl.py")
    run_stage("generate_ground_truth_jsonl.py")
    audit_main()
    run_stage("split_dataset.py")


if __name__ == "__main__":
    main()
