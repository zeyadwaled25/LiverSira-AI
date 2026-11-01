"""Run the full LiverSira-AI pipeline end-to-end for validation/reproduction.

Usage (from the project root):
    python -m scripts.run_pipeline
"""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = [
    "src.audit_dataset",
    "src.check_pdf_csv_match",
    "src.prepare_nlp_dataset",
    "src.generate_master_jsonl",
    "src.generate_ground_truth_jsonl",
    "src.audit_target_consistency",
    "src.validate_targets",
    "src.adjudicate_validation_sample",
    "src.split_dataset",
    "src.baseline_extractor",
    "evaluation.evaluate_baseline",
]


def main() -> None:
    for module in STEPS:
        print("\n" + "#" * 70)
        print(f"RUNNING: {module}")
        print("#" * 70)
        runpy.run_module(module, run_name="__main__")
    print("\n" + "#" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("#" * 70)


if __name__ == "__main__":
    main()
