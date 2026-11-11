"""Command-line entry point for the two-annotator human gold set.

Run from the repository root, one step at a time:

    python scripts/nlp/gold_set.py sample     # blinded sheets A and B + key
    # ... annotators fill annotation_sheet_A.csv and annotation_sheet_B.csv ...
    python scripts/nlp/gold_set.py agree      # agreement + adjudication sheet
    # ... a clinician completes rows marked needs_clinician=YES ...
    python scripts/nlp/gold_set.py finalize   # review CSV for adjudication
    python scripts/nlp/gold_set.py apply      # gold_standard.jsonl + GT statuses
    python scripts/nlp/reliability_report.py  # scores against the gold standard

See docs/nlp/gold_set_annotation_guidelines.md.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from liversira_ai.nlp.data.adjudication import adjudicate_cohort  # noqa: E402
from liversira_ai.nlp.data.gold_set import (  # noqa: E402
    compare_annotations,
    create_sample,
    finalize,
)

DATA = Path("data/processed/nlp")
WORK = DATA / "gold_set"  # contains report text; gitignored with the rest of data/
RESULTS = Path("results/nlp/gold_set")  # IDs-free summaries only; safe to commit

PATHS = {
    "master": DATA / "master_records.jsonl",
    "ground_truth": DATA / "ground_truth.jsonl",
    "audit": DATA / "target_consistency_audit.csv",
    "splits": DATA / "splits.json",
    "sheet_a": WORK / "annotation_sheet_A.csv",
    "sheet_b": WORK / "annotation_sheet_B.csv",
    "adjudication": WORK / "adjudication_sheet.csv",
    "review": WORK / "gold_set_review.csv",
    "gold": DATA / "gold_standard.jsonl",
    "manifest": RESULTS / "gold_set_sample_manifest.json",
    "agreement": RESULTS / "inter_annotator_agreement.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "agree", "finalize", "apply"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.step == "sample":
        key = create_sample(
            master_path=PATHS["master"],
            audit_path=PATHS["audit"],
            splits_path=PATHS["splits"],
            output_dir=WORK,
            manifest_path=PATHS["manifest"],
            seed=args.seed,
        )
        print(key.groupby(["split", "stratum"]).size().to_string())
        print(f"[DONE] {len(key)} studies. Sheets: {PATHS['sheet_a']}, {PATHS['sheet_b']}")
    elif args.step == "agree":
        agreement = compare_annotations(
            PATHS["sheet_a"], PATHS["sheet_b"], PATHS["adjudication"], PATHS["agreement"]
        )
        print(json.dumps(agreement, indent=2))
        print(f"[DONE] Adjudication sheet: {PATHS['adjudication']}")
    elif args.step == "finalize":
        review = finalize(PATHS["adjudication"], PATHS["ground_truth"], PATHS["review"])
        pending = int((review["reviewer_id"].astype(str).str.strip() == "").sum())
        print(f"[DONE] Review CSV: {PATHS['review']} ({pending} rows still need a clinician)")
    else:
        verified = adjudicate_cohort(PATHS["review"], PATHS["ground_truth"], PATHS["gold"])
        print(f"[DONE] Gold standard: {PATHS['gold']}; {verified} source labels fully verified")


if __name__ == "__main__":
    main()
