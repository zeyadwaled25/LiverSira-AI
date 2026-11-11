"""Validation-only comparison of the rule-based parser v1.1 with frozen v1.0.

* v1.1 is run on Validation studies only (``run_baseline_inference`` with the
  Validation IDs); Test reports are never parsed or scored here.
* v1.0 is read from its saved predictions, so it is exactly the frozen baseline.
* Both are scored with the same bootstrap metrics as the reliability report,
  against source-derived labels and, once it exists, the Validation part of the
  human gold standard.

Usage (from the repository root):

    python scripts/nlp/validate_parser.py
"""

import json
from pathlib import Path
from typing import Any

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp.baseline import BASELINE_VERSION, run_baseline_inference
from liversira_ai.nlp.evaluation import (
    GOLD_PATH,
    GROUND_TRUTH_PATH,
    SPLITS_PATH,
    load_jsonl,
    reference_labels,
)
from liversira_ai.nlp.parser import PARSER_VERSION
from liversira_ai.nlp.parser_comparison import (
    OUTPUT_DIR,
    RESAMPLES,
    SEED,
    V10_PATH,
    paired_presence,
    render,
    score,
)

V11_PATH = Path("results/nlp/predictions/baseline_v1_1_validation_inferences.jsonl")
V11_MANIFEST = Path("results/nlp/predictions/baseline_v1_1_validation_manifest.json")
METRICS_PATH = OUTPUT_DIR / "validation_comparison.json"
REPORT_PATH = OUTPUT_DIR / "validation_comparison.md"


def run() -> dict[str, Any]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    validation = list(splits["val"])
    test_ids = set(splits["test"])
    if test_ids & set(validation):
        raise ValueError("Validation and Test overlap")

    run_baseline_inference(GROUND_TRUTH_PATH.parent / "master_records.jsonl", V11_PATH, set(validation), V11_MANIFEST)
    v11 = load_jsonl(V11_PATH)
    if set(v11) & test_ids or set(v11) != set(validation):
        raise ValueError("v1.1 predictions must cover exactly the Validation split")
    v10 = {i: r for i, r in load_jsonl(V10_PATH).items() if i in set(validation)}

    references = {"source": reference_labels(GROUND_TRUTH_PATH, "ground_truth")}
    if GOLD_PATH.exists():
        gold = reference_labels(GOLD_PATH, "gold")
        references["gold"] = {i: g for i, g in gold.items() if i in set(validation)}

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "split": "validation",
        "test_policy": "Test reports were not parsed or scored.",
        "versions": {"v1.0": "rule-based-v1.0.0 (saved predictions)", "v1.1": f"{BASELINE_VERSION} / parser {PARSER_VERSION}"},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p) for p in (GROUND_TRUTH_PATH, SPLITS_PATH, V10_PATH, V11_PATH)
        },
        "git_commit": git_commit(),
        "method": {"interval": "percentile bootstrap over studies", "n_resamples": RESAMPLES, "seed": SEED},
        "references": {},
    }
    for name, reference in references.items():
        result["references"][name] = {
            "v1.0": score(validation, reference, v10),
            "v1.1": score(validation, reference, v11),
            "paired_presence_v11_minus_v10_on_both_answered": paired_presence(validation, reference, v10, v11),
        }
    return result


def main() -> None:
    result = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(f"[DONE] {METRICS_PATH}\n[DONE] {REPORT_PATH}")


if __name__ == "__main__":
    main()
