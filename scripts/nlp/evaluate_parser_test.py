"""Single-use Test evaluation of the frozen rule-based parser v1.1.

Safeguards:

* Refuses to run if the parser or baseline code has uncommitted changes, so
  the evaluated code is exactly the recorded commit.
* Refuses to run if Test outputs already exist: the parser is evaluated on Test
  once. Any later parser change must be judged on Validation or the gold set,
  never by re-running this script.
* Records the commit and the parser file hash before any Test report is read.

Usage (from the repository root):

    python scripts/nlp/evaluate_parser_test.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reliability_report as rr  # noqa: E402
import validate_parser as vp  # noqa: E402
from liversira_ai.nlp.baseline import BASELINE_VERSION, run_baseline_inference  # noqa: E402
from liversira_ai.nlp.metrics import format_ci  # noqa: E402
from liversira_ai.nlp.parser import PARSER_VERSION  # noqa: E402

FROZEN_CODE = [Path("src/liversira_ai/nlp/parser.py"), Path("src/liversira_ai/nlp/baseline.py")]
V11_TEST_PATH = Path("results/nlp/predictions/baseline_v1_1_test_inferences.jsonl")
V11_TEST_MANIFEST = Path("results/nlp/predictions/baseline_v1_1_test_manifest.json")
METRICS_PATH = vp.OUTPUT_DIR / "test_evaluation.json"
REPORT_PATH = vp.OUTPUT_DIR / "test_evaluation.md"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _freeze_record() -> dict[str, Any]:
    dirty = _git("status", "--porcelain", "--", *(p.as_posix() for p in FROZEN_CODE))
    if dirty:
        raise RuntimeError(f"Commit the parser before the Test run; uncommitted changes:\n{dirty}")
    existing = [p for p in (V11_TEST_PATH, METRICS_PATH, REPORT_PATH) if p.exists()]
    if existing:
        raise RuntimeError(
            "Parser v1.1 has already been evaluated on Test "
            f"({', '.join(map(str, existing))}). This evaluation is single-use."
        )
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "code_sha256": {
            p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN_CODE
        },
        "parser_version": PARSER_VERSION,
        "baseline_version": BASELINE_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run() -> dict[str, Any]:
    freeze = _freeze_record()
    splits = json.loads(rr.SPLITS_PATH.read_text(encoding="utf-8"))
    test = list(splits["test"])
    run_baseline_inference(
        rr.GROUND_TRUTH_PATH.parent / "master_records.jsonl", V11_TEST_PATH, set(test), V11_TEST_MANIFEST
    )
    v11 = rr._load_jsonl(V11_TEST_PATH)
    if set(v11) != set(test):
        raise ValueError("v1.1 Test predictions must cover exactly the Test split")
    v10 = {i: r for i, r in rr._load_jsonl(vp.V10_PATH).items() if i in set(test)}
    reference = rr._reference_labels(rr.GROUND_TRUTH_PATH, "ground_truth")
    return {
        "schema_version": "1.0",
        "split": "test",
        "status": "single_use_frozen_evaluation",
        "freeze": freeze,
        "reference": "source-derived labels (not a clinical gold standard)",
        "method": {"interval": "percentile bootstrap over studies", "n_resamples": vp.RESAMPLES, "seed": vp.SEED},
        "inputs_sha256": {
            p.as_posix(): vp._sha(p) for p in (rr.GROUND_TRUTH_PATH, rr.SPLITS_PATH, vp.V10_PATH, V11_TEST_PATH)
        },
        "v1.0": vp._score(test, reference, v10),
        "v1.1": vp._score(test, reference, v11),
        "paired_presence_v11_minus_v10_on_both_answered": vp._paired_presence(test, reference, v10, v11),
    }


def render(result: dict[str, Any]) -> str:
    # Reuse the validation layout, then replace the framing text.
    body = vp.render(
        {
            "versions": {"v1.0": "rule-based-v1.0.0 (saved predictions)", "v1.1": f"{BASELINE_VERSION} / parser {PARSER_VERSION}"},
            "method": result["method"],
            "references": {"source": {k: result[k] for k in ("v1.0", "v1.1", "paired_presence_v11_minus_v10_on_both_answered")}},
        }
    )
    tables = body.split("## Against source-derived labels", 1)[1].split("## Notes", 1)[0]
    freeze = result["freeze"]
    return "\n".join(
        [
            "# Parser v1.1 vs v1.0: Test (single frozen evaluation)",
            "",
            f"Parser {freeze['parser_version']} was frozen at commit `{freeze['git_commit'][:7]}` "
            "before any Test report was parsed, and this evaluation runs once. Test studies: "
            f"{result['v1.1']['studies']}. Reference: {result['reference']}. 95% bootstrap intervals "
            f"over studies ({result['method']['n_resamples']} resamples, seed {result['method']['seed']}).",
            "",
            "## Against source-derived labels",
            tables.rstrip(),
            "",
            "## Notes",
            "- v1.1 was developed on Validation only; these Test numbers are its unbiased estimate "
            "against source labels. Clinical accuracy still needs the human gold standard.",
            "- v1.1 leaves \"multiple lesions\" unquantified instead of setting 1, so count coverage "
            "is lower by design.",
            "- Do not change the parser in response to these numbers and re-run on Test; judge later "
            "versions on Validation and the gold set.",
            "",
        ]
    )


def main() -> None:
    result = run()
    vp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(f"[DONE] {METRICS_PATH}\n[DONE] {REPORT_PATH}")


if __name__ == "__main__":
    main()
