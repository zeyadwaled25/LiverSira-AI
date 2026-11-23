"""Freeze the Phase 8 extraction candidates and their analysis plan.

Writes results/nlp/phase8/candidates_freeze.json with the SHA-256 of every
file a candidate depends on (code, configs, weights, tokenizers), the analysis
plan, the Train-aug taxonomy and the parser v1.2 evidence report. Model weights
are also checked against the hash their training run recorded.

    python -m scripts.nlp.transformers.extraction.freeze_candidates          # write (refuses to overwrite)
    python -m scripts.nlp.transformers.extraction.freeze_candidates --check  # verify nothing changed

The Phase 8 run must call ``check()`` first and stop on any mismatch.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from liversira_ai.common.provenance import git_commit, sha256_file

FREEZE_PATH = Path("results/nlp/phase8/candidates_freeze.json")
PLAN = Path("docs/nlp/phase8_analysis_plan.md")
SEEDS = (13, 42, 87)
SHARED = [
    "src/liversira_ai/nlp/liver_text.py",
    "src/liversira_ai/nlp/baseline.py",
    "src/liversira_ai/nlp/extraction_targets.py",
    "scripts/nlp/transformers/extraction/train.py",  # field_correct, the scoring rule
    "docs/nlp/phase8_analysis_plan.md",
    "docs/nlp/challenge_b_brief.md",
    "scripts/nlp/transformers/extraction/validate_challenge_b.py",
    "docs/nlp/extraction_train_aug_taxonomy.md",
    "docs/nlp/extraction_train_aug_taxonomy.json",
    "src/liversira_ai/nlp/metrics.py",  # wilson_interval, paired_comparison
]
CANDIDATES: dict[str, dict[str, Any]] = {
    "C1": {"name": "parser v1.1", "files": ["src/liversira_ai/nlp/parser_v1_1.py"]},
    "C2": {"name": "parser v1.2", "files": ["src/liversira_ai/nlp/parser_v1_2.py",
                                            "src/liversira_ai/nlp/parser_v1_2_rules.py",
                                            "results/nlp/parser_v1_2/train_aug_report.json"]},
    "C3": {"name": "first extraction model (ClinicalBERT)", "config": "scripts/nlp/transformers/extraction/config.json",
           "code": ["src/liversira_ai/nlp/extraction_model.py"],
           "weights": "models/transformers/extraction/clinicalbert/seed{seed}",
           "run": "results/nlp/transformers/extraction/runs/clinicalbert/seed{seed}/run_manifest.json"},
    "C4": {"name": "per-lesion model (ClinicalBERT)", "config": "scripts/nlp/transformers/extraction/config_lesion.json",
           "code": ["src/liversira_ai/nlp/lesion_model.py", "src/liversira_ai/nlp/lesion_targets.py"],
           "weights": "models/transformers/extraction_lesion/clinicalbert/seed{seed}",
           "run": "results/nlp/transformers/extraction/runs_lesion/clinicalbert/seed{seed}/run_manifest.json"},
    "C5": {"name": "hybrid H = C2, falling back to C3 per field (analysis plan §4)", "uses": ["C2", "C3"]},
}
WEIGHT_FILES = ("model.safetensors", "tokenizer.json", "tokenizer_config.json")


def _hashes(paths: list[str]) -> dict[str, str]:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    return {p: sha256_file(Path(p)) for p in paths}


def collect() -> dict[str, Any]:
    files = list(SHARED)
    candidates = {}
    for cid, spec in CANDIDATES.items():
        entry: dict[str, Any] = {"name": spec["name"]}
        if "files" in spec:
            entry["files"] = spec["files"]
            files += spec["files"]
        if "config" in spec:
            entry["files"] = [spec["config"], *spec["code"]]
            files += entry["files"]
            entry["seeds"] = {}
            for seed in SEEDS:
                folder = Path(spec["weights"].format(seed=seed))
                weights = [(folder / f).as_posix() for f in WEIGHT_FILES]
                run = json.loads(Path(spec["run"].format(seed=seed)).read_text(encoding="utf-8"))
                weight_hash = sha256_file(folder / "model.safetensors")
                if weight_hash != run["weights"]["sha256"]:
                    raise SystemExit(f"{folder}: weights differ from the hash its training run recorded")
                entry["seeds"][str(seed)] = {"weights": weights, "run_manifest": spec["run"].format(seed=seed)}
                files += weights + [spec["run"].format(seed=seed)]
        if "uses" in spec:
            entry["uses"] = spec["uses"]
        candidates[cid] = entry
    return {"candidates": candidates, "sha256": _hashes(sorted(set(files)))}


def check() -> list[str]:
    """Paths whose hash no longer matches the freeze (empty list = unchanged)."""
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    return [p for p, h in frozen["sha256"].items() if not Path(p).exists() or sha256_file(Path(p)) != h]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        changed = check()
        print("unchanged" if not changed else f"CHANGED: {changed}")
        sys.exit(1 if changed else 0)
    if FREEZE_PATH.exists():
        raise SystemExit(f"{FREEZE_PATH} exists; a freeze is not overwritten (amend the plan instead)")
    record = {
        "schema_version": "1.0",
        "frozen_on": date.today().isoformat(),
        "git_commit_at_freeze": git_commit(),
        "note": "The hashes are the freeze. git_commit_at_freeze is HEAD when this was written; the frozen "
                "files may have been uncommitted at that moment. No candidate had read Test or Challenge-B.",
        "analysis_plan": PLAN.as_posix(),
        **collect(),
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {FREEZE_PATH} ({len(record['sha256'])} files)")


if __name__ == "__main__":
    main()
