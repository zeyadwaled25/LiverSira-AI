"""Freeze the Phase 8 extraction candidates and their analysis plan.

Writes results/nlp/phase8/candidates_freeze.json with the SHA-256 of every
file a candidate depends on (code, configs, weights, tokenizers), the analysis
plan, the Train-aug taxonomy and the parser v1.2 evidence report. Model weights
are also checked against the hash their training run recorded.

    python -m scripts.nlp.transformers.extraction.freeze_candidates          # write (refuses to overwrite)
    python -m scripts.nlp.transformers.extraction.freeze_candidates --amend 1 --reason "..."  # re-freeze
    python -m scripts.nlp.transformers.extraction.freeze_candidates --check  # verify nothing changed

Hashes: text files are hashed after normalizing CRLF to LF, so checkouts with
core.autocrlf do not change them; binary files (weights) are hashed as raw
bytes. The original freeze (commit 9b088a8) hashed raw bytes for every file.
A re-freeze needs a matching amendment in the analysis plan and refuses to run
once Challenge-B has been frozen.

The Phase 8 run must call ``check()`` first and stop on any mismatch.
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from liversira_ai.common.provenance import git_commit, sha256_file

FREEZE_PATH = Path("results/nlp/phase8/candidates_freeze.json")
CHALLENGE_B_FREEZE = Path("results/nlp/phase8/challenge_b_freeze.json")
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".txt", ".csv", ".toml", ".yaml", ".yml", ".cfg", ".ini"}
HASH_METHOD = "sha256; text files (by suffix) after CRLF->LF normalization, other files raw bytes"
ORIGINAL_FREEZE = {"commit": "9b088a8", "hash_method": "sha256 of raw bytes for every file (before .gitattributes)"}
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
    "C5": {"name": "hybrid H = C2, falling back to C3 per field (analysis plan §4); the primary hybrid",
           "files": ["src/liversira_ai/nlp/hybrid.py"], "uses": ["C2", "C3"]},
    "C6": {"name": "hybrid H2, exploratory (Amendment 1); appendix only, never eligible",
           "files": ["src/liversira_ai/nlp/hybrid.py", "results/nlp/phase8/amendment1_train_aug.json"],
           "uses": ["C2", "C3"]},
}
WEIGHT_FILES = ("model.safetensors", "tokenizer.json", "tokenizer_config.json")


def freeze_hash(path: Path) -> str:
    """SHA-256 used by the freeze: LF-normalized for text files, raw bytes otherwise."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _hashes(paths: list[str]) -> dict[str, str]:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    return {p: freeze_hash(Path(p)) for p in paths}


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
    return [p for p, h in frozen["sha256"].items() if not Path(p).exists() or freeze_hash(Path(p)) != h]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--amend", type=int, help="amendment number in the analysis plan that this re-freeze follows")
    parser.add_argument("--reason", help="one line, required with --amend")
    args = parser.parse_args()
    if args.check:
        changed = check()
        print("unchanged" if not changed else f"CHANGED: {changed}")
        sys.exit(1 if changed else 0)
    previous = None
    if FREEZE_PATH.exists():
        if args.amend is None:
            raise SystemExit(f"{FREEZE_PATH} exists; re-freeze only with --amend N after amending the plan")
        if CHALLENGE_B_FREEZE.exists():
            raise SystemExit("Challenge-B is frozen; the candidates can no longer be re-frozen")
        if not args.reason or f"### Amendment {args.amend} " not in PLAN.read_text(encoding="utf-8"):
            raise SystemExit(f"--reason is required and the plan must contain '### Amendment {args.amend} '")
        old = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        previous = {"freeze_file_sha256": hashlib.sha256(FREEZE_PATH.read_bytes()).hexdigest(),
                    "frozen_on": old["frozen_on"], "git_commit_at_freeze": old["git_commit_at_freeze"],
                    "amendment": old.get("amendment", 0),
                    "hash_method": old.get("hash_method", ORIGINAL_FREEZE["hash_method"])}
    record = {
        "schema_version": "1.0",
        "frozen_on": date.today().isoformat(),
        "git_commit_at_freeze": git_commit(),
        "note": "The hashes are the freeze. git_commit_at_freeze is HEAD when this was written; the frozen "
                "files may have been uncommitted at that moment. No candidate had read Test or Challenge-B.",
        "analysis_plan": PLAN.as_posix(),
        "amendment": args.amend or 0,
        "amendment_reason": args.reason,
        "challenge_b_frozen_at_this_freeze": CHALLENGE_B_FREEZE.exists(),
        "hash_method": HASH_METHOD,
        "original_freeze": ORIGINAL_FREEZE,
        "supersedes": previous,
        **collect(),
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {FREEZE_PATH} ({len(record['sha256'])} files)")


if __name__ == "__main__":
    main()
