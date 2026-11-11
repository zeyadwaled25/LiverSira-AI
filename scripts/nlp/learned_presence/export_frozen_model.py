"""Save, hash, and verify the frozen learned lesion-presence model.

The final Test evaluation reported results for a model that was never kept on
disk. This script re-fits the identical frozen configuration on the same
Train + Validation reports, saves it, and records its SHA-256 together with the
inputs and library versions in a manifest.

Reproducibility check: the re-fitted model must reproduce the persisted Test
predictions exactly (same labels, probabilities within 1e-12). This compares
against stored predictions only; no metric is recomputed and nothing is tuned.

Usage (from the repository root):

    python -m scripts.nlp.learned_presence.export_frozen_model
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp.learned_presence import load_jsonl
from scripts.nlp.learned_presence import final_evaluate_learned_presence as final

MODEL_PATH = final.MODEL_PATH
MANIFEST_PATH = final.OUTPUT_DIR / "learned_presence_word_logreg_final_model_manifest.json"
STORED_TEST_PREDICTIONS = Path("results/nlp/predictions/learned_presence_test_predictions.jsonl")
SPLIT_METADATA_PATH = Path("data/processed/nlp/split_metadata.json")
PROBABILITY_TOLERANCE = 1e-12
THRESHOLD = 0.5  # Frozen decision threshold (see the final configuration).


def _verify_against_stored_predictions(model: Pipeline) -> dict[str, Any]:
    with STORED_TEST_PREDICTIONS.open(encoding="utf-8") as handle:
        stored = [json.loads(line) for line in handle if line.strip()]
    ids = [row["id"] for row in stored]
    master = load_jsonl(final.MASTER_PATH, set(ids))
    texts = [master[record_id]["reports"]["narrative_clean"] for record_id in ids]
    probabilities = model.predict_proba(texts)[:, 1]
    stored_probabilities = np.array([row["probability_positive"] for row in stored])
    predictions = probabilities >= THRESHOLD
    label_mismatches = int(
        sum(bool(p) != bool(row["prediction"]) for p, row in zip(predictions, stored))
    )
    max_difference = float(np.max(np.abs(probabilities - stored_probabilities)))
    return {
        "reference": str(STORED_TEST_PREDICTIONS),
        "reference_sha256": sha256_file(STORED_TEST_PREDICTIONS),
        "records_compared": len(stored),
        "prediction_mismatches": label_mismatches,
        "max_abs_probability_difference": max_difference,
        "tolerance": PROBABILITY_TOLERANCE,
        "passed": label_mismatches == 0 and max_difference <= PROBABILITY_TOLERANCE,
    }


def export() -> dict[str, Any]:
    model, train_ids, _, _ = final.fit_frozen_model()
    verification = _verify_against_stored_predictions(model)
    if not verification["passed"]:
        raise RuntimeError(
            "Re-fitted model does not reproduce the stored Test predictions: "
            f"{verification}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    split_metadata = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0",
        "model": "learned_presence_word_logreg_final",
        "task": "has_lesion",
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256_file(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "threshold": THRESHOLD,
        "frozen_configuration": json.loads(final.CONFIG_PATH.read_text(encoding="utf-8")),
        "training_data": {
            "splits": ["train", "val"],
            "records": len(train_ids),
            "split_ids_sha256": split_metadata["split_ids_sha256"],
            "master_records_sha256": sha256_file(final.MASTER_PATH),
            "ground_truth_sha256": sha256_file(final.GT_PATH),
        },
        "reproducibility_check": verification,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": "scripts/nlp/learned_presence/export_frozen_model.py",
        "note": (
            "Re-fitted from the frozen configuration because the original artifact "
            "was not kept. Binary model files are gitignored; verify any copy "
            "against model_sha256 before use."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_verified_model(
    model_path: Path = MODEL_PATH, manifest_path: Path = MANIFEST_PATH
) -> Pipeline:
    """Load the frozen model only if its SHA-256 matches the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(model_path)
    if actual != manifest["model_sha256"]:
        raise ValueError(
            f"Model hash mismatch for {model_path}: expected "
            f"{manifest['model_sha256']}, found {actual}"
        )
    return joblib.load(model_path)


if __name__ == "__main__":
    result = export()
    print(json.dumps({k: result[k] for k in ("model_path", "model_sha256")}, indent=2))
    print(json.dumps(result["reproducibility_check"], indent=2))
