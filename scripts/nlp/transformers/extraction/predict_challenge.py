"""Score saved 6.2 extraction runs on another Challenge Set (liver-only filter applied).

Loads each seed's frozen weights from models/transformers/extraction/ and writes
``challenge_<name>_predictions.jsonl`` and ``challenge_<name>_metrics.json`` next
to that run's results. Weights are not changed. Test is not read.
"""

import argparse
import json
from pathlib import Path

import torch

from liversira_ai.common.provenance import sha256_file
from liversira_ai.nlp.extraction_model import ExtractionModel, encode_item, to_parser_fields
from liversira_ai.nlp.liver_text import extract_liver_text
from liversira_ai.nlp.transformer_models import CHECKPOINTS, load_tokenizer
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.train import CONFIG_PATH, MODELS_ROOT, RESULTS_ROOT, predict, read_jsonl


def score_run(model_key: str, seed: int, version: str, records: list[dict], config: dict,
              results_root: Path = RESULTS_ROOT, models_root: Path = MODELS_ROOT) -> dict:
    from safetensors.torch import load_file

    run_dir = results_root / model_key / f"seed{seed}"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    weight_file = models_root / model_key / f"seed{seed}" / "model.safetensors"
    if sha256_file(weight_file) != manifest["weights"]["sha256"]:
        raise SystemExit(f"{weight_file} does not match its run manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    checkpoint = CHECKPOINTS[model_key]
    model = ExtractionModel.from_checkpoint(checkpoint.repo_id, checkpoint.revision)
    model.load_state_dict(load_file(str(weight_file)))
    model.to(device)
    tokenizer = load_tokenizer(model_key)

    empty = {"count": None, "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}
    texts = [extract_liver_text(r["text"]).text for r in records]
    encoded = [encode_item(tokenizer, t, empty, config["max_length"]) for t in texts]
    decoded, _ = predict(model, encoded, texts, config["eval_batch_size"], device, dtype, tokenizer.pad_token_id)
    predictions = {r["challenge_id"]: to_parser_fields(d) for r, d in zip(records, decoded)}
    rows = ev.score(records, predictions)
    name = version.removeprefix("extraction_challenge_")
    with (run_dir / f"challenge_{name}_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row, d in zip(rows, decoded):
            handle.write(json.dumps({**row, "predicted": predictions[row["challenge_id"]], "confidence": d.confidence}) + "\n")
    summary = ev.summarize(rows)
    (run_dir / f"challenge_{name}_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", default="extraction_challenge_v3")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((cs.OUT_DIR / f"{args.challenge}_manifest.json").read_text(encoding="utf-8"))
    path = cs.challenge_path(args.challenge)
    if sha256_file(path) != manifest["output_sha256"]:
        raise SystemExit("Challenge Set file does not match its manifest hash")
    records = read_jsonl(path)
    for model_key in config["models"]:
        for seed in config["seeds"]:
            summary = score_run(model_key, seed, args.challenge, records, config)
            pooled = {f: round(g["pooled_targeted_accuracy"]["estimate"], 3) for f, g in summary["gate"].items()}
            print(model_key, seed, pooled, flush=True)


if __name__ == "__main__":
    main()
