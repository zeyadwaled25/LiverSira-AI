"""Fine-tune the Phase 6.2 multi-head extraction model, one seed per run.

Trains on extraction_train_v1 (Train only), keeps the epoch with the lowest
Validation loss, then predicts Validation, extraction_challenge_v2_liver and the
distractor items of extraction_challenge_v1 (liver-only filter applied). The
Challenge Sets are only predicted, never used for selection. Test is not read.

Smoke test: --limit 64 --epochs 1 --output-root <scratch dir>.
"""

import argparse
import copy
import json
import math
import platform
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp.extraction_model import (
    ExtractionModel,
    collate,
    decode,
    encode_item,
    losses,
    to_parser_fields,
)
from liversira_ai.nlp.liver_text import extract_liver_text
from liversira_ai.nlp.transformer_models import CHECKPOINTS, load_tokenizer
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import evaluate_parser as ev

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
RESULTS_ROOT = Path("results/nlp/transformers/extraction/runs")
MODELS_ROOT = Path("models/transformers/extraction")
TARGET_FIELDS = ("count", "segments", "attenuation", "diameter_cm")
DIAMETER_TOLERANCE_CM = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def field_correct(field: str, predicted: Any, target: Any) -> bool:
    """Agreement with an extraction target; ``None`` (unstated) must match ``None``."""
    if target is None or predicted is None:
        return target is None and predicted is None
    if field == "diameter_cm":
        return abs(float(predicted) - float(target)) <= DIAMETER_TOLERANCE_CM
    if field == "segments":
        return sorted(predicted) == sorted(target)
    return predicted == target


def target_accuracy(items: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy per field on all items and on lesion-positive items (target count is not "0")."""
    out: dict[str, Any] = {}
    for subset, keep in (("all", lambda t: True), ("positive", lambda t: t["count"] != "0")):
        idx = [i for i, item in enumerate(items) if keep(item["targets"])]
        out[subset] = {"n": len(idx)}
        for field in TARGET_FIELDS:
            correct = [field_correct(field, predictions[i][field], items[i]["targets"][field]) for i in idx]
            out[subset][field] = round(float(np.mean(correct)), 6) if idx else None
    return out


def challenge_items() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """extraction_challenge_v2_liver, and v1 distractor items with the liver-only filter applied."""
    v2 = read_jsonl(cs.challenge_path(cs.LIVER_VERSION))
    distractors = [
        {**r, "text": extract_liver_text(r["text"]).text}
        for r in read_jsonl(cs.challenge_path(cs.VERSION))
        if r["category"] == "distractor"
    ]
    return v2, distractors


def make_loader(encoded: list[dict[str, Any]], batch_size: int, shuffle: bool, seed: int, pad_id: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=partial(collate, pad_id=pad_id))


@torch.no_grad()
def predict(model, encoded: list[dict[str, Any]], texts: list[str], batch_size: int, device, dtype, pad_id: int):
    """Decoded fields per item and the mean summed loss."""
    model.eval()
    decoded, total, count = [], 0.0, 0
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            outputs = model(batch["input_ids"], batch["attention_mask"])
        total += sum(losses(outputs, batch).values()).item() * len(chunk)
        count += len(chunk)
        for j, item in enumerate(chunk):
            length = len(item["input_ids"])
            single = {
                "count": outputs["count"][j].float().cpu(),
                "segments": outputs["segments"][j].float().cpu(),
                "attenuation": outputs["attenuation"][j].float().cpu(),
                "diameter": outputs["diameter"][j, :length].float().cpu(),
            }
            decoded.append(decode(texts[start + j], item["offsets"], single))
    return decoded, total / max(count, 1)


def as_fields(d) -> dict[str, Any]:
    return {"count": d.count, "segments": d.segments, "attenuation": d.attenuation, "diameter_cm": d.diameter_cm}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.epochs:
        config["epochs"] = args.epochs
    checkpoint = CHECKPOINTS[args.model]
    if args.limit and not args.output_root:
        raise SystemExit("--limit is for smoke tests; pass --output-root so results/ is not touched")
    results_root = Path(args.output_root) / "results" if args.output_root else RESULTS_ROOT
    models_root = Path(args.output_root) / "models" if args.output_root else MODELS_ROOT
    out_dir = results_root / args.model / f"seed{args.seed}"
    weights_dir = models_root / args.model / f"seed{args.seed}"
    if (out_dir / "run_manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has a finished run; pass --overwrite to replace it")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16

    train_path, val_path = Path(config["train_data"]), Path(config["selection_data"])
    train, val = read_jsonl(train_path), read_jsonl(val_path)
    v2, distractors = challenge_items()
    if args.limit:
        rng = random.Random(args.seed)
        train, val, v2, distractors = (rng.sample(x, min(args.limit, len(x))) for x in (train, val, v2, distractors))

    tokenizer = load_tokenizer(args.model)
    pad_id = tokenizer.pad_token_id
    max_length = config["max_length"]
    encode = partial(encode_item, tokenizer, max_length=max_length)
    empty = {"count": None, "segments": None, "attenuation": None, "diameter_cm": None, "diameter_spans": []}
    train_enc = [encode(i["text"], i["targets"]) for i in train]
    val_enc = [encode(i["text"], i["targets"]) for i in val]
    v2_enc = [encode(r["text"], empty) for r in v2]
    distractor_enc = [encode(r["text"], empty) for r in distractors]

    model = ExtractionModel.from_checkpoint(checkpoint.repo_id, checkpoint.revision).to(device)
    from transformers import get_linear_schedule_with_warmup

    loader = make_loader(train_enc, config["batch_size"], True, args.seed, pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps = math.ceil(len(loader) / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * steps), steps)

    history, best_state, best_loss, best_epoch, stale = [], None, math.inf, None, 0
    started = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = {k: 0.0 for k in ("count", "segments", "attenuation", "diameter")}
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
            parts = losses(outputs, batch)
            loss = sum(parts.values()) / config["gradient_accumulation"]
            loss.backward()
            for k, v in parts.items():
                running[k] += v.item()
            if step % config["gradient_accumulation"] == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_decoded, val_loss = predict(model, val_enc, [i["text"] for i in val], config["eval_batch_size"],
                                        device, dtype, pad_id)
        accuracy = target_accuracy(val, [as_fields(d) for d in val_decoded])
        history.append({
            "epoch": epoch,
            "train_loss": {k: round(v / len(loader), 6) for k, v in running.items()},
            "val_loss": round(val_loss, 6),
            "val_positive_accuracy": accuracy["positive"],
            "elapsed_s": round(time.time() - started, 1),
        })
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale > config["early_stopping"]["patience"]:
                break

    model.load_state_dict(best_state)
    bs = config["eval_batch_size"]
    val_decoded, val_loss = predict(model, val_enc, [i["text"] for i in val], bs, device, dtype, pad_id)
    v2_decoded, _ = predict(model, v2_enc, [r["text"] for r in v2], bs, device, dtype, pad_id)
    distractor_decoded, _ = predict(model, distractor_enc, [r["text"] for r in distractors], bs, device, dtype, pad_id)

    v2_predictions = {r["challenge_id"]: to_parser_fields(d) for r, d in zip(v2, v2_decoded)}
    distractor_predictions = {r["challenge_id"]: to_parser_fields(d) for r, d in zip(distractors, distractor_decoded)}
    v2_rows = ev.score(v2, v2_predictions)
    distractor_rows = ev.score(distractors, distractor_predictions)
    metrics = {
        "val": target_accuracy(val, [as_fields(d) for d in val_decoded]),
        "challenge_v2_liver": ev.summarize(v2_rows),
        "challenge_v1_distractor_liver_only": ev.summarize(distractor_rows),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "val_predictions.jsonl", [
        {"id": i["id"], "targets": {k: i["targets"][k] for k in TARGET_FIELDS}, "predicted": as_fields(d),
         "confidence": d.confidence}
        for i, d in zip(val, val_decoded)
    ])
    write_jsonl(out_dir / "challenge_v2_liver_predictions.jsonl", [
        {**row, "predicted": v2_predictions[row["challenge_id"]], "confidence": d.confidence}
        for row, d in zip(v2_rows, v2_decoded)
    ])
    write_jsonl(out_dir / "challenge_v1_distractor_predictions.jsonl", [
        {**row, "predicted": distractor_predictions[row["challenge_id"]]} for row in distractor_rows
    ])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    weights_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    weight_file = weights_dir / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(weight_file))
    tokenizer.save_pretrained(weights_dir)

    manifest = {
        "schema_version": "1.0",
        "task": "extraction",
        "model": args.model,
        "checkpoint": {"repo_id": checkpoint.repo_id, "revision": checkpoint.revision, "lowercase": checkpoint.lowercase},
        "seed": args.seed,
        "config": config,
        "smoke_test_limit": args.limit,
        "data": {
            "train": len(train), "val": len(val), "challenge_v2_liver": len(v2), "challenge_v1_distractor": len(distractors),
            "train_truncated_spans": sum(e["truncated_span"] for e in train_enc),
            "val_truncated_spans": sum(e["truncated_span"] for e in val_enc),
            "test_policy": "Test was not read.",
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "history": history,
        "weights": {"path": weight_file.as_posix(), "sha256": sha256_file(weight_file)},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p)
            for p in (train_path, val_path, cs.challenge_path(cs.LIVER_VERSION), cs.challenge_path(cs.VERSION))
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "transformers": __import__("transformers").__version__,
            "deterministic_note": "cuDNN deterministic mode is on; GPU kernels can still differ slightly between runs.",
        },
        "git_commit": git_commit(),
        "train_seconds": round(time.time() - started, 1),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": out_dir.as_posix(), "best_epoch": best_epoch, "val": metrics["val"],
            "gate_v2": {f: g["pooled_targeted_accuracy"]["estimate"] for f, g in metrics["challenge_v2_liver"]["gate"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--limit", type=int, help="subsample each set (smoke tests only)")
    parser.add_argument("--output-root", help="write results/ and models/ under this directory instead")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
