"""Fine-tune the per-lesion extraction model (config_lesion.json), one seed per run.

Trains on extraction_train_lesion_v1 (Train only), keeps the epoch with the
lowest Validation loss, then predicts Validation, extraction_challenge_v2_liver,
the v1 distractor items and extraction_challenge_v3 (each on the liver-only
filter output). Challenge Sets are never used for selection. Test is not read.

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

import torch
from torch.utils.data import DataLoader

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp.extraction_model import to_parser_fields
from liversira_ai.nlp.lesion_model import LesionModel, collate, decode_item, encode_item, losses
from liversira_ai.nlp.liver_text import extract_liver_text
from liversira_ai.nlp.transformer_models import CHECKPOINTS, load_tokenizer
from scripts.nlp.transformers.extraction import build_challenge_set as cs
from scripts.nlp.transformers.extraction import evaluate_parser as ev
from scripts.nlp.transformers.extraction.train import (
    as_fields,
    challenge_items,
    read_jsonl,
    set_seed,
    target_accuracy,
    write_jsonl,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config_lesion.json"
RESULTS_ROOT = Path("results/nlp/transformers/extraction/runs_lesion")
MODELS_ROOT = Path("models/transformers/extraction_lesion")
V3 = "extraction_challenge_v3"


@torch.no_grad()
def predict(model, encoded: list[dict[str, Any]], texts: list[str], batch_size: int, device, dtype, pad_id: int,
            with_loss: bool):
    model.eval()
    decoded, details, total, count = [], [], 0.0, 0
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            if with_loss:
                out = model(batch)
                total += sum(losses(out, batch).values()).item() * len(chunk)
                count += len(chunk)
            else:
                out = model.encode(batch["input_ids"], batch["attention_mask"])
            for j, item in enumerate(chunk):
                single = {k: out[k][j] for k in ("size_tag", "count_tag", "source")}
                d, detail = decode_item(model, out["hidden"][j], single, texts[start + j], item["offsets"])
                decoded.append(d)
                details.append(detail)
    return decoded, details, (total / count if count else None)


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
    v3 = [{**r, "text": extract_liver_text(r["text"]).text} for r in read_jsonl(cs.challenge_path(V3))]
    if args.limit:
        rng = random.Random(args.seed)
        train, val, v2, distractors, v3 = (rng.sample(x, min(args.limit, len(x))) for x in (train, val, v2, distractors, v3))

    tokenizer = load_tokenizer(args.model)
    pad_id = tokenizer.pad_token_id
    encode = partial(encode_item, tokenizer, max_length=config["max_length"])
    train_enc = [encode(i["text"], i["labels"]) for i in train]
    val_enc = [encode(i["text"], i["labels"]) for i in val]
    challenge_sets = {"v2_liver": v2, "v1_distractor": distractors, "v3": v3}
    challenge_enc = {k: [encode(r["text"], None) for r in rs] for k, rs in challenge_sets.items()}

    model = LesionModel.from_checkpoint(checkpoint.repo_id, checkpoint.revision).to(device)
    from transformers import get_linear_schedule_with_warmup

    loader = DataLoader(train_enc, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=partial(collate, pad_id=pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps = math.ceil(len(loader) / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * steps), steps)

    history, best_state, best_loss, best_epoch, stale = [], None, math.inf, None, 0
    started = time.time()
    bs = config["eval_batch_size"]
    val_texts = [i["text"] for i in val]
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running: dict[str, float] = {}
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(batch)
            parts = losses(out, batch)
            (sum(parts.values()) / config["gradient_accumulation"]).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.item()
            if step % config["gradient_accumulation"] == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_decoded, _, val_loss = predict(model, val_enc, val_texts, bs, device, dtype, pad_id, with_loss=True)
        history.append({
            "epoch": epoch,
            "train_loss": {k: round(v / len(loader), 6) for k, v in running.items()},
            "val_loss": round(val_loss, 6),
            "val_positive_accuracy": target_accuracy(val, [as_fields(d) for d in val_decoded])["positive"],
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
    val_decoded, val_details, _ = predict(model, val_enc, val_texts, bs, device, dtype, pad_id, with_loss=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "val_predictions.jsonl", [
        {"id": i["id"], "targets": i["targets"], "predicted": as_fields(d), "detail": detail, "confidence": d.confidence}
        for i, d, detail in zip(val, val_decoded, val_details)
    ])
    metrics: dict[str, Any] = {"val": target_accuracy(val, [as_fields(d) for d in val_decoded])}
    for name, records in challenge_sets.items():
        decoded, details, _ = predict(model, challenge_enc[name], [r["text"] for r in records], bs, device, dtype,
                                      pad_id, with_loss=False)
        predictions = {r["challenge_id"]: to_parser_fields(d) for r, d in zip(records, decoded)}
        rows = ev.score(records, predictions)
        write_jsonl(out_dir / f"challenge_{name}_predictions.jsonl", [
            {**row, "predicted": predictions[row["challenge_id"]], "detail": detail}
            for row, detail in zip(rows, details)
        ])
        metrics[f"challenge_{name}"] = ev.summarize(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    weights_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    weight_file = weights_dir / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(weight_file))
    tokenizer.save_pretrained(weights_dir)
    manifest = {
        "schema_version": "1.0",
        "task": "extraction-per-lesion",
        "model": args.model,
        "checkpoint": {"repo_id": checkpoint.repo_id, "revision": checkpoint.revision, "lowercase": checkpoint.lowercase},
        "seed": args.seed,
        "config": config,
        "smoke_test_limit": args.limit,
        "data": {
            "train": len(train), "val": len(val), **{f"challenge_{k}": len(v) for k, v in challenge_sets.items()},
            "train_spans_dropped_by_truncation": sum(e["dropped_spans"] for e in train_enc),
            "val_spans_dropped_by_truncation": sum(e["dropped_spans"] for e in val_enc),
            "test_policy": "Test was not read.",
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "history": history,
        "weights": {"path": weight_file.as_posix(), "sha256": sha256_file(weight_file)},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p)
            for p in (train_path, val_path, cs.challenge_path(cs.LIVER_VERSION), cs.challenge_path(cs.VERSION),
                      cs.challenge_path(V3))
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "transformers": __import__("transformers").__version__,
        },
        "git_commit": git_commit(),
        "train_seconds": round(time.time() - started, 1),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": out_dir.as_posix(), "best_epoch": best_epoch, "val": metrics["val"]["positive"],
            **{k: {f: round(g["pooled_targeted_accuracy"]["estimate"] or 0, 3) for f, g in m["gate"].items()}
               for k, m in metrics.items() if k.startswith("challenge_")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
