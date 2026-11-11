"""Fine-tune one transformer for lesion presence (Phase 6.1), one seed per run.

All models share ``config.json`` (same hyperparameters, epochs, early stopping
and threshold), so the comparison differs only in the pretrained checkpoint.

* Trains on Train only. The best epoch is chosen by Validation loss.
* Writes Validation and Challenge Set predictions; Test is never read.
* Model weights go to ``models/transformers/presence/<model>/seed<seed>/``
  (gitignored, safetensors); predictions, metrics and a run manifest with the
  weight hash go to ``results/nlp/transformers/presence/<model>/seed<seed>/``.

Usage (from the repository root):

    .venv\\Scripts\\python.exe experiments/transformers/presence/train.py --model bert --seed 42

Smoke test on a few examples, written outside ``results/``:

    .venv\\Scripts\\python.exe experiments/transformers/presence/train.py --model bert --seed 42 \\
        --limit 64 --epochs 1 --output-root <scratch dir>
"""

import argparse
import copy
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from liversira_ai.common.provenance import git_commit, sha256_file
from liversira_ai.nlp import transformer_data as data
from liversira_ai.nlp.transformer_models import CHECKPOINTS, load_tokenizer

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
RESULTS_ROOT = Path("results/nlp/transformers/presence")
MODELS_ROOT = Path("models/transformers/presence")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(labels: list[bool]) -> torch.Tensor:
    """sklearn-style "balanced" weights: n / (2 * count)."""
    n, positives = len(labels), sum(labels)
    return torch.tensor([n / (2 * (n - positives)), n / (2 * positives)], dtype=torch.float32)


def summarize(labels: list[bool], probabilities: list[float], threshold: float) -> dict[str, Any]:
    y = np.array(labels, dtype=int)
    p = np.array(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 6) if b else None

    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(
        abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()
    )
    return {
        "n": len(y),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": ratio(tp + tn, len(y)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "ece_10_bins": round(float(ece), 6),
    }


def challenge_summary(records: list[dict[str, Any]], probabilities: list[float], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for category in sorted({r["category"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["category"] == category]
        positive_rate = float(np.mean([probabilities[i] >= threshold for i in idx]))
        if records[idx[0]]["scored"]:
            labels = [bool(records[i]["label"]) for i in idx]
            correct = [(probabilities[i] >= threshold) == labels[k] for k, i in enumerate(idx)]
            out[category] = {"n": len(idx), "accuracy": round(float(np.mean(correct)), 6),
                             "positive_rate": round(positive_rate, 6)}
        else:
            out[category] = {"n": len(idx), "accuracy": None, "positive_rate": round(positive_rate, 6)}
    return out


def make_loader(tokenizer, texts: list[str], labels: list[int] | None, batch_size: int, max_length: int,
                shuffle: bool, seed: int) -> DataLoader:
    encodings = tokenizer(texts, truncation=True, max_length=max_length)
    items = [
        {"input_ids": encodings["input_ids"][i], "attention_mask": encodings["attention_mask"][i],
         **({"labels": labels[i]} if labels is not None else {})}
        for i in range(len(texts))
    ]

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        out = {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (width - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] + [0] * (width - len(b["attention_mask"])) for b in batch]),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, dtype: torch.dtype,
            loss_fn: torch.nn.Module | None = None) -> tuple[list[float], float | None]:
    model.eval()
    probabilities: list[float] = []
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(**batch).logits.float()
        probabilities += torch.softmax(logits, dim=-1)[:, 1].tolist()
        if loss_fn is not None and labels is not None:
            total += loss_fn(logits, labels).item() * len(labels)
            count += len(labels)
    return probabilities, (total / count if count else None)


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.epochs:
        config["epochs"] = args.epochs
    checkpoint = CHECKPOINTS[args.model]
    results_root = Path(args.output_root) / "results" if args.output_root else RESULTS_ROOT
    models_root = Path(args.output_root) / "models" if args.output_root else MODELS_ROOT
    if args.limit and not args.output_root:
        raise SystemExit("--limit is for smoke tests; pass --output-root so results/ is not touched")
    out_dir = results_root / args.model / f"seed{args.seed}"
    weights_dir = models_root / args.model / f"seed{args.seed}"
    if (out_dir / "run_manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already has a finished run; pass --overwrite to replace it")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16

    ids = data.load_split_ids([config["train_split"], config["selection_split"]])
    train = data.load_examples(ids[config["train_split"]])
    val = data.load_examples(ids[config["selection_split"]])
    challenge = data.load_challenge()
    if args.limit:
        rng = random.Random(args.seed)
        train, val, challenge = (rng.sample(x, min(args.limit, len(x))) for x in (train, val, challenge))

    tokenizer = load_tokenizer(args.model)
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint.repo_id, revision=checkpoint.revision, num_labels=2
    ).to(device)

    train_loader = make_loader(tokenizer, [e["text"] for e in train], [int(e["label"]) for e in train],
                               config["batch_size"], config["max_length"], True, args.seed)
    val_loader = make_loader(tokenizer, [e["text"] for e in val], [int(e["label"]) for e in val],
                             config["eval_batch_size"], config["max_length"], False, args.seed)
    challenge_loader = make_loader(tokenizer, [r["text"] for r in challenge], None,
                                   config["eval_batch_size"], config["max_length"], False, args.seed)

    weights = class_weights([e["label"] for e in train]).to(device) if config["class_weight"] == "balanced" else None
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    eval_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    steps = math.ceil(len(train_loader) / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_ratio"] * steps), steps)

    history, best_state, best_loss, best_epoch, stale = [], None, math.inf, None, 0
    started = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running, batches = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            loss = loss_fn(logits, labels) / config["gradient_accumulation"]
            loss.backward()
            running += loss.item() * config["gradient_accumulation"]
            batches += 1
            if step % config["gradient_accumulation"] == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_probs, val_loss = predict(model, val_loader, device, dtype, eval_loss_fn)
        val_metrics = summarize([e["label"] for e in val], val_probs, config["threshold"])
        history.append({"epoch": epoch, "train_loss": round(running / batches, 6), "val_loss": round(val_loss, 6),
                        "val_f1": val_metrics["f1"], "val_accuracy": val_metrics["accuracy"],
                        "elapsed_s": round(time.time() - started, 1)})
        print(json.dumps(history[-1]))
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale > config["early_stopping"]["patience"]:
                break

    model.load_state_dict(best_state)
    val_probs, val_loss = predict(model, val_loader, device, dtype, eval_loss_fn)
    challenge_probs, _ = predict(model, challenge_loader, device, dtype)
    threshold = config["threshold"]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "val_predictions.jsonl", [
        {"id": e["id"], "label": e["label"], "probability": round(p, 8), "prediction": p >= threshold}
        for e, p in zip(val, val_probs)
    ])
    write_predictions(out_dir / "challenge_predictions.jsonl", [
        {"challenge_id": r["challenge_id"], "category": r["category"], "label": r["label"],
         "probability": round(p, 8), "prediction": p >= threshold}
        for r, p in zip(challenge, challenge_probs)
    ])
    metrics = {
        "val": summarize([e["label"] for e in val], val_probs, threshold),
        "challenge": challenge_summary(challenge, challenge_probs, threshold),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(weights_dir, safe_serialization=True)
    tokenizer.save_pretrained(weights_dir)
    weight_file = weights_dir / "model.safetensors"

    manifest = {
        "schema_version": "1.0",
        "task": "presence",
        "model": args.model,
        "checkpoint": {"repo_id": checkpoint.repo_id, "revision": checkpoint.revision, "lowercase": checkpoint.lowercase},
        "seed": args.seed,
        "config": config,
        "smoke_test_limit": args.limit,
        "data": {
            "train": len(train), "val": len(val), "challenge": len(challenge),
            "train_positive": sum(e["label"] for e in train),
            "test_policy": "Test was not read.",
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "history": history,
        "weights": {"path": weight_file.as_posix(), "sha256": sha256_file(weight_file)},
        "inputs_sha256": {
            p.as_posix(): sha256_file(p)
            for p in (data.MASTER_PATH, data.GROUND_TRUTH_PATH, data.SPLITS_PATH, data.CHALLENGE_PATH)
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
    return {"out_dir": out_dir.as_posix(), "best_epoch": best_epoch, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(CHECKPOINTS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, help="override config epochs (smoke tests)")
    parser.add_argument("--limit", type=int, help="subsample each split (smoke tests only)")
    parser.add_argument("--output-root", help="write results/ and models/ under this directory instead")
    parser.add_argument("--overwrite", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
