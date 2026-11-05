"""Deterministic preprocessing for the materialized SMC-LUD split."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised when the optional runtime is absent
    torch = None
    Tensor = object  # type: ignore[assignment,misc]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]


Split = Literal["train", "val", "test"]
ResizeMode = Literal["direct", "letterbox"]
NormalizationMode = Literal["none", "imagenet", "dataset"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")
LABELS = ("HCC", "Hemangioma")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InvalidImageError(ValueError):
    """Raised when an image cannot be opened, decoded, or converted to RGB."""


@dataclass(frozen=True)
class NormalizationStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int] = (224, 224)
    resize_mode: ResizeMode = "letterbox"
    interpolation: int = Image.Resampling.BILINEAR
    padding_value: int = 0
    normalization: NormalizationMode = "imagenet"
    normalization_stats_path: Path | None = None

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(size <= 0 for size in self.image_size):
            raise ValueError("image_size must contain two positive integers")
        if self.resize_mode not in ("direct", "letterbox"):
            raise ValueError(f"Unsupported resize mode: {self.resize_mode}")
        if self.normalization not in ("none", "imagenet", "dataset"):
            raise ValueError(f"Unsupported normalization mode: {self.normalization}")
        if not 0 <= self.padding_value <= 255:
            raise ValueError("padding_value must be in [0, 255]")
        if self.normalization == "dataset" and self.normalization_stats_path is None:
            raise ValueError("dataset normalization requires normalization_stats_path")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: Split
    label: str
    patient_group_id: str


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for tensor conversion and Dataset/DataLoader support. "
            "Install the project dependencies before using this module."
        )


def _load_stats(path: Path) -> NormalizationStats:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = tuple(float(value) for value in payload["mean"])
    std = tuple(float(value) for value in payload["std"])
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError(f"Invalid normalization statistics: {path}")
    return NormalizationStats(mean, std)


def load_image(path: Path) -> Image.Image:
    """Open an image, force decoding, and return a three-channel RGB image."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidImageError(f"Unable to read image: {path}") from exc


def resize_image(image: Image.Image, config: PreprocessingConfig) -> Image.Image:
    """Resize directly or letterbox without cropping."""
    width, height = config.image_size
    if config.resize_mode == "direct":
        return image.resize((width, height), resample=config.interpolation)

    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resample=config.interpolation,
    )
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    return ImageOps.expand(
        resized,
        border=(left, top, width - resized.width - left, height - resized.height - top),
        fill=(config.padding_value,) * 3,
    )


def image_to_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB image to a float tensor in CHW layout and [0, 1]."""
    _require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if tensor.shape[0] != 3 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Tensor must have three finite channels")
    if float(tensor.min()) < 0 or float(tensor.max()) > 1:
        raise ValueError("Unnormalized tensor values must be in [0, 1]")
    return tensor


def normalize_tensor(tensor: Tensor, mode: NormalizationMode,
                     stats: NormalizationStats | None = None) -> Tensor:
    """Apply exactly one supported normalization mode."""
    _require_torch()
    if mode == "none":
        return tensor
    selected = NormalizationStats(IMAGENET_MEAN, IMAGENET_STD) if mode == "imagenet" else stats
    if selected is None:
        raise ValueError("Dataset normalization requires computed statistics")
    mean = torch.tensor(selected.mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(selected.std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image(image: Image.Image, config: PreprocessingConfig) -> Tensor:
    """Apply RGB conversion, resizing, tensor conversion, and normalization."""
    tensor = image_to_tensor(resize_image(image.convert("RGB"), config))
    stats = _load_stats(config.normalization_stats_path) if config.normalization == "dataset" else None
    result = normalize_tensor(tensor, config.normalization, stats)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Normalized tensor contains NaN or Inf")
    return result


def _records_from_metadata(root: Path, split: Split) -> list[ImageRecord]:
    metadata_path = root / "patient_split_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing patient metadata sidecar: {metadata_path}")
    records: list[ImageRecord] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split or row["label"] not in LABELS:
                continue
            path = root / row["destination_rel_path"]
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                raise FileNotFoundError(f"Missing split image: {path}")
            records.append(ImageRecord(path, split, row["label"], row["patient_group_id"]))
    records.sort(key=lambda record: str(record.path).lower())
    return records


class SMCLUDDataset(Dataset):
    """PyTorch dataset backed by the immutable physical patient-level split."""

    def __init__(self, root: Path, split: Split, config: PreprocessingConfig) -> None:
        _require_torch()
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.records = _records_from_metadata(root, split)
        self.config = config
        self._stats = (
            _load_stats(config.normalization_stats_path)
            if config.normalization == "dataset"
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        record = self.records[index]
        tensor = image_to_tensor(resize_image(load_image(record.path), self.config))
        tensor = normalize_tensor(tensor, self.config.normalization, self._stats)
        if tuple(tensor.shape) != (3, *self.config.image_size):
            raise ValueError(f"Unexpected tensor shape for {record.path}: {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite tensor for {record.path}")
        return tensor, LABEL_TO_INDEX[record.label], record.patient_group_id


def create_dataloader(root: Path, split: Split, config: PreprocessingConfig,
                      batch_size: int = 32, num_workers: int = 0,
                      shuffle: bool = False) -> DataLoader:
    """Create a deterministic DataLoader; augmentation is intentionally absent."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    return DataLoader(
        SMCLUDDataset(root, split, config),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def compute_train_stats(root: Path, image_size: tuple[int, int] = (224, 224),
                        resize_mode: ResizeMode = "letterbox") -> NormalizationStats:
    """Compute channel statistics from train images only, before normalization."""
    config = PreprocessingConfig(image_size=image_size, resize_mode=resize_mode, normalization="none")
    records = _records_from_metadata(root, "train")
    if not records:
        raise ValueError("No training images found")
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for record in records:
        array = np.asarray(resize_image(load_image(record.path), config), dtype=np.float64) / 255.0
        pixels = array.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_squared_sum += np.square(pixels).sum(axis=0)
        pixel_count += len(pixels)
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_squared_sum / pixel_count - np.square(mean), 1e-12)
    return NormalizationStats(tuple(mean.tolist()), tuple(np.sqrt(variance).tolist()))
