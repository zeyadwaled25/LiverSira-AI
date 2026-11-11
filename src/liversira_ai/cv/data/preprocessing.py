"""Native-aspect preprocessing for the SMC-LUD ultrasound split.

This is the canonical CV preprocessing: EXIF orientation, RGB conversion, a
conservative trim of solid near-black edge strips, and a resize of the shorter
side that keeps the aspect ratio. Nothing is padded, cropped or stretched, so
output tensors have variable spatial size.

The earlier square letterbox pipeline is kept in ``letterbox.py``. It pads
non-square images with black, which on this dataset is almost a direct
encoding of the class label.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# Class order used by the baseline: HCC is the positive class.
CLASS_TO_INDEX = {"HCC": 0, "Hemangioma": 1}

# Conservative, deterministic policy:
# - apply EXIF orientation;
# - convert to RGB;
# - remove only complete near-black outer rows/columns;
# - resize the shorter side to 224 while preserving aspect ratio;
# - do not center-crop, stretch, or add letterbox padding.
SHORT_SIDE = 224
DARK_THRESHOLD = 3
DARK_EDGE_FRACTION = 0.999
MAX_TRIM_FRACTION_PER_EDGE = 0.02

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def trim_uniform_black_border(
    image,
    threshold=DARK_THRESHOLD,
    dark_fraction=DARK_EDGE_FRACTION,
    max_fraction=MAX_TRIM_FRACTION_PER_EDGE,
):
    """
    Remove only complete near-black strips touching the outer image frame.

    A row/column is removed only when at least 99.9% of its pixels are
    <= threshold. Fan-shaped black corners are not removed because they
    do not occupy a complete row/column. No lesion or FOV segmentation
    is attempted.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape

    row_dark_fraction = (gray <= threshold).mean(axis=1)
    col_dark_fraction = (gray <= threshold).mean(axis=0)

    max_top = int(h * max_fraction)
    max_bottom = int(h * max_fraction)
    max_left = int(w * max_fraction)
    max_right = int(w * max_fraction)

    top = 0
    while top < max_top and row_dark_fraction[top] >= dark_fraction:
        top += 1

    bottom = 0
    while bottom < max_bottom and row_dark_fraction[h - 1 - bottom] >= dark_fraction:
        bottom += 1

    left = 0
    while left < max_left and col_dark_fraction[left] >= dark_fraction:
        left += 1

    right = 0
    while right < max_right and col_dark_fraction[w - 1 - right] >= dark_fraction:
        right += 1

    # Guard against accidentally collapsing an image.
    if h - top - bottom < 32 or w - left - right < 32:
        top = bottom = left = right = 0

    cropped = image.crop((left, top, w - right, h - bottom))

    info = {
        "original_width": w,
        "original_height": h,
        "trim_top": top,
        "trim_bottom": bottom,
        "trim_left": left,
        "trim_right": right,
        "trimmed_width": cropped.width,
        "trimmed_height": cropped.height,
        "trimmed_fraction": 1.0 - (
            cropped.width * cropped.height / (w * h)
        ),
        "aspect_ratio_before": w / h,
        "aspect_ratio_after": cropped.width / cropped.height,
        "padding_pixels": 0,
    }
    return cropped, info


def resize_short_side_no_pad(image, short_side=SHORT_SIDE):
    """Resize while preserving aspect ratio; never crop or pad."""
    w, h = image.size
    scale = short_side / min(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)


def preprocess_image(image_or_path, short_side=SHORT_SIDE):
    """The single deterministic preprocessing function used in both notebooks."""
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    else:
        image = ImageOps.exif_transpose(image_or_path).convert("RGB")

    trimmed, info = trim_uniform_black_border(image)
    resized = resize_short_side_no_pad(trimmed, short_side=short_side)

    info.update({
        "output_width": resized.width,
        "output_height": resized.height,
        "output_aspect_ratio": resized.width / resized.height,
        "output_short_side": min(resized.size),
    })
    return resized, info


def image_to_normalized_tensor(image):
    """RGB PIL image -> float32 CHW tensor with ImageNet normalization."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor - mean) / std
    if not torch.isfinite(tensor).all():
        raise ValueError("Preprocessing generated NaN or Inf values.")
    return tensor


class NativeAspectUltrasoundDataset(Dataset):
    def __init__(
        self,
        frame,
        sample_weights=None,
        short_side=SHORT_SIDE,
        class_to_index=CLASS_TO_INDEX,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.sample_weights = (
            np.ones(len(self.frame), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32)
        )
        if len(self.sample_weights) != len(self.frame):
            raise ValueError("sample_weights length does not match dataset.")
        self.short_side = short_side
        self.class_to_index = class_to_index

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        record = self.frame.iloc[index]

        image, info = preprocess_image(
            record["path"],
            short_side=self.short_side
        )
        tensor = image_to_normalized_tensor(image)

        label = self.class_to_index[record["label"]]
        patient_id = str(record["patient_group_id"])
        weight = float(self.sample_weights[index])

        return (
            tensor,
            torch.tensor(label, dtype=torch.long),
            patient_id,
            torch.tensor(weight, dtype=torch.float32),
        )
