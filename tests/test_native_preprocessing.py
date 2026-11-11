"""Tests for the native-aspect CV preprocessing, on synthetic images only."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from liversira_ai.cv.data.letterbox import legacy_black_letterbox
from liversira_ai.cv.data.preprocessing import (
    CLASS_TO_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SHORT_SIDE,
    NativeAspectUltrasoundDataset,
    image_to_normalized_tensor,
    preprocess_image,
    resize_short_side_no_pad,
    trim_uniform_black_border,
)


def _gray_image(width: int, height: int, value: int = 128) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8))


def test_class_order_is_hcc_zero_hemangioma_one() -> None:
    assert CLASS_TO_INDEX == {"HCC": 0, "Hemangioma": 1}
    assert SHORT_SIDE == 224


def test_resize_keeps_aspect_ratio_without_padding() -> None:
    resized = resize_short_side_no_pad(_gray_image(600, 400), short_side=224)
    assert resized.size == (336, 224)
    assert min(np.asarray(resized).ravel()) == 128  # no black padding was added


def test_trim_removes_only_solid_black_edge_strips() -> None:
    pixels = np.full((400, 600, 3), 128, dtype=np.uint8)
    pixels[:4, :, :] = 0  # solid black strip at the top (1% of the height)
    pixels[-40:, -40:, :] = 0  # black corner: never a complete row or column
    trimmed, info = trim_uniform_black_border(Image.fromarray(pixels))
    assert (info["trim_top"], info["trim_bottom"], info["trim_left"], info["trim_right"]) == (4, 0, 0, 0)
    assert trimmed.size == (600, 396)
    assert info["padding_pixels"] == 0


def test_trim_is_capped_per_edge() -> None:
    pixels = np.full((400, 600, 3), 128, dtype=np.uint8)
    pixels[:100, :, :] = 0  # far more than the 2% cap
    _, info = trim_uniform_black_border(Image.fromarray(pixels))
    assert info["trim_top"] == 8


def test_preprocess_image_accepts_paths_and_reports_output_size(tmp_path) -> None:
    path = tmp_path / "frame.png"
    _gray_image(300, 500).save(path)
    image, info = preprocess_image(path)
    assert image.size == (224, 373)
    assert info["output_short_side"] == 224
    assert info["output_aspect_ratio"] == pytest.approx(224 / 373)


def test_normalized_tensor_layout_and_values() -> None:
    tensor = image_to_normalized_tensor(Image.new("RGB", (5, 3), (255, 0, 0)))
    assert tuple(tensor.shape) == (3, 3, 5)
    assert tensor.dtype == torch.float32
    expected = [(1 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], -IMAGENET_MEAN[1] / IMAGENET_STD[1], -IMAGENET_MEAN[2] / IMAGENET_STD[2]]
    assert tensor[:, 0, 0].tolist() == pytest.approx(expected)


def test_dataset_returns_variable_size_tensor_label_patient_and_weight(tmp_path) -> None:
    rows = []
    for name, label, size in (("a.png", "HCC", (512, 512)), ("b.png", "Hemangioma", (684, 519))):
        _gray_image(*size).save(tmp_path / name)
        rows.append({"path": tmp_path / name, "label": label, "patient_group_id": f"{label}_1"})
    dataset = NativeAspectUltrasoundDataset(pd.DataFrame(rows), sample_weights=[0.5, 1.5])

    tensor, label, patient, weight = dataset[0]
    assert tuple(tensor.shape) == (3, 224, 224)
    assert (int(label), patient, float(weight)) == (0, "HCC_1", 0.5)

    tensor, label, patient, weight = dataset[1]
    assert tuple(tensor.shape) == (3, 224, 295)
    assert (int(label), patient, float(weight)) == (1, "Hemangioma_1", 1.5)


def test_dataset_honours_explicit_short_side_and_mapping(tmp_path) -> None:
    _gray_image(200, 100).save(tmp_path / "a.png")
    frame = pd.DataFrame([{"path": tmp_path / "a.png", "label": "HCC", "patient_group_id": "p"}])
    dataset = NativeAspectUltrasoundDataset(frame, short_side=64, class_to_index={"HCC": 1, "Hemangioma": 0})
    tensor, label, _, weight = dataset[0]
    assert tuple(tensor.shape) == (3, 64, 128)
    assert int(label) == 1 and float(weight) == 1.0


def test_dataset_rejects_mismatched_weights(tmp_path) -> None:
    frame = pd.DataFrame([{"path": tmp_path / "a.png", "label": "HCC", "patient_group_id": "p"}])
    with pytest.raises(ValueError):
        NativeAspectUltrasoundDataset(frame, sample_weights=[1.0, 1.0])


def test_legacy_letterbox_pads_non_square_images_only() -> None:
    square, square_padding = legacy_black_letterbox(_gray_image(512, 512))
    wide, wide_padding = legacy_black_letterbox(_gray_image(684, 519))
    assert square.size == wide.size == (224, 224)
    assert square_padding == 0.0
    assert wide_padding == pytest.approx(1 - (224 * 170) / (224 * 224))
