from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from liversira_ai.cv.data.preprocessing import (
    InvalidImageError,
    PreprocessingConfig,
    image_to_tensor,
    load_image,
    normalize_tensor,
    resize_image,
)


def test_direct_resize_and_letterbox_preserve_requested_shape() -> None:
    image = Image.new("RGB", (200, 100), "white")
    direct = resize_image(image, PreprocessingConfig((224, 224), "direct"))
    letterbox = resize_image(image, PreprocessingConfig((224, 224), "letterbox"))
    assert direct.size == (224, 224)
    assert letterbox.size == (224, 224)
    assert letterbox.getpixel((0, 0)) == (0, 0, 0)
    assert letterbox.getpixel((112, 112)) == (255, 255, 255)


def test_invalid_image_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jpg"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(InvalidImageError):
        load_image(path)


def test_rgb_tensor_shape_and_range() -> None:
    torch = pytest.importorskip("torch")
    tensor = image_to_tensor(Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)))
    assert tuple(tensor.shape) == (3, 12, 16)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) == 0
    assert float(tensor.max()) == 0


def test_normalization_is_deterministic() -> None:
    pytest.importorskip("torch")
    tensor = image_to_tensor(Image.new("RGB", (2, 2), (128, 64, 32)))
    first = normalize_tensor(tensor, "imagenet")
    second = normalize_tensor(tensor, "imagenet")
    assert first.equal(second)
