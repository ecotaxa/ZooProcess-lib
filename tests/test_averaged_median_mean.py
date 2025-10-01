import numpy as np

from ZooProcess_lib.Legacy import averaged_median_mean


def _constant_image(height: int, width: int, value: int = 12345) -> np.ndarray:
    # Use uint16 like typical Zooscan grayscale images
    img = np.full((height, width), fill_value=value, dtype=np.uint16)
    return img


def test_averaged_median_mean_height_3088():
    # Width arbitrary but large enough; algorithm uses percentage-based cropping
    img = _constant_image(3088, 2048, value=22222)
    median, mean = averaged_median_mean(img)
    assert median == 22222
    assert mean == 22222


def test_averaged_median_mean_height_1896():
    img = _constant_image(1896, 1536, value=321)
    median, mean = averaged_median_mean(img)
    assert median == 321
    assert mean == 321
