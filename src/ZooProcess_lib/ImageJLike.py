#
# Algorithms which imitate legacy ImageJ behavior
#
import numpy as np


def convert_16bit_image_to_8bit_min_max(
    img: np.ndarray, min_val: int, max_val: int
) -> np.ndarray:
    """
    Vectorial port of ImageJ v1.41o algorithm in ij.process.TypeConverter.convertShortToByte
    """
    assert img.dtype == np.uint16
    scale = 256.0 / (max_val - min_val + 1)
    min_removed_img = img - min_val
    min_removed_img[min_removed_img < 0] = 0
    scaled_img = min_removed_img * scale + 0.5
    scaled_img[scaled_img > 255] = 255
    return scaled_img.astype(np.uint8)


def divide_round_up(img: np.ndarray, divider: int) -> np.ndarray:
    """
    Vectorial port of ImageJ v1.41o algorithm in ij.process.ShortProcessor.process for MUL case
    """
    assert img.dtype == np.uint8
    return np.floor(img / divider + 0.5).astype(np.uint8)
