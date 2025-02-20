#
# Algorithms which imitate legacy ImageJ behavior
#
from typing import Tuple

import cv2
import numpy as np

from ZooProcess_lib.img_tools import crop_if_larger


def convert_16bit_image_to_8bit_min_max(
    img: np.ndarray, min_val: int, max_val: int
) -> np.ndarray:
    """
    Vectorial port of ImageJ v1.41o algorithm in ij.process.TypeConverter.convertShortToByte
    """
    assert img.dtype == np.uint16
    max_val = min(65535, max_val)  # From ij.process.ShortProcessor.setMinAndMax
    scale = np.float64(256) / (max_val - min_val + 1)
    min_removed_img = img.astype(np.int32) - min_val
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


def images_difference(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    - ImageJ allows to do arithmetic on images of different sizes. numpy does not.
    - Difference is defined as /** dst=abs(dst-src) */ in ij.process.Blitter.java
    """
    img1_height, img1_width = image1.shape
    img2_height, img2_width = image2.shape
    # Simple behavior, the result is cropped to the smallest dimension
    res_width = min(img1_width, img2_width)
    res_height = min(img1_height, img2_height)
    # Crop (or not)
    image1 = crop_if_larger(image1, res_width, res_height)
    image2 = crop_if_larger(image2, res_width, res_height)
    return abs(image1.astype(np.int16) - image2).astype(np.uint8)


def circular_mean_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Mimic ImageJ Mean algorithm, blur with a circular kernel.
    From  ij.plugin.filter.RankFilters.doFiltering
    """
    # Below NOK for 3
    # circle = cv2.getStructuringElement(
    #     cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    # )
    # Below mimics ImageJ radius for radius 3
    assert radius == 3
    ij = 1
    circle = np.array(
        [
            [0, 0, ij, 1, ij, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, ij, 1, ij, 0, 0],
        ],
        np.uint8,
    )
    kernel = circle / circle.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def draw_line(
    image: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    color: int,
    thickness: int,
) -> None:
    """
    ImageJ macros takes floats and does a round half up to convert to int
    From  ij.macro.Functions.drawLine
    """
    assert image.dtype == np.uint8  # color is a single int
    pt1_int = (int(pt1[0] + 0.5), int(pt1[1] + 0.5))
    pt2_int = (int(pt2[0] + 0.5), int(pt2[1] + 0.5))
    cv2.line(image, pt1_int, pt2_int, (color,), thickness, lineType=cv2.LINE_4)
