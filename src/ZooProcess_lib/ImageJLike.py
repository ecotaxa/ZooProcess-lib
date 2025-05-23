#
# Algorithms which imitate legacy ImageJ behavior
#
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np

from .img_tools import cropped_if_larger


def convert_16bit_image_to_8bit_min_max(
    img: np.ndarray, min_val: int, max_val: int
) -> np.ndarray:
    """
    Vectorial port of ImageJ v1.41o algorithm in ij.process.TypeConverter.convertShortToByte
    """
    assert img.dtype == np.uint16
    max_val = min(65535, max_val)  # From ij.process.ShortProcessor.setMinAndMax
    # Full lookup table on 16bit pixel values
    all_values = np.arange(0, np.iinfo(img.dtype).max + 1, dtype=np.uint16)
    np.clip(a=all_values, a_min=min_val, a_max=max_val, out=all_values)
    val_range = max_val - min_val + 1
    scaled_values = ((all_values - min_val) * np.float32(256)) / val_range + 0.5
    np.clip(a=scaled_values, a_min=None, a_max=255, out=scaled_values)
    scaled_values = scaled_values.astype(np.uint8)
    ret = scaled_values[img]
    return ret


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
    image1 = cropped_if_larger(image1, res_width, res_height)
    image2 = cropped_if_larger(image2, res_width, res_height)
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


def bilinear_resize(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """ """
    # NOK, probably due to some rounding, maybe https://www.crisluengo.net/archives/1140/
    # ret = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # NOK either, no INTER_LINEAR_EXACT in warpAffine
    # ret = _bilinear_resize_using_warpaffine(image, new_height, new_width)
    # Still NOK, but best opencv solution, with a tolerance of 3 the results are similar to ImageJ
    # ret = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR_EXACT)
    # Looks fixed (?) in Opencv 5.x: https://github.com/opencv/opencv/issues/25937, TODO: Try again when released
    # Sill NOK, tolerance 3 -> OK-ish
    # ret = _bilinear_using_pil(image, new_height, new_width)
    # NOK
    # ret = _bilinear_using_skimage(image, new_height, new_width)
    ret = ImageJLikeResizer(image).resize(new_width, new_height)
    return ret


def _bilinear_using_skimage(image, new_height, new_width):  # pragma: no cover
    """Unused test implementation"""
    from skimage.transform import resize_local_mean

    ret = resize_local_mean(image, (new_height, new_width), grid_mode=False) * 256
    return ret


def _bilinear_using_pil(image, new_height, new_width):  # pragma: no cover
    """Unused test implementation"""
    from PIL import Image
    from PIL.Image import Resampling

    pil_image = Image.fromarray(image)
    rsz = pil_image.resize((new_width, new_height), resample=Resampling.BILINEAR)
    ret = np.array(rsz)
    return ret


def _bilinear_resize_using_opencv_warpaffine(
    image: np.ndarray, new_width: int, new_height: int
):  # pragma: no cover
    """Unused test implementation"""
    height, width = image.shape[:2]
    scale_x = new_height / height
    scale_y = new_width / width
    mat = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
    ret = cv2.warpAffine(
        image,
        mat,
        (int(width * scale_x), int(height * scale_y)),
        flags=cv2.INTER_LINEAR,
    )
    return ret


class ImageJLikeResizer(object):
    def __init__(self, image: np.ndarray) -> None:
        """
        :param image: interpolation source data
        """
        self.image = image
        self.x_bases = None  # x coordinates inside destination image
        self.x_fractions = None  # fractions for linear interpolation on x dimension

    def resize(self, dst_width: int, dst_height: int) -> np.ndarray:
        """Resizes self.image to the given dimensions. Vectorial port of ij.process.ByteProcessor.resize
            :param dst_width: The destination width.
            :param dst_height: The destination height.
        Returns:
            Resized image.
        in a loop on source ones.
        """
        assert self.image.dtype == np.uint8

        height, width = self.image.shape[:2]
        src_center_x = width / 2.0
        src_center_y = height / 2.0
        dst_center_x = dst_width / 2.0
        dst_center_y = dst_height / 2.0
        x_scale = float(dst_width) / width
        y_scale = float(dst_height) / height
        dst_center_x += x_scale / 2.0
        dst_center_y += y_scale / 2.0

        x_limit = width - 1.0
        x_limit2 = width - 1.001
        # Pre-compute x fractions which are the same for all lines
        dst_xs = []
        out = np.zeros((dst_height, dst_width), dtype=np.uint8)
        for x in range(dst_width):
            xs = (x - dst_center_x) / x_scale + src_center_x
            if xs < 0:
                xs = 0.0
            if xs >= x_limit:
                xs = x_limit2
            dst_xs.append(xs)
        x_dsts = np.array(dst_xs)
        self.x_bases = x_dsts.astype(np.uint32)
        self.x_fractions = x_dsts - self.x_bases

        y_limit = height - 1.0
        y_limit2 = height - 1.001

        ret_rows = []
        for y in range(dst_height):
            ys = (y - dst_center_y) / y_scale + src_center_y
            if ys < 0:
                ys = 0.0
            if ys >= y_limit:
                ys = y_limit2
            # if False:
            #     # Iterative code for reference
            #     row = []
            #     row_append = row.append
            #     for x in range(dst_width):
            #         xs = (x - dst_center_x) / x_scale + src_center_x
            #         if xs < 0:
            #             xs = 0.0
            #         if xs >= x_limit:
            #             xs = x_limit2
            #         row_append(ij_get_interpolated_pixel(xs, ys, image) + 0.5)
            #     dst_image[y] = row
            out[y] = self._get_interpolated_row(ys)

        return out

    def _get_interpolated_row(self, y: float) -> np.ndarray:
        """
        Interpolates all pixel values in given row using bilinear interpolation, the ImageJ way (with bugs).
            :param y: y coordinate inside destination image.
        """
        y_base = int(y)
        y_fraction = y - y_base

        lower_left, lower_right = self._extract_and_expand_rows(y_base)
        upper_left, upper_right = self._extract_and_expand_rows(y_base + 1)

        upper_diff = upper_right - upper_left
        upper_average = upper_left + self.x_fractions * upper_diff
        lower_diff = lower_right - lower_left
        lower_average = lower_left + self.x_fractions * lower_diff
        ret = lower_average + y_fraction * (upper_average - lower_average) + 0.5
        return ret.astype(self.image.dtype)

    @lru_cache(maxsize=2)  # y is slowly growing, keep last lines in cache
    def _extract_and_expand_rows(self, y: int) -> Tuple[np.ndarray, np.ndarray]:
        src_row = self.image[y]
        left_row = src_row.take(self.x_bases).astype(np.int16)
        right_row = src_row[1:].take(self.x_bases).astype(np.int16)
        return left_row, right_row

    def _ij_get_interpolated_pixel(
        self, x: float, y: float
    ) -> np.float64:  # pragma: no cover
        """
        Interpolates pixel value at given coordinates using bilinear interpolation.
            :param x: x-coordinate (float).
            :param y: y-coordinate (float).
        Unused reference implementation
        """
        xbase = int(x)
        ybase = int(y)
        xfraction = x - xbase
        yfraction = y - ybase
        # offset = ybase * width + xbase
        lower_left = self.image[ybase, xbase]
        lower_right = self.image[ybase, xbase + 1]  # Is sometimes next line
        upper_right = self.image[ybase + 1, xbase + 1]  # Is sometimes next line
        upper_left = self.image[ybase + 1, xbase]
        upper_average = upper_left + xfraction * (upper_right - upper_left)
        lower_average = lower_left + xfraction * (lower_right - lower_left)
        return lower_average + yfraction * (upper_average - lower_average)


def parseInt(num: float) -> int:
    return int(num + 0.5)
