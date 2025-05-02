from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .Border import Border
from .ImageJLike import circular_mean_blur, bilinear_resize, images_difference, draw_line
from .img_tools import (
    crophw,
    crop_right,
    clear_outside,
    load_tiff_image_and_info,
)


class BackgroundRemover:
    """
    Process removal of background from scanned sample.
    Port of algorithms from legacy Zooscan_1asep.txt macro
    """

    def __init__(self, background_process:str):
        self.background_process = background_process

    def do_from_files(self, background_file: Path, sample_file: Path) -> np.ndarray:
        """:background_file: file containing the background image, comes from CombinedBackgrounds"""
        bg_info, background_image = load_tiff_image_and_info(background_file)
        assert background_image.dtype == np.uint8
        bg_resolution = bg_info.resolution
        sample_info, sample_image = load_tiff_image_and_info(sample_file)
        assert sample_image.dtype == np.uint8
        sample_resolution = sample_info.resolution
        sample_minus_bg = self.do_from_images(background_image, bg_resolution, sample_image, sample_resolution)
        return sample_minus_bg

    def do_from_images(self, background_image, bg_resolution, sample_image, sample_resolution):
        sample_minus_bg = self._remove_bg_from_sample(
            sample_image=sample_image,
            sample_image_resolution=sample_resolution,
            bg_image=background_image,
            bg_resolution=bg_resolution,
        )
        return sample_minus_bg

    def _remove_bg_from_sample(
        self,
        sample_image: np.ndarray,
        sample_image_resolution: int,
        bg_image: np.ndarray,
        bg_resolution: int,
    ) -> np.ndarray:
        border = Border(sample_image, sample_image_resolution, self.background_process)
        (top_limit, bottom_limit, left_limit, right_limit) = border.detect()

        # TODO: below correspond to a not-debugged case "if (greycor > 2 && droite == 0) {" which
        # is met when borders are not computed.
        # limitod = border.right_limit_to_removeable_from_image()
        limitod = border.right_limit_to_removeable_from_right_limit()

        adjusted_bg = self._bg_resized_for_sample_scan(
            bg_image, bg_resolution, sample_image, sample_image_resolution
        )

        # TODO: this _only_ corresponds to "if (method == "neutral") {" in legacy
        sample_minus_background_image = images_difference(adjusted_bg, sample_image)
        # Invert 8-bit
        sample_minus_background_image = 255 - sample_minus_background_image

        sample_minus_background_image = crop_right(
            sample_minus_background_image, limitod
        )

        cleared_width = min(right_limit - left_limit, limitod)
        clear_outside(
            sample_minus_background_image,
            left_limit,
            top_limit,
            cleared_width,
            bottom_limit - top_limit,
        )

        _draw_outside_lines(
            sample_minus_background_image,
            sample_image.shape,
            right_limit,
            left_limit,
            top_limit,
            bottom_limit,
            limitod,
        )
        return sample_minus_background_image

    @staticmethod
    def _bg_resized_for_sample_scan(
        bg_image: np.ndarray,
        bg_resolution: int,
        scan_image: np.ndarray,
        scan_resolution: int,
    ) -> np.ndarray:
        """Return self, resized to accommodate the sample scan"""
        scan_height, scan_width = scan_image.shape
        bg_height, bg_width = bg_image.shape

        backratio = scan_resolution / bg_resolution
        larg = scan_width / backratio
        haut = scan_height / backratio

        fondx0 = bg_width - larg
        fondy0 = bg_height - haut

        # TODO: What happens on ImageJ side?
        fondy0 = max(fondy0, 0)
        haut = min(haut, bg_image.shape[0])

        image_cropped = crophw(bg_image, fondx0, fondy0, larg, haut)

        # IJ macro: run("Mean...", "radius=3");
        image_mean = circular_mean_blur(image_cropped, 3)

        L = int(bg_width * backratio)
        H = int(bg_height * backratio)
        image_resized = bilinear_resize(image_mean, L, H)

        return image_resized


def _draw_outside_lines(
    image: np.array,
    sample_dims: Tuple[int, int],
    right_limit,
    left_limit,
    top_limit,
    bottom_limit,
    limitod,
):

    height, width = sample_dims
    if limitod < width:
        width = limitod
    if right_limit != width and right_limit < limitod:
        draw_line(image, (right_limit, 0), (right_limit, height / 4), 0, 1)
        draw_line(image, (right_limit, height / 4 + 4), (right_limit, height), 0, 1)
    if left_limit != 0:
        draw_line(image, (left_limit, 0), (left_limit, height / 4), 0, 1)
        draw_line(image, (left_limit, height / 4 + 4), (left_limit, height), 0, 1)
    if bottom_limit != height:
        draw_line(image, (0, bottom_limit), (width / 4, bottom_limit), 0, 1)
        draw_line(image, (width / 4 + 4, bottom_limit), (width, bottom_limit), 0, 1)
    if top_limit != 0:
        draw_line(image, (0, top_limit), (width / 4, top_limit), 0, 1)
        draw_line(image, (width / 4 + 4, top_limit), (width, top_limit), 0, 1)
