from __future__ import annotations

from typing import Tuple

import numpy as np

from .Border import Border
from .ImageJLike import circular_mean_blur, bilinear_resize, images_difference
from .img_tools import (
    crophw,
    crop_right,
    clear_outside,
    draw_outside_lines,
)


class Background:
    """Port of algorithms from legacy Zooscan_1asep.txt macro"""

    def __init__(
        self,
        image: np.ndarray,
        # image_path: Path,
        resolution=300,
        # sample_scan_resolution=2400,
        # output_path=None,  # TODO: Remove
    ):
        """
        :param image: np.ndarray (H, W)
        :param image_path: the file the image comes from
        :param resolution: present background resolution
        :param sample_scan_resolution: associated scan resolution
        """
        self.image = image
        # self.name = image_path.name

        # self.output_path = output_path

        self.blancres = resolution
        # self.frametypeback = None
        # if self.name.find("_large_"):
        #     self.frametypeback = "large"
        # if self.name.find("_narrow_"):
        #     self.frametypeback = "narrow"

        # self.backratio = self.compute_backratio(sample_scan_resolution)

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.lf = self.width
        self.hf = self.height

        # self.Lm = self.L * 0.95
        # self.LM = self.L * 1.05
        # self.Hm = self.H * 0.95
        # self.HM = self.H * 1.05

    def __str__(self):
        return f"""
            name: {self.name}
            blancres: {self.blancres}

            L : {self.L:8}   H : {self.H:8}
            lf: {self.lf:8}   hf: {self.hf:8}
            LM: {self.LM:8}   HM: {self.HM:8}
            Lm: {self.Lm:8}   Hm: {self.Hm:8}

            ratio: {self.backratio}
        """

    def compute_backratio(self, scan_resolution: int) -> float:
        return scan_resolution / self.blancres

    def _resized_for_sample_scan(
        self,
        sample_scan_cropx: int,
        sample_scan_cropy: int,
        sample_scan_resolution: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return self, resized to accommodate the sample scan"""

        backratio = self.compute_backratio(sample_scan_resolution)
        larg = sample_scan_cropx / backratio
        haut = sample_scan_cropy / backratio

        fondx0 = self.lf - larg
        fondy0 = self.hf - haut
        # TODO: What happens on ImageJ side?
        fondy0 = max(fondy0, 0)
        haut = min(haut, self.image.shape[0])

        print(f"fond: {fondx0},{fondy0} {haut},{larg}")

        image_cropped = crophw(self.image, fondx0, fondy0, larg, haut)

        # IJ macro: run("Mean...", "radius=3");
        image_mean = circular_mean_blur(image_cropped, 3)

        L = int(self.width * backratio)
        H = int(self.height * backratio)
        image_resized = bilinear_resize(image_mean, L, H)

        # saveimage(
        #     image_resized, self.name, "resized", ext="tiff", path=self.output_path
        # )

        return image_cropped, image_mean, image_resized

    def removed_from(
        self,
        sample_image: np.ndarray,
        sample_image_resolution: int,
        processing_method: str,
    ) -> np.ndarray:
        border = Border(sample_image, sample_image_resolution, processing_method)
        (top_limit, bottom_limit, left_limit, right_limit) = border.detect()

        # TODO: below correspond to a not-debugged case "if (greycor > 2 && droite == 0) {" which
        # is met when borders are not computed.
        # limitod = border.right_limit_to_removeable_from_image()
        limitod = border.right_limit_to_removeable_from_right_limit()

        cropped_bg, mean_bg, adjusted_bg = self._resized_for_sample_scan(
            sample_image.shape[1], sample_image.shape[0], sample_image_resolution
        )

        # saveimage(cropped_bg, "/tmp/zooprocess/cropped_bg.tif")
        # ref_cropped_bg = loadimage(Path("/tmp/zooprocess/fond_cropped_legacy.tif"))
        # diff_actual_with_ref_and_source(ref_cropped_bg, cropped_bg, ref_cropped_bg)
        # assert np.array_equal(ref_cropped_bg, cropped_bg)
        # saveimage(mean_bg, "/tmp/zooprocess/mean_bg.tif")
        # ref_mean_bg = loadimage(Path("/tmp/zooprocess/fond_apres_mean.tif"))
        # diff_actual_with_ref_and_source(ref_mean_bg, mean_bg, ref_mean_bg)
        # assert np.array_equal(ref_mean_bg, mean_bg)
        # saveimage(adjusted_bg, Path("/tmp/zooprocess/resized_bg.tif"))
        # ref_resized_bg = loadimage(Path("/tmp/zooprocess/fond_apres_resize.tif"))
        # diff_actual_with_ref_and_source(ref_resized_bg, adjusted_bg, ref_resized_bg)
        # if not np.array_equal(ref_resized_bg, adjusted_bg):
        #     nb_errors = diff_actual_with_ref_and_source(
        #         ref_resized_bg,
        #         adjusted_bg,
        #         last_background_image,
        #         tolerance=0,
        #     )
        #     if nb_errors > 0:
        #         assert False
        #
        # TODO: this _only_ corresponds to "if (method == "neutral") {" in legacy
        sample_minus_background_image = images_difference(adjusted_bg, sample_image)
        # Invert 8-bit
        sample_minus_background_image = 255 - sample_minus_background_image

        # ref_after_sub_bg = loadimage(Path("/tmp/zooprocess/fond_apres_subs.tif"))
        # diff_actual_with_ref_and_source(ref_after_sub_bg, sample_minus_background_image, ref_after_sub_bg)
        # assert np.array_equal(ref_after_sub_bg, sample_minus_background_image)

        sample_minus_background_image = crop_right(
            sample_minus_background_image, limitod
        )
        # ref_after_sub_and_crop_bg = loadimage(Path("/tmp/zooprocess/fond_apres_subs_et_crop.tif"))
        # diff_actual_with_ref_and_source(ref_after_sub_and_crop_bg, sample_minus_background_image, ref_after_sub_and_crop_bg)
        # assert np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image)
        cleared_width = min(right_limit - left_limit, limitod)
        clear_outside(
            sample_minus_background_image,
            left_limit,
            top_limit,
            cleared_width,
            bottom_limit - top_limit,
        )
        # ref_after_sub_and_crop_bg = loadimage(
        #     Path("/tmp/zooprocess/fond_apres_subs_et_crop_et_clear.tif")
        # )
        # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
        #     diff_actual_with_ref_and_source(
        #         ref_after_sub_and_crop_bg,
        #         sample_minus_background_image,
        #         ref_after_sub_and_crop_bg,
        #     )
        #     assert False
        draw_outside_lines(
            sample_minus_background_image,
            sample_image.shape,
            right_limit,
            left_limit,
            top_limit,
            bottom_limit,
            limitod,
        )
        # ref_after_sub_and_crop_bg = loadimage(
        #     Path("/tmp/zooprocess/fond_apres_subs_et_crop_et_clear_et_lignes.tif")
        # )
        # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
        #     diff_actual_with_ref_and_source(
        #         ref_after_sub_and_crop_bg,
        #         sample_minus_background_image,
        #         ref_after_sub_and_crop_bg,
        #     )
        #     assert False
        return sample_minus_background_image
