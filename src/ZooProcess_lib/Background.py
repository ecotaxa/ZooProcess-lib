from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .ImageJLike import circular_mean_blur, bilinear_resize
from .img_tools import crophw, crop, saveimage, draw_box


class Background:
    """Port of algorithms from legacy Zooscan_1asep.txt macro"""

    def __init__(
        self,
        image: np.ndarray,
        image_path: Path,
        resolution=300,
        sample_scan_resolution=2400,
        output_path=None,  # TODO: Remove
    ):
        """
        :param image: np.ndarray (H, W)
        :param image_path: the file the image comes from
        :param resolution: present background resolution
        :param sample_scan_resolution: associated scan resolution
        """
        self.image = image
        self.name = image_path.name

        self.output_path = output_path

        self.blancres = resolution
        self.frametypeback = None
        if self.name.find("_large_"):
            self.frametypeback = "large"
        if self.name.find("_narrow_"):
            self.frametypeback = "narrow"

        self.backratio = self.compute_backratio(sample_scan_resolution)

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.L = int(self.width * self.backratio)
        self.H = int(self.height * self.backratio)
        self.lf = self.width
        self.hf = self.height

        self.Lm = self.L * 0.95
        self.LM = self.L * 1.05
        self.Hm = self.H * 0.95
        self.HM = self.H * 1.05

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

    def mean(self):
        x1 = self.width / 10
        y1 = self.height / 10
        x2 = self.width - x1
        y2 = self.height - y1
        w = x2 - x1
        h = y2 - y1
        w = 10

        image_cropped = crop(self.image, left=x1, top=y1, right=x2, bottom=y2)

        image_drew = draw_box(self.image, x=x1, y=y1, w=w, h=h)
        saveimage(image_drew, self.name, "drew", path=self.output_path)

        image_median = cv2.medianBlur(image_cropped, 3)
        # image_median = image_drew

        mean = np.mean(image_median, axis=None)
        print(mean)
        return mean

    def compute_backratio(self, scan_resolution=2400):
        return scan_resolution / self.blancres

    def voxel(self, scan_image):
        # backratio = self.backratio(scan_image)
        pass

    def resized_for_sample_scan(
        self, sample_scan_cropx: int, sample_scan_cropy: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        larg = sample_scan_cropx / self.backratio
        haut = sample_scan_cropy / self.backratio
        fondx0 = self.lf - larg
        fondy0 = self.hf - haut
        # TODO: What happens on ImageJ side?
        fondy0 = max(fondy0, 0)
        haut = min(haut, self.image.shape[0])

        print(f"fond: {fondx0},{fondy0} {haut},{larg}")

        image_cropped = crophw(self.image, fondx0, fondy0, larg, haut)
        # macro: run("Mean...", "radius=3");
        image_mean = circular_mean_blur(image_cropped, 3)

        # img2 = np.resize(image_mean, (image_mean.shape[0] + 1, image_mean.shape[1]))

        image_resized = bilinear_resize(image_mean, self.L, self.H)

        # saveimage(
        #     image_resized, self.name, "resized", ext="tiff", path=self.output_path
        # )

        return image_cropped, image_mean, image_resized
