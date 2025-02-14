from typing import Tuple

import numpy as np


def median_mean(image: np.ndarray):
    # Exact same as ImageJ "Measure" output on full 8-bit tiff background image
    median = np.median(image, axis=None)
    mean = np.mean(image, axis=None)
    return median, mean


def resized_float(largeur, hauteur) -> Tuple[float, float, float, float]:
    BX = largeur * 0.03
    BY = hauteur * 0.05
    W = largeur * 0.94
    H = hauteur * 0.93
    return BX, BY, W, H
