#
# Algorithms found in ZooProcess legacy and differing (maybe?) from the usual meaning of their output
#
import math
from typing import Tuple

import numpy as np

from .img_tools import crophw


def averaged_median_mean(image: np.ndarray) -> Tuple[float, float]:
    """
    A measurement on the image which is the average of median and mean of horizontal
    slices from the image, after removing some border.
    Return: (median, mean)
    """
    height = image.shape[0]
    width = image.shape[1]

    BX, BY, W, H = resized_float(width, height)
    step = math.floor(H / 20)
    By = BY
    k = 0
    mediansum = 0.0
    meansum = 0.0
    while By < H + step:
        BX = int(BX)
        By = int(By)
        W = int(W)
        img = crophw(image, BX, By, W, step)

        median = np.median(img, axis=None)
        mediansum = mediansum + median

        mean = np.mean(img, axis=None)
        meansum = meansum + mean
        k += 1
        By += step

    median = mediansum / k
    mean = meansum / k

    return median, mean


def resized_float(largeur, hauteur) -> Tuple[float, float, float, float]:
    BX = largeur * 0.03
    BY = hauteur * 0.05
    W = largeur * 0.94
    H = hauteur * 0.93
    return BX, BY, W, H
