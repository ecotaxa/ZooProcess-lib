#
# Algorithms found in ZooProcess legacy and differing (maybe?) from the usual meaning of their output
#
import math
from typing import Tuple

import numpy as np

from ZooProcess_lib.img_tools import crophw
from ZooProcess_lib.img_tools_2 import resized_float


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
        step = int(step)
        img = crophw(image, By, BX, step, W)

        median = np.median(img, axis=None)
        mediansum = mediansum + median

        mean = np.mean(img, axis=None)
        meansum = meansum + mean
        k += 1
        By += step

    median = mediansum / k
    mean = meansum / k

    return median, mean
