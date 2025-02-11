from typing import Tuple

import numpy as np

from img_tools import crophw


def convert_16bit_image_to_8bit_min_max(img: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    assert img.dtype == np.uint16
    # Vectorial port of ImageJ algorithm in TypeConverter.convertShortToByte
    scale = 256. / (max_val - min_val + 1)
    min_removed_img = img - min_val
    min_removed_img[min_removed_img < 0] = 0
    assert ~(min_removed_img < 0).all()
    scaled_img = min_removed_img * scale + 0.5
    scaled_img[scaled_img > 255] = 255
    ret = scaled_img.astype(np.uint8)
    return ret


def median_mean(image: np.ndarray):
    # Exact same as ImageJ "Measure" output on full 8-bit tiff background image
    median = np.median(image, axis=None)
    mean = np.mean(image, axis=None)
    return median, mean


def picheral_median_2(image: np.ndarray) -> Tuple[float, float]:
    """
    Return: (median, mean)
    """
    import math
    # from 16to8bit import resize
    height = image.shape[0]
    width = image.shape[1]
    print(f"size {width}x{height}")

    # BX = width*0.03
    # BY = height*0.05
    # W = width *0.94
    # H = height*0.93

    BX, BY, W, H = resized_float(width, height)
    print(f"BX,BY,W,H = {BX},{BY},{W},{H}")
    step = math.floor(H / 20)
    print(f"step: {step}")
    By = BY
    k = 0
    mediansum = 0.0
    meansum = 0.0
    while By < H + step:
        BX = int(BX)
        By = int(By)
        W = int(W)
        step = int(step)
        # print(f"crop ({BX},{By})x({W},{step})")
        img = crophw(image, BX, By, W, step)

        # median,mean = mesure(img)

        median = np.median(img, axis=None)
        mediansum = mediansum + median

        # mean = getResult("Mean",k);
        mean = np.mean(img, axis=None)
        meansum = meansum + mean
        print(f"median: {median}, mean: {mean}")
        k = k + 1
        By = By + step

    median = mediansum / k
    mean = meansum / k

    print(f"median: {median}, mean: {mean}")

    return (median, mean)


def resized_float(largeur, hauteur) -> Tuple[float, float, float, float]:
    BX = largeur * 0.03
    BY = hauteur * 0.05
    W = largeur * 0.94
    H = hauteur * 0.93
    return BX, BY, W, H
