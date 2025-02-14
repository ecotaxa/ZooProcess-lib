
import numpy as np
from ZooProcess_lib.img_tools import (
    crophw,
    saveimage,
)


def picheral_median(image:np.ndarray):
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

    BX = 0
    BY = 0
    W = width
    H = height


    # BX,BY,W,H = resize(width,height)  # floor(BX),floor(BY),ceil(W),ceil(H)
    print(f"BX,BY,W,H = {BX},{BY},{W},{H}")
    step = math.floor(H/20)
    print(f"step: {step}")
    By = BY
    k = 0
    mediansum = 0.0
    meansum = 0.0
    while k < 20 : #By < H+step :
        BX = int(BX)
        By = int(By)
        W = int(W)
        step = int(step)
        # print(f"crop ({BX},{By})x({W},{step})")
        img = crophw(image, BX, By, W, step)

        # saveimage(img, "test", str(k), ext="tiff")

        # if img == np.empty : break

        # median,mean = mesure(img)

        median = np.median(img, axis=None)
        mediansum = mediansum + median

        # mean = getResult("Mean",k);
        mean = np.mean(img, axis=None)
        meansum = meansum + mean
        print(f"median: {median}, mean: {mean}")
        k = k + 1
        By = By + step	
        print(f"k: {k} By: {By} < {H+step} ")

    median = mediansum / k
    mean = meansum / k

    print(f"median: {median}, mean: {mean}")

    return (median,mean)

