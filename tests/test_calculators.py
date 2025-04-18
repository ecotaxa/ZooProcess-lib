from math import exp

import numpy as np

from ZooProcess_lib.calculators.Custom import fractal_mp
from ZooProcess_lib.img_tools import loadimage
from .test_segmenter import FEATURES_DIR


def test_ij_like_EDM():
    """EDM aka EDT found elsewhere is not the same as the one computed in IJ"""
    img = loadimage(FEATURES_DIR / "mask_holes_2855_223.png")
    img = 1 - (img.astype(np.uint8) // 255)
    meas, areas = fractal_mp(img)
    from_ij = [
        1470,
        1848,
        2058,
        2292,
        2457,
        2662,
        2842,
        3040,
        3233,
        3451,
        3659,
        3887,
        4115,
        4356,
        4834,
        5084,
        5609,
        6147,
        6717,
        7600,
        8228,
        9220,
        10255,
        11341,
        12893,
        14537,
    ]
    assert [round(exp(a)) for a in areas] == from_ij
    assert meas == 1.3501
