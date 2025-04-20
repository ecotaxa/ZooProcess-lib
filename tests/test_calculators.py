from math import exp

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Features import Features
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.calculators.Custom import fractal_mp, ij_perimeter
from ZooProcess_lib.img_tools import loadimage
from .test_sample import MEASURES_TYPES, round_measurements
from .test_segmenter import FEATURES_DIR
from .test_segmenter import MEASURES_DIR
from .test_utils import read_result_csv


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


def test_ij_like_perimeter():
    """EDM aka EDT found elsewhere is not the same as the one computed in IJ"""
    img = loadimage(FEATURES_DIR / "mask_holes_2855_223.png")
    img_conv = 1 - img // 255
    perim = ij_perimeter(img_conv)
    assert round(perim, 3) == 359.772


@pytest.mark.parametrize(
    "img",
    [
        "2855_223",  # The first crop in apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2
        "13892_13563",
    ],
)
def test_ij_like_measures(img: str):
    image = loadimage(FEATURES_DIR / f"crop_{img}.png")
    # Add a white border, otherwise the particle touches a border and is gone
    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255,))
    THRESHOLD = 243
    segmenter = Segmenter(image, 0.3, 100, THRESHOLD)
    rois = segmenter.find_blobs(Segmenter.METH_TOP_CONTOUR_SPLIT)
    assert len(rois) == 1
    feat = Features(image=image, roi=rois[0], threshold=THRESHOLD)
    exp = read_result_csv(MEASURES_DIR / f"meas_{img}.csv", MEASURES_TYPES)
    # round_measurements(exp)
    act = [feat.as_legacy()]
    round_measurements(act)
    for k in ("BX", "BY", "XStart", "YStart"):  # Remove image-related features
        del exp[0][k]
        del act[0][k]
    assert act == exp
