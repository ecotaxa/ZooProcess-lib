from math import exp

import numpy as np
import pytest

from ZooProcess_lib.Features import Features, TYPE_BY_ECOTAXA, ECOTAXA_TSV_ROUNDINGS
from ZooProcess_lib.ROI import ecotaxa_tsv_unq
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.calculators.Custom import fractal_mp, ij_perimeter
from ZooProcess_lib.img_tools import loadimage
from .data_dir import MEASURES_DIR, FEATURES_DIR
from .data_tools import (
    to_legacy_rounding,
    FEATURES_TOLERANCES,
    report_and_fix_tolerances,
    DERIVED_FEATURES_TOLERANCES,
    diff_features_lists,
)
from .test_utils import read_ecotaxa_tsv


@pytest.mark.skip(
    "we choose a slightly different algorithm for EDM, see comments in fractal_mp()"
)
def test_ij_like_EDM():
    """EDM aka EDT found elsewhere is not the same as the one computed in IJ"""
    img = loadimage(MEASURES_DIR / "mask_holes_2855_223.png")
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
    assert round(meas, 4) == 1.3501


def test_ij_like_perimeter():
    """Verify a perimeter to match the expected one from legacy pp"""
    img = loadimage(MEASURES_DIR / "mask_holes_2855_223.png")
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
def test_ij_like_features(img: str):
    image = loadimage(MEASURES_DIR / f"crop_{img}.png")
    THRESHOLD = 243
    segmenter = Segmenter(0.3, 100, THRESHOLD)
    rois, _ = segmenter.find_ROIs_in_cropped_image(
        image, 2400, Segmenter.METH_TOP_CONTOUR_SPLIT
    )
    assert len(rois) == 1
    feat = Features(image=image, resolution=2400, roi=rois[0], threshold=THRESHOLD)
    exp = read_ecotaxa_tsv(FEATURES_DIR / f"ecotaxa_{img}.tsv", TYPE_BY_ECOTAXA)
    act = [feat.as_ecotaxa()]
    to_legacy_rounding(act, ECOTAXA_TSV_ROUNDINGS)
    for k in (
        "object_bx",
        "object_by",
        "object_xstart",
        "object_ystart",
    ):  # Transfer image-related features
        act[0][k] = exp[0][k]
    tolerance_problems = []
    ECOTAXA_TOLERANCES = {
        "object_" + k.lower(): v
        for k, v in (FEATURES_TOLERANCES | DERIVED_FEATURES_TOLERANCES).items()
    }
    if exp != act:
        different, not_in_reference, not_in_actual = diff_features_lists(
            exp, act, ecotaxa_tsv_unq
        )
        tolerance_problems = report_and_fix_tolerances(different, ECOTAXA_TOLERANCES)
    assert act == exp
    assert tolerance_problems == []
