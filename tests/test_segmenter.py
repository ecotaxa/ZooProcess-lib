from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pytest

from ZooProcess_lib.ROI import feature_unq
from ZooProcess_lib.img_tools import loadimage
from ZooProcess_lib.segmenters.ConnectedComponents import (
    ConnectedComponentsSegmenter,
    CC,
)
from .test_sample import sort_by_coords, assert_valid_diffs

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
SEGMENTER_DIR = IMAGES_DIR / "segmenter"

from ZooProcess_lib.Segmenter import Segmenter


@pytest.mark.skip("interface changed")
def test_denoiser():
    obj = 255
    bgd = 0
    img = np.array(
        [
            [bgd, bgd, bgd, bgd, obj, obj, obj],
            [bgd, obj, bgd, bgd, obj, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, obj, bgd, bgd, bgd, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
        ],
        np.uint8,
    )
    exp = np.array(
        [
            [bgd, bgd, bgd, bgd, obj, obj, obj],
            [bgd, bgd, bgd, bgd, obj, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
        ],
        np.uint8,
    )
    res = Segmenter.denoise_for_segment(img)
    np.testing.assert_equal(res, exp)


def test_holes_62388():
    # This image biggest particle has a pattern in holes, their contour touches the particle itself
    image = loadimage(SEGMENTER_DIR / "cc_62388.png")
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,))
    part = Segmenter(image, 0.1, 100000).find_blobs(Segmenter.METH_CONNECTED_COMPONENTS)
    areas = sorted([p.features["Area"] for p in part], reverse=True)
    assert areas == [1675, 342, 244, 207]
    part = Segmenter(image, 0.1, 100000).find_blobs(Segmenter.METH_TOP_CONTOUR_SPLIT)
    features = sorted([p.features for p in part], key=feature_unq, reverse=True)
    assert features == [
        {"Area": 244, "BX": 146, "BY": 91, "Height": 26, "Width": 81},
        {"Area": 1407, "BX": 102, "BY": 110, "Height": 34, "Width": 67},
        {"Area": 207, "BX": 67, "BY": 7, "Height": 18, "Width": 26},
        {"Area": 342, "BX": 14, "BY": 96, "Height": 30, "Width": 25},
        {"Area": 1675, "BX": 4, "BY": 4, "Height": 148, "Width": 258},
    ]

