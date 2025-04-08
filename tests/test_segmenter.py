from pathlib import Path

import cv2
import numpy as np
import pytest

from ZooProcess_lib.img_tools import loadimage

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
