import numpy as np

from ZooProcess_lib.Segmenter import Segmenter


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
