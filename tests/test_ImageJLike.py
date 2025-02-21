import numpy as np

from ZooProcess_lib.ImageJLike import bilinear_resize
from ZooProcess_lib.img_tools import loadimage
from tests.test_utils import diff_actual_with_ref_and_source


def test_resize():
    src_image = loadimage("/tmp/fond_apres_mean.tif")  # TODO: Add into repo (compressed)
    dst_image = bilinear_resize(src_image, 25000, 14992)
    ref_image = loadimage("/tmp/fond_apres_resize.tif")
    assert dst_image.shape == ref_image.shape
    # assert np.array_equal(ref_image, dst_image)
    # for i in range(dst_image.shape[0]):
    #     assert np.array_equal(ref_image[i], dst_image[i])
    if not np.array_equal(ref_image, dst_image):
        nb_real_errors = diff_actual_with_ref_and_source(
            ref_image,
            dst_image,
            ref_image,
            tolerance=0,
        )
        if nb_real_errors > 0:
            assert False


if __name__ == "__main__":
    test_resize()
