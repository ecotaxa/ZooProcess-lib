#
# Reverse engineering of macro Zooscan_convert.txt from Zooprocess legacy
#
from pathlib import Path

import cv2

from .img_tools import loadimage, rotate90c, Lut, minAndMax
from .Legacy import averaged_median_mean
from .ImageJLike import convert_16bit_image_to_8bit_min_max

CV2_VERTICAL_FLIP_CODE = 0
CV2_NO_TIFF_COMPRESSION = 1


def Zooscan_convert(from_file: Path, to_file: Path, lut: Lut):
    """
    :param from_file: e.g. 20241216_0926_back_large_raw_2.tif
    :param to_file: e.g. 20241216_0926_back_large_2.tif in another directory
    :param lut: processing configuration
    """
    is_a_background = from_file.name[13:19] == "_back_"
    assert is_a_background, "Only bg processing for now!"

    raw_image = loadimage(from_file, type=cv2.IMREAD_UNCHANGED)
    (marc_median, marc_mean) = averaged_median_mean(raw_image)
    (min_rec, max_rec) = minAndMax(marc_median, lut)
    image8bit = convert_16bit_image_to_8bit_min_max(raw_image, min_rec, max_rec)
    image_rotated = rotate90c(image8bit)
    image_flipped = cv2.flip(image_rotated, CV2_VERTICAL_FLIP_CODE)

    dest_file = to_file.as_posix()
    cv2.imwrite(dest_file, image_flipped, params=(cv2.IMWRITE_TIFF_COMPRESSION, CV2_NO_TIFF_COMPRESSION))
