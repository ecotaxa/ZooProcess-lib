#
# Reverse engineering of macro Zooscan_background_5.txt from Zooprocess legacy
#
from pathlib import Path
from typing import List

import cv2

from .ImageJLike import divide_round_up
from .img_tools import Lut, loadimage

CV2_VERTICAL_FLIP_CODE = 0
CV2_NO_TIFF_COMPRESSION = 1


def Zooscan_combine_backgrounds(from_files: List[Path], to_file: Path):
    """Process is e.g.:
    IN:
    20250204_1003_back_large_raw_2.tif
    20250204_1003_back_large_raw_1.tif
    INTERMEDIATE:
    20250204_1003_back_large_2.tif
    20250204_1003_back_large_1.tif
    OUT:
    20250204_1003_back_large_manual_log.txt
    20250204_1003_background_large_manual.tif
    :param from_files: Intermediate files here.
    """
    dest_image = combine_backgrounds(from_files)

    dest_file = to_file.as_posix()
    cv2.imwrite(
        dest_file,
        dest_image,
        params=(cv2.IMWRITE_TIFF_COMPRESSION, CV2_NO_TIFF_COMPRESSION),
    )


def combine_backgrounds(from_files):
    assert len(from_files) == 2
    back_1 = loadimage(from_files[0], type=cv2.IMREAD_UNCHANGED)
    back_2 = loadimage(from_files[1], type=cv2.IMREAD_UNCHANGED)
    dest_image = divide_round_up(back_1, 2) + divide_round_up(back_2, 2)
    return dest_image
