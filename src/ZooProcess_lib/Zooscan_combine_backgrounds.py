#
# Reverse engineering of macro Zooscan_background_5.txt from Zooprocess legacy
#
from pathlib import Path
from typing import List

import cv2

from .ImageJLike import divide_round_up
from .img_tools import loadimage, saveimage, image_info, load_tiff_image_and_info

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
    :param to_file: Resulting file.
    """
    assert len(from_files) == 2
    info_1, back_1 = load_tiff_image_and_info(from_files[0])
    info_2, back_2 = load_tiff_image_and_info(from_files[1])
    assert info_1.resolution == info_2.resolution
    dest_image = divide_round_up(back_1, 2)
    dest_image += divide_round_up(back_2, 2)
    dest_file = to_file.as_posix()
    saveimage(dest_image, dest_file, dpi=info_1.resolutions)
