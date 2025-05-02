#
# Reverse engineering of macro Zooscan_background_5.txt from Zooprocess legacy
#
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .ImageJLike import divide_round_up
from .img_tools import saveimage, load_tiff_image_and_info

CV2_VERTICAL_FLIP_CODE = 0
CV2_NO_TIFF_COMPRESSION = 1


class BackgroundCombiner:
    """
    Add two background images to get a "manual" background.
    """

    def __init__(self):
        pass

    def do_files(self, background_files: List[Path], to_file: Path) -> None:
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
        :param background_files: Intermediate files here.
        :param to_file: Resulting file.
        """
        assert len(background_files) == 2
        info_1, back_1 = load_tiff_image_and_info(background_files[0])
        info_2, back_2 = load_tiff_image_and_info(background_files[1])
        dest_image, dest_resolution = self.do_from_images(
            [(back_1, info_1.resolution), (back_2, info_2.resolution)]
        )
        dest_file = to_file.as_posix()
        saveimage(dest_image, dest_file, dpi=(dest_resolution, dest_resolution))

    def do_from_images(
        self, backs: List[Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, int]:
        assert len(backs) == 2
        assert backs[0][1] == backs[1][1]  # Resolutions
        dest_image = divide_round_up(backs[0][0], 2)
        dest_image += divide_round_up(backs[1][0], 2)
        return dest_image, backs[0][1]
