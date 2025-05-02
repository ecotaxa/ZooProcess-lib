from pathlib import Path

import cv2
import numpy as np

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.img_tools import loadimage
from data_dir import BACK_TIME, BACKGROUND_DIR


def test_combine_backgrounds(tmp_path):
    """Ensure we combine like legacy the scanned background images"""
    bg_scan_date = BACK_TIME
    processor = Processor()
    source_files = [
        Path(BACKGROUND_DIR, f"{bg_scan_date}_back_large_{index}.tif") for index in (1, 2)
    ]
    assert [source_file.exists() for source_file in source_files]
    reference_bg_file = Path(BACKGROUND_DIR, f"{bg_scan_date}_background_large_manual.tif")
    assert reference_bg_file.exists()
    output_path = tmp_path / reference_bg_file.name
    processor.bg_combiner.do_files(source_files, output_path)
    expected_image = loadimage(reference_bg_file, type=cv2.IMREAD_UNCHANGED)
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)
    assert np.array_equal(expected_image, actual_image)
