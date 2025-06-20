import cv2
import numpy as np

from ZooProcess_lib.LegacyMeta import LutFile
from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.img_tools import loadimage
from data_dir import BACKGROUND_DIR, CONFIG_DIR


def test_identical_converted_8bit_background(tmp_path):
    """Ensure we convert like legacy the scanned background images"""
    source_bg_file = BACKGROUND_DIR / "20240529_0946_back_large_raw_1.tif"
    assert source_bg_file.exists()
    reference_bg_file = BACKGROUND_DIR / "20240529_0946_back_large_1.tif"
    assert reference_bg_file.exists()
    output_path = tmp_path / source_bg_file.name
    lut = LutFile.read(CONFIG_DIR / "lut.txt")
    processor = Processor.from_legacy_config(None, lut)
    processor.converter.do_file_to_file(source_bg_file, output_path)
    expected_image = loadimage(reference_bg_file, type=cv2.IMREAD_UNCHANGED)
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)
    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image, actual_image)
