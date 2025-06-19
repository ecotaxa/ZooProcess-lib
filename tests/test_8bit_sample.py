import cv2
import numpy as np

from ZooProcess_lib.ImageJLike import convert_16bit_image_to_8bit_min_max
from ZooProcess_lib.LegacyMeta import LutFile
from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.img_tools import loadimage
from data_dir import RAW_DIR, CONFIG_DIR, SAMPLE_DIR


def test_identical_converted_8bit_sample(tmp_path):
    """Ensure we convert like legacy the scanned background images"""
    raw_sample_file = RAW_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_raw_1.tif"

    output_path = tmp_path / raw_sample_file.name
    lut = LutFile.read(CONFIG_DIR / "lut.txt")
    processor = Processor.from_legacy_config(None, lut)
    processor.converter.do_file_to_file(raw_sample_file, output_path)

    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)

    ref_8bit_sample_file = SAMPLE_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1.tif"
    assert ref_8bit_sample_file.exists()
    expected_image = loadimage(ref_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)

    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image, actual_image)


def test_convert_16bit_image_to_8bit_min_max():
    # Below min should en up in 0s
    img = np.array([766, 762, 745, 778, 798, 790], np.uint16)
    out = convert_16bit_image_to_8bit_min_max(img, 805, 65536)
    assert out.tolist() == [0, 0, 0, 0, 0, 0]
