import cv2
import numpy as np

from ZooProcess_lib.ImageJLike import convert_16bit_image_to_8bit_min_max
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import loadimage
from tests.env_fixture import projects
from tests.projects_for_test import APERO2000


def test_identical_converted_8bit_sample(projects, tmp_path):
    """Ensure we convert like legacy the scanned background images"""
    folder = ZooscanFolder(projects, APERO2000)
    TP = ZooscanProject(projects, APERO2000)
    sample = "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4"
    index = 1  # TODO: should come from get_names() below

    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)

    output_path = tmp_path / raw_sample_file.name
    Zooscan_convert(raw_sample_file, output_path, TP.readLut())
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)

    ref_8bit_sample_file = folder.zooscan_scan.get_file_produced_from(
        raw_sample_file.name
    )
    assert ref_8bit_sample_file.exists()
    expected_image = loadimage(ref_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)

    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image, actual_image)


def test_convert_16bit_image_to_8bit_min_max():
    # Below min should en up in 0s
    img = np.array([766, 762, 745, 778, 798, 790], np.uint16)
    out = convert_16bit_image_to_8bit_min_max(img, 805, 65536)
    assert out.tolist() == [0, 0, 0, 0, 0, 0]
