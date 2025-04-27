import cv2
import numpy as np

from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import loadimage
from tests.env_fixture import projects
from tests.projects_for_test import APERO


def test_identical_converted_8bit_background(projects, tmp_path):
    """Ensure we convert like legacy the scanned background images"""
    project_folder = APERO
    TP = ZooscanProject(projects, project_folder)
    source_bg_file = TP.getRawBackgroundFile("20241216_0926", 2)
    assert source_bg_file.exists()
    reference_bg_file = TP.getProcessedBackgroundFile(source_bg_file)
    assert reference_bg_file.exists()
    output_path = tmp_path / source_bg_file.name
    Zooscan_convert(source_bg_file, output_path, TP.readLut())
    expected_image = loadimage(reference_bg_file, type=cv2.IMREAD_UNCHANGED)
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)
    assert np.array_equal(expected_image, actual_image)
