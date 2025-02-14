import cv2
import numpy as np

from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.Zooscan_combine_backgrounds import Zooscan_combine_backgrounds
from ZooProcess_lib.img_tools import loadimage
from tests.env_fixture import projects
from tests.projects_for_test import APERO


def test_combine_backgrounds(projects, tmp_path):
    """Ensure we combine like legacy the scanned background images"""
    project_folder = APERO
    TP = ZooscanProject(projects, project_folder)
    scan_date = "20241216_0926"
    source_files = [
        TP.getProcessedBackgroundFile(TP.getRawBackgroundFile(scan_date, index))
        for index in (1, 2)
    ]
    assert [source_file.exists() for source_file in source_files]
    reference_bg_file = TP.getCombinedBackgroundFile(scan_date, "manual")
    assert reference_bg_file.exists()
    output_path = tmp_path / reference_bg_file.name
    Zooscan_combine_backgrounds(source_files, output_path, TP.readLut())
    expected_image = loadimage(reference_bg_file, type=cv2.IMREAD_UNCHANGED)
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)
    assert np.array_equal(expected_image, actual_image)
