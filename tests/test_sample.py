import cv2
import numpy as np

from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import load_zipped_image, loadimage, image_info, get_date_time_digitized
from tests.env_fixture import projects
from tests.projects_for_test import APERO2000


def test_raw_to_work(projects, tmp_path):
    """Ensure we can mimic raw sample scanned to work equivalent"""
    folder = ZooscanFolder(projects, APERO2000)
    TP = ZooscanProject(projects, APERO2000)
    sample = "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4"
    index = 1  # TODO: should come from get_names() below

    # load_final_ref_image(folder, index, sample)

    # Load the last background used at time of scan operation
    dates = folder.zooscan_back.get_dates()
    assert len(dates) > 0

    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    # img_info = image_info(raw_sample_file)
    # digitized_at = get_date_time_digitized(img_info)
    raw_image = loadimage(raw_sample_file, type=cv2.IMREAD_UNCHANGED)
    assert raw_image.dtype == np.uint16
    # last_background_file = folder.zooscan_back.get_last_background_before(digitized_at)
    # last_background = loadimage(last_background_file, type=cv2.IMREAD_UNCHANGED)
    # assert last_background.dtype == np.uint8

    output_path = tmp_path / raw_sample_file.name
    Zooscan_convert(raw_sample_file, output_path, TP.readLut())
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)

    ref_8bit_sample_file = folder.zooscan_scan.get_file_produced_from(raw_sample_file.name)
    assert ref_8bit_sample_file.exists()
    expected_image = loadimage(ref_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)

    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image[0], actual_image[0])
    assert np.array_equal(expected_image, actual_image)



def load_final_ref_image(folder, index, sample):
    assert sample in [
        a_sample["name"] for a_sample in folder.zooscan_scan.raw.get_names()
    ]
    work_files_in_sample = folder.zooscan_scan.work.get_files(sample, index)
    zipped_raw = work_files_in_sample.get("rawz")
    source_zip = zipped_raw
    assert source_zip.exists()
    reference_image = load_zipped_image(source_zip)
    assert reference_image.dtype == np.uint8

