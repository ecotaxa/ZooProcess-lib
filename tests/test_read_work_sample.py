import numpy as np

from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import load_zipped_image
from tests.env_fixture import projects
from tests.projects_for_test import APERO2000


def test_read_sample_in_work(projects, tmp_path):
    """Ensure we can read tiff-in-zip"""
    folder = ZooscanFolder(projects, APERO2000)
    sample = "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_4_sur_4"
    index = 1  # TODO: should come from get_names() below
    assert sample in [
        a_sample["name"] for a_sample in folder.zooscan_scan.raw.get_names()
    ]
    work_files_in_sample = folder.zooscan_scan.work.get_files(sample, index)
    zipped_raw = work_files_in_sample.get("combz")
    source_zip = zipped_raw
    assert source_zip.exists()
    reference_image = load_zipped_image(source_zip)
    assert reference_image.dtype == np.uint8
