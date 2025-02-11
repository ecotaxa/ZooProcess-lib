import unittest
from pathlib import Path

import cv2
import numpy as np

from ProjectClass import ProjectClass
from Zooscan_convert import Zooscan_convert
from img_tools import loadimage, picheral_median, Lut
from img_tools_2 import median_mean, picheral_median_2

my_piqv_home = "/home/laurent/Devs/from_Lab/ZooProcess/ImageJ4zoo/P:"
my_test_folder = "/home/laurent/Devs/from_Lab/ZooProcess/TestOut"


class test_8bit_LS(unittest.TestCase):
    project_folder = "Zooscan_apero_tha_bioness_2_sn033"
    TP = ProjectClass(project_folder, piqvhome=my_piqv_home, testfolder=my_test_folder)

    bg_name = "20241216_0926_back_large"
    bg_index = "_2"

    raw_bg_name = bg_name + "_raw" + bg_index + ".tif"
    bg_raw_file = TP.back / raw_bg_name
    out_8bit_name = bg_name + bg_index + ".tif"
    bg_ref_8bit_file = TP.back / out_8bit_name

    output_path = Path(TP.testfolder, "8bit")
    output_path.mkdir(parents=False, exist_ok=True)
    bg_8bit_file = output_path / out_8bit_name

    def test_setup(self):
        assert self.output_path.exists()
        assert self.bg_raw_file.exists()
        assert self.bg_ref_8bit_file.exists()

    def test_bg_convert(self):
        source_image = loadimage(self.bg_raw_file)
        (marc_median, marc_mean) = picheral_median_2(source_image)
        assert (float(marc_median), round(float(marc_mean), 4)) == (41331.45, 40792.4142)  # Values from ImageJ run
        lut = Lut()
        Zooscan_convert(self.bg_raw_file, self.bg_8bit_file, lut)
        expected_image = loadimage(self.bg_ref_8bit_file, type=cv2.IMREAD_UNCHANGED)
        actual_image = loadimage(self.bg_8bit_file, type=cv2.IMREAD_UNCHANGED)
        (median_exp, mean_exp) = median_mean(expected_image)
        (median_act, mean_act) = median_mean(actual_image)
        assert expected_image.shape == actual_image.shape
        assert (median_exp, mean_exp) == (median_act, mean_act)
        assert np.array_equal(expected_image, actual_image)
