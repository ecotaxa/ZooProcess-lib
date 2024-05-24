


import unittest
import pytest


import numpy as np

from pathlib import Path
import cv2

from Background import Background
from ProjectClass import ProjectClass

from Border import Border

from img_tools import (
        crop, crop_scan, crophw, cropnp,
        loadimage, saveimage, 
        picheral_median, 
        converthisto16to8, convertImage16to8bit, 
        minAndMax, 
        rotate90c, rotate90cc,
        normalize, normalize_back,
        separate_apply_mask,
        draw_contours, draw_boxes, draw_boxes_filtered,
        generate_vignettes,
        mkdir,
        getPath,
        resize,
        rolling_ball_black_background,
    
    )


class test_8bit(unittest.TestCase):

    project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
    TP = ProjectClass(project_folder)

    # back_name = "20141003_1144_back_large_1.tif" 
    # back_name = "20141003_1144_back_large" 
    # sample = "test_01_tot"

    use_raw = True

    bg_name = "20141003_1144_back_large"
    name = "test_01_tot"
    # back_name = "20141003_1144_back_large_raw_1.tif" 
    # sample = "test_01_tot_raw_1.tif"
    back_name = bg_name + "_raw" + "_1" + ".tif" 
    sample = name + "_raw" + "_1" + ".tif"

    output_path = Path(TP.testfolder,"8bit")



    def test_convertion(self):

        