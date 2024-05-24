
import unittest
import pytest
import numpy as np

from pathlib import Path
import cv2

from Background import Background
from ProjectClass import ProjectClass

from Border import Border

from img_tools import (
        # crop, crop_scan, crophw, cropnp,
        loadimage, saveimage, 
        # picheral_median, 
        # converthisto16to8, convertImage16to8bit, 
        # minAndMax, 
        # rotate90c, rotate90cc,
        # normalize, normalize_back,
        # separate_apply_mask,
        # draw_contours, draw_boxes, draw_boxes_filtered,
        # generate_vignettes,
        # mkdir,
        # resize,
        # rolling_ball_black_background,
    )

from median import picheral_median

class test_median(unittest.TestCase):

    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)
    
    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_picheral_median(self):

        back_name = "20240112_1518_back_large_1.tif"

        back_file = Path(self.TP.back, back_name)
        print(f"file: {back_file}")
        back_image = loadimage(back_file.as_posix())

        median, mean = picheral_median(back_image)

        print(f"median: {median}, mean: {mean}")



from to8bit import convertion

from pathlib import Path
from ProjectClass import ProjectClass
from img_tools import loadimage

from median import picheral_median

def test_picheral_median(file):
    from img_tools import picheral_median as median
    # project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    # TP = ProjectClass(project_folder)

    # # back_name = "20240112_1518_back_large_1.tif"
    # # back_file = Path(TP.back, back_name)
    # back_file = Path(TP.back, filename)
    # print(f"file: {back_file}")
    # back_image = loadimage(back_file.as_posix())
    back_image = loadimage(file.as_posix())
    median, mean = median(back_image)
    print(f"median: {median}, mean: {mean}")


def test_picheral_median_local(file):
    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    # # back_name = "20240112_1518_back_large_1.tif"
    # # back_file = Path(TP.back, back_name)
    # back_file = Path(TP.back, filename)
    # print(f"file: {back_file}")
    # back_image = loadimage(back_file.as_posix())
    back_image = loadimage(file.as_posix())

    median, mean = picheral_median(back_image)
    # median, mean = median(back_image)

    print(f"median: {median}, mean: {mean}")


from img_tools import getPath

def test_median():

    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    back_name = "20240112_1518_back_large_1.tif"
    # back_file = Path(TP.back, back_name)

    # output_path = TP.testfolder
    # # saveimage(scan_unbordered, sample, "unbordered", ext="tiff", path=output_path)
    back_file = Path(getPath(back_name, "unbordered", ext="tiff", path=TP.testfolder))

    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(TP.rawscan, sample + "_raw" + "_1" + ".tif")
    # image = loadimage(rawscan_file.as_posix())

    rawscan_file = Path(getPath(sample , "unbordered", ext="tiff", path=TP.testfolder))
    # image = loadimage(rawscan_file.as_posix())

    test_picheral_median(back_file)
    test_picheral_median_local(back_file)

    test_picheral_median(rawscan_file)
    test_picheral_median_local(rawscan_file)


def test_en_8bit():
    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    back_name = "20240112_1518_back_large_1.tif"
    back_file = Path(getPath(back_name, "resized", ext="tiff", path=TP.testfolder))
    back_image = loadimage(back_file.as_posix())

    image_back_8bit = convertion(back_image, back_name, TP = TP)
    saveimage(image_back_8bit, back_name, "8bit", ext="jpg", path=TP.testfolder)


    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(getPath(sample , "unbordered", ext="tiff", path=TP.testfolder))
    # image = loadimage(rawscan_file.as_posix())  
    # image_sample_8bit = convertion(image, sample)
    # saveimage(image_sample_8bit, sample, "8bit", ext="jpg", path=TP.testfolder)
    rawscan_file = Path(getPath(sample , "treated", ext="tiff", path=TP.testfolder))
    image_sample_8bit = loadimage(rawscan_file.as_posix())  

    
    image_substracted = np.subtract(image_sample_8bit, image_back_8bit)
    saveimage(image_substracted, sample, "substracted", ext="tiff", path=TP.testfolder)

    image_substracted2 = np.subtract(image_back_8bit, image_sample_8bit)
    saveimage(image_substracted2, sample, "substracted2", ext="tiff", path=TP.testfolder)
    print("Done")

if __name__ == '__main__':


    # test_en_8bit()

    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    back_name = "20240112_1518_back_large_1.tif"
    back_file = Path(getPath(back_name, "resized", ext="tiff", path=TP.testfolder))
    back_image = loadimage(back_file.as_posix())

    image_back_8bit = convertion(back_image, back_name, TP = TP)
    saveimage(image_back_8bit, back_name, "8bit", ext="jpg", path=TP.testfolder)


    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(getPath(sample , "unbordered", ext="tiff", path=TP.testfolder))
    # image = loadimage(rawscan_file.as_posix())  
    # image_sample_8bit = convertion(image, sample)
    # saveimage(image_sample_8bit, sample, "8bit", ext="jpg", path=TP.testfolder)
    rawscan_file = Path(getPath(sample , "treated", ext="tiff", path=TP.testfolder))
    image_sample_8bit = loadimage(rawscan_file.as_posix())  

    
    image_substracted = np.subtract(image_sample_8bit, image_back_8bit)
    saveimage(image_substracted, sample, "substracted", ext="tiff", path=TP.testfolder)

    image_substracted2 = np.subtract(image_back_8bit, image_sample_8bit)
    saveimage(image_substracted2, sample, "substracted2", ext="tiff", path=TP.testfolder)
    print("Done")


