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
        resize,
        rolling_ball_black_background,
    
    )


class test_background(unittest.TestCase):

    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_init(self):
        
        back_name = "20240112_1518_back_large_1.tif" 

        back_file = Path(self.TP.back, back_name)
        print(f"file: {back_file}")
        back_image = loadimage( back_file.as_posix())

        background = Background(back_image, back_name, output_path=self.TP.testfolder)

        print(f"{background}")

        background.redim()
        

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_mean(self):
        
        back_name = "20240112_1518_back_large_1.tif" 

        back_file = Path(self.TP.back, back_name)
        print(f"file: {back_file}")
        back_image = loadimage( back_file.as_posix())

        background = Background(back_image, back_name, output_path=self.TP.testfolder)

        print(f"{background}")

        background.mean()
        

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_resize_background_but_(self):

        back_name = "20240112_1518_back_large_1.tif" 
        back_file = Path(self.TP.back, back_name)
        print(f"file: {back_file}")
        back_image = loadimage( back_file.as_posix())
        output_path = self.TP.testfolder
        border = Border(back_image)
        border.output_path = output_path
        border.name = back_name
        border.draw_image = loadimage(back_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()
        image_unbordered = crop(back_image, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)

        background = Background(image_unbordered, back_name, output_path=self.TP.testfolder)

        image_resized = background.redim()

        background.mean()





    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_all(self):

        back_name = "20240112_1518_back_large_1.tif" 
        back_file = Path(self.TP.back, back_name)
        print(f"file: {back_file}")
        back_image = rotate90c(loadimage( back_file.as_posix()))
        output_path = self.TP.testfolder
        border = Border(back_image)
        border.output_path = output_path
        border.name = back_name
        border.draw_image = rotate90c(loadimage(back_file.as_posix()))

        limitetop, limitbas, limitegauche, limitedroite = border.detect()
        print(f"back limite t={limitetop}, b={limitbas}, l={limitegauche}, r={limitedroite}")

        image_back_unbordered = crop(back_image, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        saveimage(image_back_unbordered, back_name, "unbordered", ext="tiff", path=output_path)


        # a faire apres
        # background = Background(image_back_unbordered, back_name, output_path=self.TP.testfolder)
        # image_resized = background.redim() 



        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        rawscan_file = Path(self.TP.rawscan, sample + "_raw" + "_1" + ".tif")
        image = loadimage(rawscan_file.as_posix())
        scan_border = Border(image)
        scan_border.output_path = output_path
        scan_border.name = sample
        scan_border.draw_image = loadimage(rawscan_file.as_posix())

        scan_limit_top, scan_limit_bottom, scan_limit_left, scan_limit_right = scan_border.detect()
        # limitetop, limitbas, limitegauche, limitedroite = scan_border.detect()
        # print(f"scan limite {scan_limite}")
        print(f"scan limite t={scan_limit_top}, b={scan_limit_bottom}, l={scan_limit_left}, r={scan_limit_right}")
        print(f"scan limite top={scan_limit_top}, b={scan_limit_bottom}, l={scan_limit_left}, r={scan_limit_right}")
        
        scan_unbordered = cropnp(image, left=scan_limit_left, top=scan_limit_top, right=scan_limit_right, bottom=scan_limit_bottom)
        saveimage(scan_unbordered, sample, "unbordered", ext="tiff", path=output_path)


        image_back_blurred = cv2.medianBlur(image_back_unbordered, 3)

        H = scan_unbordered.shape[0]
        L = scan_unbordered.shape[1]
        # scale factor
        # fx = 
        # fy = 
        interpolation = cv2.INTER_LINEAR
        image_back_resized = cv2.resize( image_back_blurred, dsize=(L, H), interpolation=interpolation)
        saveimage(image_back_resized, back_name, "resized", ext="tiff", path=output_path)

        print(f"scan_unbordered shape: {scan_unbordered.shape}")
        print(f"image_back_resized shape: {image_back_resized.shape}")

        # image_substracted = scan_unbordered - image_back_resized
        # image_substracted = cv2.subtract(scan_unbordered, image_back_resized)

        # image_substracted = cv2.subtract(scan_unbordered, image_back_resized, dest=None, mask=None, dtype=np.uint8)
        image_substracted = np.subtract(scan_unbordered, image_back_resized)
        saveimage(image_substracted, sample, "substracted", ext="tiff", path=output_path)

        image_substracted2 = np.subtract(image_back_resized, scan_unbordered)
        saveimage(image_substracted2, sample, "substracted2", ext="tiff", path=output_path)

        # background.mean()

