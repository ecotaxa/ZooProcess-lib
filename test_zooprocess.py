
import unittest
from ZooscanFolder import Zooscan_Project, Zooscan_sample_scan

from ProjectClass import ProjectClass
from zooprocess import ZooProcess

from tools import is_file_exist
from pathlib import Path

import pytest

import numpy as np
import cv2
import os

import Border

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
        draw_box
    )

debug = False

class Test_ZooProcess(unittest.TestCase):

    # project_folder = "Zooscan_dyfamed_wp2_2023_biotom_sn001"
    project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
    TP = ProjectClass(project_folder)

    def test_openproject(self):

        ut = ZooProcess(self.TP.home,self.TP.project_name)

        self.assertEqual(ut.path, self.TP.home)
        self.assertEqual(ut.name, self.TP.project_name)

        self.assertIsInstance(ut.project, Zooscan_Project)

        check = ut.check_folders()
        self.assertEqual(check,True,"Folder do not exist")


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_convert_raw(self):

        output_path = self.TP.testfolder
        # sample = "dyfamed_20230111_200m_d1"
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        # scan_file = Path(self.TP.folder, "Zooscan_scan", "_work", sample + "_1" + ".tif")
        scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        # /Users/sebastiengalvagno/piqv/plankton/zooscan_monitoring/Zooscan_dyfamed_wp2_2023_biotom_sn001/Zooscan_scan

        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")

        ut = ZooProcess(self.TP.home, self.TP.project_name, output_path)

        # scan = Zooscan_sample_scan(self.TP.project_name, 1, self.TP.folder)
        # self.assertEqual(len(scans),2,"#of scans are not equal")
        
        # image = ut.normalize_rawscan(scans[0])
        image = ut.normalize_rawscan(rawscan_file)

        scanfile_exist = is_file_exist(scan_file)
        self.assertEqual(scanfile_exist, True, f"scanfile_exist do not exist.\n-----------------------\n{scan_file}\n-----------------------\n")

    def test_background(self):
        print("do nothing")
        pass


   
    # Zooscan_1asep.txt:396 
    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_bord_gauche(self):

        resolution = 2400
        step = resolution / 240

        output_path = self.TP.testfolder
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]

        print(f'image size width: {width} height: {height}')

        limit = width
        limitgauche = limit
        k = width * 0.05
        print(f"-- k: {k}")
        # a = 0
        # makeRectangle(k,height/2 - height*0.125, 10 * step,height*0.25);

        if debug: print(f"dim: {k}, {height/2 - height*0.125}, {step}, {height*0.25}")

        img = crophw(image, top=k, left=height/2 - height*0.125, height=step, width=height*0.25)

        draw_img = cv2.merge([image, image, image])
        draw_image = draw_box(draw_img, x=k, y=height/2 - height*0.125, h=height*0.25, w=step, color=(255,0,0)) 
        # saveimage(draw_image, sample, "draw", ext="tiff", path=output_path)

        if debug: print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=output_path)

        print(f"Mean of cropped image: {mean}")
        greytaux = 0.9
        greyvalb = int(mean * greytaux)
        print(f"greyvalb: {greyvalb}")

        while k > 0:
            if debug: print(f"dim: {k}, {height/2 - height*0.125}, {step}, {height*0.25}")
            # img = crophw(image, top=k, left=height/2 - height*0.125, width=step, height=height*0.25)
            img = crophw(image, top=k, left=height/2 - height*0.125, height=step, width=height*0.25)
            # print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=output_path)

            if debug: print(f"dim: {k}, {height/2 - height*0.125}, {step}, {height*0.25} mean:{mean} <? {greyvalb}")
            draw_image = draw_box(draw_img, x=k, y=height/2 - height*0.125, h=height*0.25, w=step, color=(255,0,0)) 
            # print(f"Mean of cropped image: {mean}")
            # if mean < mean * 0.95: ??????? où j'ai pris ça ?????
            if mean < greyvalb:
                limit = k
                limit = limit + step
                limitgauche = int(limit)
                # k = 0
                break
            # a += 1
            k = k - step

        print(f"Limit left: {limitgauche}")
        saveimage(draw_image, sample, "drew", ext="tiff", path=output_path)



    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_bord_droit(self):

        resolution = 2400
        step = resolution/240

        output_path = self.TP.testfolder
        crop_output_path = os.path.join(output_path, "crops")
        if not os.path.exists(crop_output_path):
            os.makedirs(crop_output_path)

        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]

        print(f'image size width: {width} height: {height}')

        limit = width
        limitedroite = limit
        k = max( width * 0.75 , width - resolution/2 )
        print(f"k: {k}")

        img = crophw(image, top=k, left=int(height/4), width=int(height*0.25), height=int(10 * step))
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        draw_img = loadimage(rawscan_file.as_posix())
        draw_image = draw_box(draw_img, x=k, y=height/4, h=height*0.25, w=step, color=(255,0,0)) 

        saveimage(img, sample, "crop_right" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)

        print(f"Mean of cropped image: {mean}")
        greytaux = 0.9
        greyvalb = int(mean * greytaux)

        while k <= width :
            img = crophw(image, top=k, left=int(height/4), width=int(height*0.25), height=int(step))
            print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            print(f"Mean of cropped image: {mean}")
            draw_image = draw_box(draw_image, x=k, y=height/4, h=height*0.25, w=step, color=(255,0,0)) 
            if mean < greyvalb: # on arrete
                limit = k
                limit = limit - step
                limitedroite = int(limit)
                # k = width
                break
            k = k + step

        print(f"Limit right: {limitedroite}")


    @pytest.mark.skip(reason="Skipping this taest for now because of XYZ reason.")  
    def test_bord_bas(self):

        resolution = 2400
        step = resolution/240

        output_path = self.TP.testfolder
        crop_output_path = os.path.join(output_path, "crops")
        if not os.path.exists(crop_output_path):
            os.makedirs(crop_output_path)
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]

        print(f'image size width: {width} height: {height}')

        limit = height
        limitbas = limit
        k = height * 0.95
        print(f"k: {k}")
        img = crophw(image, top=width / 6, left=k, height=width*0.15, width=step)
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        print(f"Mean of cropped image: {mean}")
        saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)

        draw_img = loadimage(rawscan_file.as_posix())
        draw_image = draw_box(draw_img, y=k, x=width/6, w=width*0.15, h=step, color=(255,0,0)) 


        greytaux = 0.9
        greyvalb = int(mean * greytaux)
        print(f"greyvalb: {greyvalb}")

        while k <= height :
            img = crophw(image, width / 6, k, width*0.15, step)
            print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            print(f"Mean of cropped image: {mean}")
            saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)
            draw_image = draw_box(draw_image, y=k, x=width/6, w=width*0.15, h=step, color=(255,0,0)) 
            if mean < greyvalb: # on arrete
                limit = k
                limit = limit - step
                limitebas = int(limit)
                # k = width
                break
            k = k + step

        print(f"Limit bottom: {limitebas}")
        saveimage(draw_image, sample, "drew", ext="tiff", path=output_path)

    @pytest.mark.skip(reason="Skipping this taest for now because of XYZ reason.")  
    def test_bord_haut(self):
        resolution = 2400
        step = resolution/240

        output_path = self.TP.testfolder
        crop_output_path = os.path.join(output_path, "crops")
        if not os.path.exists(crop_output_path):
            os.makedirs(crop_output_path)
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]

        print(f'image size width: {width} height: {height}')

        limit = height
        limitetop = limit
        k = height * 0.05
        print(f"k: {k}")
        # img = crophw(image, left=width / 2 - width * .25, top=k, width=width * 0.2, height=10 * step)
        # img = crophw(image, top=width / 2 - width * .25, left=k, height=width * 0.2, width=10 * step)
        img = crophw(image, top=width / 2 - width * .25, left=k, height=width * 0.2, width=10 * step)
        # img = crophw(image, top=height / 2 - height * .25, left=k, width=height * 0.2, height=10 * step)
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        print(f"Mean of cropped image: {mean}")
        saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)
        # draw_img = cv2.merge([image, image, image])
        draw_img = loadimage(rawscan_file.as_posix())
        # draw_image = draw_box(draw_img, x=k, y=height/2 - height*0.125, w=height*0.25, h=step, color=(255,0,0)) 
        draw_image = draw_box(draw_img, y=k, x=width/2 - width/8, w=width*0.25, h=step, color=(255,0,0)) 

        greytaux = 0.9
        greyvalb = int(mean * greytaux)
        print(f"greyvalb: {greyvalb}")

        while k > 0 :
            # img = crophw(image, left=width / 2 - width * .25, top=k, width=width * 0.2, height=step)
            img = crophw(image, top=width / 2 - width * .25, left=k, height=width * 0.2, width=step)
            # img = crophw(image, top=height / 2 - height * .25, left=k, width=height * 0.2, height=step)

            print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            print(f"Mean of cropped image: {mean}")
            saveimage(img, sample, "crop" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)
            # draw_image = draw_box(draw_img, x=k, y=height/2 - height*0.125, w=height*0.25, h=step, color=(255,0,0)) 
            draw_image = draw_box(draw_image, y=k, x=width/2 - width/8, w=width/4, h=step, color=(255,0,0)) 

            if mean < greyvalb: # or mean == 255: # on arrete
                limit = k
                limit = limit + step #(width-limit)/10
                limitetop = int(limit)
                # k = width
                break
            k = k - step

        print(f"Limit top: {limitetop}")
        saveimage(draw_image, sample, "drew", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_border(self):

        # resolution = 2400
        # step = resolution/240

        output_path = self.TP.testfolder
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        # scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]
        print(f'image size width: {width} height: {height}')

        border = Border.Border(image)
        border.output_path = output_path
        border.name = sample
        border.draw_image = loadimage(rawscan_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()

        print("-----")
        print(f"Limit top: {limitetop}")
        print(f"Limit bottom: {limitbas}")
        print(f"limit left: {limitegauche}")
        print(f"Limit right: {limitedroite}")

        img = crop(image, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        # img = cropnp(image, top=limitetop, left=limitegauche, bottom=limitbas, right=limitedroite)
        print(f"shape of cropped image: {img.shape}")
        self.assertGreater(img.shape[0], 0, "height is null")
        self.assertGreater(img.shape[1], 0 , "width is null")
        saveimage(img, sample, "cropped", ext="tiff", path=output_path)
        # saveimage(image, sample, "image", ext="tiff", path=output_path)


    #@pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_border_background(self):
        # resolution = 2400
        # step = resolution/240

        output_path = self.TP.testfolder
        # sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        sample = "20240112_1518_back_large" 

        # scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_back", sample + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]
        print(f'image size width: {width} height: {height}')

        border = Border.Border(image)
        border.output_path = output_path
        border.name = sample
        border.draw_image = loadimage(rawscan_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()

        print("-----")
        print(f"Limit top: {limitetop}")
        print(f"Limit bottom: {limitbas}")
        print(f"limit left: {limitegauche}")
        print(f"Limit right: {limitedroite}")

        # img = crop(image, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        img = cropnp(image, top=limitetop, left=limitegauche, bottom=limitbas, right=limitedroite)
        print(f"shape of cropped image: {img.shape}")
        self.assertGreater(img.shape[0], 0, "height is null")
        self.assertGreater(img.shape[1], 0 , "width is null")
        saveimage(img, sample, "cropped", ext="tiff", path=output_path)
        # saveimage(image, sample, "image", ext="tiff", path=output_path)
        

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_left_border_background(self):

        output_path = self.TP.testfolder
        # sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        sample = "20240112_1518_back_large" 

        # scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_back", sample + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]
        print(f'image size width: {width} height: {height}')

        border = Border.Border(image)
        border.output_path = output_path
        border.name = sample
        border.draw_image = loadimage(rawscan_file.as_posix())

        left_limit = border.left_limit()
        print(f"left: {left_limit}")


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_bord_droit_background(self):

        resolution = 2400
        step = resolution/240

        output_path = self.TP.testfolder
        crop_output_path = os.path.join(output_path, "crops")
        if not os.path.exists(crop_output_path):
            os.makedirs(crop_output_path)
        
        sample = "20240112_1518_back_large" 
        scan_file = Path(self.TP.folder, "Zooscan_scan", sample + "_1" + ".tif")

        rawscan_file = Path(self.TP.folder, "Zooscan_back", sample + "_1" + ".tif")
        print(f"rawscan_file: {rawscan_file}")

        image = loadimage(rawscan_file.as_posix())
        height = image.shape[0]
        width = image.shape[1]

        print(f'image size width: {width} height: {height}')

        limit = width
        limitedroite = limit
        k = max( width * 0.75 , width - resolution/2 )
        print(f"k: {k}")
        img = crophw(image, top=k, left=int(height/4), height=int(height*0.25), width=int(10 * step))
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        saveimage(img, sample, "crop_right_back" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)
        print(f"Mean of cropped image: {mean}")
        draw_img = loadimage(rawscan_file.as_posix())
        # draw_image = draw_box(draw_img, x=k, y=height/2 - height*0.125, w=height*0.25, h=step, color=(255,0,0)) 
        draw_image = draw_box(draw_img, y=height/4, x=k, h=height/4, w=step, color=(255,0,0)) 

        greytaux = 0.5
        greyvalb = int(mean * greytaux)

        while k <= width :
            img = crophw(image, k, int(height/4), int(height*0.25), int(step))
            print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            print(f"Mean of cropped image: {mean}")
            saveimage(img, sample, "crop_right_back" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=crop_output_path)
            draw_image = draw_box(draw_image, y=height/4, x=k, h=height/4, w=step, color=(255,0,0)) 
            if mean < greyvalb: # on arrete
                limit = k
                limit = limit - step
                limitedroite = int(limit)
                # k = width
                break
            k = k + step

        print(f"Limit right: {limitedroite}")
        saveimage(draw_image, sample, "drew_right", ext="tiff", path=output_path)
