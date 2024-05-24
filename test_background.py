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
        output_path = Path(self.TP.testfolder,"back")
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


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_subtracted_to8bit(self):
        from to8bit import convertion

        output_path = Path(self.TP.testfolder,"back")

        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        image_substracted = loadimage(substracted_file.as_posix())

        image_back_8bit = convertion(image_substracted, sample, TP = self.TP )
        saveimage(image_back_8bit, sample, "convertion_8bit", ext="jpg", path=output_path)

    
    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_togray(self):

        output_path = Path(self.TP.testfolder,"back")

        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        image_gray = loadimage(substracted_file.as_posix())

        saveimage(image_gray, sample, "gray", ext="jpg", path=output_path)


        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_saturation(self):
        """
        test mais c'est nul
        """
        from PIL import Image
        import numpy as np
    
        Image.MAX_IMAGE_PIXELS = 375000000

        output_path = Path(self.TP.testfolder,"back")
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"

        print("test_saturation")

        # Build uint16 grayscale Tiff image out of uint8 JPEG image, for testing.
        ################################################################################

        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        print(f"substracted_file:", substracted_file.as_posix())

        img8 = Image.open(substracted_file.as_posix()).convert('L')  # Read image and convert to Grayscale.
        img16 = Image.fromarray(np.asarray(img8).astype(np.uint16)*255)
        path = Path(getPath(sample , "uint16_img", ext="tiff", path=output_path))
        # img16.save('uint16_img.tif')
        img16.save(path.as_posix())
        ################################################################################

        # uint16_img = Image.open('uint16_img.tif')  # Read input image (uint16 grayscale)
        uint16_img = Image.open(path.as_posix())  # Read input image (uint16 grayscale)
        data = np.asarray(uint16_img)  # Convert to NumPy array - assume data applies self.data from the question

        # Select a threshold, that above it, a pixel is considered saturated
        # The threshold is specific to the camera (may by 4000 for example - we can't know from the correct value from question).
        saturation_threshold = 64000

        # Build a mask (NumPy array) with True where images is saturated, and False where not.
        sat_mask = data > saturation_threshold

        imax2 = np.amax(data)  # 64770

        if imax2 > 0:
            scale = 255.0/imax2
            #data2 = (data.astype(float) * scale).astype('u2')
            # For making the solution less confusing, convert data2 to uint8 type
            # When displaying and image, uint8 is prefered, since the display uses 8 bits per color channel (unless using HDR, but that irrelevant for the answer).
            data2 = (data.astype(float) * scale).astype('uint8')
        else:
            data2 = data2.astype('uint8')

        # Converting from single channel (grayscale) to RGB (3 color channels) where red=green=blue (we need RGB for marking with red).
        data2_rgb = np.dstack((data2, data2, data2))
        #sat_mask = np.dstack((sat_mask, sat_mask, sat_mask))

        # Put red color where mask is True
        data2_rgb[sat_mask] = np.array([255, 0, 0], np.uint8)  # [255, 0, 0] applies red color, in case the image is uint16, use [65535, 0, 0].

        path = Path(getPath(sample , "data2_rgb", ext="png", path=output_path))
        Image.fromarray(np.asarray(data2_rgb)).save(path.as_posix())  # Save image for testing

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_saturation2(self):
        """
        la partie saturation fonctionne mais pas le passage en gray
        en meme temps l'image est deja en gray
        """
        from PIL import Image
        import numpy as np
    
        Image.MAX_IMAGE_PIXELS = 375000000

        output_path = Path(self.TP.testfolder,"back")
        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"

        print("test_saturation")

        # Build uint16 grayscale Tiff image out of uint8 JPEG image, for testing.
        ################################################################################

        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        print(f"substracted_file:", substracted_file.as_posix())

        # img8 = Image.open(substracted_file.as_posix()).convert('L')  # Read image and convert to Grayscale.
        # img16 = Image.fromarray(np.asarray(img8).astype(np.uint16)*255)
        # path = Path(getPath(sample , "uint16_img", ext="tiff", path=output_path))
        # # img16.save('uint16_img.tif')
        # img16.save(path.as_posix())
        # ################################################################################

        # uint16_img = Image.open('uint16_img.tif')  # Read input image (uint16 grayscale)
        # uint16_img = Image.open(path.as_posix())  # Read input image (uint16 grayscale)
        uint16_img = Image.open(substracted_file.as_posix())  # Read input image (uint16 grayscale)
        data = np.asarray(uint16_img)  # Convert to NumPy array - assume data applies self.data from the question

        # Select a threshold, that above it, a pixel is considered saturated
        # The threshold is specific to the camera (may by 4000 for example - we can't know from the correct value from question).
        saturation_threshold = 25000

        # Build a mask (NumPy array) with True where images is saturated, and False where not.
        sat_mask = data > saturation_threshold

        imax2 = np.amax(data)  # 64770

        if imax2 > 0:
            scale = 255.0/imax2
            #data2 = (data.astype(float) * scale).astype('u2')
            # For making the solution less confusing, convert data2 to uint8 type
            # When displaying and image, uint8 is prefered, since the display uses 8 bits per color channel (unless using HDR, but that irrelevant for the answer).
            data2 = (data.astype(float) * scale).astype('uint8')
        else:
            data2 = data2.astype('uint8')

        # Converting from single channel (grayscale) to RGB (3 color channels) where red=green=blue (we need RGB for marking with red).
        data2_rgb = np.dstack((data2, data2, data2))
        #sat_mask = np.dstack((sat_mask, sat_mask, sat_mask))

        # Put red color where mask is True
        data2_rgb[sat_mask] = np.array([255, 0, 0], np.uint8)  # [255, 0, 0] applies red color, in case the image is uint16, use [65535, 0, 0].

        path = Path(getPath(sample , "data2_rgb" + "_" + str(saturation_threshold), ext="png", path=output_path))
        Image.fromarray(np.asarray(data2_rgb)).save(path.as_posix())  # Save image for testing


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_split_color(self):

        output_path = Path(self.TP.testfolder,"back")

        sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        # image_gray = loadimage(substracted_file.as_posix())

        image = cv2.imread(substracted_file.as_posix())
        b,g,r = cv2.split(image)

        saveimage(b, sample, "blue", ext="jpg", path=output_path)
        saveimage(g, sample, "green", ext="jpg", path=output_path)
        saveimage(r, sample, "red", ext="jpg", path=output_path)


   