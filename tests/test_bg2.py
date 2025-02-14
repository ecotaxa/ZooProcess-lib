
import unittest
import pytest
import numpy as np

from pathlib import Path
import cv2

from ZooProcess_lib.Background import Background
from ZooProcess_lib.ZooscanProject import ZooscanProject

from ZooProcess_lib.Border import Border

from ZooProcess_lib.img_tools import (
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

    project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
    TP = ZooscanProject(project_folder)

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

    output_path = Path(TP.testfolder)

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_all2(self):

        output_path = Path(self.TP.testfolder,"back")

        # back_name = "20141003_1144_back_large_1.tif" 
        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")

        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        # print(f"file: {back_file}")
        # back_image = loadimage( back_file.as_posix())
        # back_name = "20141003_1144_back_large.tif" 
        # back_file = Path(self.TP.back, self.back_name + "_raw" + "_1" + ".tif")
        # back_image = rotate90c(loadimage( back_file.as_posix()))
        border = Border(back_image)
        border.output_path = output_path
        border.name = self.back_name
        if self.use_raw:
            border.draw_image = rotate90c(loadimage(back_file.as_posix()))
        else:
            border.draw_image = loadimage(back_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()
        print(f"back limite t={limitetop}, b={limitbas}, l={limitegauche}, r={limitedroite}")

        image_back_unbordered = crop(back_image, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        saveimage(image_back_unbordered, self.back_name, "unbordered", ext="tiff", path=output_path)


        # a faire apres
        # background = Background(image_back_unbordered, back_name, output_path=self.TP.testfolder)
        # image_resized = background.redim() 



        # sample = "test_01_tot"
        # rawscan_file = Path(self.TP.rawscan, self.sample + "_raw" + "_1" + ".tif")
        if self.use_raw:
            rawscan_file = Path(self.TP.rawscan, self.sample)
            image = rotate90c(loadimage(rawscan_file.as_posix()))
        else:
            rawscan_file = Path(self.TP.scan, self.sample)
            image = loadimage(rawscan_file.as_posix())
                   
        scan_border = Border(image)
        scan_border.output_path = output_path
        scan_border.name = self.sample
        if self.use_raw:
            scan_border.draw_image = rotate90c(loadimage(rawscan_file.as_posix()))
        else:
            scan_border.draw_image = loadimage(rawscan_file.as_posix())

        # ici regle de 3 pour avir les dimensions pour copuper le scan

        scan_limit_top = image.shape[0] * limitetop / back_image.shape[0]
        scan_limit_bottom = image.shape[0] * limitbas / back_image.shape[0]
        scan_limit_left = image.shape[1] * limitegauche / back_image.shape[1]
        scan_limit_right = image.shape[1] * limitedroite / back_image.shape[1]
        # scan_limit_top, scan_limit_bottom, scan_limit_left, scan_limit_right = scan_border.detect()

        # limitetop, limitbas, limitegauche, limitedroite = scan_border.detect()
        # print(f"scan limite {scan_limite}")
        print(f"scan limite t={scan_limit_top}, b={scan_limit_bottom}, l={scan_limit_left}, r={scan_limit_right}")
        print(f"scan limite top={scan_limit_top}, b={scan_limit_bottom}, l={scan_limit_left}, r={scan_limit_right}")
        
        scan_unbordered = cropnp(image, left=scan_limit_left, top=scan_limit_top, right=scan_limit_right, bottom=scan_limit_bottom)
        saveimage(scan_unbordered, self.sample, "unbordered", ext="tiff", path=output_path)


        image_back_blurred = cv2.medianBlur(image_back_unbordered, 3)

        H = scan_unbordered.shape[0]
        L = scan_unbordered.shape[1]
        # scale factor
        # fx = 
        # fy = 
        interpolation = cv2.INTER_LINEAR
        image_back_resized = cv2.resize( image_back_blurred, dsize=(L, H), interpolation=interpolation)
        saveimage(image_back_resized, self.back_name, "resized", ext="tiff", path=output_path)

        print(f"scan_unbordered shape: {scan_unbordered.shape}")
        print(f"image_back_resized shape: {image_back_resized.shape}")

        # image_substracted = scan_unbordered - image_back_resized
        # image_substracted = cv2.subtract(scan_unbordered, image_back_resized)

        # image_substracted = cv2.subtract(scan_unbordered, image_back_resized, dest=None, mask=None, dtype=np.uint8)
        image_substracted = np.subtract(scan_unbordered, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted", ext="tiff", path=output_path)

        image_substracted2 = np.subtract(image_back_resized, scan_unbordered)
        saveimage(image_substracted2, self.sample, "substracted2", ext="tiff", path=output_path)

        # background.mean()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_subtracted_to8bit(self):
        from to8bit import convertion

        output_path = Path(self.TP.testfolder,"back")

        sample = "test_01_tot"
        substracted_file = Path(getPath(sample , "substracted2", ext="tiff", path=output_path))
        image_substracted = loadimage(substracted_file.as_posix())

        image_back_8bit = convertion(image_substracted, sample, TP = self.TP )
        saveimage(image_back_8bit, sample, "convertion_8bit", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_scan_to8bit(self):
        from to8bit import convertion

        output_path = Path(self.TP.testfolder)

        # sample = "test_01_tot"
        # rawscan_file = Path(self.TP.rawscan, sample + "_raw" + "_1" + ".tif")
        # image = rotate90c(loadimage(rawscan_file.as_posix()))
        if self.use_raw:
            rawscan_file = Path(self.TP.rawscan, self.sample)
            image = rotate90c(loadimage(rawscan_file.as_posix()))
        else:
            rawscan_file = Path(self.TP.scan, self.sample)
            image = loadimage(rawscan_file.as_posix())

        image_back_8bit = convertion(image, self.sample, TP = self.TP )
        saveimage(image_back_8bit, self.sample, "convertion_8bit", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit(self):
        from to8bit import convertion

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        image_back_8bit = convertion(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion_8bit_picheral_img_tools", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit_convert(self):
        from to8bit import convert_s, convert

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        from img_tools import picheral_median

        median,mean = picheral_median(back_image)
        print(f"median: {median}, mean: {mean}")
        median2 = np.median(back_image)
        print(f"median2: {median2}")
        min, max = minAndMax(median)
        print(f"min: {min}, max: {max}")
        min, max = minAndMax(median2)
        print(f"min: {min}, max: {max}")

        min255 = min * 255 / 65536
        max255 = max * 255 / 65536

        print(f"min255: {int(min255)}, max255: {int(max255)}")

        min255 = 6
        max255 = 254

        print(f"min255: {int(min255)}, max255: {int(max255)}")

        # image_back_8bit = convert(back_image, int(min255), int(max255), np.uint8)
        # image_back_8bit = convert(back_image, 0, 254, np.uint8)
        image_back_8bit = convert(back_image, int(min), int(max), np.uint8)

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion_8bit_convert_s_forced_hacked", ext="tiff", path=output_path)

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_scan_to8bit_convert(self):
        from to8bit import convert_s, convert, convert_forced

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            rawscan_file = Path(self.TP.rawscan, self.sample)
            image = rotate90c(loadimage(rawscan_file.as_posix()))
        else:
            rawscan_file = Path(self.TP.scan, self.sample)
            image = loadimage(rawscan_file.as_posix())

        from img_tools import picheral_median

        median,mean = picheral_median(image)

        print(f"median: {median}, mean: {mean}")
        median2 = np.median(image)
        print(f"median2: {median2}")
        min, max = minAndMax(median)
        print(f"min: {min}, max: {max}")
        min, max = minAndMax(median2)
        print(f"min: {min}, max: {max}")

        min255 = min * 255 / 65536
        max255 = max * 255 / 65536
    

        print(f"min255: {int(min255)}, max255: {int(max255)}")

        # min255 = 6
        max255 = 254

        # print(f"min255: {int(min255)}, max255: {int(max255)}")

        image_back_8bit = convert_forced(image, int(min255), int(max255), np.uint8)
        # image_back_8bit = convert(image, 0, 254, np.uint8)
        # image_back_8bit = convert(image, int(min), int(max), np.uint8)

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.sample, "convertion_8bit_convert_s_forced_picheral", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit_convert_mm(self):
        from to8bit import convert_mm

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        from img_tools import picheral_median

        median,mean = picheral_median(back_image)
        print(f"median: {median}, mean: {mean}")
        median2 = np.median(back_image)
        print(f"median2: {median2}")
        min, max = minAndMax(median2)
        print(f"min: {min}, max: {max}")

        # min255 = min * 255 / 65536
        # max255 = max * 255 / 65536

        # print(f"min255: {int(min255)}, max255: {int(max255)}")

        # min255 = 6
        # max255 = 254

        # print(f"min255: {int(min255)}, max255: {int(max255)}")


        # image_back_8bit = convert(back_image, int(min255), int(max255), np.uint8)
        image_back_8bit = convert_mm(back_image, 0, 255, min, max, np.uint8)

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion_8bit_convert_forced_0_255", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_scan_to8bit_convert_mm(self):
        from to8bit import convert_mm

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        # if self.use_raw:
        #     back_image = rotate90c(loadimage( back_file.as_posix()))
        # else:
        #     back_image = loadimage( back_file.as_posix())
        
        rawscan_file = Path(self.TP.rawscan, self.sample)
        image = rotate90c(loadimage(rawscan_file.as_posix()))

        from img_tools import picheral_median

        median,mean = picheral_median(image)
        print(f"median: {median}, mean: {mean}")
        median2 = np.median(image)
        print(f"median2: {median2}")
        min, max = minAndMax(median2)
        print(f"min: {min}, max: {max}")

        min255 = min * 255 / 65536
        max255 = max * 255 / 65536

        # print(f"min255: {int(min255)}, max255: {int(max255)}")

        # min255 = 6
        # max255 = 254

        # print(f"min255: {int(min255)}, max255: {int(max255)}")


        image_back_8bit = convert_mm(image, int(min255), int(max255), np.uint8)
        # image_back_8bit = convert_mm(image, 0, 255, min, max, np.uint8)

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        # saveimage(image_back_8bit, self.sample, "convertion_8bit_convert_forced" + "_" + str(min)  "_" str(max), ext="tiff", path=output_path)
        saveimage(image_back_8bit, self.sample, "convertion_8bit_convert_forced", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit_convert2(self):
        """
        bug in convert2
        """
        from to8bit import convert2

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        from img_tools import picheral_median

        median,mean = picheral_median(back_image)
        min, max = minAndMax(median)
        print(f"min: {min}, max: {max}")

        image_back_8bit = convert2(back_image, min, max )

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion_8bit_convert2", ext="tiff", path=output_path)




    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit_convertion2(self):
        from to8bit import convertion2

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        image_back_8bit = convertion2(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion2_8bit", ext="tiff", path=output_path)

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_resize_back(self):
        output_path = Path(self.TP.testfolder)




    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_substract(self):

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.testfolder, self.back_name)
        back_image = loadimage( self.back_name, "convertion_8bit", ext="tiff", path=output_path)

        # rawscan_file = Path(self.TP.rawscan, self.sample + "_raw" + "_1" + ".tif")
        image = loadimage(self.sample , "convertion_8bit", ext="tiff", path=output_path)


        H = image.shape[0]
        L = image.shape[1]
        # scale factor
        # fx = 
        # fy = 
        interpolation = cv2.INTER_LINEAR
        image_back_resized = cv2.resize( back_image, dsize=(L, H), interpolation=interpolation)
        saveimage(image_back_resized, self.back_name, "resized", ext="tiff", path=output_path)

        print(f"scan_unbordered shape: {image.shape}")
        print(f"image_back_resized shape: {image_back_resized.shape}")


        image_substracted = np.subtract(image, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted_from_8bit", ext="tiff", path=output_path)

        image_substracted2 = np.subtract(image_back_resized, image)
        saveimage(image_substracted2, self.sample, "substracted2_from_8bit", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_back_to8bit_convert_s(self):
        from to8bit import convert_s

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.back, self.back_name)
        # back_image = loadimage( back_file.as_posix())

        back_file = Path(self.TP.back, self.back_name)
        print(f"file: {back_file}")
        if self.use_raw:
            back_image = rotate90c(loadimage( back_file.as_posix()))
        else:
            back_image = loadimage( back_file.as_posix())

        from img_tools import picheral_median

        median,mean = picheral_median(back_image)
        print(f"median: {median}, mean: {mean}")
        median2 = np.median(back_image)
        print(f"median2: {median2}")
        min, max = minAndMax(median)
        print(f"min: {min}, max: {max}")

        min255 = min * 255 / 65536
        max255 = max * 255 / 65536

        print(f"min255: {int(min255)}, max255: {int(max255)}")

        # min255 = 6
        # max255 = 254
        # print(f"min255: {int(min255)}, max255: {int(max255)}")


        # image_back_8bit = convert_s(back_image, int(min255), int(max255), np.uint8)
        image_back_8bit = convert_s(back_image, int(min), int(max), np.uint8)

        # image_back_8bit = convert(back_image, self.back_name, TP = self.TP )
        saveimage(image_back_8bit, self.back_name, "convertion_8bit_convert_forced", ext="tiff", path=output_path)


    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_substract(self):

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.testfolder, self.back_name)
        back_image = loadimage( self.back_name, "convertion_8bit_convert_forced_0_255", ext="tiff", path=output_path)

        # rawscan_file = Path(self.TP.rawscan, self.sample + "_raw" + "_1" + ".tif")
        image = loadimage(self.sample , "convertion_8bit_convert_forced_0_255", ext="tiff", path=output_path)


        H = image.shape[0]
        L = image.shape[1]
        # scale factor
        # fx = 
        # fy = 
        interpolation = cv2.INTER_LINEAR
        image_back_resized = cv2.resize( back_image, dsize=(L, H), interpolation=interpolation)
        saveimage(image_back_resized, self.back_name, "resized", ext="tiff", path=output_path)

        print(f"scan_unbordered shape: {image.shape}")
        print(f"image_back_resized shape: {image_back_resized.shape}")


        image_substracted = np.subtract(image, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted_from_8bit_convert_forced_0_255", ext="tiff", path=output_path)

        image_substracted2 = np.subtract(image_back_resized, image)
        saveimage(image_substracted2, self.sample, "substracted2_from_8bit_convert_forced_0_255", ext="tiff", path=output_path)




    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_substract_convert_s_forced_hacked(self):

        output_path = Path(self.TP.testfolder)

        # back_file = Path(self.TP.testfolder, self.back_name)
        back_image = loadimage( self.back_name, "convertion_8bit_convert_s_forced_hacked", ext="tiff", path=output_path)

        # rawscan_file = Path(self.TP.rawscan, self.sample + "_raw" + "_1" + ".tif")
        image = loadimage(self.sample , "convertion_8bit_convert_s_forced_hacked", ext="tiff", path=output_path)


        H = image.shape[0]
        L = image.shape[1]
        # scale factor
        # fx = 
        # fy = 
        interpolation = cv2.INTER_LINEAR
        image_back_resized = cv2.resize( back_image, dsize=(L, H), interpolation=interpolation)
        saveimage(image_back_resized, self.back_name, "resized", ext="tiff", path=output_path)

        print(f"scan_unbordered shape: {image.shape}")
        print(f"image_back_resized shape: {image_back_resized.shape}")


        image_substracted = np.subtract(image, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted_from_8bit_convert_s_forced_hacked", ext="tiff", path=output_path)

        image_substracted2 = np.subtract(image_back_resized, image)
        saveimage(image_substracted2, self.sample, "substracted2_from_8bit_convert_s_forced_hacked", ext="tiff", path=output_path)


    # @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_rr(self):

        from to8bit import resize

        # DAPI = loadimage(self.sample , "convertion_8bit_convert_s_forced_hacked", ext="tiff", path=self.output_path)
        DAPI = loadimage(self.sample, path=self.TP.rawscan)

        min = 6
        max = 254

        DAPI_8bit_d = cv2.normalize(DAPI, None, min, max, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # plt.imshow(DAPI_8bit_d, cmap='gray')
        saveimage(DAPI_8bit_d, self.sample, "DAPI_8bit_d", ext="tiff", path=self.output_path)

        back = loadimage(self.back_name,path=self.TP.back)
        back_8bit_d = cv2.normalize(back, None, min, max, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        saveimage(back_8bit_d, self.sample, "DAPI_Back_8bit_d", ext="tiff", path=self.output_path)

        image_back_resized = resize(DAPI, back_8bit_d)
        saveimage(image_back_resized, self.back_name, "resized", ext="tiff", path=self.output_path)

        image_substracted = np.subtract(DAPI_8bit_d, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted_from_DAPI", ext="tiff", path=self.output_path)

        image_substracted2 = np.subtract(image_back_resized, DAPI_8bit_d)
        saveimage(image_substracted2, self.sample, "substracted2_from_DAPI", ext="tiff", path=self.output_path)


    def test_applied_back_filter_on_scan(self):

        from to8bit import resize, filters

        scan_image = loadimage(self.sample, path=self.TP.rawscan)
        back_image = loadimage(self.back_name,path=self.TP.back)

        imin,imax,min,max = filters(scan_image)
        print( f"imin: {imin}, imax: {imax} - min: {min}, max: {max}" )

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # print(f"a: {a}, b: {b}")
        back_image_8bit = (a * back_image + b).astype(np.uint8)
        scan_image_8bit = (a * scan_image + b).astype(np.uint8)

        image_back_resized = resize(scan_image_8bit, back_image_8bit)

        image_substracted = np.subtract(scan_image_8bit, image_back_resized)
        saveimage(image_substracted, self.sample, "substracted_back_filter", ext="tiff", path=self.output_path)

        image_substracted2 = np.subtract(image_back_resized, scan_image_8bit)
        saveimage(image_substracted2, self.sample, "substracted2_back_filter", ext="tiff", path=self.output_path)

        image_back_rotated = rotate90c(image_back_resized)
        image_scan_rotated = rotate90c(image_substracted2)

        saveimage(image_back_rotated, self.back_name, "image_back_rotated", ext="tiff", path=self.output_path)
        saveimage(image_scan_rotated, self.sample, "image_scan_rotated", ext="tiff", path=self.output_path)

        # ajouter un flip horizontal pour que l'utilsateur voit l'image comme son scan

        border = Border(image_back_rotated)
        border.output_path = self.output_path
        border.name = self.back_name
        # reload the image_back_rotated to don't mutated the original (a copy is need to work)
        # make a byte copy will probably most efficiant
        back_file = Path(getPath(self.back_name, extraname="image_back_rotated", ext="tiff", path=self.output_path))
        border.draw_image = loadimage(back_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()
        print(f"back limite t={limitetop}, b={limitbas}, l={limitegauche}, r={limitedroite}")

        # on se fiche de cette image
        # image_back_unbordered = crop(image_back_rotated, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        # saveimage(image_back_unbordered, self.back_name, "unbordered", ext="tiff", path=self.output_path)

        image_scan_unbordered = crop(image_scan_rotated, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        saveimage(image_scan_unbordered, self.sample, "unbordered", ext="tiff", path=self.output_path)

        # transforme en masque (binarisation par seuillage)
        
        # ret,mask = cv2.threshold(image_scan_unbordered,100,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_min = 0 # 225 # 237 # 200 # 243 # 220 # 243 # 200 # 126 # 243 # 0 # 100
        thresh_max = 255 # 241 # 250 # 243 # 255 # 243
        th, mask = cv2.threshold(image_scan_unbordered,thresh_min,thresh_max,cv2.THRESH_BINARY)
        saveimage(mask, self.sample, "unbordered_mask" + "_" + str(thresh_min) + "_" + str(thresh_max), ext="tiff", path=self.output_path)
        print(f"th: {th}")

        th, mask = cv2.threshold(image_scan_unbordered,thresh_min,thresh_max,cv2.THRESH_OTSU)
        saveimage(mask, self.sample, "unbordered_otsu" + "_" + str(thresh_min) + "_" + str(thresh_max), ext="tiff", path=self.output_path)
        print(f"th: {th}")

        th, mask = cv2.threshold(image_scan_unbordered,thresh_min,thresh_max,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        saveimage(mask, self.sample, "unbordered_bin_inv_otsu" + "_" + str(thresh_min) + "_" + str(thresh_max), ext="tiff", path=self.output_path)
        print(f"th: {th}")

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of Contours found = " + str(len(contours)))

        image_3channels = draw_contours(image_scan_unbordered, contours)
        saveimage(image_3channels,self.sample, "draw_contours_on_image", path=self.output_path)

        white_mask = np.full(mask.shape[:2],255, np.uint8)
        image_3channels = draw_contours(white_mask, contours)
        saveimage(image_3channels, self.sample, "draw_contours", path=self.output_path)

        organism_size = 50

        # acceptable size
        def filter(h,w)-> bool:
            if h < organism_size and w < organism_size: return False
            return True

        image_3channels = draw_boxes_filtered(image_scan_unbordered, contours,filter)
        saveimage(image_3channels, self.sample, "draw_boxes_filtered_on_image", path=self.output_path)

        image_3channels = draw_boxes_filtered(white_mask, contours,filter, add_number=True)
        saveimage(image_3channels, self.sample, "draw_boxes_filtered", path=self.output_path)

        vignettepath = Path(self.output_path,"vignettes")
        filelist = generate_vignettes(image_scan_unbordered,contours,filter, path=vignettepath)