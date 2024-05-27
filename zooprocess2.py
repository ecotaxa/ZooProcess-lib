

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


class zooprocessv10:

    """
    How to use it:

    Define your project location
    ```
    project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
    TP = ProjectClass(project_folder)
    ```

    Define the background and the scan to analyze:
    ```
    scan_name = "test_01_tot"
    bg_name = "20141003_1144_back_large"
    ```

    Instantiate the class:
    ```
    zooprocess = zooprocessv10(TP, scan_name, bg_name)
    ```
    
    to debug: and write the file in a particular folder
    overwrite
    ```
    zooprocess.output_path = TP.testfolder
    ```

    Run the analyze:
    zooprocess.process()

    """

    # back_name = "20141003_1144_back_large_1.tif" 
    # back_name = "20141003_1144_back_large" 
    # sample = "test_01_tot"

    use_raw = True
    use_average = True

    # bg_name = "20141003_1144_back_large"
    # name = "test_01_tot"
    # back_name = "20141003_1144_back_large_raw_1.tif" 
    # sample = "test_01_tot_raw_1.tif"
    # back_name = bg_name + "_raw" + "_1" + ".tif" 
    # sample = name + "_raw" + "_1" + ".tif"

    # output_path = Path(self.TP.testfolder)

    def __init__(self, TP: ProjectClass, scan_name, back_name):
        self.TP = TP
        self.name = scan_name
        self.bg_name = back_name

        self.sample = self.name + "_raw" + "_1" + ".tif"
        self.back_name = self.bg_name + "_raw" + "_1" + ".tif" 
        self.back_name_2 = self.bg_name + "_raw" + "_2" + ".tif" 

        output_path = Path(self.TP.testfolder)


    def background(self,scan_image, imin,imax,min,max):
        from to8bit import resize

        back_image = loadimage(self.back_name, path=self.TP.back)

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # print(f"a: {a}, b: {b}")
        back_image_8bit = (a * back_image + b).astype(np.uint8)
        scan_image_8bit = (a * scan_image + b).astype(np.uint8)
        image_back_resized = resize(scan_image_8bit, back_image_8bit)
        
        return image_back_resized

    def background_average(self,scan_image, imin,imax,min,max):

        from to8bit import resize

        back_image = loadimage(self.back_name, path=self.TP.back)
        back_image_2 = loadimage(self.back_name_2, path=self.TP.back)

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # print(f"a: {a}, b: {b}")
        back_image_8bit = (a * back_image + b).astype(np.uint8)
        back_image_2_8bit = (a * back_image_2 + b).astype(np.uint8)

        scan_image_8bit = (a * scan_image + b).astype(np.uint8)

        image_back_resized = resize(scan_image_8bit, back_image_8bit)
        image_back_2_resized = resize(scan_image_8bit, back_image_2_8bit)

        image_back_median_resized = (image_back_resized / 2 + image_back_2_resized / 2 ).astype(np.uint8)
        
        return image_back_median_resized


    def process(self):

        from to8bit import resize, filters

        scan_image = loadimage(self.sample, path=self.TP.rawscan)
        # back_image = loadimage(self.back_name, path=self.TP.back)
        # back_image_2 = loadimage(self.back_name_2, path=self.TP.back)

        imin,imax,min,max = filters(scan_image)
        print( f"imin: {imin}, imax: {imax} - min: {min}, max: {max}" )

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # print(f"a: {a}, b: {b}")
        # back_image_8bit = (a * back_image + b).astype(np.uint8)
        # back_image_2_8bit = (a * back_image_2 + b).astype(np.uint8)

        scan_image_8bit = (a * scan_image + b).astype(np.uint8)

        # image_back_resized = resize(scan_image_8bit, back_image_8bit)
        # image_back_2_resized = resize(scan_image_8bit, back_image_2_8bit)

        # image_back_median_resized = (image_back_resized + image_back_2_resized) / 2

        if self.use_average:
            image_back_resized = self.background_average(scan_image, imin,imax,min,max)
        else:
            image_back_resized = self.background(scan_image, imin,imax,min,max)

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
        mkdir(vignettepath)
        filelist = generate_vignettes(image_scan_unbordered,contours,filter, path=vignettepath)

