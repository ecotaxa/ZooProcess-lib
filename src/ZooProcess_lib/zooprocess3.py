from pathlib import Path

import cv2
import numpy as np

from Border import Border
from ZooscanProject import ZooscanProject
from img_tools import (
    crop, loadimage, saveimage,
    picheral_median,
    minAndMax,
    rotate90c, draw_contours, draw_boxes_filtered,
    generate_vignettes, mkdir,
    getPath,
)


def Analyze_sample(TP: ZooscanProject, sample: str):
    print(f"Analyze: {TP.project_name} - {sample}")

    output = Path(TP.testfolder, sample)
    print(f"output:  {output.as_posix()}")
    print(f"output:  {output.name}")  # {output.as_posix()}
    if output.exists():
        print(f"Sample already analyzed: {sample}")
        return

    bg_pattern = TP.getBackgroundUsed(sample + "_1")
    bg_name = bg_pattern + "_back_large"

    z = zooprocessv10(TP, sample, bg_name)

    mkdir(output)
    z.output_path = output

    z.process()


# def areaFilter(contour):
#     if area < organism_size_min or area > organism_size_max: return False
#     return True

# def filterSize(contour):
#     h=100
#     w=100
#     x,y,width,height = cv2.boundingRect(contour)
#     if width < w or height < h:
#         return False
#     else:
#         return True

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

    def __init__(self, TP: ZooscanProject, scan_name, back_name):
        self.TP = TP
        self.name = scan_name
        self.bg_name = back_name

        self.sample = self.name + "_raw" + "_1" + ".tif"
        self.back_name = self.bg_name + "_raw" + "_1" + ".tif"
        self.back_name_2 = self.bg_name + "_raw" + "_2" + ".tif"

        output_path = Path(self.TP.testfolder)

    def background(self, scan_image, imin, imax, min, max):
        from to8bit import resize

        back_image = loadimage(self.back_name, path=self.TP.back)

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # print(f"a: {a}, b: {b}")
        back_image_8bit = (a * back_image + b).astype(np.uint8)
        scan_image_8bit = (a * scan_image + b).astype(np.uint8)
        image_back_resized = resize(scan_image_8bit, back_image_8bit)

        return image_back_resized

    def convert_to_8bit(self, image):

        median, mean = picheral_median(image)
        imin, imax = minAndMax(median)

        min = 0
        max = 254

        a = (max - min) / (imax - imin)
        b = max - a * imax
        # b = imax - a * max

        back_image_8bit = (a * image + b).astype(np.uint8)

        return back_image_8bit

    def normalize(self, scan_image):

        from to8bit import resize

        back_image_1 = loadimage(self.back_name, path=self.TP.back)
        back_image_2 = loadimage(self.back_name_2, path=self.TP.back)

        back_image_1_8bit = self.convert_to_8bit(back_image_1)
        back_image_2_8bit = self.convert_to_8bit(back_image_2)

        saveimage(back_image_1_8bit, self.back_name, "8bit", ext="tiff", path=self.output_path)
        saveimage(back_image_2_8bit, self.back_name, "8bit", ext="tiff", path=self.output_path)

        scan_image_8bit = self.convert_to_8bit(scan_image)
        saveimage(scan_image_8bit, self.sample, "8bit", ext="tiff", path=self.output_path)

        image_back_1_resized = resize(scan_image_8bit, back_image_1_8bit)
        image_back_2_resized = resize(scan_image_8bit, back_image_2_8bit)

        image_back_median_resized = (image_back_1_resized / 2 + image_back_2_resized / 2).astype(np.uint8)

        # image_back_median_resized = cv2.medianBlur(image_back_median_resized, 3)

        image_back_rotated = rotate90c(image_back_median_resized)
        image_scan_rotated = rotate90c(scan_image_8bit)

        saveimage(image_back_rotated, self.back_name, "image_back_rotated", ext="tiff", path=self.output_path)
        saveimage(image_scan_rotated, self.sample, "image_scan_rotated", ext="tiff", path=self.output_path)

        # ajouter un flip horizontal pour que l'utilsateur voit l'image comme son scan

        return image_scan_rotated, image_back_rotated

    def crop(self, image):
        border = Border(image)
        border.output_path = self.output_path
        border.name = self.back_name

        # reload the image_back_rotated to don't mutated the original (a copy is need to work)
        # make a byte copy will probably most efficiant
        back_file = Path(getPath(self.back_name, extraname="image_back_rotated", ext="tiff", path=self.output_path))
        border.draw_image = loadimage(back_file.as_posix())

        limitetop, limitbas, limitegauche, limitedroite = border.detect()

        return limitetop, limitbas, limitegauche, limitedroite

    def process(self):

        scan_image = loadimage(self.sample, path=self.TP.rawscan)

        print(f"scan_image shape: {scan_image.shape}")

        # normalise : scan en 8 bit rotated, mix back1 & back 2 8 bits 
        image_scan_normed, image_back_normed = self.normalize(scan_image)

        print(f"image_scan_normed shape: {image_scan_normed.shape}")
        print(f"image_back_normed shape: {image_back_normed.shape}")

        # image_substracted = np.subtract(image_scan_normed, image_back_normed)
        # saveimage(image_substracted, self.sample, "substracted_back", ext="tiff", path=self.output_path)

        # image_substracted2 = np.subtract(image_back_normed, image_scan_normed)
        # saveimage(image_substracted2, self.sample, "substracted_back_inv", ext="tiff", path=self.output_path)

        # On recherche les bords de l'image de fond
        limitetop, limitbas, limitegauche, limitedroite = self.crop(image_back_normed)
        print(f"back limite t={limitetop}, b={limitbas}, l={limitegauche}, r={limitedroite}")

        # on se fiche de cette image
        # image_back_unbordered = crop(image_back_rotated, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        # saveimage(image_back_unbordered, self.back_name, "unbordered", ext="tiff", path=self.output_path)

        # on crop le scan original
        image_scan_unbordered = crop(rotate90c(scan_image), left=limitetop, top=limitegauche, right=limitbas,
                                     bottom=limitedroite)
        saveimage(image_scan_unbordered, self.sample, "unbordered", ext="tiff", path=self.output_path)

        # on crop les 8 bit
        image_scan_normed_cropped = crop(image_scan_normed, left=limitetop, top=limitegauche, right=limitbas,
                                         bottom=limitedroite)
        saveimage(image_scan_normed_cropped, self.sample, "image_scan_normed_unbordered", ext="tiff",
                  path=self.output_path)

        image_back_normed_cropped = crop(image_back_normed, left=limitetop, top=limitegauche, right=limitbas,
                                         bottom=limitedroite)
        saveimage(image_back_normed_cropped, self.sample, "image_back_normed_unbordered", ext="tiff",
                  path=self.output_path)

        # # on crop le scan 8 bits
        # image_scan_normed_unbordered = crop(image_scan_normed, left=limitetop, top=limitegauche, right=limitbas, bottom=limitedroite)
        # saveimage(image_scan_unbordered, self.sample, "unbordered_8bit", ext="tiff", path=self.output_path)

        image_substracted = np.subtract(image_scan_normed_cropped, image_back_normed_cropped)
        saveimage(image_substracted, self.sample, "substracted_back", ext="tiff", path=self.output_path)

        image_substracted2 = np.subtract(image_back_normed_cropped, image_scan_normed_cropped)
        saveimage(image_substracted2, self.sample, "substracted_back_inv", ext="tiff", path=self.output_path)

        # transforme en masque (binarisation par seuillage de l'image 8 bits)
        # mask_inv = self.thresholding_with_inv_otsu(image_scan_normed_unbordered, path=self.output_path)
        # saveimage(mask_inv, self.sample, "mask_inv", ext="tiff", path=self.output_path)
        mask = self.thresholding_with_inv_otsu(image_substracted, path=self.output_path)
        saveimage(mask, self.sample, "mask", ext="tiff", path=self.output_path)

        contours = self.contours(mask)
        self.draw_contours_filtered(image_scan_unbordered, contours, organism_size_min=100,
                                    organism_size_max=100)  # , filterSize)
        # self.draw_contours(image_scan_unbordered, contours)

        # self.vignettes(image_scan_unbordered, contours)
        self.vignettes(image_scan_normed_cropped, contours)

        print(f"Done open {self.output_path}")

    def aire_filter(self, resolution, minsize, maxsize) -> tuple:
        pixel = 25.4 / resolution;
        Smmin = (3.1416 / 4) * pow(minsize, 2);
        Spmin = round(Smmin / (pow(pixel, 2)));
        Smmax = (3.1416 / 4) * pow(maxsize, 2);
        Spmax = round(Smmax / (pow(pixel, 2)));

        return (Spmin, Spmax)

    def threshold_binary(self, image, thresh_min=0, thresh_max=255, path: Path = None):
        th, mask = cv2.threshold(image, thresh_min, thresh_max, cv2.THRESH_BINARY)
        if path:
            saveimage(mask, self.sample, "unbordered_mask" + "_" + str(thresh_min) + "_" + str(thresh_max), ext="tiff",
                      path=path)
        print(f"threshold at: {th}")
        return mask

    def thresholding_with_otsu(self, image, thresh_min=0, thresh_max=255, path: Path = None):
        th, mask = cv2.threshold(image, thresh_min, thresh_max, cv2.THRESH_OTSU)
        saveimage(mask, self.sample, "unbordered_otsu" + "_" + str(thresh_min) + "_" + str(thresh_max), ext="tiff",
                  path=path)
        print(f"threshold at: {th}")
        return mask

    def thresholding_with_inv_otsu(self, image, thresh_min=0, thresh_max=255, path: Path = None):
        th, mask = cv2.threshold(image, thresh_min, thresh_max, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        saveimage(mask, self.sample, "unbordered_bin_inv_otsu" + "_" + str(thresh_min) + "_" + str(thresh_max),
                  ext="tiff", path=path)
        print(f"threshold at: {th}")
        return mask

    def contours(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of Contours found = " + str(len(contours)))
        return contours

    def draw_contours(self, image_scan_unbordered, contours):
        image_3channels = draw_contours(image_scan_unbordered, contours)
        saveimage(image_3channels, self.sample, "draw_contours_on_image2", path=self.output_path)

        # white_mask = np.full(mask.shape[:2],255, np.uint8)
        white_mask = np.full(image_scan_unbordered.shape[:2], 255, np.uint8)
        image_3channels = draw_contours(white_mask, contours)
        saveimage(image_3channels, self.sample, "draw_contours2", path=self.output_path)

    # def draw_contours_filtered(self, image_scan_unbordered, contours): #, filter):

    #     # filtered_contours = list(filter(lambda c: filter(c), contours))
    #     filtersize = 50
    #     def f (contour):
    #         x,y,w,h = cv2.boundingRect(contour)
    #         if w < filtersize and h < filtersize:
    #             return False
    #         return True

    #     filtered_contours = list(filter(f, contours))

    #     # filtered_contours = filter(filter, contours)

    #     white_mask = np.full(image_scan_unbordered.shape[:2], 255, np.uint8)
    #     image_3channels = draw_contours(white_mask, filtered_contours)
    #     saveimage(image_3channels, self.sample, "draw_contours_filtered", path=self.output_path)
    #     return image_3channels

    def draw_contours_filtered(self, image_scan_unbordered, contours, organism_size_min=100, organism_size_max=10000):
        # filtersize = 50
        # def f (contour):
        #     x,y,w,h = cv2.boundingRect(contour)
        #     if w < filtersize and h < filtersize:
        #         return False
        #     return True

        # # c = filter(f, contours)
        # c = list(filter(f, contours))

        # threshold_area = 10000     #threshold area 
        # contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)   
        # for cnt in contours:        
        #     area = cv2.contourArea(cnt)         
        #     if area > threshold_area:                   
        #         #Put your code in here

        # image_3channels = cv2.merge([image_scan_unbordered, image_scan_unbordered, image_scan_unbordered])

        nb = 0
        c = []

        # def filter(h,w)-> bool:
        #     if h < organism_size and w < organism_size: return False
        #     return True

        # def areaFilter(area):
        #     print(f"area: {area}")
        #     return True
        #     if area < organism_size_min or area > organism_size_max: return False
        #     return True

        def filter(contour) -> bool:
            # x,y,w,h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < organism_size_min and area > organism_size_max: return False
            return True

        for i in range(0, len(contours)):
            # mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
            x, y, w, h = cv2.boundingRect(contours[i])
            # area = cv2.minAreaRect(contours[i])
            area = cv2.contourArea(contours[i])
            # if h < 100 or w < 100: continue
            # if filter(h,w) :
            # if areaFilter(area):
            if filter(contours[i]):
                c.append(contours[i])
                nb = nb + 1
                # cv2.drawContours()
                # append_contours(image_3channels,contours,index=i)

        self.draw_contours(image_scan_unbordered, c)

        print(f"Numnber of Contours found = {len(contours)}")
        print(f"Number of filtered Contours found = {nb}")
        print(f"Number of filtered Contours found = {len(c)}")

        # saveimage(image_3channels, self.sample, "draw_contours_filtered_on_image", path=self.output_path)

    def vignettes(self, image_scan_unbordered, contours, organism_size=100):
        # acceptable size
        # def filter(h,w)-> bool:
        #     return True
        #     if h < organism_size and w < organism_size: return False
        #     return True

        def filter(contour) -> bool:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"h: {h} w: {w}")
            if h == 1 or w == 1: return False
            if h < organism_size and w < organism_size: return False
            return True

        image_3channels = draw_boxes_filtered(image_scan_unbordered, contours, filter)
        saveimage(image_3channels, self.sample, "draw_boxes_filtered_on_image", path=self.output_path)

        # white_mask = np.full(mask.shape[:2],255, np.uint8)
        white_mask = np.full(image_scan_unbordered.shape[:2], 255, np.uint8)
        image_3channels = draw_boxes_filtered(white_mask, contours, filter, add_number=True)
        saveimage(image_3channels, self.sample, "draw_boxes_filtered", path=self.output_path)

        vignettepath = Path(self.output_path, "vignettes")
        mkdir(vignettepath)
        filelist = generate_vignettes(image_scan_unbordered, contours, filter, path=vignettepath)

        print(f"Vignettes générées: {len(filelist)}")
        print(f"{filelist}")
