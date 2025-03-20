from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

from ZooProcess_lib.ImageJLike import parseInt
from ZooProcess_lib.img_tools import saveimage, draw_contours, crophw


class Segmenter(object):
    """
    Divide an image into segments and store the result sub-images.
    """

    THRESH_MAX = 243
    RESOLUTION = 2400
    Wlimit = 20000
    Hlimit = 6500
    overlap = 0.6

    def __init__(self, image: ndarray, minsize: float, maxsize: float):
        assert image.dtype == np.uint8
        self.image = image
        self.height, self.width = image.shape[:2]
        pixel = 25.4 / self.RESOLUTION
        Smmin = (3.1416 / 4) * pow(minsize, 2)
        Smmax = (3.1416 / 4) * pow(maxsize, 2)
        # s_p_* are in pixel^2
        self.s_p_min = round(Smmin / (pow(pixel, 2)))
        self.s_p_max = round(Smmax / (pow(pixel, 2)))

    def process(self):
        # Threshold the source image to have a b&w mask
        # thresh_min = 0 # implied in opencv call
        thresh_max = self.THRESH_MAX
        _th, msk1 = cv2.threshold(self.image, thresh_max, 255, cv2.THRESH_BINARY)
        self.sanity_check(msk1)
        if self.width > self.Wlimit and self.height > self.Hlimit:
            # Process image in 2 parts: TODO: see if useful at all.
            # O = parseInt(self.width * self.overlap)
            # M = self.width - O
            # msk1 = crophw(msk1, 0, 0, O, self.height)
            saveimage(msk1, Path("/tmp/temp0_msk1.tif"))
            # Required measurements:
            #       area bounding area_fraction limit decimal=2
            # Result:
            #       Area	BX	BY	Width	Height	%Area	XStart	YStart
            self.find_particles(msk1)
        else:
            assert False, "Single image KO"

    def find_particles(self, mask: ndarray):
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        mask = 255 - mask  # Opencv looks for white objects on black background
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print("Number of Contours found = " + str(len(contours)))
        filtered_contours = []
        for a_contour in contours:
            area = cv2.contourArea(a_contour)
            if area < self.s_p_min:
                continue
            filtered_contours.append(a_contour)
        print("Number of filtered contours = " + str(len(filtered_contours)))
        image_3channels = draw_contours(self.image, filtered_contours)
        saveimage(image_3channels, Path("/tmp/contours.tif"))

    @staticmethod
    def sanity_check(mask: ndarray):
        min_bwratio = 25
        histo = np.unique(mask, return_counts=True)
        assert list(histo[0]) == [0, 255]
        nb_black, nb_white = list(histo[1])
        bwratiomeas = nb_black / nb_white
        print(f"bwratiomeas: {bwratiomeas}")
        if bwratiomeas > min_bwratio / 100:
            print(
                f"########### WARNING : More than {min_bwratio}% of the segmented image is black ! \nThe associated background image maybe NOK."
            )
