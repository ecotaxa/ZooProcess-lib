import math
from typing import List, Dict

import cv2
import numpy as np
from numpy import ndarray


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
            # saveimage(msk1, Path("/tmp/temp0_msk1.tif"))
            # Required measurements:
            #       area bounding area_fraction limit decimal=2
            # Result:
            #       Area	BX	BY	Width	Height	%Area	XStart	YStart
            return self.find_particles(msk1)
        else:
            assert False, "Single image KO"

    def process2(self):
        # Threshold the source image to have a b&w mask
        # thresh_min = 0 # implied in opencv call
        thresh_max = self.THRESH_MAX
        _th, msk1 = cv2.threshold(self.image, thresh_max, 255, cv2.THRESH_BINARY)
        self.sanity_check(msk1)
        # saveimage(msk1, Path("/tmp/temp0_msk1.tif"))
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        return self.find_particles_via_cc(msk1)

    def find_particles_via_cc(self, mask: ndarray) -> List[Dict]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        mask = 255 - mask  # Opencv looks for white objects on black background
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        ret = []
        # _unique, counts = np.unique(labels, return_counts=True)
        for a_cc in range(retval):
            area = int(stats[a_cc, cv2.CC_STAT_AREA])
            if area < self.s_p_min:
                continue
            if area > self.s_p_max:
                continue
            # componentMask = (labels == a_cc)
            x = int(stats[a_cc, cv2.CC_STAT_LEFT])
            y = int(stats[a_cc, cv2.CC_STAT_TOP].astype(int))
            w = int(stats[a_cc, cv2.CC_STAT_WIDTH].astype(int))
            h = int(stats[a_cc, cv2.CC_STAT_HEIGHT].astype(int))
            ret.append({"BX": x, "BY": y, "Width": w, "Height": h, "Area": int(area)})
        return ret

    def find_particles(self, mask: ndarray) -> List[Dict]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        mask = 255 - mask  # Opencv looks for white objects on black background
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        print("Number of Contours found = " + str(len(contours)))
        ret = []
        filtered_contours = []
        for a_contour in contours:
            if a_contour.shape == (1, 1, 2):  # Single-point "contour"
                continue
            x, y, w, h = cv2.boundingRect(a_contour)
            if w * h < self.s_p_min:
                # Even if contour was around a filled rectangle it would not meet min
                # Don't bother computing exact area
                continue
            area = self.pixels_inside_contour(a_contour, x, y, w, h)
            if area < self.s_p_min:
                continue
            if area > self.s_p_max:
                continue
            ret.append({"BX": x, "BY": y, "Width": w, "Height": h, "Area": area})
            filtered_contours.append(a_contour)
        # image_3channels = draw_contours(self.image, filtered_contours)
        # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret

    @staticmethod
    def sanity_check(mask: ndarray):
        min_bwratio = 25
        nb_white = np.count_nonzero(mask)
        nb_black = mask.shape[0] * mask.shape[1] - nb_white
        bwratiomeas = nb_black / nb_white
        print(f"bwratiomeas: {bwratiomeas}")
        if bwratiomeas > min_bwratio / 100:
            print(
                f"########### WARNING : More than {min_bwratio}% of the segmented image is black ! \nThe associated background image maybe NOK."
            )

    @staticmethod
    def pixels_inside_contour(contour, x, y, w, h) -> int:
        contour_canvas = np.zeros([h, w], np.uint8)
        cv2.drawContours(
            image=contour_canvas,
            contours=[contour],
            contourIdx=0,
            color=255,
            thickness=cv2.FILLED,
            offset=(-x, -y),
        )
        return np.count_nonzero(contour_canvas)
