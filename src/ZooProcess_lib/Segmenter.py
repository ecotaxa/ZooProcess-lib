from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

from ZooProcess_lib.img_tools import saveimage


class Segmenter(object):
    """
    Divide an image into segments and store the result sub-images.
    """

    THRESH_MAX = 243

    def __init__(self, image: ndarray):
        assert image.dtype == np.uint8
        self.image = image

    def process(self):
        # Threshold the source image to have a b&w mask
        thresh_min = 0
        thresh_max = self.THRESH_MAX
        _th, msk1 = cv2.threshold(self.image, thresh_max, 255, cv2.THRESH_BINARY)
        saveimage(msk1, Path("/tmp/msk.jpg"))
        self.sanity_check(msk1)
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        pass

    def sanity_check(self, mask: ndarray):
        min_bwratio = 25
        histo = np.unique(mask, return_counts=True)
        assert list(histo[0]) == [0, 255]
        nb_black, nb_white = list(histo[1])
        bwratiomeas = nb_black / nb_white
        print(f"bwratiomeas: {bwratiomeas}")
        if bwratiomeas > min_bwratio / 100:
            print(
                "########### WARNING : More than "
                + min_bwratio
                + "% of the segmented image is black ! \nThe associated background image maybe NOK."
            )
