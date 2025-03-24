from typing import List, Dict, TypedDict

import cv2
import numpy as np
from numpy import ndarray

from ZooProcess_lib.EllipseFitter import EllipseFitter
from ZooProcess_lib.img_tools import cropnp, saveimage


class Blob(TypedDict):
    # Enclosing rectangle
    BX: int
    BY: int
    Width: int
    Height: int
    # Number of pixels inside the blob
    Area: int
    # Quoting some forum: (XStart,YStart) are the coordinates of the first boundary point
    # of particles found by the particle analyzer.
    XStart: int
    YStart: int
    # Area fraction : For thresholded images is the percentage of pixels in the image or selection
    # that have been highlighted in red using Image▷Adjust▷Threshold… [T]↑.
    # For non-thresholded images is the percentage of non-zero pixels. Uses the heading %Area.
    # %Area: int
    Major: float
    Minor: float
    Angle: float


class Segmenter(object):
    """
    Divide an image into segments and store the result sub-images.
    """

    THRESH_MAX = 243
    RESOLUTION = 2400
    # Constants for 2-image processing. Historical.
    # Wlimit = 20000
    # Hlimit = 6500
    # overlap = 0.6

    METH_CONTOUR = 1
    METH_CONNECTED_COMPONENTS = 2

    def __init__(self, image: ndarray, minsize: float, maxsize: float):
        assert image.dtype == np.uint8
        self.image = image
        self.height, self.width = image.shape[:2]
        pixel = 25.4 / self.RESOLUTION
        sm_min = (3.1416 / 4) * pow(minsize, 2)
        sm_max = (3.1416 / 4) * pow(maxsize, 2)
        # s_p_* are in pixel^2
        self.s_p_min = round(sm_min / (pow(pixel, 2)))
        self.s_p_max = round(sm_max / (pow(pixel, 2)))
        self.contours: List[ndarray] = []
        self.blobs: List[Blob] = []
        self.contour_masks: List[ndarray] = []

    def find_blobs(self, method: int = METH_CONTOUR):
        # Threshold the source image to have a b&w mask
        thresh_max = self.THRESH_MAX
        _th, msk1 = cv2.threshold(self.image, thresh_max, 255, cv2.THRESH_BINARY)
        self.sanity_check(msk1)
        # if self.width > self.Wlimit and self.height > self.Hlimit:
        # Process image in 2 parts: TODO: see if useful at all.
        # O = parseInt(self.width * self.overlap)
        # M = self.width - O
        # msk1 = crophw(msk1, 0, 0, O, self.height)
        # saveimage(msk1, Path("/tmp/temp0_msk1.tif"))
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        if method == self.METH_CONNECTED_COMPONENTS:
            # Faster, but areas don't match with ImageJ. Left for future investigations
            self.blobs = self.find_particles_via_cc(msk1)
        else:
            self.blobs = self.find_particles(msk1)
        return self.blobs

    def find_particles(self, mask: ndarray) -> List[Blob]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        inv_mask = 255 - mask  # Opencv looks for white objects on black background
        contours, hierarchy = cv2.findContours(
            inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        print("Number of Contours found = " + str(len(contours)))
        ret: List[Blob] = []
        single_point_contour_shape = (1, 1, 2)
        for a_contour in contours:
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                continue
            x, y, w, h = cv2.boundingRect(a_contour)
            if w * h < self.s_p_min:
                # Even if contour was around a filled rectangle it would not meet min criterion
                # -> don't bother drawing the contour, which is expensive
                continue
            contour_mask = self.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < self.s_p_min:
                continue
            if area > self.s_p_max:
                continue
            ret.append(
                {
                    "BX": x,
                    "BY": y,
                    "Width": w,
                    "Height": h,
                    "Area": area,
                }
            )
            self.contour_masks.append(contour_mask)
            self.contours.append(a_contour)
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
    def draw_contour(contour, x, y, w, h) -> ndarray:
        contour_canvas = np.zeros([h, w], np.uint8)
        cv2.drawContours(
            image=contour_canvas,
            contours=[contour],
            contourIdx=0,
            color=255,
            thickness=cv2.FILLED,
            offset=(-x, -y),
        )
        return contour_canvas

    def find_particles_via_cc(self, mask: ndarray) -> List[Dict]:
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

    def split_by_blobs(self):
        assert self.blobs, "No blobs"
        for ndx, (a_blob, its_mask, its_contour) in enumerate(
            zip(self.blobs, self.contour_masks, self.contours)
        ):
            width = a_blob["Width"]
            height = a_blob["Height"]
            bx = a_blob["BX"]
            by = a_blob["BY"]
            # For filtering out horizontal lines
            ratiobxby = width / height
            print("ratiobxby", ratiobxby)
            vignette = cropnp(
                self.image, top=by, left=bx, bottom=by + height, right=bx + width
            )
            vignette = np.bitwise_or(vignette, 255 - its_mask)
            # Compute more features
            # Xstart
            # First pixel in shape seems OK for this measurement
            x_start = a_blob["BX"] + int(np.argmax(its_mask == 255))
            a_blob["XStart"] = x_start
            a_blob["YStart"] = a_blob["BY"]
            # %Area
            pct_area = (
                100
                - (np.count_nonzero(vignette <= self.THRESH_MAX) * 100) / a_blob["Area"]
            )
            a_blob["%Area"] = round(pct_area, 3)
            # major, minor, angle
            if False:
                # Very different from ref. and drawing them gives strange results sometimes
                vignette_contour = its_contour - (a_blob["BX"], a_blob["BY"])
                min_ellipse = cv2.fitEllipse(vignette_contour)
                cv2.ellipse(vignette, min_ellipse, 0)
                cv2.drawContours(vignette, [vignette_contour], 0, 0)
            else:
                fitter = EllipseFitter()
                fitter.fit(its_mask)
                a_blob["Major"] = round(fitter.major, 3)
                a_blob["Minor"] = round(fitter.minor, 3)
                a_blob["Angle"] = round(fitter.angle, 3)
                # fitter.draw_ellipse(vignette)
                vignette = cv2.ellipse(
                    img=vignette,
                    center=(int(fitter.x_center), int(fitter.y_center)),
                    axes=(int(fitter.minor / 2), int(fitter.major / 2)),
                    angle=90 - fitter.angle,
                    startAngle=0,
                    endAngle=360,
                    color=(0,),
                )

            saveimage(vignette, "/tmp/vignette_%s.png" % ndx)
