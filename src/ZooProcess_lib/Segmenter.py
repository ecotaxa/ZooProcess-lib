import dataclasses
import math
from typing import List, TypedDict, Tuple, Optional

import cv2
import numpy as np
from numpy import ndarray

from .EllipseFitter import EllipseFitter
from .img_tools import cropnp


class Features(TypedDict):
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


@dataclasses.dataclass(frozen=True)
class ROI(object):
    features: Features
    mask: ndarray
    contour: Optional[ndarray] = None


def features_are_equal(features: Features, another_blob: Features):
    return features["BX"] == another_blob["BX"] and features["BY"] == another_blob["BY"]


class Segmenter(object):
    """
    Divide an image into segments and store the result sub-images.
    """

    THRESH_MAX = 243
    RESOLUTION = 2400
    # Constants for 2-image processing. Historical.
    Wlimit = 20000
    Hlimit = 6500
    overlap = 0.6

    METH_CONTOUR = 1
    METH_CONNECTED_COMPONENTS = 2
    LEGACY_COMPATIBLE = 8

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

    def find_blobs(self, method: int = METH_CONTOUR) -> List[ROI]:
        # Threshold the source image to have a b&w mask
        thresh_max = self.THRESH_MAX
        _th, msk1 = cv2.threshold(self.image, thresh_max, 255, cv2.THRESH_BINARY)
        self.sanity_check(msk1)
        if method & self.LEGACY_COMPATIBLE:
            # Process image in 2 overlapping parts, split vertically.
            # There is a side effect in that objects in the middle 10% band can appear or disappear if too large.
            if self.width > self.Wlimit and self.height > self.Hlimit:
                overlap_size = int(self.width * self.overlap)
                left_mask = cropnp(
                    msk1, top=0, left=0, bottom=self.height, right=overlap_size
                )
                left_rois = self.find_particles(left_mask, self.s_p_min, self.s_p_max)
                right_mask = cropnp(
                    msk1,
                    top=0,
                    left=self.width - overlap_size,
                    bottom=self.height,
                    right=self.width,
                )
                right_rois = self.find_particles(right_mask, self.s_p_min, self.s_p_max)
                for features in right_rois:
                    features.features["BX"] += self.width - overlap_size
                to_add = []
                for a_roi in right_rois:
                    for another_roi in left_rois:
                        if features_are_equal(a_roi.features, another_roi.features):
                            break
                    else:
                        to_add.append(a_roi)
                return left_rois + to_add
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        if method & self.METH_CONNECTED_COMPONENTS:
            # Faster, but areas don't match with ImageJ. Left for future investigations
            return self.find_particles_via_cc(msk1, self.s_p_min, self.s_p_max)
        else:
            return self.find_particles(msk1, self.s_p_min, self.s_p_max)

    @staticmethod
    def find_particles(mask: ndarray, s_p_min: int, s_p_max: int) -> List[ROI]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        inv_mask = 255 - mask  # Opencv looks for white objects on black background
        height, width = inv_mask.shape[:2]
        # ImageJ can ignore around borders but the side lines prevent proper detection using openCV
        # y_limit1, x_limit1, y_limit2, x_limit2 = self.undo_border_lines(inv_mask)
        # print(x_limit1, y_limit1, x_limit2, y_limit2)
        contours, (hierarchy,) = cv2.findContours(
            inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 1:
            # In some cases and despite previous steps, the border of the scan goes fully round the image, so
            # there is a single contour!
            # Fix by removing it.
            first_pixel = np.argmax(inv_mask[0] == 255)
            cv2.floodFill(inv_mask, None, (first_pixel, 0), (0,))
            contours, (hierarchy,) = cv2.findContours(
                inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

        print("Number of Contours found = " + str(len(contours)))
        ret: List[ROI] = []
        roots = set()
        single_point_contour_shape = (1, 1, 2)
        for contour_id, (a_contour, its_hierarchy) in enumerate(
            zip(contours, hierarchy)
        ):
            (
                next_contour,
                previous_contour,
                child_contour,
                parent_contour,
            ) = its_hierarchy
            assert parent_contour < contour_id  # Ensure we've seen the parent before
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                continue
            x, y, w, h = cv2.boundingRect(a_contour)
            # Eliminate if touching the border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                continue
            if w * h < s_p_min:
                # Even if contour was around a filled rectangle it would not meet min criterion
                # -> don't bother drawing the contour, which is expensive
                continue
            contour_mask = Segmenter.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                continue
            if area > s_p_max:
                continue
            ret.append(
                ROI(
                    features={
                        "BX": x,
                        "BY": y,
                        "Width": w,
                        "Height": h,
                        "Area": area,
                    },
                    mask=contour_mask,
                    contour=a_contour,
                )
            )
            # image_3channels = draw_contours(self.image, self.contours)
            # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret

    def undo_border_lines(self, inv_mask: ndarray) -> Tuple[int, int, int, int]:
        # Find the 2 border dots in each dimension
        (left, right) = np.where(inv_mask[0] == 255)[0]
        (top, bottom) = np.where(inv_mask[:, 0] == 255)[0]
        # Clear them
        cv2.line(
            img=inv_mask,
            pt1=(0, top),
            pt2=(self.width - 1, top),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(0, bottom),
            pt2=(self.width - 1, bottom),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(left, 0),
            pt2=(left, self.height - 1),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(right, 0),
            pt2=(right, self.height - 1),
            color=(0,),
            thickness=1,
        )
        return top, left, bottom, right

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
            color=(255,),
            thickness=cv2.FILLED,
            offset=(-x, -y),
        )
        return contour_canvas

    @staticmethod
    def find_particles_via_cc(mask: ndarray, s_p_min: int, s_p_max: int) -> List[ROI]:
        inv_mask = 255 - mask  # Opencv looks for white objects on black background
        # y_limit1, x_limit1, y_limit2, x_limit2 = self.undo_border_lines(inv_mask)
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        ret = []
        for a_cc in range(retval):
            area = int(stats[a_cc, cv2.CC_STAT_AREA])  # Area excluding eventual holes

            if area > s_p_max:
                continue

            x = int(stats[a_cc, cv2.CC_STAT_LEFT])
            y = int(stats[a_cc, cv2.CC_STAT_TOP])
            w = int(stats[a_cc, cv2.CC_STAT_WIDTH])
            h = int(stats[a_cc, cv2.CC_STAT_HEIGHT])

            if w * h < s_p_min:
                # Even if contour was around a filled rectangle it would not meet min criterion
                # -> don't bother drawing the contour, which is expensive
                continue

            # Eliminate if touching the border
            # if (
            #     x == x_limit1 + 1
            #     or y == y_limit1 + 1
            #     or x + w == x_limit2
            #     or y + h == y_limit2
            # ):
            #     continue

            sub_img = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
            sub_mask = (sub_img == a_cc).astype(dtype=np.uint8) * 255
            filled_mask = Segmenter.filled_mask(sub_mask)
            area = (filled_mask == 255).sum()

            if area < s_p_min:
                continue
            if area > s_p_max:
                continue

            ret.append(
                ROI(
                    features={
                        "BX": x,
                        "BY": y,
                        "Width": w,
                        "Height": h,
                        "Area": int(area),
                    },
                    mask=filled_mask,
                    contour=None,
                )
            )
        return ret

    @staticmethod
    def filled_mask(sub_mask: ndarray) -> ndarray:
        height, width = sub_mask.shape
        sub_mask2 = cv2.copyMakeBorder(
            sub_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
        )
        cv2.floodFill(sub_mask2, None, (0, 0), (128,))
        orig_mask = cropnp(
            image=sub_mask2, top=1, left=1, bottom=height + 1, right=width + 1
        )
        orig_mask[orig_mask == 0] = 255
        orig_mask[orig_mask == 128] = 0
        return orig_mask

    def split_by_blobs(self, rois: List[ROI]):
        assert rois, "No ROIs"
        for ndx, a_roi in enumerate(rois):
            features = a_roi.features
            width = features["Width"]
            height = features["Height"]
            bx = features["BX"]
            by = features["BY"]
            # For filtering out horizontal lines
            ratiobxby = width / height
            print("ratiobxby", ratiobxby)
            # assert ratiobxby < 40
            vignette = cropnp(
                self.image, top=by, left=bx, bottom=by + height, right=bx + width
            )
            vignette = np.bitwise_or(vignette, 255 - a_roi.mask)
            # Compute more features
            # Xstart
            # First white pixel in first line of shape seems OK for this measurement
            x_start = features["BX"] + int(np.argmax(a_roi.mask == 255))
            features["XStart"] = x_start
            features["YStart"] = features["BY"]
            # %Area
            pct_area = (
                100
                - (np.count_nonzero(vignette <= self.THRESH_MAX) * 100)
                / features["Area"]
            )
            features["%Area"] = round(pct_area, 3)
            # major, minor, angle
            if False:
                # Very different from ref. and drawing them gives strange results sometimes
                # From some docs, we must not expect anything reasonable if the shape is not close from an ellipse
                vignette_contour = a_roi.contour - (features["BX"], features["BY"])
                min_ellipse = cv2.fitEllipse(vignette_contour)
                cv2.ellipse(vignette, min_ellipse, 0)
                cv2.drawContours(vignette, [vignette_contour], 0, 0)
            if False:
                # Angle matches with ref, but for some reason the axis are sometimes very different, and
                # visual check shows sometimes the ellipse extremely shifted. Maybe a centroid issue.
                import skimage

                props = skimage.measure.regionprops(a_roi.mask, cache=False)[0]
                features["Major"] = round(props.axis_major_length, 3)
                features["Minor"] = round(props.axis_minor_length, 3)
                features["Angle"] = round(math.degrees(props.orientation) + 90, 3)
                vignette = cv2.ellipse(
                    img=vignette,
                    center=(int(props.centroid[0]), int(props.centroid[1])),
                    axes=(
                        int(props.axis_minor_length / 2),
                        int(props.axis_major_length / 2),
                    ),
                    angle=90 - (math.degrees(props.orientation) + 90),
                    startAngle=0,
                    endAngle=360,
                    color=(0,),
                    thickness=1,
                )
            # Port of ImageJ algo
            fitter = EllipseFitter()
            fitter.fit(a_roi.mask)
            features["Major"] = round(fitter.major, 3)
            features["Minor"] = round(fitter.minor, 3)
            features["Angle"] = round(fitter.angle, 3)
            # fitter.draw_ellipse(vignette)
            # vignette = cv2.ellipse(
            #     img=vignette,
            #     center=(int(fitter.x_center), int(fitter.y_center)),
            #     axes=(int(fitter.minor / 2), int(fitter.major / 2)),
            #     angle=90 - fitter.angle,
            #     startAngle=0,
            #     endAngle=360,
            #     color=(0,),
            #     thickness=2,
            # )

            # saveimage(vignette, "/tmp/vignette_%s.png" % ndx)
