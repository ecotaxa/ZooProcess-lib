from typing import List, Tuple

import cv2
import numpy as np
from numpy import ndarray

from .ROI import ROI, roi_unq
from .img_tools import cropnp
from .segmenters.ConnectedComponents import ConnectedComponentsSegmenter
from .segmenters.ExternalContours import ExternalContoursSegmenter
from .segmenters.RecursiveContours import RecursiveContoursSegmenter


class Segmenter(object):
    """
    Divide an image into segments and produce ROIs.
    """

    # Constants for 2-image processing. Historical.
    Wlimit = 20000
    Hlimit = 6500
    overlap = 0.6
    # For filtering horizontal lines
    max_w_to_h_ratio = 40

    METH_TOP_CONTOUR = 1
    METH_TOP_CONTOUR_SPLIT = 2
    METH_CONTOUR_TREE = 4
    METH_CONNECTED_COMPONENTS = 8
    METH_CONNECTED_COMPONENTS_SPLIT = 16
    LEGACY_COMPATIBLE = 64

    def __init__(
        self,
        minsize: float,
        maxsize: float,
        threshold: int,
    ):
        self.minsize = minsize
        self.maxsize = maxsize
        self.threshold = threshold

    def find_ROIs_in_cropped_image(
        self, image: ndarray, resolution: int, method: int = METH_TOP_CONTOUR_SPLIT
    ) -> Tuple[List[ROI], List[int]]:
        """Find ROIs in a sub-image, aka cropped image.
        The only difference with below is that particles touching the border are _not_ eliminated.
        """
        border = 1
        # Add a white border, otherwise when a particle touches a border it is eliminated
        image = cv2.copyMakeBorder(
            image, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,)
        )
        rois, stats = self.find_ROIs_in_image(image, resolution, method)
        ret = [ROI(r.x - border, r.y - border, r.mask) for r in rois]
        return ret, stats

    def find_ROIs_in_image(
        self, image: ndarray, resolution: int, method: int = METH_TOP_CONTOUR_SPLIT
    ) -> Tuple[List[ROI], List[int]]:
        """:return: (list of ROIs, elimination statistics i.e. number of particles eliminated at different stages)"""
        pixel = 25.4 / resolution
        sm_min = (3.1416 / 4) * pow(self.minsize, 2)
        sm_max = (3.1416 / 4) * pow(self.maxsize, 2)
        # part_size_* are in pixel^2
        part_size_min: int = round(sm_min / (pow(pixel, 2)))
        part_size_max: int = round(sm_max / (pow(pixel, 2)))
        assert image.dtype == np.uint8
        height, width = image.shape[:2]
        # Threshold the source image to have a b&w mask
        # mask is white objects on black background
        _th, inv_mask = cv2.threshold(image, self.threshold, 1, cv2.THRESH_BINARY_INV)
        # saveimage(inv_mask, "/tmp/inv_mask.tif")
        self.sanity_check(inv_mask)
        if method & self.LEGACY_COMPATIBLE:
            assert (
                method - self.LEGACY_COMPATIBLE != 0
            ), "Sub-method is mandatory for legacy mode"
            if width > self.Wlimit and height > self.Hlimit:
                return self.find_particles_legacy_way(
                    inv_mask, method, part_size_min, part_size_max
                )
        return self.find_particles_with_method(
            inv_mask, method, part_size_min, part_size_max
        )

    def find_particles_legacy_way(
        self, inv_mask: ndarray, method: int, part_size_min: int, part_size_max: int
    ) -> Tuple[List[ROI], List[int]]:
        # Process image in 2 overlapping parts, split vertically.
        # There is a good side effect that, when borders are all around the image, it works.
        # BUT as well the bad side effect that objects in the middle 20% band can appear if embedded
        # or disappear if too large.
        height, width = inv_mask.shape[:2]
        overlap_size = int(width * self.overlap)
        left_mask = cropnp(inv_mask, top=0, left=0, bottom=height, right=overlap_size)
        left_rois, left_stats = self.find_particles_with_method(
            left_mask, method, part_size_min, part_size_max
        )
        right_mask = cropnp(
            inv_mask,
            top=0,
            left=width - overlap_size,
            bottom=height,
            right=width,
        )
        right_rois, right_stats = self.find_particles_with_method(
            right_mask, method, part_size_min, part_size_max
        )
        # Fix coordinates from right pane
        for right_roi in right_rois:
            right_roi.x += width - overlap_size
        # Merge ROI lists
        key_func = lambda r: roi_unq(r)
        right_by_key = {key_func(ri): ri for ri in right_rois}
        left_by_key = {key_func(le): le for le in left_rois}
        assert len(left_by_key) == len(left_rois)
        assert len(right_by_key) == len(right_rois)
        for right_key, right_roi in right_by_key.items():
            if right_key in left_by_key:
                _left_roi = left_by_key[right_key]  # Unused so far
        # Add different
        for left_key, left_roi in left_by_key.items():
            if left_key not in right_by_key:
                right_by_key[left_key] = left_roi
        return list(right_by_key.values()), [
            ls + rs for ls, rs in zip(left_stats, right_stats)
        ]

    def find_particles_with_method(
        self, inv_mask: ndarray, method: int, part_size_min: int, part_size_max: int
    ) -> Tuple[List[ROI], List[int]]:
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        if method & self.METH_CONNECTED_COMPONENTS:
            # Most reliable from execution/complexity points of view
            return ConnectedComponentsSegmenter().find_particles_via_cc(
                inv_mask,
                part_size_min,
                part_size_max,
                self.max_w_to_h_ratio,
                with_split=False,
            )
        elif method & self.METH_CONNECTED_COMPONENTS_SPLIT:
            return ConnectedComponentsSegmenter().find_particles_via_cc(
                inv_mask,
                part_size_min,
                part_size_max,
                self.max_w_to_h_ratio,
                with_split=True,
            )
        elif method & self.METH_CONTOUR_TREE:
            # Universal, but very slow on noisy images, collapses from seconds to ten of minutes
            return RecursiveContoursSegmenter.find_particles_contour_tree(
                inv_mask, part_size_min, part_size_max, self.max_w_to_h_ratio
            )
        elif method & self.METH_TOP_CONTOUR_SPLIT:
            # Avoid the borders problem
            return ExternalContoursSegmenter.find_particles_contours(
                inv_mask,
                part_size_min,
                part_size_max,
                self.max_w_to_h_ratio,
                split=True,
            )
        else:
            # Fails on 4-borders images (0 contour found)
            return ExternalContoursSegmenter.find_particles_contours(
                inv_mask,
                part_size_min,
                part_size_max,
                self.max_w_to_h_ratio,
                split=False,
            )

    @staticmethod
    def undo_border_lines(inv_mask: ndarray) -> Tuple[int, int, int, int]:
        height, width = inv_mask.shape[:2]
        # Find the 2 border dots in each dimension
        (left, right) = np.where(inv_mask[0] != 0)[0]
        (top, bottom) = np.where(inv_mask[:, 0] != 0)[0]
        # Clear them
        cv2.line(
            img=inv_mask,
            pt1=(0, top),
            pt2=(width - 1, top),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(0, bottom),
            pt2=(width - 1, bottom),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(left, 0),
            pt2=(left, height - 1),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(right, 0),
            pt2=(right, height - 1),
            color=(0,),
            thickness=1,
        )
        return top, left, bottom, right

    @staticmethod
    def sanity_check(inv_mask: ndarray):
        min_bwratio = 25
        nb_black = np.count_nonzero(inv_mask)
        nb_white = inv_mask.shape[0] * inv_mask.shape[1] - nb_black
        bwratiomeas = nb_black / nb_white
        print(f"bwratiomeas: {bwratiomeas}")
        if bwratiomeas > min_bwratio / 100:
            # Note: below message refers to non-inverted mask
            print(
                f"########### WARNING : More than {min_bwratio}% of the segmented image is black ! \nThe associated background image maybe NOK."
            )

    @staticmethod
    def denoise_particles_via_cc(inv_mask: ndarray, part_size_min: int):
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        assert (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ) == (0, 1, 2, 3, 4)
        ret = []
        print("cc filter, initial: ", retval)
        eliminated_1 = 0
        eliminated_l = 0
        eliminated_r = 0
        for cc_id in range(retval):
            x, y, w, h, area_excl_holes = [int(m) for m in stats[cc_id]]
            # Single point
            if area_excl_holes == 1:
                inv_mask[y, x] = 0
                eliminated_1 += 1
                continue
            # More frequent exclusion reasons first
            if w == 1 or h == 1:
                cv2.line(
                    img=inv_mask, pt1=(x, y), pt2=(x + w - 1, y + h - 1), color=(0,)
                )
                eliminated_l += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < part_size_min:
                sub_labels = cropnp(
                    image=labels, top=y, left=x, bottom=y + h, right=x + w
                )
                # noinspection PyUnresolvedReferences # seen as bool by PyCharm
                sub_mask = (sub_labels == cc_id).astype(
                    dtype=np.uint8
                ) * 255  # 255=shape, 0=not in shape
                inv_mask[y : y + h, x : x + w] = np.bitwise_xor(
                    inv_mask[y : y + h, x : x + w], sub_mask
                )
                eliminated_r += 1
                continue
        left = retval - (eliminated_1 + eliminated_l + eliminated_r)
        print(
            "cc filter, eliminated: ",
            eliminated_1,
            eliminated_l,
            eliminated_r,
            " left: ",
            left,
        )
        return ret
