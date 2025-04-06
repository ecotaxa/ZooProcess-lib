import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy import ndarray

from ..ROI import ROI
from ..img_tools import cropnp, saveimage


class ConnectedComponentsSegmenter:
    def __init__(self, image):
        self.image = image

    @classmethod
    def find_particles_via_cc(
        cls, inv_mask: ndarray, s_p_min: int, s_p_max: int, max_w_to_h_ratio: float
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        labels, retval, stats = cls.extract_ccs(inv_mask)

        print("Number of cc found: ", retval)
        filtering_stats = [0] * 7
        maybe_kept, filtering_stats[0:2] = cls.prefilter(stats, s_p_min)
        ret = []
        # Note: The labels matrix is used for marking exclusion zones as well
        for x, y, w, h, area_excl_holes, cc_id in maybe_kept:
            # Proceed to more expensive filtering

            first_on = np.argmax(labels[y, x : x + w] == cc_id)
            if first_on == 0:
                # Shape was erased, i.e. excluded
                filtering_stats[2] += 1
                continue
            holes, filled_mask = cls.get_regions(

            holes, obj_mask = cls.get_regions(
                labels, cc_id, x, y, w, h, area_excl_holes
            )
            area = area_excl_holes + np.count_nonzero(holes)

            # Eliminate if touching any border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                cls.prevent_inclusion(labels, holes, x, y, w, h)
                filtering_stats[3] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                # print("Excluded region: ", w, h, area_excl_holes, area, s_p_min)
                filtering_stats[4] += 1
                continue
            if area > s_p_max:
                cls.prevent_inclusion(labels, holes, x, y, w, h)
                filtering_stats[5] += 1
                continue
            # Horizontal stripes from scanner bed movement
            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[6] += 1
                continue

            cls.prevent_inclusion(labels, holes, x, y, w, h)

            ret.append(
                ROI(
                    features={
                        "BX": int(x),
                        "BY": int(y),
                        "Width": int(w),
                        "Height": int(h),
                        "Area": int(area),
                    },
                    mask=obj_mask + holes,
                    contour=None,
                )
            )
        print("Initial CCs", retval, "filter stats", filtering_stats, "left", len(ret))
        return ret

    @classmethod
    def prevent_inclusion(cls, shape: ndarray, mask: ndarray, x, y, w, h):
        """
        Mark exclusion zone in shape. "0" in shape means allowed, so warp a bit outside.
        """
        shape[y : y + h, x : x + w] += mask

    @classmethod
    def prefilter(
        cls, cc_stats: ndarray, s_p_min: float
    ) -> Tuple[ndarray, Tuple[int, int]]:
        assert (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ) == (0, 1, 2, 3, 4)
        # Add index after stats, starting at 1, as the first 'component' is the whole image.
        offs = 1
        indices = np.reshape(
            np.arange(start=offs, stop=len(cc_stats), dtype=np.uint32),
            (len(cc_stats) - offs, 1),
        )
        ret = np.concatenate((cc_stats[offs:], indices), axis=1)
        # 1-pixel line, including single point
        by_size_1 = ret[:, cv2.CC_STAT_WIDTH] > 1
        size_flt = len(ret) - int(np.sum(by_size_1))
        ret = ret[by_size_1]
        by_size_2 = ret[:, cv2.CC_STAT_HEIGHT] > 1
        size_flt += len(ret) - int(np.sum(by_size_2))
        ret = ret[by_size_2]
        # Even if contour was around a filled rectangle it would not meet min criterion
        by_area = ret[:, cv2.CC_STAT_WIDTH] * ret[:, cv2.CC_STAT_HEIGHT] > int(s_p_min)
        area_flt = len(ret) - int(np.sum(by_area))
        ret = ret[by_area]
        return ret, (size_flt, area_flt)

    @classmethod
    def extract_ccs(cls, inv_mask):
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        return labels, retval, stats

    @staticmethod
    def get_regions(
        labels: ndarray, cc_id: int, x: int, y: int, w: int, h: int, area_excl: int
    ) -> Tuple[ndarray, ndarray]:
        # before = time.time()
        height, width = labels.shape[:2]
        empty_ratio = h * w // area_excl
        # Compute filled area
        if empty_ratio > 20:
            # It's a bit faster to draw the hole shapes inside sparse shapes
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_contours(
                labels, cc_id, x, y, w, h, height, width
            )
        else:
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_floodfill(labels, cc_id, x, y, w, h, width,
                                                                                       height)
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_floodfill(
                labels, cc_id, x, y, w, h, width, height
            )
        return holes, sub_mask

    @staticmethod
    def get_regions_using_floodfill(
        labels: ndarray, cc_id: int, x, y, w, h, width, height
    ) -> Tuple[ndarray, ndarray]:
        if x == 0 or y == 0 or x + w == width or y + h == height:
            # The shape touches an image border
            sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                dtype=np.uint8
            ) * 255  # 0=not in shape (either around shape or inside), 255=shape
            # Draw lines around for the floodFill to spread all around
            obj_mask = cv2.copyMakeBorder(
                obj_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
            )
        else:
            # Wa can save the copyMakeBorder above by picking around the existing lines
            sub_labels = cropnp(
                image=labels,
                top=y - 1,
                left=x - 1,
                bottom=y + h + 2,
                right=x + w + 2,
            )
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                dtype=np.uint8
            ) * 255  # 0=not in shape (either around shape or inside), 255=shape
        holes = 255 - obj_mask  # 255=around 0=shape 255=holes
        cv2.floodFill(
            holes, mask=None, seedPoint=(0, 0), newVal=(0,)
        )  # 0=around 0=shape 255=holes
        sub_mask = cropnp(image=obj_mask, top=1, left=1, bottom=h + 1, right=w + 1)
        holes = cropnp(image=holes, top=1, left=1, bottom=h + 1, right=w + 1)
        return holes, sub_mask

    @staticmethod
    def get_regions_using_contours(
        labels: ndarray, cc_id: int, x, y, w, h, height, width
    ) -> Tuple[ndarray, ndarray]:
        sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
        # noinspection PyUnresolvedReferences
        obj_mask = (sub_labels == cc_id).astype(
            dtype=np.uint8
        ) * 255  # 0=not in shape (either around shape or inside), 255=shape
        contours, _ = cv2.findContours(
            obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        # Despite all the efforts made to avoid it (e.g. small holes in border lines so contour detection algo
        # can sneak inside particles area), sometimes the first contour is an _outer_ one. It can be fitting perfectly
        # around the border lines to image borders, or, when there are tiny holes in border lines, all holes could be covered
        # in by some particle or noise.
        # Historical algorithm is immune to this problem, because when taking 2 slices, one vertical side is completely opened
        # and contour detection will find everything. But it raises other issues like lost big particles.
        #
        # OTOH, sometimes there could be only 3 full borders, resulting in same contour, but an inner
        # one as usually.
        #
        # First case manifest itself as an inner contour filling nearly all the image. It's geometrically OK as a
        # hole inside the shape, but we get rid of it.
        #
        # ConnectedComponentsSegmenter.debug_cc_comp_contour(obj_mask, contours)
        if x == 0 and y == 0 and x + w == width and y + h == height:
            big_area_threshold = height * width * 8 / 10
            to_remove = ConnectedComponentsSegmenter.find_contours_above(
                contours, big_area_threshold
            )
            if to_remove is not None:
                # print("4 borders")
                contours = list(contours)
                del contours[to_remove]
        # Holes are in second level of RETR_CCOMP method output
        holes = np.zeros_like(obj_mask)  # 0:non-hole 255:hole
        cv2.drawContours(
            image=holes,
            contours=contours[1:],
            contourIdx=-1,
            color=(255,),
            thickness=cv2.FILLED,  # FILLED -> inside + contour
        )  # 0=not in hole, 255=hole
        # Above 'holes' is not pixel-exact as the edges b/w particle and holes is drawn, eliminate them
        holes ^= obj_mask
        sub_mask = obj_mask
        return holes, sub_mask


    @staticmethod
    def find_contours_above(contours, big_area_threshold):
        for contour_ndx in range(1, len(contours)):
            if cv2.contourArea(contours[contour_ndx]) > big_area_threshold:
                return contour_ndx
        return None

    @staticmethod
    def debug_cc_comp_contour(obj_mask, contours):
        dbg_img = np.zeros_like(obj_mask)
        dbg_img_3chan = cv2.merge([dbg_img, dbg_img, dbg_img])
        cv2.drawContours(dbg_img_3chan, contours[0:1], -1, (255, 0, 0), cv2.FILLED)
        cv2.drawContours(dbg_img_3chan, contours[1:], -1, (0, 255, 0), cv2.FILLED)
        saveimage(dbg_img_3chan, Path("/tmp/zooprocess/contours.tif"))
