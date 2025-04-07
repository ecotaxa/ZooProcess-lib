import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from numpy import ndarray

from ..ROI import ROI
from ..img_tools import cropnp, saveimage


class CC:
    __slots__ = "x", "y", "w", "h", "touching", "entire"
    x: int
    y: int
    w: int
    h: int
    touching: bool  # The CC touches one border at least
    entire: bool  # The CC touches all borders

    def __init__(self, x: int, y: int, w: int, h: int, width: int, height: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.touching = x == 0 or y == 0 or x + w == width or y + h == height
        self.entire = w == width and h == height


class ConnectedComponentsSegmenter:
    def __init__(self, image):
        self.image = image

    @classmethod
    def find_particles_via_cc(
        cls, inv_mask: ndarray, s_p_min: int, s_p_max: int, max_w_to_h_ratio: float
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        hnoise = cls.horizontal_noise_ratio(inv_mask)  # takes a few tens of ms
        denoised = False
        if hnoise >= 1:
            before = cls.denoise(inv_mask, s_p_min)
            denoised = True
            labels, retval, stats = cls.extract_ccs(inv_mask)
            print("Noisy! number of initial CCs:", before, "then found: ", retval)
        else:
            labels, retval, stats = cls.extract_ccs(inv_mask)
            print("Number of cc found: ", retval)
        filtering_stats = [0] * 8
        maybe_kept, filtering_stats[0:3] = cls.prefilter(stats, s_p_min, not denoised)
        ret = []
        # Note: The labels matrix is used for marking exclusion zones as well
        for x, y, w, h, area_excl_holes, cc_id in maybe_kept:
            # Proceed to more expensive filtering

            first_on = np.argmax(labels[y, x : x + w] == cc_id)
            if first_on == 0:
                # Shape was erased, i.e. excluded
                filtering_stats[3] += 1
                continue

            cc = CC(x, y, w, h, width, height)

            holes, obj_mask = cls.get_regions(labels, cc_id, cc, area_excl_holes)
            area = area_excl_holes + np.count_nonzero(holes)

            # Eliminate if touching any border
            if cc.touching:
                cls.prevent_inclusion(labels, holes, cc)
                filtering_stats[4] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                # print("Excluded region: ", w, h, area_excl_holes, area, s_p_min)
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                cls.prevent_inclusion(labels, holes, cc)
                filtering_stats[6] += 1
                continue
            # Horizontal stripes from scanner bed movement
            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[7] += 1
                continue

            cls.prevent_inclusion(labels, holes, cc)

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
    def prevent_inclusion(cls, shape: ndarray, mask: ndarray, cc: CC):
        """
        Mark exclusion zone in shape. "0" in shape means allowed, so warp a bit outside.
        """
        shape[cc.y : cc.y + cc.h, cc.x : cc.x + cc.w] += mask

    @classmethod
    def prefilter(
        cls, cc_stats: ndarray, s_p_min: float, do_square: bool
    ) -> Tuple[ndarray, Tuple[int, int, int]]:
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
        # Even if all pixels formed a 1-pixel-wide square, adding the hole inside would not make enough
        if do_square:
            min_pixels = int(math.sqrt(s_p_min))
            by_holes_area = ret[:, cv2.CC_STAT_AREA] > min_pixels
            holes_area_flt = len(ret)
            ret = ret[by_holes_area]
            holes_area_flt -= len(ret)
        else:
            holes_area_flt = 0
        # Even if contour was around a filled rectangle it would not meet min criterion
        by_area = ret[:, cv2.CC_STAT_WIDTH] * ret[:, cv2.CC_STAT_HEIGHT] > int(s_p_min)
        area_flt = len(ret)
        ret = ret[by_area]
        area_flt -= len(ret)
        # 1-pixel lines
        # TODO: a OR here (np.where or np.select?)
        by_size_1 = ret[:, cv2.CC_STAT_WIDTH] > 1
        size_flt = len(ret)
        ret = ret[by_size_1]
        by_size_2 = ret[:, cv2.CC_STAT_HEIGHT] > 1
        ret = ret[by_size_2]
        size_flt -= len(ret)
        return ret, (holes_area_flt, area_flt, size_flt)

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
        labels: ndarray, cc_id: int, cc: CC, area_excl: int
    ) -> Tuple[ndarray, ndarray]:
        # before = time.time()
        empty_ratio = cc.h * cc.w // area_excl
        # Compute filled area
        if empty_ratio > 20:
            # It's a bit faster to draw the hole shapes inside sparse shapes
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_contours(
                labels, cc_id, cc
            )
        else:
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_floodfill(
                labels, cc_id, cc
            )
        return holes, sub_mask

    @staticmethod
    def get_regions_using_floodfill(
        labels: ndarray, cc_id: int, cc: CC
    ) -> Tuple[ndarray, ndarray]:
        if cc.touching:
            # The shape touches an image border
            sub_labels = cropnp(
                image=labels, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
            )
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                np.uint8
            )  # 0=not in shape (either around shape or inside), 1=shape
            # Draw lines around for the floodFill to spread all around
            obj_mask = cv2.copyMakeBorder(
                obj_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
            )
        else:
            # Wa can save the copyMakeBorder above by picking around the existing lines
            sub_labels = cropnp(
                image=labels,
                top=cc.y - 1,
                left=cc.x - 1,
                bottom=cc.y + cc.h + 2,
                right=cc.x + cc.w + 2,
            )
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                np.uint8
            )  # 0=not in shape (either around shape or inside), 1=shape
        holes = 1 - obj_mask  # 1=around 0=shape 1=holes
        cv2.floodFill(
            holes, mask=None, seedPoint=(0, 0), newVal=(0,)
        )  # 0=around 0=shape 1=holes
        sub_mask = cropnp(
            image=obj_mask, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1
        )
        holes = cropnp(image=holes, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1)
        return holes, sub_mask

    @staticmethod
    def get_regions_using_contours(
        labels: ndarray,
        cc_id: int,
        cc: CC,
    ) -> Tuple[ndarray, ndarray]:
        # return np.ones((cc.h, cc.w), dtype=np.uint8), np.ones(
        #     (cc.h, cc.w), dtype=np.uint8
        # )
        # ESSAI sur le masque directement pour éviter la recopie
        sub_labels = cropnp(
            image=labels, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
        )
        # noinspection PyUnresolvedReferences
        obj_mask = (sub_labels == cc_id).astype(
            np.uint8
        )  # 0=not in shape (either around shape or inside), 1=shape
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
        if cc.entire:
            big_area_threshold = cc.h * cc.w * 8 / 10
            to_remove = ConnectedComponentsSegmenter.find_contours_above(
                contours, big_area_threshold
            )
            if to_remove is not None:
                # print("4 borders")
                contours = list(contours)
                del contours[to_remove]
        # Holes are in second level of RETR_CCOMP method output
        holes = np.zeros_like(obj_mask)
        cv2.drawContours(
            image=holes,
            contours=contours[1:],
            contourIdx=-1,
            color=(1,),
            thickness=cv2.FILLED,  # FILLED → inside + contour
        )  # 0=not in hole, 1=hole
        # Above 'holes' is not pixel-exact as the edges b/w particle and holes is drawn, eliminate them
        holes ^= obj_mask
        return holes, obj_mask

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

    @classmethod
    def horizontal_noise_ratio(cls, inv_mask: ndarray) -> int:
        """Number of != pixels from one line to another, in central region of the image"""
        height, width = inv_mask.shape[:2]
        excluded = int(height * 0.9) // 2
        orig = cropnp(image=inv_mask, top=excluded, bottom=-excluded)
        below = cropnp(image=inv_mask, top=excluded + 1, bottom=-excluded + 1)
        diff = orig ^ below
        ret = int(np.sum(diff) * 100 / ((height - excluded) * width))
        return ret

    @classmethod
    def denoise(cls, inv_mask: ndarray, s_p_min: float) -> int:
        """Remove connected components which cannot end up in final result"""
        min_pixels = int(math.sqrt(s_p_min))
        retval, labels = cv2.connectedComponents(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S
        )

        _unique, counts = np.unique(labels, return_counts=True)
        to_erase = np.nonzero(counts <= min_pixels)
        labels2 = np.isin(labels, to_erase)
        inv_mask[labels2] = 0
        return int(retval)
