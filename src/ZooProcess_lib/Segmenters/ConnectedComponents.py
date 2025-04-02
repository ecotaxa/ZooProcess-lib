from typing import List, Tuple

import cv2
import numpy as np
from numpy import ndarray

from ..ROI import ROI
from ..img_tools import cropnp


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
        maybe_kept, filtering_stats[0:1] = cls.prefilter(stats, s_p_min)
        fitting_criteria = []
        in_shape = np.zeros((height, width), dtype=np.uint8)
        for x, y, w, h, area_excl_holes, cc_id in maybe_kept:
            # The cc bounding rect is the whole image minus borders, filter it but don't exclude inside
            # TODO: 2 sub-cases: either the whole area is included,
            #  or there is a band around the whole image
            # if (
            #     x == 0
            #     and y == 0
            #     and x + w == width
            #     and y + h == height
            #     # and area_excl_holes > w * h / 2  # TODO: Not accurate
            # ):
            #     filtering_stats[3] += 1
            #     continue

            # Proceed to more expensive filtering
            holes, filled_mask = cls.get_regions(labels, cc_id, x, y, w, h)
            area = area_excl_holes + np.count_nonzero(holes)
            start_x = x + np.argmax(filled_mask == 255)

            if in_shape[y, start_x] != 0:
                # Excluded
                filtering_stats[2] += 1
                continue

            # Eliminate if touching any border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                cls.prevent_inclusion(in_shape, filled_mask, x, y, w, h)
                filtering_stats[3] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                filtering_stats[4] += 1
                continue
            if area > s_p_max:
                cls.prevent_inclusion(in_shape, filled_mask, x, y, w, h)
                filtering_stats[5] += 1
                continue

            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[6] += 1
                continue

            cls.prevent_inclusion(in_shape, filled_mask, x, y, w, h)

            fitting_criteria.append(
                ROI(
                    features={
                        "BX": int(x),
                        "BY": int(y),
                        "Width": int(w),
                        "Height": int(h),
                        "Area": int(area),
                    },
                    mask=filled_mask,
                    contour=None,
                )
            )
        # ret = cls.remove_in_holes(fitting_criteria, width, height)
        ret = fitting_criteria
        print("Initial", retval, "filter stats", filtering_stats, "left", len(ret))
        return ret

    @classmethod
    def prevent_inclusion(cls, in_shape, filled_mask, x, y, w, h):
        in_shape[y : y + h, x : x + w] = np.bitwise_or(
            in_shape[y : y + h, x : x + w], filled_mask
        )

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
        # Add index after stats, starting at 1 as the first 'component' is the whole image
        offs = 1
        indices = np.reshape(
            np.arange(start=offs, stop=len(cc_stats), dtype=np.uint32),
            (len(cc_stats) - offs, 1),
        )
        ret = np.concatenate((cc_stats[offs:], indices), axis=1)
        # 1-pixel line, including single point
        by_size_1 = ret[:, cv2.CC_STAT_WIDTH] > 1
        ret = ret[by_size_1]
        size_flt = len(by_size_1)
        by_size_2 = ret[:, cv2.CC_STAT_HEIGHT] > 1
        ret = ret[by_size_2]
        size_flt += len(by_size_2)
        # Even if contour was around a filled rectangle it would not meet min criterion
        by_area = ret[:, cv2.CC_STAT_WIDTH] * ret[:, cv2.CC_STAT_HEIGHT]
        ret = ret[by_area > s_p_min]
        area_flt = len(by_area)
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
        labels: ndarray,
        cc_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[ndarray, ndarray]:
        # print("get_regions size:", w * h)
        # Compute filled area
        if False:
            sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                dtype=np.uint8
            )  # 0=not in shape (either around shape or inside), 1=shape
            contours, (hierarchy,) = cv2.findContours(
                obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            sub_mask3 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=sub_mask3,
                contours=contours,
                contourIdx=0,
                color=(255,),
                thickness=cv2.FILLED,
            )  # 0=not in filled shape, 255=filled shape
            # Holes are in second-level of RETR_CCOMP method output
            holes3 = np.zeros_like(obj_mask)  # 0:non-hole 255:hole
            cv2.drawContours(
                image=holes3,
                contours=contours[1:],
                contourIdx=-1,
                color=(255,),
                thickness=cv2.FILLED,  # filled -> inside + contour
            )  # 0=not in filled shape, 255=filled shape
            cv2.drawContours(
                image=holes3,
                contours=contours[1:],
                contourIdx=-1,
                color=(0,),
                thickness=1,  # fix the "contour' part of cv2.FILLED above
            )  # 0=not in filled shape, 255=filled shape
            return holes3, sub_mask3
        if True:
            sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                dtype=np.uint8
            ) * 255  # 0=not in shape (either around shape or inside), 255=shape
            obj_mask2 = cv2.copyMakeBorder(
                obj_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
            )
            cv2.floodFill(
                image=obj_mask2,
                mask=None,
                seedPoint=(0, 0),
                newVal=(128,),
                flags=4,  # 4-pixel connectivity, don't cross a cc border
            )  # 0=not part of shape but inside it i.e. holes, 255=shape, 128=outside border
            sub_mask = cropnp(image=obj_mask2, top=1, left=1, bottom=h + 1, right=w + 1)
            holes = sub_mask == 0  # False:non-hole True:hole
            sub_mask[holes] = 255
            sub_mask[sub_mask == 128] = 0

        return holes, sub_mask
