from typing import List, Set, Tuple

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
        print("Number of cc found: ", retval)
        embedded = set()
        filtering_stats = [0] * 8
        for cc_id in range(1, retval):  # First 'component' is the whole image
            if cc_id in embedded:
                filtering_stats[0] += 1
                continue
            x, y, w, h, area_excl_holes = [int(m) for m in stats[cc_id]]
            # More frequent exclusion reasons first
            if w == 1 or h == 1:
                filtering_stats[1] += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < s_p_min:
                filtering_stats[2] += 1
                continue
            # The cc bounding rect is the whole image minus borders, filter it but don't exclude inside
            # TODO: 2 sub-cases: either the whole area is included,
            #  or there is a band around the whole image
            if (
                x == 0
                and y == 0
                and x + w == width
                and y + h == height
                # and area_excl_holes > w * h / 2  # TODO: Not accurate
            ):
                filtering_stats[3] += 1
                continue
            # Proceed to more expensive filtering
            sub_labels, holes, filled_mask = cls.get_regions(labels, cc_id, x, y, w, h)
            area = area_excl_holes + holes.sum()
            # Eliminate if touching some border (but not all of them)
            if x == 0 or y == 0 or x + w == width or y + h == height:
                # embedded.update(hole_ids)
                cls.forbid_inside_objects(sub_labels * holes, cc_id, embedded)
                filtering_stats[4] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                # embedded.update(hole_ids)
                cls.forbid_inside_objects(sub_labels * holes, cc_id, embedded)
                filtering_stats[6] += 1
                continue

            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[7] += 1
                continue

            # embedded.update(hole_ids)
            cls.forbid_inside_objects(sub_labels * holes, cc_id, embedded)

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
        print("Initial", retval, "filter stats", filtering_stats, "left", len(ret))
        return ret

    @staticmethod
    def get_regions(
        labels: ndarray,
        cc_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        # print("get_regions size:", w * h)
        # Compute filled area
        sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
        obj_mask = (sub_labels == cc_id).astype(
            dtype=np.uint8
        ) * 255  # 0=not in shape (either around shape or inside), 255=shape
        if True:
            contours, (hierarchy,) = cv2.findContours(
                obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            # contours, (hierarchy,) = cv2.findContoursLinkRuns(obj_mask)
            sub_mask3 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=sub_mask3,
                contours=contours,
                contourIdx=0,
                color=(255,),
                thickness=cv2.FILLED,
            )  # 0=not in filled shape, 255=filled shape
            holes3 = np.zeros_like(obj_mask)
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
            holes3 = holes3 == 255
            # if x == 0 and y == 0:
            #     saveimage(sub_mask3, "/tmp/contour_sub.tif")
            # holes2 = np.bitwise_xor(sub_mask, sub_mask3) == 255
            # holes_id = np.unique(np.where(sub_labels > cc_id)).tolist()
            return sub_labels, holes3, sub_mask3  # , holes_id
        if False:
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            sub_mask2 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=sub_mask2,
                contours=contours,
                contourIdx=0,
                color=(255,),
                thickness=cv2.FILLED,
            )  # 0=not in filled shape, 255=filled shape
            holes2 = np.bitwise_xor(obj_mask, sub_mask2) == 255
            return sub_labels, holes2, sub_mask2  # , holes_id
        if False:
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

        if x == 0:
            pass

        return sub_labels, holes, sub_mask

    @staticmethod
    def forbid_inside_objects(other: ndarray, contour_id: int, embedded: Set[int]):
        # Check for embedded objects
        in_holes = np.unique(other).tolist()
        if len(in_holes) > 1:
            in_holes.remove(0)
            # for a_hole in in_holes:
            #     assert a_hole > contour_id
            embedded.update(in_holes)
