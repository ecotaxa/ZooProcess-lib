from typing import List

import cv2
import numpy as np
from numpy import ndarray

from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.Segmenters.ExternalContours import ExternalContoursSegmenter


class RecursiveContoursSegmenter(object):
    @classmethod
    def find_particles_contour_tree(
        cls, inv_mask: ndarray, s_p_min: int, s_p_max: int, max_w_to_h_ratio: float
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        # In some cases and despite previous steps, the border of the scan goes fully round the image, so
        # there is a single contour!
        # image_3channels = draw_contours(inv_mask, contours, thickness=1)
        # saveimage(image_3channels, Path("/tmp/contours.tif"))
        # Fix by removing it.
        # first_pixel = np.argmax(inv_mask[0] == 255)
        # saveimage(inv_mask, "/tmp/bef_flood.tif")
        # cv2.floodFill(
        #     image=inv_mask,
        #     mask=None,
        #     seedPoint=(first_pixel, 0),
        #     newVal=(0,),
        #     flags=8,  # 8-pixel connectivity, like contour detection does
        # )
        # Segmenter.undo_border_lines(inv_mask)
        # cv2.drawContours(
        #     image=inv_mask,
        #     contours=contours,
        #     contourIdx=0,
        #     color=(0,),
        #     thickness=4
        # )
        # Breach the border
        # cv2.line(inv_mask, (0, 0), (300, 300), (0,), thickness=2)
        # inv_mask = cv2.copyMakeBorder(
        #     inv_mask, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(0,)
        # )
        # saveimage(inv_mask, "/tmp/aft_flood.tif")
        # approx = cv2.CHAIN_APPROX_SIMPLE # 8 min on complicated image
        approx = (
            cv2.CHAIN_APPROX_TC89_L1
        )  # 4 min on complicated image (and different output)
        approx = (
            cv2.CHAIN_APPROX_TC89_KCOS
        )  # 4 min on complicated image (and different output)
        approx = cv2.CHAIN_APPROX_NONE  # 4 min on complicated image
        contours, (hierarchy,) = cv2.findContours(inv_mask, cv2.RETR_TREE, approx)
        # return []
        # root_children_contours = []
        # for a_contour, its_hierarchy in zip(contours, hierarchy):
        #     (
        #         next_contour,
        #         previous_contour,
        #         child_contour,
        #         parent_contour,
        #     ) = its_hierarchy
        #     # In RETR_CCOMP mode we have 2 hierarchies, -1 is enclosing one, other is holes on
        #     if parent_contour == -1:
        #         root_children_contours.append(a_contour)
        # contours = root_children_contours
        # contours, (hierarchy,) = cv2.findContoursLinkRuns(
        #     inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        # )
        # image_3channels = draw_contours(inv_mask, contours, thickness=3)
        # saveimage(image_3channels, Path("/tmp/contours2.tif"))
        # image_3channels2 = draw_contours(inv_mask, root_children_contours, thickness=3)
        # saveimage(image_3channels2, Path("/tmp/contours3.tif"))
        # contours = root_children_contours

        print("Number of RETR_TREE Contours found = " + str(len(contours)))
        ret: List[ROI] = []
        accepted_parents = {-1: 0}  # contour_id, level
        single_point_contour_shape = (1, 1, 2)
        # Optimization
        accepted_parents_get = accepted_parents.get
        for contour_id, (a_contour, its_hierarchy) in enumerate(
            zip(contours, hierarchy)
        ):
            parent_contour = int(its_hierarchy[3])
            # assert parent_contour < contour_id  # Ensure we've seen the parent before
            level = accepted_parents_get(parent_contour)
            if level is None:
                continue
            # More frequent exclusion reasons first
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                continue  # Too small, really
            x, y, w, h = cv2.boundingRect(a_contour)
            # Eliminate if touching the border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                # Keep descending, maybe an embedded shape fits
                accepted_parents[contour_id] = accepted_parents[parent_contour] + 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < s_p_min:
                continue
            # Compute filled area
            contour_mask = ExternalContoursSegmenter.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                continue
            elif area > s_p_max:
                # Keep descending, maybe an embedded shape fits
                accepted_parents[contour_id] = accepted_parents[parent_contour] + 1
                continue
            if level % 2 != 0:
                # Is a contour around a hole
                continue
            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                continue
            roi = ROI(
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
            ret.append(roi)
            # image_3channels = draw_contours(self.image, self.contours)
            # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret


