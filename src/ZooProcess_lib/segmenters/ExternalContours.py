from typing import List

import cv2
import numpy as np

from .ConnectedComponents import ConnectedComponentsSegmenter
from ..ROI import ROI


class ExternalContoursSegmenter:
    @classmethod
    def find_particles(
        cls, inv_mask: np.ndarray, s_p_min: int, s_p_max: int, max_w_to_h_ratio: float
    ) -> List[ROI]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        height, width = inv_mask.shape[:2]
        contours, _ = cv2.findContours(
            inv_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,  # same as cv2.CHAIN_APPROX_NONE but returns less data
        )
        if len(contours) <= 1:
            print("0 or 1 contour!")
            return ConnectedComponentsSegmenter.find_particles_via_cc(
                inv_mask, s_p_min, s_p_max, max_w_to_h_ratio
            )

        print("Number of RETR_EXTERNAL Contours found = " + str(len(contours)))
        ret: List[ROI] = []
        single_point_contour_shape = (1, 1, 2)
        filtering_stats = [0] * 8
        for a_contour in contours:
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                filtering_stats[0] += 1
                continue
            x, y, w, h = cv2.boundingRect(a_contour)
            # Eliminate if touching the border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                filtering_stats[1] += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            # -> don't bother drawing the contour, which is expensive
            if w * h < s_p_min:
                filtering_stats[2] += 1
                continue
            contour_mask = cls.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                filtering_stats[6] += 1
                continue
            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[7] += 1
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
        print(
            "Initial contours", len(contours), "filter stats", filtering_stats, "left", len(ret)
        )
        # image_3channels = draw_contours(self.image, self.contours)
        # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret

    @staticmethod
    def draw_contour(contour, x, y, w, h) -> np.ndarray:
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
