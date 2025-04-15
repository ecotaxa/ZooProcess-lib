from typing import List

import cv2
import numpy as np
from numpy import ndarray


def enclosing_rect(stats: List[ndarray]) -> ndarray:
    """Return the smallest rectangle enclosing all rectangles of list,
    with area of returned value"""
    min_x, min_y = 100000, 100000
    max_x, max_y = -1, -1
    sum_area = 0
    for x, y, width, height, area in stats:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + width)
        max_y = max(max_y, y + height)
        sum_area += area
    return np.asarray(
        [min_x, min_y, max_x - min_x, max_y - min_y, sum_area],
        dtype=np.uint32,  # Fails when max_x or max_y is -1 and that's very OK
    )


def all_relative_to(stats: List[ndarray], enclosing: ndarray) -> List[ndarray]:
    ret = []
    x_ncl, y_ncl = enclosing[:2]
    for a_stat in stats:
        a_stat_copy = a_stat.copy()
        a_stat_copy[cv2.CC_STAT_LEFT] -= x_ncl
        a_stat_copy[cv2.CC_STAT_TOP] -= y_ncl
        ret.append(a_stat_copy)
    return ret


def all_translated_by(stats: List[ndarray], delta_x: int) -> List[ndarray]:
    ret = []
    for a_stat in stats:
        a_stat_copy = a_stat.copy()
        a_stat_copy[cv2.CC_STAT_LEFT] += delta_x
        ret.append(a_stat_copy)
    return ret
