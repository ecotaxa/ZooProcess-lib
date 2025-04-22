import math
from math import log, floor

import cv2
import numpy as np

from .EDM import euclidean_distance_map
from ..calculators.Wand import Wand


def fractal_mp(mask: np.ndarray):
    """Sum of EDM of the mask and its inverse + regression"""
    larger_mask = enlarged_mask(mask)

    edm1_round = ij_like_EDM(larger_mask)
    edm2_round = ij_like_EDM(1 - larger_mask)

    # EDM(mask) + EDM(inverse(mask))
    edm_sum = edm1_round + edm2_round

    areas, logs = sum_areas_and_logs(edm_sum)

    suma = sum(areas)
    sumg = sum(logs)
    moyenneg = sumg / len(logs)
    moyennea = suma / len(areas)
    # Slope computation
    secartg = 0
    secarta = 0
    for a_log, an_area in zip(logs, areas):
        ecartgcar = pow(a_log - moyenneg, 2)
        secartg += ecartgcar
        ecartacar = pow(an_area - moyennea, 2)
        secarta += ecartacar
    stdg = secartg * 1 / (len(logs))
    stdg = pow(stdg, 0.5)
    stda = secarta * 1 / (len(areas))
    stda = pow(stda, 0.5)
    ret = 2 - stda / stdg
    return ret, areas


def sum_areas_and_logs(edm_sum: np.ndarray):
    lg = 0
    iterations = 40
    logs = []
    areas = []
    for k in range(1, iterations + 1):
        y = round(pow(1.1, k))
        if lg != y:
            lg = y
            area = number_of_pixels_below(edm_sum, lg)
            areas.append(log(area))
            logs.append(log(2 * lg))
            # print(
            #     "idx ",
            #     index,
            #     "lg ",
            #     lg,
            #     " -> area ",
            #     area,
            #     " -> log(area) ",
            #     areas[index],
            # )
    return areas, logs


def ij_like_EDM(mask):
    """Compute Euclidian Distance Map of the mask, returned as an uint8 image."""
    edm = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_3)
    # edm = euclidean_distance_map(mask) # Accurate but really, really slow
    edm_round = edm + 0.5  # float -> int Java rounding
    edm_round[edm_round > 255] = 255  # Saturate
    return edm_round.astype(np.uint8)


def number_of_pixels_below(image: np.ndarray, threshold: int) -> int:
    return np.count_nonzero(image <= threshold)


def enlarged_mask(mask: np.ndarray) -> np.ndarray:
    """Return a centered copy of mask inside a larger frame"""
    H, L = mask.shape
    if L >= 200 and H >= 200:
        Lf = 2 * L
        Hf = 2 * H
    else:
        Lf = 4 * L
        Hf = 4 * H
    Xs = floor(Lf / 2 - L / 2)
    Ys = floor(Hf / 2 - H / 2)
    ret = np.zeros((Hf, Lf), dtype=np.uint8)
    ret[Ys : Ys + H, Xs : Xs + L] = mask
    return ret


def get_traced_perimeter(
    x_points: np.ndarray, y_points: np.ndarray, n_points: int
) -> float:
    """
    Returns the perimeter length of ROIs created using the
    wand tool and the particle analyzer. The algorithm counts
    edge pixels as 1 and corner pixels as sqrt(2). It does this by
    calculating the total length of the ROI boundary and subtracting
    2-sqrt(2) for each non-adjacent corner. For example, a 1x1 pixel
    ROI has a boundary length of 4 and 2 non-adjacent edges so the
    perimeter is 4-2*(2-sqrt(2)). A 2x2 pixel ROI has a boundary length
    of 8 and 4 non-adjacent edges so the perimeter is 8-4*(2-sqrt(2)).
    """
    sum_dx = 0
    sum_dy = 0
    n_corners = 0
    dx1 = x_points[0] - x_points[n_points - 1]
    dy1 = y_points[0] - y_points[n_points - 1]
    side1 = abs(dx1) + abs(dy1)  # One of these is 0
    corner = False

    for i in range(n_points):
        next_i = i + 1
        if next_i == n_points:
            next_i = 0
        dx2 = x_points[next_i] - x_points[i]
        dy2 = y_points[next_i] - y_points[i]
        sum_dx += abs(dx1)
        sum_dy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)

        if side1 > 1 or not corner:
            corner = True
            n_corners += 1
        else:
            corner = False

        dx1 = dx2
        dy1 = dy2
        side1 = side2

    return sum_dx + sum_dy - (n_corners * (2 - math.sqrt(2)))


def ij_perimeter(mask: np.ndarray) -> float:
    wand = Wand(mask)
    x_start = int(np.argmax(mask != 0))
    wand.auto_outline(int(x_start), 0)
    ret = get_traced_perimeter(wand.xpoints, wand.ypoints, wand.npoints)
    return float(ret)
