from math import log, floor

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_bf


def fractal_mp(mask: np.ndarray):
    """Sum of EDM of the mask and its inverse + linear (?) regression"""
    larger_mask = enlarged_mask(mask)

    edm1_round = ij_like_EDM(larger_mask)
    edm2_round = ij_like_EDM(1 - larger_mask)

    edm_sum = edm1_round + edm2_round

    areas, logs = sum_areas_and_logs(edm_sum)

    suma = sum(logs)
    sumg = sum(areas)
    moyenneg = sumg / len(logs) + 1
    moyennea = suma / len(areas) + 1
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
    index = 0
    logs = []
    areas = []
    for k in range(1, iterations + 1):
        y = round(pow(1.1, k))
        if lg != y:
            lg = y
            area = number_of_pixels_below(edm_sum, lg)
            areas.append(log(area))
            logs.append(log(2 * lg))
            # print("idx ", index, " -> area ", area, " -> log(area) ", areas[index])
            index += 1
    return areas, logs


def ij_like_EDM(larger_mask):
    edm = cv2.distanceTransform(larger_mask, cv2.DIST_L2, cv2.DIST_MASK_5)
    # edm = distance_transform_edt(larger_mask)
    # edm = distance_transform_bf(larger_mask)
    # if bidou:
    #     edm2 = (65535 - (edm * 128)).astype(np.uint16)
    #     edm3 = edm2 + 21 / 41
    #     edm4 = edm3 / 128
    #     edm_round = (edm2 + 0.5).astype(np.uint8)
    # else:
    #     edm_round = (edm + 0.5).astype(np.uint8)
    edm_round = (edm + 0.5).astype(np.uint8)
    return edm_round


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
