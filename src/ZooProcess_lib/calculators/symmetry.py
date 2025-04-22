import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from ZooProcess_lib.ImageJLike import parseInt
from ZooProcess_lib.img_tools import saveimage


def imagej_like_symmetry(
    mask: np.ndarray,
    x_centroid: np.float64,
    y_centroid: np.float64,
    angle: float,
    area: int,
    pixel_size: float,
):
    """
    Params:
    mask: binary image with 0=background, 255=foreground
    w, h: initial width and height (float or int)
    bx, by: top-left corner of the initial selection (float or int)
    x, y: initial center point (float or int)
    angle: rotation angle (float or int)
    pixel: pixel size (float)
    area: reference area (float or int)
    """
    # nansum: counter (int, must be initialized)
    height, width = mask.shape

    # ---------------- centering ----------------------
    # print("--- Centering ---")
    diag = math.sqrt(width * width + height * height)  # Ensure float for pow
    diag *= 1.2
    diag_int = int(diag)  # Image dimensions must be integers

    # Create the new image for vignette A
    vignette_a = np.zeros((diag_int, diag_int), dtype=np.uint8)

    # Calculate the "pasting" position for centering around the centroid
    pos_x = (diag / 2) - x_centroid
    pos_y = (diag / 2) - y_centroid

    pos_x, pos_y = parseInt(pos_x), parseInt(pos_y)
    # ImageJ "paste", which is imitated below, has big tolerances in that:
    #  - if source does not fit in dest rectangle, but fits in the image, it will copy the source in the middle
    #    of destination image, ignoring the dest rectangle
    #  - if source does not fit in dest image, it will crop the source so it fits into dest rectangle/the image
    # Using numpy it just fails as there is not enough "space". Imitating ImageJ behavior would lead to a badly
    # centered object, the centroid would not be in the middle of the image, which would void a bit the measurements.
    try:
        vignette_a[pos_y : pos_y + height, pos_x : pos_x + width] = mask
    except ValueError:
        image_3channels = cv2.merge([mask, mask, mask])
        cv2.drawMarker(image_3channels, (int(x_centroid), int(y_centroid)), (255, 0, 0))
        saveimage(image_3channels, Path(f"/tmp/vignette_a_{pos_x}_{pos_y}_problem.png"))
        return 0, 0, 0

    vignette_a = rotate_image(vignette_a, -angle + 180, (0,))
    # vignette_a particle is now horizontal on its largest axis
    # The image is no longer binary [0,255] as the rotation introduced interpolations

    # if 100 <= diag <= 400:
    # saveimage(vignette_a, Path(f"/tmp/vignette_a_{pos_x}_{pos_y}.png"))
    # vignette_a = loadimage(Path("/tmp/vignetteA-ref-orig80.tif"))
    # vignette_a = vignette_a[:, :, 0]

    h_rot, w_rot = vignette_a.shape

    # Variables for the vertical search
    c = 0  # Counter for found pairs of points
    # --------- Interval normalization by pixel size --------------
    step = math.floor(0.1 / pixel_size)
    num_steps = int(1 + w_rot / step)
    # num_steps = w_rot
    point_a = [0] * num_steps
    point_b = [0] * num_steps
    dif = [0] * num_steps
    flag = 0  # Reset flag for each column

    for x in range(0, w_rot, step):
        for y in range(h_rot - 1):
            pa = vignette_a[y, x]
            pb = vignette_a[y + 1, x]
            pd = abs(pa - pb)
            if pd > 100 and flag == 0:  # First transition found
                flag = 1
                point_a[c] = y
                continue
            if pd > 100 and flag == 1:  # Second transition found
                point_b[c] = y
                c += 1  # Increment c ONLY when a valid pair is found
                break  # A and B, i.e. start and end, were found for this Y
        flag = 0

    # Calculate differences only for the valid pairs
    for k in range(c):
        dif[k] = point_b[k] - point_a[k]

    # ---------- max and mean of the difference -----------
    mean_df = 0
    max_val = 0

    # Spec says: "Thickness ratio : relation between the maximum thickness of an object and
    # the average thickness of the object excluding the maximum". Seems that the "c - 1" below assumes
    # that largest measurement is at the end of the list/array.
    for k in range(c - 1):  # Iterate over all calculated differences
        mean_df += dif[k]
        if dif[k] > max_val:
            max_val = dif[k]
    if c > 0:
        mean_df = mean_df / c
    else:
        mean_df = 0  # Avoid division by zero

    # Calculate the ratio, handle division by zero
    if mean_df != 0:
        thick_ratio = float(max_val) / mean_df
    else:
        thick_ratio = float("nan")  # Reproduces the isNaN behavior

    # ------------- axis 1 (Horizontal Symmetry) ----------------------------
    vignette_b = np.copy(vignette_a)
    horiz_flipped_vignette_b = cv2.flip(vignette_b, 0)
    diff_a_and_b = vignette_a - horiz_flipped_vignette_b

    (h_diff_histo, _) = np.histogram(diff_a_and_b, 256, range=(0, 255))
    # h_diff_histo[0] contains the count of pixels with value 0 (same pixels on both sides)
    h_area_sym = h_diff_histo[0]
    if area != 0:
        symetry_h = (h_area_sym / 2.0) / area
    else:
        symetry_h = float("nan")  # Or another error value

    # ------------- axis 2 (Vertical Symmetry) ----------------------------
    vignette_c = np.copy(vignette_a)
    vert_flipped_vignette_c = cv2.flip(vignette_c, 1)

    # Original ImageJ code has a "Make Binary" here for B before the diff, so we end up comparing a
    # binary image with a non-binary one (greys in A introduced by the rotation)
    diff_a_and_c = vignette_a - vert_flipped_vignette_c

    (v_diff_histo, _) = np.histogram(diff_a_and_c, 256, range=(0, 255))
    v_area_sym = v_diff_histo[0]  # Black pixels (zero difference)
    if area != 0:
        symetry_v = (v_area_sym / 2.0) / area  # Float division
    else:
        symetry_v = float("nan")

    # Return results
    if math.isnan(thick_ratio):
        thick_ratio = 1  # Behavior from original code for NaN
        # nansum += 1

    return symetry_h, symetry_v, thick_ratio


def make_binary(crop: np.ndarray, threshold: int) -> np.ndarray:
    crop[crop < threshold] = 0
    crop[crop >= threshold] = 255
    crop = 255 - crop
    return crop


def rotate_image(image: np.ndarray, angle: float, background: Tuple):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image,
        rot_mat,
        image.shape[1::-1],
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    return result
