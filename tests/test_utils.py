import csv
from pathlib import Path
from typing import List, Dict, Type

import cv2
import numpy as np

from ZooProcess_lib.Zooscan_convert import CV2_VERTICAL_FLIP_CODE
from ZooProcess_lib.img_tools import loadimage, rotate90cc, saveimage


def reverse_convert(raw_sample_file):
    """Inverse transformation as convert to 8 bits, to have comparable pixels in debug"""
    raw_sample_image = loadimage(raw_sample_file, type=cv2.IMREAD_UNCHANGED)
    raw_sample_image = cv2.flip(raw_sample_image, CV2_VERTICAL_FLIP_CODE)
    return rotate90cc(raw_sample_image)


def diff_actual_with_ref_and_source(
    expected_image, actual_image, raw_sample_image, tolerance=0
):
    e_cum = 0
    for l in range(expected_image.shape[0]):
        e_cum += print_diff(
            expected_image, actual_image, l, raw_sample_image, tolerance
        )
        if e_cum > 10000:
            print("Stop for image")
            break
    return e_cum


def save_diff_image(
    expected_image: np.ndarray, actual_image: np.ndarray, path: Path
) -> None:
    abs_diff = np.abs(expected_image.astype(np.int16) - actual_image)
    # expand differences in luminance domain
    abs_diff *= 100
    abs_diff = 255 - (np.clip(abs_diff, 0, 255).astype(np.uint8))
    # if abs_diff.shape[2] == 4:  # Hard-code the alpha channel to fully solid
    #     abs_diff[:, :, 3] = 255
    saveimage(abs_diff, path)


def print_line(array, index):
    line = array[index]
    print("-------------")
    print(",".join([str(int(elem)) for elem in line]))


def print_diff(array_e, array_a, line_num, comp_img, tolerance) -> int:
    ret = 0
    line_e = array_e[line_num].tolist()
    line_a = array_a[line_num].tolist()
    for ndx, elem in enumerate(line_e):
        delta = abs(line_a[ndx] - elem)
        if delta > tolerance:
            print(
                f"lin {line_num} diff @{ndx}: seen {line_a[ndx]} vs exp {elem}, src {comp_img[line_num][ndx]}"
            )
            ret += 1
            if ret > 20:
                print("Stop for line")
                break
    return ret


def read_measures_csv(csv_file: Path, typings: Dict[str, Type]) -> List[Dict]:
    ret = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for a_line in reader:
            to_add = {}
            for k, v in typings.items():
                if k in a_line:
                    typing = float if typings[k] == np.float64 else typings[k]
                    try:
                        to_add[k] = typing(a_line[k])
                    except ValueError as e:
                        # Some theoretically int features are stored as floats in CSVs
                        flt = float(a_line[k])
                        if int(flt) == flt:
                            to_add[k] = typing(flt)
            ret.append(to_add)
    return ret


def read_ecotaxa_tsv(
    csv_file: Path,
    typings: Dict[str, Type],
) -> List[Dict]:
    ret = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        first_line = True
        for a_line in reader:
            if first_line:
                first_line = False
                continue
            to_add = {}
            for k, v in typings.items():
                if k in a_line:
                    typing = float if typings[k] == np.float64 else typings[k]
                    to_add[k] = typing(a_line[k])
                    # print("TSV: ", a_line[k], to_add[k])
            ret.append(to_add)
    return ret
