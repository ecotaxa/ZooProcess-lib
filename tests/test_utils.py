from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Dict, Type, Tuple

import cv2
import numpy as np

from ZooProcess_lib.Converter import CV2_VERTICAL_FLIP_CODE
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


def diff_actual_with_ref(
    expected_image, actual_image, tolerance=0
):
    e_cum = 0
    for l in range(expected_image.shape[0]):
        e_cum += print_diff_no_source(
            expected_image, actual_image, l, tolerance
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


def print_diff_no_source(array_e, array_a, line_num, tolerance) -> int:
    ret = 0
    line_e = array_e[line_num].tolist()
    line_a = array_a[line_num].tolist()
    for ndx, elem in enumerate(line_e):
        delta = abs(line_a[ndx] - elem)
        if delta > tolerance:
            print(
                f"lin {line_num} diff @{ndx}: seen {line_a[ndx]} vs exp {elem}"
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


def visual_diffs(different, not_in_reference, not_in_actual, sample, tgt_img):
    for a_diff in different:
        a_ref, an_act = a_diff
        print(a_ref)
        dict_diff = {
            k: str(v) + " vs " + str(an_act[k])
            for k, v in a_ref.items()
            if an_act[k] != v
        }
        print("<->", an_act, " : ", dict_diff)
        cv2.rectangle(
            tgt_img,
            (an_act["BX"], an_act["BY"]),
            (an_act["BX"] + an_act["Width"], an_act["BY"] + an_act["Height"]),
            (0,),
            1,
        )
        # for a_roi in found_rois:
        #     height, width = a_roi.mask.shape
        #     if an_act["Width"] == width and an_act["Height"] == height:
        #         Signal the diff with the mask shifted a bit
        # tgt_img[
        #     an_act["BY"] + 100 : an_act["BY"] + 100 + height,
        #     an_act["BX"] - 150 : an_act["BX"] - 150 + width,
        # ] = (
        #     a_roi.mask * 255
        # )
    for num, an_act in enumerate(not_in_reference):
        # vig = cropnp(
        #     image=vis1,
        #     top=an_act["BY"],
        #     left=an_act["BX"],
        #     bottom=an_act["BY"] + an_act["Height"],
        #     right=an_act["BX"] + an_act["Width"],
        # )
        print(f"extra {num}:{an_act}")
        # saveimage(vig, f"/tmp/zooprocess/diff_{num}.png")
        cv2.rectangle(
            tgt_img,
            (an_act["BX"], an_act["BY"]),
            (an_act["BX"] + an_act["Width"], an_act["BY"] + an_act["Height"]),
            (0,),
            1,
        )
    # for num, a_ref in enumerate(expected):
    #     cv2.rectangle(
    #         tgt_img,
    #         (a_ref["BX"], a_ref["BY"]),
    #         (a_ref["BX"] + a_ref["Width"], a_ref["BY"] + a_ref["Height"]),
    #         (0,),
    #         1,
    #     )
    for num, a_ref in enumerate(not_in_actual):
        print(f"missing ref {num}:{a_ref}")
        cv2.rectangle(
            tgt_img,
            (a_ref["BX"], a_ref["BY"]),
            (a_ref["BX"] + a_ref["Width"], a_ref["BY"] + a_ref["Height"]),
            (0,),
            4,
        )
    if len(different) or len(not_in_actual) or len(not_in_reference):
        saveimage(tgt_img, f"/tmp/zooprocess/dif_on_{sample}.tif")


def compare_vignettes(ref_thumbs_dir: Path, act_thumbs_dir: Path, threshold: int):
    # Tolerate a different extension as long as bitmaps are similar
    ref_images = list_images_in(ref_thumbs_dir)
    act_images = list_images_in(act_thumbs_dir)
    ref_images = color_vignettes_removed(ref_images)
    # Basic matches, it will fail for projects with the historical bug on the 20% band
    assert len(ref_images) == len(
        act_images
    ), f"Different number of vignettes, expected {len(ref_images)} actual {len(act_images)}"
    # assert set(ref_images.keys()) == set(act_images.keys())
    # Tricky matches as we have != numbering
    ref_by_size = categorize_by_size(ref_images)
    act_by_size = categorize_by_size(act_images)
    in_error = False
    for a_size, ref_images in ref_by_size.items():
        if a_size not in act_by_size:
            print(f"Size {a_size} not found in actual thumbnails")
            in_error = True
            continue
        maybe_same = act_by_size[a_size]
        unique = False
        if len(ref_images) == 1 and len(maybe_same) == 1:
            unique = True
        for a_ref_name, a_ref_img in ref_images.items():
            for act_name, act_img in maybe_same.items():
                # Ref jpegs are lossy (see some histograms above 243), we need to compare with a tolerance
                abs_diff = np.abs(a_ref_img.astype(np.int16) - act_img.astype(np.int16))
                max_diff = np.max(abs_diff)
                if max_diff in (1, 2):
                    print(f"Matched {a_ref_name}.jpg with {act_name}.png")
                    break
                diff_summ = np.sum(abs_diff)
                obj_pixels = np.count_nonzero(act_img <= threshold)
                avg_diff = diff_summ / obj_pixels
                # if avg_diff <= 12:
                #     break
                if unique:
                    print(
                        f"imagej {ref_thumbs_dir / a_ref_name}.jpg {act_thumbs_dir / act_name}.png diff={diff_summ} max={max_diff} pixels={obj_pixels} avg={avg_diff}"
                    )
            else:
                print(f"Image {a_ref_name} not matched in actual thumbnails")
                in_error = True
    assert not in_error, "Error in vignettes comparison, see above for details."


def list_images_in(image_dir: Path):
    ret = {}  # key: base name, value:image read np.ndarray
    for a_file in os.listdir(image_dir):
        if a_file.endswith(".jpg") or a_file.endswith(".png"):
            img_file = os.path.join(image_dir, a_file)
            img_data = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            file_without_ext = a_file[:-4]
            ret[file_without_ext] = img_data
            # Apparently jpg encoding damaged pure white
            # if a_file.endswith(".jpg"):
            #     img_data[img_data >= 254] = 255
    return ret


def color_vignettes_removed(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    ret = {}
    for img_name, img_data in images.items():
        if "_color_" in img_name:
            continue
        ret[img_name] = img_data
    return ret


def categorize_by_size(
    images: Dict[str, np.ndarray]
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    ret = {}
    for img_name, img_data in images.items():
        height, width = img_data.shape[:2]
        key = (height, width)
        if key not in ret:
            ret[key] = {img_name: img_data}
        else:
            ret[key][img_name] = img_data
    return ret
