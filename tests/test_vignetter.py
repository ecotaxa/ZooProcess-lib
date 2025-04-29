import os
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Extractor import Extractor
from ZooProcess_lib.Lut import Lut
from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.ZooscanFolder import ZooscanConfig
from ZooProcess_lib.Zooscan_combine_backgrounds import Zooscan_combine_backgrounds
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import (
    load_tiff_image_and_info,
    loadimage,
    add_separated_mask,
)
from test_utils import diff_actual_with_ref_and_source
from .data_dir import BACK_TIME, BACKGROUND_DIR, CONFIG_DIR, RAW_DIR, WORK_DIR


def sort_like_legacy(rois: List[ROI], limit: int):
    # Looks (from ecotaxa TSVs) that the sort is by BY first, then BX, but in 2 chunks separated by image height
    rois.sort(key=lambda roi: (roi.y + (0 if roi.x < limit else 1000000), roi.x))


def test_thumbnail_generator(tmp_path):
    # A bit of e2e testing as well, see if we can do from _only_ raw images up to thumbnails
    lut = Lut.read(CONFIG_DIR / "lut.txt")
    conf = ZooscanConfig.read(CONFIG_DIR / "process_install_both_config.txt")
    # Backgrounds pre-processing
    bg_scan_date = BACK_TIME
    bg_raw_files = [
        Path(BACKGROUND_DIR, f"{bg_scan_date}_back_large_raw_{index}.tif")
        for index in (1, 2)
    ]
    eight_bit_bgs = [tmp_path / raw_bg_file.name for raw_bg_file in bg_raw_files]
    [
        Zooscan_convert(raw_bg_file, output_path, lut)
        for raw_bg_file, output_path in zip(bg_raw_files, eight_bit_bgs)
    ]
    combined_bg_file = Path(tmp_path, f"{bg_scan_date}_background_large_manual.tif")
    Zooscan_combine_backgrounds(eight_bit_bgs, combined_bg_file)
    # Sample pre-processing
    raw_sample_file = RAW_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_raw_1.tif"
    eight_bit_sample = tmp_path / raw_sample_file.name
    Zooscan_convert(raw_sample_file, eight_bit_sample, lut)
    # Background removal
    bg_info, background_image = load_tiff_image_and_info(combined_bg_file)
    sample_info, sample_image = load_tiff_image_and_info(eight_bit_sample)
    background = Background(background_image, resolution=bg_info.resolution)
    sample_scan = background.removed_from(
        sample_image=sample_image,
        processing_method="select",
        sample_image_resolution=sample_info.resolution,
    )
    # Add separator mask, it is present in test data
    sep_image = loadimage(
        WORK_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1_sep.gif",
        type=cv2.COLOR_BGR2GRAY,
    )
    sample_scan = add_separated_mask(sample_scan, sep_image)
    # Segmentation
    segmenter = Segmenter(
        sample_scan,
        sample_info.resolution,
        conf.minsizeesd_mm,
        conf.maxsizeesd_mm,
        conf.upper,
    )
    rois = segmenter.find_blobs()
    sort_like_legacy(rois, limit=sample_info.height)
    # Thumbnails generation
    thumbs_dir = tmp_path / "thumbs"
    os.makedirs(thumbs_dir)
    extractor = Extractor(
        sample_scan,
        sample_info.resolution,
        conf.longline_mm,
        rois,
        thumbs_dir,
        "apero2023_tha_bioness_017_st66_d_n1_d3_1",
    )
    extractor.extract_all()
    # Reference thumbnails
    ref_thumbs_dir = tmp_path / "ref_thumbs"  # TODO, a zip and extract it
    ref_thumbs_dir = WORK_DIR
    compare_vignettes(ref_thumbs_dir, thumbs_dir)


def compare_vignettes(ref_thumbs_dir: Path, act_thumbs_dir: Path):
    # Tolerate a different extension as long as bitmaps are equal
    ref_images = list_images_in(ref_thumbs_dir)
    act_images = list_images_in(act_thumbs_dir)
    # Basic matches
    assert len(ref_images) == len(act_images)
    assert set(ref_images.keys()) == set(act_images.keys())
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
        for a_ref_name, a_ref_img in ref_images.items():
            for act_img in maybe_same.values():
                nb_real_errors = diff_actual_with_ref_and_source(
                    a_ref_img,
                    act_img,
                    act_img,
                    tolerance=1,  # In case there is some debug to do, of course with 0 it's strict equality
                )
                print(f"nb_real_errors={nb_real_errors}")
                # if np.array_equal(a_ref_img, act_img):
                #     break
            else:
                print(f"Image {a_ref_name} not found in actual thumbnails")
                in_error = True
    assert not in_error


def list_images_in(image_dir: Path):
    ret = {}  # key: base name, value:image read np.ndarray
    for a_file in os.listdir(image_dir):
        if a_file.endswith(".jpg") or a_file.endswith(".png"):
            img_file = os.path.join(image_dir, a_file)
            img_data = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            file_without_ext = a_file[:-4]
            ret[file_without_ext] = img_data
            # Apparently jpg encoding damaged pure white
            if a_file.endswith(".jpg"):
                img_data[img_data >= 254] = 255
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
