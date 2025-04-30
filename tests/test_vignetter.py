import os
import zipfile
from pathlib import Path

import cv2

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Extractor import Extractor
from ZooProcess_lib.Lut import Lut
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.ZooscanFolder import ZooscanConfig
from ZooProcess_lib.Zooscan_combine_backgrounds import Zooscan_combine_backgrounds
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import (
    load_tiff_image_and_info,
    loadimage,
    add_separated_mask,
)
from .data_dir import BACK_TIME, BACKGROUND_DIR, CONFIG_DIR, RAW_DIR, WORK_DIR
from .data_tools import sort_ROIs_like_legacy
from .test_utils import compare_vignettes


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
    sort_ROIs_like_legacy(rois, limit=sample_info.height)
    # Thumbnails generation
    thumbs_dir = tmp_path / "thumbs"
    os.makedirs(thumbs_dir)
    extractor = Extractor(
        sample_scan,
        sample_info.resolution,
        conf.upper,
        conf.longline_mm,
        rois,
        thumbs_dir,
        "apero2023_tha_bioness_017_st66_d_n1_d3_1",
    )
    extractor.extract_all()
    # Reference thumbnails
    vignettes_dir = tmp_path / "vignettes"
    with zipfile.ZipFile(WORK_DIR / "vignettes.zip", "r") as zip_ref:
        zip_ref.extractall(vignettes_dir)
    ref_thumbs_dir = vignettes_dir
    compare_vignettes(ref_thumbs_dir, thumbs_dir, conf.upper)
