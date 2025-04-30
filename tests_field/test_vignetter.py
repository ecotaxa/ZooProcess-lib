import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import pytest
from PIL import Image

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Extractor import Extractor
from ZooProcess_lib.Features import (
    legacy_measures_list_from_roi_list,
)
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.Zooscan_combine_backgrounds import Zooscan_combine_backgrounds
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
from ZooProcess_lib.img_tools import (
    load_tiff_image_and_info,
    loadimage,
    add_separated_mask,
    image_info,
    get_date_time_digitized,
)
from tests.data_tools import (
    sort_ROIs_like_legacy,
    to_legacy_format,
    sort_by_coords,
    read_box_measurements,
    BOX_MEASUREMENTS,
)
from tests.test_utils import compare_vignettes
from .env_fixture import projects
from .projects_for_test import POINT_B_JB
from .projects_repository import tested_samples, all_samples_in

# The test takes ages, by randomizing the order there are better chances to see problems early
shuffled = tested_samples.copy()
random.shuffle(shuffled)


@pytest.mark.parametrize(
    "project, sample",
    shuffled,
    ids=[sample for (_prj, sample) in shuffled],
)
def test_thumbnail_generator(projects, project, sample, tmp_path):
    assert_same_vignettes(project, projects, sample, tmp_path)


def assert_same_vignettes(project, projects, sample, tmp_path):
    # A bit of e2e testing as well, see if we can do from _only_ raw images up to thumbnails
    folder = ZooscanFolder(projects, project)
    lut = folder.zooscan_config.read_lut()
    conf = folder.zooscan_config.read()
    index = 1  # TODO: should come from get_names() below
    ref_box_measures = read_box_measurements(folder, sample, index)
    # Read raw sample scan
    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    img_info = image_info(Image.open(raw_sample_file))
    digitized_at = get_date_time_digitized(img_info)
    if digitized_at is None:
        file_stats = raw_sample_file.stat()  # TODO: Encapsulate this
        digitized_at = datetime.fromtimestamp(file_stats.st_mtime)
    assert digitized_at is not None
    # Backgrounds pre-processing
    bg_raw_files = folder.zooscan_back.get_last_raw_backgrounds_before(digitized_at)
    eight_bit_bgs = [tmp_path / raw_bg_file.name for raw_bg_file in bg_raw_files]
    [
        Zooscan_convert(raw_bg_file, output_path, lut)
        for raw_bg_file, output_path in zip(bg_raw_files, eight_bit_bgs)
    ]
    combined_bg_file = Path(tmp_path, f"{digitized_at}_background_large_manual.tif")
    Zooscan_combine_backgrounds(eight_bit_bgs, combined_bg_file)
    # Sample pre-processing
    eight_bit_sample = tmp_path / raw_sample_file.name
    Zooscan_convert(raw_sample_file, eight_bit_sample, lut)
    # Background removal
    bg_info, background_image = load_tiff_image_and_info(combined_bg_file)
    sample_info, sample_image = load_tiff_image_and_info(eight_bit_sample)
    background = Background(background_image, resolution=bg_info.resolution)
    sample_scan = background.removed_from(
        sample_image=sample_image,
        processing_method="select" if "triatlas" in project else "",
        sample_image_resolution=sample_info.resolution,
    )
    # Always add separator mask, if present
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    sep_file = work_files.get("sep")
    if sep_file is not None:
        sep_image = loadimage(sep_file, type=cv2.COLOR_BGR2GRAY)
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
    # Get box measurements, no need to compare thumbnails if they don't match
    actual_measures = to_legacy_format(
        legacy_measures_list_from_roi_list(
            sample_scan, conf.resolution, rois, conf.upper, BOX_MEASUREMENTS.keys()
        )
    )
    sort_by_coords(actual_measures)
    assert actual_measures == ref_box_measures
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
        sample + "_" + str(index),
    )
    extractor.extract_all()
    # Reference thumbnails
    ref_thumbs_dir = folder.zooscan_scan.work.path / (sample + "_" + str(index))
    compare_vignettes(ref_thumbs_dir, thumbs_dir, conf.upper)


dev_samples = [(p, s) for (p, s) in all_samples_in(POINT_B_JB) if "197809" in s]


@pytest.mark.parametrize("project, sample", dev_samples)
def test_dev_thumbnail_generator(projects, project, sample, tmp_path):
    assert_same_vignettes(project, projects, sample, tmp_path)
