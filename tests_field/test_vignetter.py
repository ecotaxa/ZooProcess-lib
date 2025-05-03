import os
import random
import shutil
from datetime import datetime

import cv2
import pytest
from PIL import Image

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import (
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
from .projects_for_test import POINT_B_JB, APERO1, TRIATLAS
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
    conf = folder.zooscan_config.read()
    lut = folder.zooscan_config.read_lut()
    processor = Processor(conf, lut)
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
    bg_converted_files = [
        processor.converter.do_file_to_image(a_raw_bg_file)
        for a_raw_bg_file in bg_raw_files
    ]
    combined_bg_image, bg_resolution = processor.bg_combiner.do_from_images(
        bg_converted_files
    )
    # Sample pre-processing
    eight_bit_sample_image, sample_resolution = processor.converter.do_file_to_image(
        raw_sample_file
    )
    # Background removal
    sample_scan = processor.bg_remover.do_from_images(
        combined_bg_image, bg_resolution, eight_bit_sample_image, sample_resolution
    )
    # Always add separator mask, if present
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    sep_file = work_files.get("sep")
    if sep_file is not None:
        sep_image = loadimage(sep_file, type=cv2.COLOR_BGR2GRAY)
        sample_scan = add_separated_mask(sample_scan, sep_image)
    # Segmentation
    rois = processor.segmenter.find_ROIs_in_image(
        sample_scan,
        sample_resolution,
    )
    sort_ROIs_like_legacy(rois, limit=sample_scan.shape[0])
    # Get box measurements, no need to compare thumbnails if they don't match
    actual_measures = to_legacy_format(
        processor.calculator.legacy_measures_list_from_roi_list(
            sample_scan, conf.resolution, rois, BOX_MEASUREMENTS.keys()
        )
    )
    sort_by_coords(actual_measures)
    assert actual_measures == ref_box_measures
    # Thumbnails generation
    thumbs_dir = tmp_path / "thumbs"
    os.makedirs(thumbs_dir)
    processor.extractor.extract_all_from_image(
        sample_scan,
        sample_resolution,
        rois,
        thumbs_dir,
        sample + "_" + str(index),
    )
    # Reference thumbnails
    ref_thumbs_dir = folder.zooscan_scan.work.path / (sample + "_" + str(index))
    compare_vignettes(ref_thumbs_dir, thumbs_dir, conf.upper)
    # Cleanup if all went well
    shutil.rmtree(thumbs_dir)


dev_samples = [(p, s) for (p, s) in all_samples_in(POINT_B_JB)]  # if "197809" in s
dev_samples = [
    (TRIATLAS, "m158_mn15_n3_d3")
]  # Extra vignette not filtered by W/H ratio
dev_samples = all_samples_in(APERO1)[:8]


@pytest.mark.parametrize("project, sample", dev_samples)
def test_dev_thumbnail_generator(projects, project, sample, tmp_path):
    assert_same_vignettes(project, projects, sample, tmp_path)
