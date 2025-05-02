from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import (
    image_info,
    get_date_time_digitized,
    loadimage,
    load_tiff_image_and_info,
)
from tests.test_utils import save_diff_image, diff_actual_with_ref_and_source
from .env_fixture import projects
from .projects_for_test import TRIATLAS
from .projects_repository import tested_samples
from .test_sample import load_final_ref_image

tested_samples = [
    (TRIATLAS, "m158_mn06_n3_d3")
]  # Extra vignette not filtered by W/H ratio

@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (prj, sample) in tested_samples],
)
def test_raw_to_work(projects, tmp_path, project, sample):
    """Ensure we can mimic sample - background -> work vis1 equivalent"""
    folder = ZooscanFolder(projects, project)
    processor = Processor.from_legacy_config(folder.zooscan_config.read(), folder.zooscan_config.read_lut())

    index = 1  # TODO: should come from get_names() below

    # Load the last background used at time of scan operation
    # dates = folder.zooscan_back.get_dates()
    # assert len(dates) > 0

    # Read raw sample scan, just for its date
    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    img_info = image_info(Image.open(raw_sample_file))
    digitized_at = get_date_time_digitized(img_info)
    if digitized_at is None:
        file_stats = raw_sample_file.stat()  # TODO: Encapsulate this
        digitized_at = datetime.fromtimestamp(file_stats.st_mtime)
    assert digitized_at is not None

    # Read 8bit sample scan
    eight_bit_sample_file = folder.zooscan_scan.get_file_produced_from(
        raw_sample_file.name
    )
    assert eight_bit_sample_file.exists()
    sample_info, eight_bit_sample_image = load_tiff_image_and_info(
        eight_bit_sample_file
    )
    assert eight_bit_sample_image.dtype == np.uint8

    # Read 8bit combined background scan
    last_background_file = folder.zooscan_back.get_last_background_before(digitized_at)
    bg_info, last_background_image = load_tiff_image_and_info(last_background_file)
    assert last_background_image.dtype == np.uint8

    sample_minus_background_image = processor.bg_remover.do_from_files(last_background_file, eight_bit_sample_file)

    # Compare with stored reference (vis1.zip)
    _, expected_final_image = load_final_ref_image(folder, sample, index)
    assert sample_minus_background_image.shape == expected_final_image.shape

    # saveimage(sample_minus_background_image, "/tmp/zooprocess/final_with_bg.tif")
    # compare
    # Always add separator mask, if present
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    sep_file = work_files.get("sep")
    if sep_file is not None:
        assert sep_file.exists()
        sep_image = loadimage(sep_file, type=cv2.COLOR_BGR2GRAY)
        assert sep_image.dtype == np.uint8
        assert sep_image.shape == sample_minus_background_image.shape
        # TODO: extract all this, checks on the mask, etc, etc.
        sample_minus_background_image_plus_sep = (
            sample_minus_background_image.astype(np.uint16) + sep_image
        )
        sample_minus_background_image = np.clip(
            sample_minus_background_image_plus_sep, 0, 255
        ).astype(np.uint8)

    if not np.array_equal(expected_final_image, sample_minus_background_image):
        save_diff_image(
            expected_final_image,
            sample_minus_background_image,
            Path("/tmp/zooprocess/diff.jpg"),
        )
        # assert False
        nb_real_errors = diff_actual_with_ref_and_source(
            expected_final_image,
            sample_minus_background_image,
            sample_minus_background_image,
            tolerance=0,
        )
        if nb_real_errors > 0:
            assert False
        # assert np.array_equal(sample_minus_background_image[0], expected_final_image[0])

    # assert expected_image.shape == actual_image.shape
