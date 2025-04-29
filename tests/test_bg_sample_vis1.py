from pathlib import Path

import cv2
import numpy as np

from ZooProcess_lib.Background import Background
from ZooProcess_lib.img_tools import (
    loadimage,
    load_zipped_image,
    load_tiff_image_and_info, add_separated_mask,
)
from data_dir import BACKGROUND_DIR, SAMPLE_DIR, WORK_DIR
from tests.test_utils import save_diff_image, diff_actual_with_ref_and_source


def test_background_plus_sample_to_vis1(tmp_path):
    """Ensure we can mimic sample - background -> work vis1 equivalent"""
    # Read 8bit sample scan
    eight_bit_sample_file = SAMPLE_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1.tif"
    assert eight_bit_sample_file.exists()
    sample_info, eight_bit_sample_image = load_tiff_image_and_info(
        eight_bit_sample_file
    )
    assert eight_bit_sample_image.dtype == np.uint8

    # Read 8bit combined background scan
    last_background_file = BACKGROUND_DIR / "20240529_0946_background_large_manual.tif"
    bg_info, last_background_image = load_tiff_image_and_info(last_background_file)
    assert last_background_image.dtype == np.uint8
    background = Background(last_background_image, resolution=bg_info.resolution)

    sample_minus_background_image = background.removed_from(
        sample_image=eight_bit_sample_image,
        processing_method="",
        sample_image_resolution=sample_info.resolution,
    )

    # Compare with stored reference (vis1.zip)
    _, expected_final_image = load_zipped_image(
        WORK_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1_vis1.zip"
    )
    assert sample_minus_background_image.shape == expected_final_image.shape

    # Add separator mask, it is present in test data
    sep_image = loadimage(
        WORK_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1_sep.gif",
        type=cv2.COLOR_BGR2GRAY,
    )
    # TODO if useful in V10: extract all this, checks on the mask, etc, etc.
    sample_minus_background_image = add_separated_mask(
        sample_minus_background_image, sep_image
    )

    if not np.array_equal(expected_final_image, sample_minus_background_image):
        save_diff_image(
            expected_final_image,
            sample_minus_background_image,
            Path("/tmp/zooprocess/diff.jpg"),
        )
        nb_real_errors = diff_actual_with_ref_and_source(
            expected_final_image,
            sample_minus_background_image,
            sample_minus_background_image,
            tolerance=0,  # In case there is some debug to do, of course with 0 it's strict equality
        )
        if nb_real_errors > 0:
            assert False
        # assert np.array_equal(sample_minus_background_image[0], expected_final_image[0])

    assert expected_final_image.shape == sample_minus_background_image.shape


