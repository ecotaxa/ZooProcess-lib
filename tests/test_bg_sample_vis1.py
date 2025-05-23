from pathlib import Path

import cv2
import numpy as np

from ZooProcess_lib.LegacyConfig import ZooscanConfig
from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.img_tools import (
    loadimage,
    load_zipped_image,
    add_separated_mask,
)
from data_dir import BACKGROUND_DIR, SAMPLE_DIR, WORK_DIR, CONFIG_DIR
from tests.test_utils import save_diff_image, diff_actual_with_ref_and_source


def test_background_plus_sample_to_vis1(tmp_path):
    """Ensure we can mimic sample - background -> work vis1 equivalent"""
    conf = ZooscanConfig.read(CONFIG_DIR / "process_install_both_config.txt")
    processor = Processor(conf)
    # Read 8bit sample scan
    eight_bit_sample_file = SAMPLE_DIR / "apero2023_tha_bioness_017_st66_d_n1_d3_1.tif"
    # Read 8bit combined background scan
    last_background_file = BACKGROUND_DIR / "20240529_0946_background_large_manual.tif"

    sample_minus_background_image = processor.bg_remover.do_from_files(last_background_file, eight_bit_sample_file)

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


