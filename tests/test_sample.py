import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Border import Border
from ZooProcess_lib.ImageJLike import images_difference
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import (
    load_zipped_image,
    loadimage,
    image_info,
    get_date_time_digitized,
    crop_right,
    clear_outside,
    draw_outside_lines,
)
from tests.env_fixture import projects
from tests.projects_for_test import (
    APERO2000,
    APERO,
    IADO,
    TRIATLAS,
    APERO_REDUCED,
)
from tests.test_utils import (
    save_diff_image,
    diff_actual_with_ref_and_source,
    read_result_csv,
)


# from tests.projects_for_test import APERO2000_REDUCED as APERO2000


def test_8bit_sample_border(projects, tmp_path):
    """Ensure we compute borders to remove like legacy"""
    folder = ZooscanFolder(projects, APERO2000)
    sample = "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4"
    index = 1
    src_8bit_sample_file = folder.zooscan_scan.get_8bit_file(sample, index)
    assert src_8bit_sample_file.exists()
    src_image = loadimage(src_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)
    border = Border(src_image)
    (top, bottom, left, right) = border.detect()
    # ImageJ debug on same image, macro Zooscan_1asep.txt
    Right_limit = 24520
    Left_limit = 380
    Upper_limit = 330
    Bottom_limit = 14600
    assert (top, bottom, left, right) == (
        Upper_limit,
        Bottom_limit,
        Left_limit,
        Right_limit,
    )


APERO2000_samples = [
    # "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1", # Corrupted ZIP
    # "apero2023_tha_bioness_sup2000_013_st46_d_n4_d2_1_sur_1", # Corrupted ZIP
    # "apero2023_tha_bioness_sup2000_016_st55_d_n9_d2_1_sur_1", # Corrupted ZIP
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d1_1_sur_1",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_1_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_2_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_4_sur_4",
]
APERO_samples = [
    "apero2023_tha_bioness_013_st46_d_n3_d1_1_sur_1",
    "apero2023_tha_bioness_013_st46_d_n3_d1_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n4_d2_1_sur_2",
    "apero2023_tha_bioness_013_st46_d_n3_d2_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n4_d2_2_sur_2",
    "apero2023_tha_bioness_013_st46_d_n3_d3",
    "apero2023_tha_bioness_014_st46_n_n4_d3",
    "apero2023_tha_bioness_013_st46_d_n4_d1_1_sur_1",
    # "apero2023_tha_bioness_014_st46_n_n5_d1_1_sur_1", # Corrupted ZIP
    "apero2023_tha_bioness_013_st46_d_n4_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n5_d2_1_sur_2",
    "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n5_d2_2_sur_2",
    "apero2023_tha_bioness_013_st46_d_n4_d3",
    "apero2023_tha_bioness_014_st46_n_n5_d3",
    # "apero2023_tha_bioness_013_st46_d_n5_d1_1_sur_1", # Corrupted ZIP
    "apero2023_tha_bioness_014_st46_n_n6_d1_1_sur_1",
    "apero2023_tha_bioness_013_st46_d_n5_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n6_d2_1_sur_2",
    "apero2023_tha_bioness_013_st46_d_n5_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n6_d2_2_sur_2",
    "apero2023_tha_bioness_013_st46_d_n5_d3",
    "apero2023_tha_bioness_014_st46_n_n6_d3",
    "apero2023_tha_bioness_013_st46_d_n6_d1_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n7_d1_1_sur_4",
    "apero2023_tha_bioness_013_st46_d_n6_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n7_d1_2_sur_4",
    "apero2023_tha_bioness_013_st46_d_n6_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n7_d1_3_sur_4",
    "apero2023_tha_bioness_013_st46_d_n6_d3",
    "apero2023_tha_bioness_014_st46_n_n7_d1_4_sur_4",
    "apero2023_tha_bioness_013_st46_d_n7_d2_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n7_d2_1_sur_2",
    "apero2023_tha_bioness_013_st46_d_n7_d3",
    "apero2023_tha_bioness_014_st46_n_n7_d2_2_sur_2",
    "apero2023_tha_bioness_013_st46_d_n8_d1_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n9_d1_1_sur_8",
    "apero2023_tha_bioness_013_st46_d_n8_d2_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n9_d1_2_sur_8",
    "apero2023_tha_bioness_013_st46_d_n8_d3",
    "apero2023_tha_bioness_014_st46_n_n9_d1_3_sur_8",
    "apero2023_tha_bioness_013_st46_d_n9_d2_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n9_d1_4_sur_8",
    "apero2023_tha_bioness_013_st46_d_n9_d3",
    "apero2023_tha_bioness_014_st46_n_n9_d1_5_sur_8",
    "apero2023_tha_bioness_014_st46_n_n1_d1_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n9_d1_6_sur_8",
    "apero2023_tha_bioness_014_st46_n_n1_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d1_7_sur_8",
    "apero2023_tha_bioness_014_st46_n_n1_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d1_8_sur_8",
    "apero2023_tha_bioness_014_st46_n_n1_d3",
    "apero2023_tha_bioness_014_st46_n_n9_d2_1_sur_8",
    "apero2023_tha_bioness_014_st46_n_n2_d1_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d2_2_sur_8",
    "apero2023_tha_bioness_014_st46_n_n2_d1_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d2_3_sur_8",
    "apero2023_tha_bioness_014_st46_n_n2_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d2_4_sur_8",
    "apero2023_tha_bioness_014_st46_n_n2_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d2_5_sur_8",
    "apero2023_tha_bioness_014_st46_n_n2_d3",
    "apero2023_tha_bioness_014_st46_n_n9_d2_6_sur_8",
    "apero2023_tha_bioness_014_st46_n_n3_d1_1_sur_1",
    "apero2023_tha_bioness_014_st46_n_n9_d2_7_sur_8",
    "apero2023_tha_bioness_014_st46_n_n3_d2_1_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d2_8_sur_8",
    "apero2023_tha_bioness_014_st46_n_n3_d2_2_sur_2",
    "apero2023_tha_bioness_014_st46_n_n9_d3",
    "apero2023_tha_bioness_014_st46_n_n3_d3",
    "apero2023_tha_bioness_014_st46_n_n4_d1_1_sur_1",
]
IADO_samples = [
    "s_17_1_tot",
    "s_17_2_tot",
    "s_17_3_tot",
    "s_17_4_tot",
    "s_17_5_tot",
    "s_17_6_tot",
    "s_21_a_tot",
    "s_21_c_tot",
    "t_17_10_tot",
    "t_17_1_tot",
    "t_17_3_tot",
    "t_17_7_tot",
    "t_18_4b_tot",
    "t_18_5_tot",
    "t_22_10_tot",
    "t_22_12_5_tot",
    "t_22_1_tot",
    "t_22_2_tot",
    "t_22_3_tot",
    "t_22_4_tot",
    "t_22_5_tot",
    "t_22_6_tot",
    "t_22_7_tot",
    "t_22_8_tot",
    "t_22_9_tot",
]
TRIATLAS_samples = [
    "m158_mn03_n1_d2",
    "m158_mn03_n1_d3",
    "m158_mn03_n2_d2",
    "m158_mn03_n2_d3",
    "m158_mn03_n3_d1",
    "m158_mn03_n3_d3",
    "m158_mn03_n4_d1",
    "m158_mn03_n4_d2",
    "m158_mn03_n4_d3",
    "m158_mn03_n5_d1_1_sur_4",
    "m158_mn03_n5_d1_2_sur_4",
    "m158_mn03_n5_d1_3_sur_4",
    "m158_mn03_n5_d1_4_sur_4",
    "m158_mn03_n5_d2",
    "m158_mn04_n1_d1",
    "m158_mn04_n1_d2",
    "m158_mn04_n2_d1",
    "m158_mn04_n2_d2",
    "m158_mn04_n2_d3",
    "m158_mn04_n3_d1",
    "m158_mn04_n3_d2",
    "m158_mn04_n4_d2",
    "m158_mn04_n4_d3",
    "m158_mn04_n5_d1_1_sur_4",
    "m158_mn04_n5_d1_2_sur_4",
    "m158_mn04_n5_d1_3_sur_4",
    "m158_mn04_n5_d1_4_sur_4",
    "m158_mn05_n1_d1",
    "m158_mn05_n1_d3",
    "m158_mn05_n2_d1",
    "m158_mn05_n2_d2",
    "m158_mn05_n3_d2",
    "m158_mn05_n3_d3",
    "m158_mn05_n4_d2",
    "m158_mn05_n5_d1_1_sur_4",
    "m158_mn05_n5_d1_2_sur_4",
    "m158_mn05_n5_d1_3_sur_4",
    "m158_mn06_n2_d1_1_sur_2",
    "m158_mn06_n2_d1_2_sur_2",
    "m158_mn06_n2_d2",
    "m158_mn06_n3_d1",
    "m158_mn06_n3_d3",
    "m158_mn06_n4_d3",
    "m158_mn06_n5_d1_1_sur_8",
    "m158_mn06_n5_d2",
    "m158_mn06_n5_d3",
    "m158_mn10_n1_d1",
    "m158_mn10_n1_d2",
    "m158_mn10_n1_d3",
    "m158_mn10_n2_d1",
    "m158_mn10_n2_d3",
    "m158_mn10_n3_d1",
    "m158_mn10_n3_d2",
    "m158_mn10_n3_d3",
    "m158_mn10_n4_d2",
    "m158_mn10_n4_d3",
    "m158_mn10_n5_d1_1_sur_2",
    "m158_mn10_n5_d3",
    "m158_mn11_n1_d1",
    "m158_mn11_n1_d3",
    "m158_mn11_n2_d1",
    "m158_mn11_n2_d2",
    "m158_mn11_n2_d3",
    "m158_mn11_n3_d1",
    "m158_mn11_n3_d2",
    "m158_mn11_n4_d1",
    "m158_mn11_n4_d2",
    "m158_mn11_n4_d3",
    "m158_mn11_n5_d1",
    "m158_mn11_n5_d3",
    "m158_mn14_n1_d1",
    "m158_mn14_n1_d2",
    "m158_mn14_n1_d3",
    "m158_mn14_n2_d1",
    "m158_mn14_n2_d3",
    "m158_mn14_n3_d2",
    "m158_mn14_n4_d1",
    "m158_mn14_n5_d1",
    "m158_mn14_n5_d2",
    "m158_mn15_n1_d1",
    "m158_mn15_n1_d3",
    "m158_mn15_n2_d1",
    "m158_mn15_n2_d2",
    "m158_mn15_n3_d2",
    "m158_mn15_n3_d3",
    "m158_mn15_n4_d1",
    "m158_mn15_n4_d2",
    "m158_mn18_n1_d1",
    "m158_mn18_n1_d2",
    "m158_mn18_n1_d3",
    "m158_mn18_n2_d1_3_sur_4",
    "m158_mn18_n2_d1_4_sur_4",
    "m158_mn18_n2_d2",
    "m158_mn18_n3_d1",
    "m158_mn18_n3_d2",
    "m158_mn18_n3_d3",
    "m158_mn18_n4_d1",
    "m158_mn18_n4_d2",
    "m158_mn18_n4_d3",
    "m158_mn18_n5_d2",
    "m158_mn18_n5_d3",
    "m158_mn19_n1_d1",
    "m158_mn19_n1_d2",
    "m158_mn19_n1_d3",
    "m158_mn19_n2_d1_1_sur_2",
    "m158_mn19_n2_d1_2_sur_2",
    "m158_mn19_n2_d2",
    "m158_mn19_n2_d3",
    "m158_mn19_n5_d1_1_sur_5",
    "m158_mn19_n5_d1_4_sur_5",
    "m158_mn19_n5_d1_5_sur_5",
    "m158_mn19_n5_d2",
    "m158_mn19_n5_d3",
]

APERO2000_tested_samples = zip([APERO2000] * 100, APERO2000_samples)
# APERO2000_tested_samples = []
APERO_tested_samples = zip([APERO] * 100, APERO_samples)
# APERO_tested_samples = zip([APERO] * 100, APERO_samples[-5:-4])
IADO_tested_samples = zip([IADO] * 100, IADO_samples)
TRIATLAS_tested_samples = zip([TRIATLAS] * 150, TRIATLAS_samples)
tested_samples = (
    list(APERO2000_tested_samples)
    + list(APERO_tested_samples)
    + list(IADO_tested_samples)
    + list(TRIATLAS_tested_samples)
)


@pytest.mark.parametrize("project, sample", tested_samples)
def test_raw_to_work(projects, tmp_path, project, sample):
    """Ensure we can mimic sample - background -> work vis1 equivalent"""
    folder = ZooscanFolder(projects, project)

    index = 1  # TODO: should come from get_names() below

    # Load the last background used at time of scan operation
    # dates = folder.zooscan_back.get_dates()
    # assert len(dates) > 0

    # Read raw sample scan, just for its date
    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    img_info = image_info(raw_sample_file)
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
    eight_bit_sample_image = loadimage(eight_bit_sample_file, type=cv2.IMREAD_UNCHANGED)
    assert eight_bit_sample_image.dtype == np.uint8

    # Read 8bit combined background scan
    last_background_file = folder.zooscan_back.get_last_background_before(digitized_at)
    last_background_image = loadimage(last_background_file, type=cv2.IMREAD_UNCHANGED)
    assert last_background_image.dtype == np.uint8

    border = Border(eight_bit_sample_image, "select" if "triatlas" in project else "")
    (top_limit, bottom_limit, left_limit, right_limit) = border.detect()
    # TODO: below correspond to a not-debugged case "if (greycor > 2 && droite == 0) {" which
    # is met when borders are not computed.
    # limitod = border.right_limit_to_removeable_from_image()
    limitod = border.right_limit_to_removeable_from_right_limit()
    # assert limitod == 24568  # From ImageJ debug
    # assert right_limit == 24214

    bg = Background(last_background_image, last_background_file)
    cropped_bg, mean_bg, adjusted_bg = bg.resized_for_sample_scan(
        eight_bit_sample_image.shape[1], eight_bit_sample_image.shape[0]
    )

    # saveimage(cropped_bg, "/tmp/cropped_bg.tif")
    # ref_cropped_bg = loadimage(Path("/tmp/fond_cropped_legacy.tif"))
    # diff_actual_with_ref_and_source(ref_cropped_bg, cropped_bg, ref_cropped_bg)
    # assert np.array_equal(ref_cropped_bg, cropped_bg)

    # saveimage(mean_bg, "/tmp/mean_bg.tif")
    # ref_mean_bg = loadimage(Path("/tmp/fond_apres_mean.tif"))
    # diff_actual_with_ref_and_source(ref_mean_bg, mean_bg, ref_mean_bg)
    # assert np.array_equal(ref_mean_bg, mean_bg)

    # saveimage(adjusted_bg, Path("/tmp/resized_bg.tif"))
    # ref_resized_bg = loadimage(Path("/tmp/fond_apres_resize.tif"))
    # diff_actual_with_ref_and_source(ref_resized_bg, adjusted_bg, ref_resized_bg)
    # if not np.array_equal(ref_resized_bg, adjusted_bg):
    #     nb_errors = diff_actual_with_ref_and_source(
    #         ref_resized_bg,
    #         adjusted_bg,
    #         last_background_image,
    #         tolerance=0,
    #     )
    #     if nb_errors > 0:
    #         assert False
    #

    # TODO: this _only_ corresponds to "if (method == "neutral") {" in legacy
    sample_minus_background_image = images_difference(
        adjusted_bg, eight_bit_sample_image
    )
    # Invert 8-bit
    sample_minus_background_image = 255 - sample_minus_background_image

    # ref_after_sub_bg = loadimage(Path("/tmp/fond_apres_subs.tif"))
    # diff_actual_with_ref_and_source(ref_after_sub_bg, sample_minus_background_image, ref_after_sub_bg)
    # assert np.array_equal(ref_after_sub_bg, sample_minus_background_image)

    sample_minus_background_image = crop_right(sample_minus_background_image, limitod)

    # ref_after_sub_and_crop_bg = loadimage(Path("/tmp/fond_apres_subs_et_crop.tif"))
    # diff_actual_with_ref_and_source(ref_after_sub_and_crop_bg, sample_minus_background_image, ref_after_sub_and_crop_bg)
    # assert np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image)

    cleared_width = min(right_limit - left_limit, limitod)
    sample_minus_background_image = clear_outside(
        sample_minus_background_image,
        left_limit,
        top_limit,
        cleared_width,
        bottom_limit - top_limit,
    )

    # ref_after_sub_and_crop_bg = loadimage(
    #     Path("/tmp/fond_apres_subs_et_crop_et_clear.tif")
    # )
    # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
    #     diff_actual_with_ref_and_source(
    #         ref_after_sub_and_crop_bg,
    #         sample_minus_background_image,
    #         ref_after_sub_and_crop_bg,
    #     )
    #     assert False

    draw_outside_lines(
        sample_minus_background_image,
        eight_bit_sample_image.shape,
        right_limit,
        left_limit,
        top_limit,
        bottom_limit,
        limitod,
    )

    # ref_after_sub_and_crop_bg = loadimage(
    #     Path("/tmp/fond_apres_subs_et_crop_et_clear_et_lignes.tif")
    # )
    # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
    #     diff_actual_with_ref_and_source(
    #         ref_after_sub_and_crop_bg,
    #         sample_minus_background_image,
    #         ref_after_sub_and_crop_bg,
    #     )
    #     assert False

    # Compare with stored reference (vis1.zip)
    expected_final_image = load_final_ref_image(folder, sample, index)
    assert sample_minus_background_image.shape == expected_final_image.shape

    # saveimage(sample_minus_background_image, "/tmp/final_with_bg.tif")
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
            expected_final_image, sample_minus_background_image, Path("/tmp/diff.jpg")
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


def load_final_ref_image(folder, sample, index):
    assert sample in [
        a_sample["name"] for a_sample in folder.zooscan_scan.raw.get_names()
    ]
    work_files_in_sample = folder.zooscan_scan.work.get_files(sample, index)
    zipped_combined = work_files_in_sample.get("combz")
    assert zipped_combined.exists()
    reference_image = load_zipped_image(zipped_combined)
    return reference_image


@pytest.mark.parametrize(
    "project, sample",
    [(APERO_REDUCED, "apero2023_tha_bioness_014_st46_n_n9_d2_8_sur_8")],
)
def test_segmentation(projects, tmp_path, project, sample):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    vis1 = load_final_ref_image(folder, sample, index)
    # macro: setThreshold(0, 129);
    # run("Threshold", "thresholded remaining black");
    # TODO: below from Zooscan_config/process_install_both_config.txt
    minsizeesd_mm = 1.5
    maxsizeesd_mm = 100
    ref = read_result_csv(
        Path("/tmp/Results.xls"),
        {
            "BX": int,
            "BY": int,
            "Width": int,
            "Height": int,
            "Area": int,
            "XStart": int,
            "YStart": int,
        },
    )
    sort_by_dist(ref)
    segmenter = Segmenter(vis1, minsizeesd_mm, maxsizeesd_mm)
    found = segmenter.find_blobs()
    segmenter.split_by_blobs()
    sort_by_dist(found)
    assert found[1:] == ref  # TODO: There is a full image border in openCV output


def sort_by_dist(features: List[Dict]):
    features.sort(key=lambda f: math.sqrt(math.pow(f["BX"], 2) + math.pow(f["BY"], 2)))
