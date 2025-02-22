import cv2
import numpy as np
import pytest

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Border import Border
from ZooProcess_lib.ImageJLike import images_difference
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
from tests.projects_for_test import APERO2000


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
    "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1",
    "apero2023_tha_bioness_sup2000_013_st46_d_n4_d2_1_sur_1",
    "apero2023_tha_bioness_sup2000_016_st55_d_n9_d2_1_sur_1",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d1_1_sur_1",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_1_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_2_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4",
    "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_4_sur_4",
]


@pytest.mark.parametrize("sample", APERO2000_samples)
def test_raw_to_work(projects, tmp_path, sample):
    """Ensure we can mimic sample - background -> work vis1 equivalent"""
    folder = ZooscanFolder(projects, APERO2000)
    # sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"  # Bad ZIP?

    index = 1  # TODO: should come from get_names() below

    # Load the last background used at time of scan operation
    # dates = folder.zooscan_back.get_dates()
    # assert len(dates) > 0

    # Read raw sample scan, just for its date
    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    img_info = image_info(raw_sample_file)
    digitized_at = get_date_time_digitized(img_info)

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

    border = Border(eight_bit_sample_image)
    (top_limit, bottom_limit, left_limit, right_limit) = border.detect()
    # TODO: below correspond to a not-debugged case "if (greycor > 2 && droite == 0) {"
    # limitod = border.right_limit_to_removeable_from_image()
    limitod = border.right_limit_to_removeable_from_right_limit()
    # assert limitod == 24568  # From ImageJ debug

    bg = Background(last_background_image, last_background_file)
    cropped_bg, mean_bg, adjusted_bg = bg.resized_for_sample_scan(
        eight_bit_sample_image.shape[1], eight_bit_sample_image.shape[0]
    )

    # saveimage(cropped_bg, "/tmp/cropped_bg.tif")
    # ref_cropped_bg = loadimage("/tmp/fond_cropped_legacy.tif")
    # diff_actual_with_ref_and_source(ref_cropped_bg, cropped_bg, ref_cropped_bg)
    # assert np.array_equal(ref_cropped_bg, cropped_bg)

    # saveimage(mean_bg, "/tmp/mean_bg.tif")
    # ref_mean_bg = loadimage("/tmp/fond_apres_mean.tif")
    # diff_actual_with_ref_and_source(ref_mean_bg, mean_bg, ref_mean_bg)
    # assert np.array_equal(ref_mean_bg, mean_bg)
    #

    # saveimage(adjusted_bg, "/tmp/resized_bg.tif")
    # ref_resized_bg = loadimage("/tmp/fond_apres_resize.tif")
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

    # TODO: this _only_ corresponds to "if (method == "neutral") {" in legacy
    sample_minus_background_image = images_difference(
        adjusted_bg, eight_bit_sample_image
    )
    # Invert 8-bit
    sample_minus_background_image = 255 - sample_minus_background_image

    # ref_after_sub_bg = loadimage("/tmp/fond_apres_subs.tif")
    # diff_actual_with_ref_and_source(ref_after_sub_bg, sample_minus_background_image, ref_after_sub_bg)
    # assert np.array_equal(ref_after_sub_bg, sample_minus_background_image)

    sample_minus_background_image = crop_right(sample_minus_background_image, limitod)

    # ref_after_sub_and_crop_bg = loadimage("/tmp/fond_apres_subs_et_crop.tif")
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

    # ref_after_sub_and_crop_bg = loadimage("/tmp/fond_apres_subs_et_crop_et_clear.tif")
    # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
    #     diff_actual_with_ref_and_source(ref_after_sub_and_crop_bg, sample_minus_background_image, ref_after_sub_and_crop_bg)
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

    # ref_after_sub_and_crop_bg = loadimage("/tmp/fond_apres_subs_et_crop_et_clear_et_lignes.tif")
    # if not np.array_equal(ref_after_sub_and_crop_bg, sample_minus_background_image):
    #     diff_actual_with_ref_and_source(ref_after_sub_and_crop_bg, sample_minus_background_image, ref_after_sub_and_crop_bg)
    #     assert False

    # Compare with stored reference (vis1.zip)
    expected_final_image = load_final_ref_image(folder, sample, index)
    assert sample_minus_background_image.shape == expected_final_image.shape

    # saveimage(sample_minus_background_image, "/tmp/final_with_bg.tif")
    # compare
    if not np.array_equal(expected_final_image, sample_minus_background_image):
        # Try to add separator mask
        work_files = folder.zooscan_scan.work.get_files(sample, index)
        sep_file = work_files["sep"]
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
        assert np.array_equal(expected_final_image, sample_minus_background_image)

        # save_diff_image(expected_final_image, sample_minus_background_image, Path("/tmp/diff.jpg"))
        # assert False
        # nb_real_errors = diff_actual_with_ref_and_source(
        #     expected_final_image,
        #     sample_minus_background_image,
        #     sample_minus_background_image,
        #     tolerance=0,
        # )
        # if nb_real_errors > 0:
        #     assert False
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


def test_segmentation(projects, tmp_path):
    before_seg = loadimage("/tmp/final_bg.tif")
    # macro: setThreshold(0, 129);
    # run("Threshold", "thresholded remaining black");
