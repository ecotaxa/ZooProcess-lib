from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Background import Background
from ZooProcess_lib.Border import Border
from ZooProcess_lib.ImageJLike import images_difference
from ZooProcess_lib.Segmenter import Segmenter, feature_unq, Features
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import (
    load_zipped_image,
    loadimage,
    image_info,
    get_date_time_digitized,
    crop_right,
    clear_outside,
    draw_outside_lines, saveimage,
)
from .env_fixture import projects, read_home
from .projects_for_test import (
    APERO2000,
    APERO,
    IADO,
    TRIATLAS,
    APERO1,
)
from .test_utils import (
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


def all_samples_in(project: str, but_not=()) -> list[tuple[str, str]]:
    folder = ZooscanFolder(read_home(), project)
    scans = folder.zooscan_scan.list_samples()
    return [(project, a_scan) for a_scan in sorted(scans) if a_scan not in but_not]


tested_samples = (
    all_samples_in(
        APERO2000,
        [
            "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_sup2000_013_st46_d_n4_d2_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_sup2000_016_st55_d_n9_d2_1_sur_1",  # Corrupted ZIP
        ],
    )
    + all_samples_in(
        APERO,
        [
            "apero2023_tha_bioness_014_st46_n_n5_d1_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_013_st46_d_n5_d1_1_sur_1",  # Corrupted ZIP
        ],
    )
    + all_samples_in(IADO)
    + all_samples_in(TRIATLAS)
    + all_samples_in(APERO1)
)
APERO_tested_samples = all_samples_in(APERO1)
APERO_tested_samples_raw_to_work = all_samples_in(
    APERO1,
    [
        "apero2023_tha_bioness_005_st20_d_n7_d1_1_sur_1",  # output diff
        "apero2023_tha_bioness_005_st20_d_n7_d2_3_sur_4",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d1_1_sur_2",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d1_2_sur_2",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d2_1_sur_1",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d3",  # output diff
        "apero2023_tha_bioness_017_st66_d_n1_d2_1_sur_4",  # tiff problem?
        "apero2023_tha_bioness_018_st66_n_n1_d1_1_sur_2",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n1_d1_2_sur_2",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n3_d2_1_sur_1",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n3_d3",  # AttributeError
    ],
)


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (prj, sample) in tested_samples],
)
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
    # [(APERO_REDUCED, "apero2023_tha_bioness_014_st46_n_n9_d2_8_sur_8")],
    # [(APERO_REDUCED2, "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2")],
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_segmentation(projects, tmp_path, project, sample):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    measures = work_files["meas"]
    measures_types = {
        "BX": int,
        "BY": int,
        "Width": int,
        "Height": int,
        "Area": int,
        "%Area": float,
        "XStart": int,
        "YStart": int,
        "Major": float,
        "Minor": float,
        "Angle": float,
    }
    ref = read_result_csv(measures, measures_types)
    # This filter is _after_ measurements in ImageJ
    ref = [
        a_ref
        for a_ref in ref
        if a_ref["Width"] / a_ref["Height"] < Segmenter.max_w_to_h_ratio
    ]
    sort_by_coords(ref)
    if "%Area" in measures_types:
        for a_ref in ref:
            a_ref["%Area"] = round(
                a_ref["%Area"], 3
            )  # Sometimes there are more decimals in measurements
    if "Angle" in measures_types:
        for a_ref in ref:
            a_ref["Angle"] = round(
                a_ref["Angle"], 3
            )  # Sometimes there are more decimals in measurements
    # TODO: Add threshold (AKA 'upper= 243' in config) here
    segmenter = Segmenter(vis1, conf.minsizeesd_mm, conf.maxsizeesd_mm)
    # found_rois = segmenter.find_blobs(
    #     Segmenter.LEGACY_COMPATIBLE | Segmenter.METH_CONNECTED_COMPONENTS
    # )
    found_rois = segmenter.find_blobs(
        # Segmenter.LEGACY_COMPATIBLE
    )
    # found_rois = segmenter.find_blobs(Segmenter.LEGACY_COMPATIBLE)
    # found_rois = segmenter.find_blobs()
    segmenter.split_by_blobs(found_rois)

    found = [a_roi.features for a_roi in found_rois]
    sort_by_coords(found)
    if found != ref:
        different, not_in_act, not_in_ref = visual_diffs(found, ref, sample, vis1)
    assert found == ref


one_contour = [
    (APERO, "apero2023_tha_bioness_013_st46_d_n4_d1_1_sur_1"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n4_d2_1_sur_2"),
    # (APERO, "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2"), # also missing
    (APERO, "apero2023_tha_bioness_013_st46_d_n5_d2_2_sur_2"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n6_d2_1_sur_2"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n6_d3"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n7_d2_1_sur_1"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n7_d3"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n8_d2_1_sur_1"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n8_d3"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n9_d3"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n1_d1_1_sur_1"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n2_d1_2_sur_2"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n4_d1_1_sur_1"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n4_d2_1_sur_2"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d1_1_sur_4"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d1_4_sur_4"),
    (TRIATLAS, "m158_mn05_n1_d2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d2_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n3_d3"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n4_d3"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n5_d2_1_sur_8"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n5_d3"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n7_d2_1_sur_4"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n8_d1_1_sur_1"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n8_d2_1_sur_1"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n9_d1_1_sur_1"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n2_d2_3_sur_4"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n2_d3"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n3_d3"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n5_d1_1_sur_1"),
    # (APERO1, "apero2023_tha_bioness_006_st20_n_n7_d1_2_sur_2"), # also big diff
    (APERO1, "apero2023_tha_bioness_006_st20_n_n7_d2_4_sur_4"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n8_d2_3_sur_4"),
    # (APERO1, "apero2023_tha_bioness_006_st20_n_n9_d2_1_sur_4"), # TODO: Check the image, 3,7M of contours
    (APERO1, "apero2023_tha_bioness_006_st20_n_n9_d2_2_sur_4"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n9_d2_4_sur_4"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n1_d1_1_sur_1"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n1_d2_4_sur_4"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n1_d3"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n2_d2_1_sur_1"),
    # (APERO1, "apero2023_tha_bioness_017_st66_d_n3_d2_1_sur_1"), # Also in missing
    (APERO1, "apero2023_tha_bioness_017_st66_d_n5_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n5_d2_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n6_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n9_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n2_d1_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n6_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n8_d2_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n9_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n9_d2_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_018_st66_n_n9_d3"),
]

extra_big = [
    (APERO, "apero2023_tha_bioness_014_st46_n_n6_d1_1_sur_1"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d1_2_sur_4"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n2_d1_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n7_d1_2_sur_2"),
    (TRIATLAS, "m158_mn18_n2_d1_1_sur_4"),
]

missingd = [
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d2_1_sur_2"),
    (APERO, "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d1_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d1_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n3_d1_3_sur_4"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n3_d2_1_sur_1"),
]

wrong_mask_maybe_gives_no_roi_when_legacy_has = [
    (APERO1, "apero2023_tha_bioness_013_st46_d_n1_d2_1_sur_1"),
]

more_than25 = [(APERO1, "apero2023_tha_bioness_018_st66_n_n8_d3")]


@pytest.mark.parametrize(
    "project, sample",
    one_contour,
    ids=[sample for (_prj, sample) in one_contour],
    # [
    #     (APERO1, "apero2023_tha_bioness_005_st20_d_n2_d1_2_sur_2"),
    #     (APERO, "apero2023_tha_bioness_013_st46_d_n7_d3"), # Single contour
    # (APERO, "apero2023_tha_bioness_014_st46_n_n6_d1_1_sur_1"),
    # (TRIATLAS, "m158_mn18_n2_d1_1_sur_4"),
    # (APERO, "apero2023_tha_bioness_014_st46_n_n7_d1_2_sur_4"), # 1 missing object
    # (APERO, "apero2023_tha_bioness_014_st46_n_n7_d2_1_sur_2"), # 1 missing object
    # ],
    # project = 'Zooscan_apero_tha_bioness_sup2000_sn033'
    # sample = 'apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4'
)
def test_algo_diff(projects, tmp_path, project, sample):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()

    # TODO: Add threshold (AKA 'upper= 243' in config) here
    segmenter = Segmenter(vis1, conf.minsizeesd_mm, conf.maxsizeesd_mm)

    # found_rois_new = segmenter.find_blobs(Segmenter.METH_CONNECTED_COMPONENTS)
    found_rois_new = segmenter.find_blobs()
    found_feats_new = [a_roi.features for a_roi in found_rois_new]
    sort_by_coords(found_feats_new)

    found_rois_compat = segmenter.find_blobs(Segmenter.LEGACY_COMPATIBLE)
    found_feats_compat = [a_roi.features for a_roi in found_rois_compat]
    sort_by_coords(found_feats_compat)

    if found_feats_compat != found_feats_new:
        different, not_in_compat, not_in_new = diff_dict_lists(
            found_feats_compat, found_feats_new, feature_unq
        )
        assert different == []
        # for a_diff in different:
        #     a_ref, an_act = a_diff
        #     print(a_ref)
        #     print("->", an_act)
        if len(not_in_compat) > 0:
            for num, a_new in enumerate(not_in_compat):
                # vig = cropnp(
                #     image=vis1,
                #     top=an_act["BY"],
                #     left=an_act["BX"],
                #     bottom=an_act["BY"] + an_act["Height"],
                #     right=an_act["BX"] + an_act["Width"],
                # )
                print(f"in new only {num}:{a_new}")
                # saveimage(vig, f"/tmp/diff_{num}.png")
                draw_roi(vis1, a_new, 4)
        # if len(not_in_new) > 0:
        #     for num, a_compat in enumerate(not_in_new):
        #         # vig = cropnp(
        #         #     image=vis1,
        #         #     top=an_act["BY"],
        #         #     left=an_act["BX"],
        #         #     bottom=an_act["BY"] + an_act["Height"],
        #         #     right=an_act["BX"] + an_act["Width"],
        #         # )
        #         print(f"in compat only {num}:{a_compat}")
        #         # saveimage(vig, f"/tmp/diff_{num}.png")
        #         draw_roi(vis1, a_compat, 8)
        # if len(not_in_compat) + len(not_in_new) > 0:
        #     saveimage(vis1, f"/tmp/dif_on_{sample}.tif")
        # assert not_in_compat == [], "Extra NEW"
    # if len(not_in_act) > 0:
    # for num, a_ref in enumerate(ref):
    #     cv2.rectangle(
    #         vis1,
    #         (a_ref["BX"], a_ref["BY"]),
    #         (a_ref["BX"] + a_ref["Width"], a_ref["BY"] + a_ref["Height"]),
    #         (0,),
    #         1,
    #     )
    # for num, a_ref in enumerate(not_in_act):
    #     print(f"missing ref {num}:{a_ref}")
    #     cv2.rectangle(
    #         vis1,
    #         (a_ref["BX"], a_ref["BY"]),
    #         (a_ref["BX"] + a_ref["Width"], a_ref["BY"] + a_ref["Height"]),
    #         (0,),
    #         4,
    #     )
    #     saveimage(vis1, "/tmp/diff.tif")
    assert found_feats_new == found_feats_compat
    # assert different == []
    # assert not_in_act == []
    # assert not_in_ref == []


def draw_roi(image: np.ndarray, features: Features, thickness: int = 1):
    cv2.rectangle(
        image,
        (features["BX"], features["BY"]),
        (features["BX"] + features["Width"], features["BY"] + features["Height"]),
        (0,),
        thickness,
    )


def sort_by_coords(features: List[Dict]):
    features.sort(key=feature_unq)

@pytest.mark.parametrize(
    "project, sample",
    wrong_mask_maybe_gives_no_roi_when_legacy_has,
    ids=[sample for (_prj, sample) in wrong_mask_maybe_gives_no_roi_when_legacy_has],
)
def test_nothing_found(projects, tmp_path, project, sample):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()
    ref = read_measurements(folder, sample, index)
    # TODO: Add threshold (AKA 'upper= 243' in config) here
    segmenter = Segmenter(vis1, conf.minsizeesd_mm, conf.maxsizeesd_mm)
    # found_rois = segmenter.find_blobs(
    #     Segmenter.LEGACY_COMPATIBLE | Segmenter.METH_CONNECTED_COMPONENTS
    # )
    found_rois = segmenter.find_blobs(Segmenter.METH_CONNECTED_COMPONENTS)
    # found_rois = segmenter.find_blobs(Segmenter.METH_RETR_TREE)
    # found_rois = segmenter.find_blobs(Segmenter.LEGACY_COMPATIBLE)
    # found_rois = segmenter.find_blobs()
    segmenter.split_by_blobs(found_rois)

    found = [a_roi.features for a_roi in found_rois]
    sort_by_coords(found)
    # if found != ref:
    #     different, not_in_act, not_in_ref = visual_diffs(found, ref, sample, vis1)
    assert found == ref


def visual_diffs(actual, expected, sample, tgt_img):
    different, not_in_ref, not_in_act = diff_dict_lists(expected, actual, feature_unq)
    for a_diff in different:
        a_ref, an_act = a_diff
        print(a_ref)
        print("->", an_act)
    for num, an_act in enumerate(not_in_ref):
        # vig = cropnp(
        #     image=vis1,
        #     top=an_act["BY"],
        #     left=an_act["BX"],
        #     bottom=an_act["BY"] + an_act["Height"],
        #     right=an_act["BX"] + an_act["Width"],
        # )
        print(f"extra {num}:{an_act}")
        # saveimage(vig, f"/tmp/diff_{num}.png")
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
    for num, a_ref in enumerate(not_in_act):
        print(f"missing ref {num}:{a_ref}")
        cv2.rectangle(
            tgt_img,
            (a_ref["BX"], a_ref["BY"]),
            (a_ref["BX"] + a_ref["Width"], a_ref["BY"] + a_ref["Height"]),
            (0,),
            4,
        )
    if len(different) or len(not_in_act) or len(not_in_ref):
        saveimage(tgt_img, f"/tmp/dif_on_{sample}.tif")
    return different, not_in_act, not_in_ref


def read_measurements(project_folder, sample, index):
    work_files = project_folder.zooscan_scan.work.get_files(sample, index)
    measures = work_files["meas"]
    measures_types = {
        "BX": int,
        "BY": int,
        "Width": int,
        "Height": int,
        "Area": int,
        "%Area": float,
        "XStart": int,
        "YStart": int,
        "Major": float,
        "Minor": float,
        "Angle": float,
    }
    ref = read_result_csv(measures, measures_types)
    # This filter is _after_ measurements in ImageJ
    ref = [
        a_ref
        for a_ref in ref
        if a_ref["Width"] / a_ref["Height"] < Segmenter.max_w_to_h_ratio
    ]
    sort_by_coords(ref)
    if "%Area" in measures_types:
        for a_ref in ref:
            a_ref["%Area"] = round(
                a_ref["%Area"], 3
            )  # Sometimes there are more decimals in measurements
    if "Angle" in measures_types:
        for a_ref in ref:
            a_ref["Angle"] = round(
                a_ref["Angle"], 3
            )  # Sometimes there are more decimals in measurements
    return ref


def diff_dict_lists(
    ref: List[Dict], act: List[Dict], key_func: Callable[[Dict], Any]
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    different = []
    not_in_act = []
    refs_by_key = {key_func(a_ref): a_ref for a_ref in ref}
    acts_by_key = {key_func(an_act): an_act for an_act in act}
    for ref_key, a_ref in refs_by_key.items():
        in_act = acts_by_key.get(ref_key)
        if in_act is None:
            not_in_act.append(a_ref)
        else:
            if in_act != a_ref:
                different.append((a_ref, in_act))
            acts_by_key.pop(ref_key)
    return different, list(acts_by_key.values()), not_in_act
