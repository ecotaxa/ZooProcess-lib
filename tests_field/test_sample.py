# Pt d'entrée "Segmentation après rajout fond blanc"
# Rajouter un offset sur le pt d'entrée -> rend les ROI relatives
# Pt d'entrée "masque de"
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Border import Border
from ZooProcess_lib.Features import (
    Features,
    feature_unq,
)
from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import (
    load_zipped_image,
    load_tiff_image_and_info,
)
from ZooProcess_lib.tools import measure_time
from tests.data_tools import (
    FEATURES_TOLERANCES,
    report_and_fix_tolerances,
    diff_features_lists,
    read_measurements,
    to_legacy_format,
    sort_by_coords,
)
from tests.test_utils import visual_diffs
from .env_fixture import projects
from .projects_for_test import (
    APERO2000,
    APERO,
)
from .projects_repository import (
    tested_samples,
    wrong_mask_maybe_gives_no_roi_when_legacy_has,
)


def test_8bit_sample_border(projects, tmp_path):
    """Ensure we compute borders to remove like legacy"""
    folder = ZooscanFolder(projects, APERO2000)
    sample = "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4"
    index = 1
    src_8bit_sample_file = folder.zooscan_scan.get_8bit_file(sample, index)
    assert src_8bit_sample_file.exists()
    src_info, src_image = load_tiff_image_and_info(src_8bit_sample_file)
    border = Border(src_image, resolution=src_info.resolution)
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


def load_final_ref_image(folder, sample, index):
    assert sample in [
        a_sample["name"] for a_sample in folder.zooscan_scan.raw.get_names()
    ]
    work_files_in_sample = folder.zooscan_scan.work.get_files(sample, index)
    zipped_combined = work_files_in_sample.get("combz")
    assert zipped_combined.exists()
    ref_info, reference_image = load_zipped_image(zipped_combined)
    # Note: the resolution is absent from the zipped TIFF
    return ref_info, reference_image


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_segmentation(projects, tmp_path, project, sample):
    assert_segmentation(
        projects,
        project,
        sample,
        Segmenter.METH_TOP_CONTOUR_SPLIT,
    )


def assert_segmentation(projects, project, sample, method):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    info_vis1, vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()
    lut = folder.zooscan_config.read_lut()
    processor = Processor(conf, lut)
    ref_feats = read_measurements(folder, sample, index)
    found_rois = processor.segmenter.find_ROIs_in_image(vis1, conf.resolution, method)
    # found_rois = list(filter(lambda r: r.mask.shape == (45, 50), found_rois))

    act_feats = to_legacy_format(
        processor.calculator.legacy_measures_list_from_roi_list(
            vis1, conf.resolution, found_rois
        )
    )
    sort_by_coords(act_feats)
    tolerance_problems = []
    if act_feats != ref_feats:
        different, not_in_reference, not_in_actual = diff_features_lists(
            ref_feats, act_feats, feature_unq
        )
        tolerance_problems = report_and_fix_tolerances(different, FEATURES_TOLERANCES)
        fix_valid_diffs(act_feats, not_in_reference, not_in_actual, info_vis1.width)
        if act_feats != ref_feats:
            different, not_in_reference, not_in_actual = diff_features_lists(
                ref_feats, act_feats, feature_unq
            )  # Diff again, to exhibit differences which were not fixed
            [draw_roi_mask(vis1, a_roi) for a_roi in found_rois]
            visual_diffs(different, not_in_reference, not_in_actual, sample, vis1)
    assert act_feats == ref_feats
    assert tolerance_problems == []


def test_linear_response_time(projects, tmp_path):
    method = Segmenter.METH_TOP_CONTOUR_SPLIT
    assert_linear_response_time(projects, tmp_path, tested_samples, method)


def assert_linear_response_time(projects, tmp_path, test_set, method):
    """Assert that response time does not explode, even for tricky/unusual cases.
    Note: Cannot be done in pytest parametrized test, which are isolated"""
    spent_times = []
    for num_test, (project, sample) in enumerate(test_set):
        folder = ZooscanFolder(projects, project)
        index = 1  # TODO: should come from get_names() below
        _, vis1 = load_final_ref_image(folder, sample, index)
        conf = folder.zooscan_config.read()
        processor = Processor(conf, None)
        spent, found_rois = measure_time(
            processor.segmenter.find_ROIs_in_image, vis1, 2400, method
        )
        # Minimal & fast
        assert found_rois != []
        spent_times.append(spent)
        if num_test % 10 == 0:
            do_perf_stats(spent_times)
        print(f"test #{num_test}: spent time: {spent:.2f}s")
    median, np_times, stddev = do_perf_stats(spent_times)
    z_scores = (np_times - median) / stddev
    max_zscore = 3
    for a_sample, its_time, its_score in zip(test_set, spent_times, z_scores):
        if abs(its_score) > max_zscore:
            print("Fails for sample:", a_sample, "time:", its_time, "score:", its_score)
    assert np.argmax(np.abs(z_scores) > max_zscore) == 0


def do_perf_stats(spent_times):
    np_times = np.array(spent_times)
    min_, max_, mean, median, stddev = [
        round(m, 2)
        for m in (
            np.min(spent_times),
            np.max(spent_times),
            np.average(np_times),
            np.median(np_times),
            np.std(np_times),
        )
    ]
    print(
        "min:", min_, "max:", max_, "mean:", mean, "median:", median, "stddev:", stddev
    )
    return median, np_times, stddev


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_algo_diff(projects, tmp_path, project, sample):
    """
    Read ROIs using legacy algorithm and CC one.
    The differences have to fit in expected patterns.
    """
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    info_vis1, vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()
    processor = Processor(conf, None)

    found_rois_new = processor.segmenter.find_ROIs_in_image(
        vis1, conf.resolution, Segmenter.METH_CONNECTED_COMPONENTS
    )
    ref_feats = processor.calculator.legacy_measures_list_from_roi_list(
        vis1, conf.resolution, found_rois_new, conf.upper
    )
    sort_by_coords(ref_feats)

    found_rois_compat = processor.segmenter.find_ROIs_in_image(
        vis1, conf.resolution, Segmenter.LEGACY_COMPATIBLE | Segmenter.METH_TOP_CONTOUR
    )
    act_feats = processor.calculator.legacy_measures_list_from_roi_list(
        vis1, conf.resolution, found_rois_compat, conf.upper
    )
    sort_by_coords(act_feats)

    if act_feats != ref_feats:
        different, not_in_reference, not_in_actual = diff_features_lists(
            ref_feats, act_feats, feature_unq
        )
        fix_valid_diffs(act_feats, not_in_reference, not_in_actual, info_vis1.width)
    assert act_feats == ref_feats


def fix_valid_diffs(act_feats, not_in_legacy, not_in_new, image_width):
    actual_was_modified = False
    enclosing_rectangles = [
        (
            a_new["BX"],
            a_new["BY"],
            a_new["BX"] + a_new["Width"],
            a_new["BY"] + a_new["Height"],
        )
        for a_new in act_feats
    ]
    # Boundaries of the problematic area
    central_band_end = int(image_width * Segmenter.overlap)
    central_band_start = image_width - int(image_width * Segmenter.overlap)

    # We have in 'new' version extra particles which were removed from legacy
    # as they touch both borders of a 20% vertical band in the middle of the image.
    # So in either processed band they are eliminated.
    for a_new in not_in_legacy:
        x1, x2 = a_new["BX"], a_new["BX"] + a_new["Width"]
        if x1 <= central_band_start and x2 >= central_band_end:
            # OK, identified as 'validly extra', remove as we have no comparison point
            act_feats.remove(a_new)

    # We have in 'new' version missing particles, which were wrongly included in legacy.
    # e.g. Big object A 'embeds' small object B and is crossed by only central_band_start line
    #    (if crossed by both lines it's above case).
    #      Legacy:
    #           A in split in 2 by central_band_start computed above, giving Aleft and Aright.
    #           B ends up in Aright (in central band).
    #           While processing right band, Aright is eliminated as it touches a border.
    #           BUT B is not seen as 'inside Aright' so it survives.
    #           As legacy finds OK particle A in full (while processing left band), we
    #               have B which is included in A, impossible otherwise.
    #       New:
    #           A is detected and B is seen as 'embedded', so B is eliminated.
    for a_compat in not_in_new:
        x1, x2 = a_compat["BX"], a_compat["BX"] + a_compat["Width"]
        y1, y2 = a_compat["BY"], a_compat["BY"] + a_compat["Height"]
        ok_geo = (
            central_band_start <= x1 <= central_band_end
            and central_band_start <= x2 <= central_band_end
        )
        # Not 100% accurate as we should compare masks, but enough for a test
        parent = [
            a_rect
            for a_rect in enclosing_rectangles
            if a_rect[0] <= x1 <= a_rect[2]
            and a_rect[0] <= x2 <= a_rect[2]
            and a_rect[1] <= y1 <= a_rect[3]
            and a_rect[1] <= y2 <= a_rect[3]
        ]
        if ok_geo and len(parent) >= 1:
            act_feats.append(a_compat)
            actual_was_modified = True
    if actual_was_modified:
        sort_by_coords(act_feats)


def draw_roi(image: np.ndarray, features: Features, thickness: int = 1):
    cv2.rectangle(
        image,
        (features["BX"], features["BY"]),
        (features["BX"] + features["Width"], features["BY"] + features["Height"]),
        (0,),
        thickness,
    )


def draw_roi_mask(image: np.ndarray, roi: ROI):
    (whole, _) = cv2.findContours(roi.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y = roi.x, roi.y
    assert len(whole) == 1
    cv2.drawContours(
        image=image,
        contours=whole,
        contourIdx=-1,
        color=(10,),
        thickness=5,
        offset=(x, y),
    )


@pytest.mark.parametrize(
    "segmentation_method",
    [
        # Segmenter.METH_TOP_CONTOUR, # This one indeed returns nothing
        Segmenter.LEGACY_COMPATIBLE | Segmenter.METH_TOP_CONTOUR,
        Segmenter.METH_CONNECTED_COMPONENTS,
        Segmenter.METH_TOP_CONTOUR_SPLIT,
    ],
)
@pytest.mark.parametrize(
    "project, sample",
    wrong_mask_maybe_gives_no_roi_when_legacy_has,
    ids=[sample for (_prj, sample) in wrong_mask_maybe_gives_no_roi_when_legacy_has],
)
def test_nothing_found(projects, tmp_path, project, sample, segmentation_method):
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    _, vis1 = load_final_ref_image(folder, sample, index)
    conf = folder.zooscan_config.read()
    processor = Processor(conf, None)
    ref = read_measurements(folder, sample, index)
    # found_rois = segmenter.find_blobs(
    #     Segmenter.LEGACY_COMPATIBLE | Segmenter.METH_CONNECTED_COMPONENTS
    # )
    found_rois = processor.segmenter.find_ROIs_in_image(
        vis1, conf.resolution, segmentation_method
    )
    # found_rois = segmenter.find_blobs(Segmenter.METH_RETR_TREE)
    # found_rois = segmenter.find_blobs(Segmenter.LEGACY_COMPATIBLE)
    # found_rois = segmenter.find_blobs()

    found = to_legacy_format(
        processor.calculator.legacy_measures_list_from_roi_list(
            vis1, conf.resolution, found_rois
        )
    )
    sort_by_coords(found)
    tolerance_problems = []
    if found != ref:
        different, not_in_reference, not_in_actual = diff_features_lists(
            ref, found, feature_unq
        )
        tolerance_problems = report_and_fix_tolerances(different, FEATURES_TOLERANCES)
    assert found == ref
    assert tolerance_problems == []


def test_dev_linear_response_time(projects, tmp_path):
    test_set = tested_samples
    assert_linear_response_time(projects, tmp_path, test_set)


dev_samples = [
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_2_sur_8",
    )
]


@pytest.mark.parametrize("project, sample", dev_samples)
def test_dev_segmentation(projects, tmp_path, project, sample):
    assert_segmentation(
        projects,
        project,
        sample,
        Segmenter.METH_TOP_CONTOUR_SPLIT,
    )
