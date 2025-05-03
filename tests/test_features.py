from ZooProcess_lib.Features import Features, feature_unq
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.img_tools import loadimage
from .data_dir import FEATURES_DIR, SEGMENTER_DIR
from .data_tools import (
    read_measures_from_file,
    to_legacy_format,
    sort_by_coords,
    diff_features_lists,
    report_and_fix_tolerances,
    FEATURES_TOLERANCES,
)
from .test_utils import visual_diffs


def test_features_on_simplified_scan():
    image_path = SEGMENTER_DIR / "apero2023_tha_bioness_013_st46_d_n8_d3_1_vis1.png"
    image = loadimage(image_path)
    features_file = FEATURES_DIR / "apero2023_tha_bioness_013_st46_d_n8_d3_1_meas.txt"
    THRESHOLD = 243
    RESOLUTION = 2400
    segmenter = Segmenter(0.3, 100, THRESHOLD)
    roi_list, _ = segmenter.find_ROIs_in_image(
        image, RESOLUTION, Segmenter.METH_TOP_CONTOUR_SPLIT
    )
    if False:
        the_roi = [r for r in roi_list if (r.x, r.y) == (95, 14000)][0]
        features = [Features(image, the_roi, THRESHOLD)]
    else:
        features = [Features(image, RESOLUTION, a_roi, THRESHOLD) for a_roi in roi_list]
    features_as_legacy = to_legacy_format([f.as_measures() for f in features])
    sort_by_coords(features_as_legacy)
    ref_features = read_measures_from_file(features_file)
    damaged_features = {
        "IntDen",
        "Max",
        "Mean",
        "Kurt",
        "Skew",
        "StdDev",
        "Mode",
        "Median",
    }  # The image was damaged, TODO: Re-generate one
    NOT_SAME_TOLERANCES = FEATURES_TOLERANCES.copy()
    for dam in damaged_features:
        if dam in NOT_SAME_TOLERANCES:
            NOT_SAME_TOLERANCES.pop(dam)
        for feat in ref_features:
            if dam in feat:
                feat.pop(dam)
        for feat in features_as_legacy:
            if dam in feat:
                feat.pop(dam)
    if not ref_features == features_as_legacy:
        different, not_in_reference, not_in_actual = diff_features_lists(
            ref_features, features_as_legacy, feature_unq
        )
        tolerance_problems = report_and_fix_tolerances(different, NOT_SAME_TOLERANCES)
        visual_diffs(
            different,
            not_in_reference,
            not_in_actual,
            "apero2023_tha_bioness_013_st46_d_n8_d3",
            image,
        )
    assert ref_features == features_as_legacy
    assert tolerance_problems == []
