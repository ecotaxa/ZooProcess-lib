from typing import Dict, Tuple

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Features import legacy_measures_list_from_roi_list
from ZooProcess_lib.ROI import feature_unq
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.img_tools import loadimage
from ZooProcess_lib.segmenters.ConnectedComponents import (
    ConnectedComponentsSegmenter,
    CC,
)
from .data_dir import SEGMENTER_DIR
from .test_sample import sort_by_coords


@pytest.mark.skip("interface changed")
def test_denoiser():
    obj = 255
    bgd = 0
    img = np.array(
        [
            [bgd, bgd, bgd, bgd, obj, obj, obj],
            [bgd, obj, bgd, bgd, obj, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, obj, bgd, bgd, bgd, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
        ],
        np.uint8,
    )
    exp = np.array(
        [
            [bgd, bgd, bgd, bgd, obj, obj, obj],
            [bgd, bgd, bgd, bgd, obj, obj, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
            [bgd, bgd, bgd, bgd, bgd, bgd, bgd],
        ],
        np.uint8,
    )
    res = Segmenter.denoise_for_segment(img)
    np.testing.assert_equal(res, exp)


def test_holes_62388():
    # This image biggest particle has a pattern in holes, their contour touches the particle itself
    image = loadimage(SEGMENTER_DIR / "cc_62388.png")
    image = cv2.copyMakeBorder(image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255,))
    parts = Segmenter(image, 0.1, 100000, 243).find_blobs(
        Segmenter.METH_TOP_CONTOUR_SPLIT
    )
    features = sorted(
        legacy_measures_list_from_roi_list(
            image, parts, 243, {"Area", "BX", "BY", "Height", "Width"}
        ),
        key=feature_unq,
        reverse=True,
    )
    assert features == [
        {"Area": 244, "BX": 146, "BY": 91, "Height": 26, "Width": 81},
        {"Area": 1407, "BX": 102, "BY": 110, "Height": 34, "Width": 67},
        {"Area": 207, "BX": 67, "BY": 7, "Height": 18, "Width": 26},
        {"Area": 342, "BX": 14, "BY": 96, "Height": 30, "Width": 25},
        {"Area": 1675, "BX": 4, "BY": 4, "Height": 148, "Width": 258},
    ]


def test_compare_split_methods():
    image = loadimage(
        SEGMENTER_DIR / "apero2023_tha_bioness_013_st46_d_n8_d3_1_vis1.png"
    )
    THRESHOLD = 243
    segmenter = Segmenter(image, 0.3, 100, THRESHOLD)

    found_rois_new = segmenter.find_blobs(Segmenter.METH_TOP_CONTOUR_SPLIT)
    found_feats_new = legacy_measures_list_from_roi_list(
        image, found_rois_new, THRESHOLD
    )
    sort_by_coords(found_feats_new)

    found_rois_old = segmenter.find_blobs(Segmenter.METH_CONTOUR_TREE)
    found_feats_compat = legacy_measures_list_from_roi_list(
        image, found_rois_old, THRESHOLD
    )
    sort_by_coords(found_feats_compat)

    assert found_feats_compat == found_feats_new


def geoloc(stats) -> Dict[Tuple, Tuple]:
    ret = {}
    for ndx, (x_ref, y_ref, w_ref, h_ref, a_ref) in enumerate(stats):
        key = (int(x_ref), int(y_ref), int(w_ref), int(h_ref))
        ret[key] = (a_ref, ndx)
    return ret


def mask_at(labels: np.ndarray, coords, label):
    x, y, w, h = coords
    sub_labels = labels[y : y + h, x : x + w]
    return sub_labels == label


def cc_from_coord(coord) -> CC:
    x, y, w, h = coord
    return CC(x, y, w, h, -1, -1, 0, 0, 1000, 1000)


def test_split_image():
    # Assert validity of splitting an image vertically in 2, in CC mode,
    # and reconstructing the CCs
    thresh_max = 243
    image = loadimage(SEGMENTER_DIR / "s_17_3_tot_1_vis1.png")
    _th, inv_mask = cv2.threshold(image, thresh_max, 1, cv2.THRESH_BINARY_INV)
    (stripe,) = ConnectedComponentsSegmenter.extract_ccs(inv_mask)
    ref_labels, retval1, ref_stats = stripe.labels, stripe.retval, stripe.stats
    stripes = ConnectedComponentsSegmenter.extract_ccs_vertical_split(inv_mask)
    (l_labels, l_ret, l_stats) = stripes[0].labels, stripes[0].retval, stripes[0].stats
    (c_labels, c_ret, c_stats) = stripes[1].labels, stripes[1].retval, stripes[1].stats
    (r_labels, r_ret, r_stats) = stripes[2].labels, stripes[2].retval, stripes[2].stats
    assert (
        l_ret + c_ret + r_ret >= retval1
    )  # Split objects appear at least twice but filtered out


def test_splitting_image_conserves_data():
    # Assert validity of splitting an image vertically in 2 and reconstructing the CCs
    thresh_max = 243
    image = loadimage(SEGMENTER_DIR / "s_17_3_tot_1_vis1.png")
    _th, inv_mask = cv2.threshold(image, thresh_max, 1, cv2.THRESH_BINARY_INV)
    (stripe,) = ConnectedComponentsSegmenter.extract_ccs(inv_mask)
    ref_labels, retval1, ref_stats = stripe.labels, stripe.retval, stripe.stats
    vs_stripe = ConnectedComponentsSegmenter.extract_ccs_vertical_split(
        inv_mask, for_test=True
    )[0]
    (
        vs_labels,
        retval2,
        vs_stats,
    ) = (
        vs_stripe.labels,
        vs_stripe.retval,
        vs_stripe.stats,
    )
    assert retval2 >= retval1  # Split objects appear at least twice
    # Equivalence of outputs
    ref_stats_by_coord = geoloc(ref_stats)
    vs_stats_by_coord = geoloc(vs_stats)
    print("cc2 done, comparing")
    for coord in ref_stats_by_coord.keys():
        ref_area, ref_ndx = ref_stats_by_coord[coord]
        vs_area, vs_ndx = vs_stats_by_coord[coord]
        # 'feature' equivalence
        assert ref_area == vs_area
        # Masks equality
        ref_mask = mask_at(inv_mask, coord, ref_ndx)
        ref_pix_count = np.count_nonzero(ref_mask)
        vs_mask = mask_at(inv_mask, coord, vs_ndx)
        vs_pix_count = np.count_nonzero(vs_mask)
        np.testing.assert_equal(ref_mask, vs_mask)
        assert ref_pix_count == vs_pix_count
        #
        cc = cc_from_coord(coord)
        holes_ref, submask_ref = ConnectedComponentsSegmenter.get_regions(
            ref_labels, ref_ndx, cc, ref_area
        )
        holes_vs, submask_vs = ConnectedComponentsSegmenter.get_regions(
            vs_labels, vs_ndx, cc, vs_area
        )
        if np.any(submask_ref != submask_vs):
            print(f"submask diff for {ref_ndx} AKA {vs_ndx}")
        if np.any(holes_ref != holes_vs):
            print(f"holes mask diff for {ref_ndx} AKA {vs_ndx}")
        np.testing.assert_equal(holes_ref, holes_vs)
        np.testing.assert_equal(submask_ref, submask_vs)
    for coord in vs_stats_by_coord.keys():
        if coord in ref_stats_by_coord:
            continue
        print(coord, vs_stats_by_coord[coord])
