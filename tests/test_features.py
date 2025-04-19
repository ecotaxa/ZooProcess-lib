from pathlib import Path

from ZooProcess_lib.Features import Features
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.img_tools import loadimage, load_zipped_image, saveimage
from .test_sample import (
    read_measures_from_file,
    round_measurements,
    sort_by_coords,
    visual_diffs,
)
from .test_segmenter import MEASURES_DIR
from .test_segmenter import SEGMENTER_DIR


def test_features_on_simplified_scan():
    image_path = SEGMENTER_DIR / "apero2023_tha_bioness_013_st46_d_n8_d3_1_vis1.png"
    image = loadimage(image_path)
    features_file = MEASURES_DIR / "apero2023_tha_bioness_013_st46_d_n8_d3_1_meas.txt"
    THRESHOLD = 243
    segmenter = Segmenter(image, 0.3, 100, THRESHOLD)
    roi_list = segmenter.find_blobs(Segmenter.METH_TOP_CONTOUR_SPLIT)
    features = [Features(image, p, THRESHOLD) for p in roi_list]
    legacy_features = round_measurements([f.as_legacy() for f in features])
    sort_by_coords(legacy_features)
    ref_features = read_measures_from_file(features_file)
    if not ref_features == legacy_features:
        visual_diffs(
            ref_features,
            legacy_features,
            "apero2023_tha_bioness_013_st46_d_n8_d3",
            image,
            roi_list,
        )
    assert ref_features == legacy_features
