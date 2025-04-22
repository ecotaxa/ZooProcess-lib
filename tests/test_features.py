from ZooProcess_lib.Features import Features
from ZooProcess_lib.Segmenter import Segmenter
from ZooProcess_lib.img_tools import loadimage
from .test_sample import (
    read_measures_from_file,
    to_legacy_format,
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
    if False:
        the_roi = [r for r in roi_list if (r.x, r.y) == (95, 14000)][0]
        features = [Features(image, the_roi, THRESHOLD)]
        segmenter.split_by_blobs([the_roi])
    else:
        features = [Features(image, a_roi, THRESHOLD) for a_roi in roi_list]
    features_as_legacy = to_legacy_format([f.as_legacy() for f in features])
    sort_by_coords(features_as_legacy)
    ref_features = read_measures_from_file(features_file)
    if not ref_features == features_as_legacy:
        visual_diffs(
            ref_features,
            features_as_legacy,
            "apero2023_tha_bioness_013_st46_d_n8_d3",
            image,
            roi_list,
        )
    assert ref_features == features_as_legacy
