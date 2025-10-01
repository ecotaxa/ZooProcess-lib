import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder, WRK_MSK1
from ZooProcess_lib.img_tools import (
    get_creation_date,
    save_gif_image,
    load_image,
)
from test_utils import save_diff_image, diff_actual_with_ref
from .env_fixture import projects
from .projects_for_test import POINT_B_JB, APERO1, TRIATLAS
from .projects_repository import tested_samples, all_samples_in

# The test takes ages, by randomizing the order there are better chances to see problems early
shuffled = tested_samples.copy()
random.shuffle(shuffled)


@pytest.mark.parametrize(
    "project, sample",
    shuffled,
    ids=[sample for (_prj, sample) in shuffled],
)
def test_msk_generator(projects, project, sample, tmp_path):
    assert_same_msk(project, projects, sample, tmp_path)


def assert_same_msk(project, projects, sample, tmp_path):
    folder = ZooscanProjectFolder(projects, project)
    conf = folder.zooscan_config.read()
    lut = folder.zooscan_config.read_lut()
    processor = Processor(conf, lut)
    index = 1  # TODO: should come from get_names() below

    legacy_files = folder.zooscan_scan.work.get_files(sample, index)
    if WRK_MSK1 not in legacy_files:
        return
    legacy_msk_path = legacy_files[WRK_MSK1]

    # Read raw sample scan
    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    digitized_at = get_creation_date(raw_sample_file)
    assert digitized_at is not None
    # Backgrounds pre-processing
    bg_raw_files = folder.zooscan_back.get_last_raw_backgrounds_before(digitized_at)
    bg_converted_files = [
        processor.converter.do_file_to_image(a_raw_bg_file, True)
        for a_raw_bg_file in bg_raw_files
    ]
    combined_bg_image, bg_resolution = processor.bg_combiner.do_from_images(
        bg_converted_files
    )
    # Sample pre-processing
    eight_bit_sample_image, sample_resolution = processor.converter.do_file_to_image(
        raw_sample_file, False
    )
    # Background removal
    sample_scan = processor.bg_remover.do_from_images(
        combined_bg_image, bg_resolution, eight_bit_sample_image, sample_resolution
    )

    mask = processor.segmenter.get_mask_from_image(sample_scan)
    tmp_path = tempfile.mktemp(suffix=".gif")
    save_gif_image(mask, Path(tmp_path))
    print("Mask saved to {} vs {}".format(Path(tmp_path), legacy_msk_path))
    ref_gif = load_image(legacy_msk_path, imread_mode=cv2.IMREAD_GRAYSCALE)
    assert mask.shape == ref_gif.shape
    assert mask.dtype == ref_gif.dtype
    diff_actual_with_ref(ref_gif, mask, tolerance=0)
    # save_diff_image(
    #     ref_gif,
    #     mask,
    #     Path("/tmp/zooprocess/diff.jpg"),
    # )
    # assert np.array_equal(mask, ref_gif)


dev_samples = [(p, s) for (p, s) in all_samples_in(POINT_B_JB)]  # if "197809" in s
dev_samples = [
    (TRIATLAS, "m158_mn15_n3_d3")
]  # Extra vignette not filtered by W/H ratio
dev_samples = all_samples_in(APERO1)[:8]


@pytest.mark.parametrize("project, sample", dev_samples)
def test_dev_msk_generator(projects, project, sample, tmp_path):
    assert_same_msk(project, projects, sample, tmp_path)
