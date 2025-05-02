import random

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.img_tools import loadimage
from .env_fixture import projects
from .projects_repository import tested_samples

# The test takes ages, by randomizing the order there are better chances to see problems early
shuffled = tested_samples.copy()
random.shuffle(shuffled)

@pytest.mark.parametrize(
    "project, sample",
    shuffled,
    ids=[sample for (_prj, sample) in shuffled],
)
def test_identical_converted_8bit_sample(projects, project, sample, tmp_path):
    """Ensure we convert like legacy the scanned background images"""
    folder = ZooscanFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    processor = Processor.from_legacy_config(folder.zooscan_config.read(), folder.zooscan_config.read_lut())

    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)
    actual_image, _ = processor.converter.do_file_to_image(raw_sample_file)

    ref_8bit_sample_file = folder.zooscan_scan.get_file_produced_from(
        raw_sample_file.name
    )
    assert ref_8bit_sample_file.exists()
    expected_image = loadimage(ref_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)

    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image, actual_image)
