import random

import cv2
import numpy as np
import pytest

from ZooProcess_lib.ZooscanFolder import ZooscanFolder
from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.Zooscan_convert import Zooscan_convert
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
    TP = ZooscanProject(projects, project)
    index = 1  # TODO: should come from get_names() below

    raw_sample_file = folder.zooscan_scan.raw.get_file(sample, index)

    output_path = tmp_path / raw_sample_file.name
    Zooscan_convert(raw_sample_file, output_path, TP.readLut())
    actual_image = loadimage(output_path, type=cv2.IMREAD_UNCHANGED)

    ref_8bit_sample_file = folder.zooscan_scan.get_file_produced_from(
        raw_sample_file.name
    )
    assert ref_8bit_sample_file.exists()
    expected_image = loadimage(ref_8bit_sample_file, type=cv2.IMREAD_UNCHANGED)

    assert expected_image.shape == actual_image.shape
    assert np.array_equal(expected_image, actual_image)
