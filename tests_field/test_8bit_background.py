
import cv2
import numpy as np
import pytest

from ZooProcess_lib.Processor import Processor
from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder
from ZooProcess_lib.img_tools import loadimage
from .env_fixture import projects, read_home
from .projects_for_test import IADO, APERO2000, APERO, TRIATLAS, APERO1

all_projects = [IADO, APERO2000, APERO, TRIATLAS, APERO1]


def all_backgrounds(but_not=()) -> list[tuple[str, str, int]]:
    ret = []
    for a_project in all_projects:
        folder = ZooscanProjectFolder(read_home(), a_project)
        for scan_date in folder.zooscan_back.get_dates():
            for idx in (1, 2):
                source_bg_file = folder.zooscan_back.get_raw_background_file(
                    scan_date, idx
                )
                if not source_bg_file.exists():
                    continue
                reference_bg_file = folder.zooscan_back.get_processed_background_file(
                    scan_date, idx
                )
                if not reference_bg_file.exists():
                    continue
                for an_exclusion in but_not:
                    if str(reference_bg_file).endswith(an_exclusion):
                        exclude = True
                        break
                else:
                    exclude = False
                if exclude:
                    continue
                ret.append((a_project, scan_date, idx))
    return ret


@pytest.mark.parametrize(
    "project, scan_date, idx",
    all_backgrounds(
        (
            "Zooscan_apero_tha_bioness_2_sn033/Zooscan_back/20241212_1130_back_large_2.tif",
            # TODO: Visually KO, too dark. Leftover by accident?
        )
    ),
)
def test_identical_converted_8bit_background(
    projects, project, scan_date, idx, tmp_path
):
    """Ensure we convert like legacy the scanned background images for all projects"""
    folder = ZooscanProjectFolder(projects, project)
    source_bg_file = folder.zooscan_back.get_raw_background_file(scan_date, idx)
    reference_bg_file = folder.zooscan_back.get_processed_background_file(
        scan_date, idx
    )
    expected_image = loadimage(reference_bg_file, type=cv2.IMREAD_UNCHANGED)
    processor = Processor.from_legacy_config(folder.zooscan_config.read(), folder.zooscan_config.read_lut())
    actual_image, _ = processor.converter.do_file_to_image(source_bg_file)
    assert np.array_equal(expected_image, actual_image)
