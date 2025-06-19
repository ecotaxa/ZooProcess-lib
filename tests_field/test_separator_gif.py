import cv2
import pytest

from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder, WRK_SEP
from ZooProcess_lib.img_tools import (
    load_image,
)
from .env_fixture import projects
from .projects_repository import tested_samples


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_read_separators(projects, project, sample):
    assert_sep_gif_ok(project, projects, sample)


def assert_sep_gif_ok(project, projects, sample):
    folder = ZooscanProjectFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    # Read separator mask
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    sep_file = work_files.get(WRK_SEP)
    if sep_file is not None:
        sep_image = load_image(sep_file, imread_mode=cv2.IMREAD_GRAYSCALE)  # 1.2s avg
        # save_gif_image(sep_image, Path(f"/tmp/{sample}.gif"))  # 2s avg
