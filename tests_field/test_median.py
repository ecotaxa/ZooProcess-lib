from pathlib import Path

import numpy as np
import pytest

from ZooProcess_lib.Legacy import averaged_median_mean
from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.img_tools import getPath
from ZooProcess_lib.img_tools import loadimage
from ZooProcess_lib.img_tools import (
    # crop, crop_scan, crophw, cropnp,
    saveimage,
    # picheral_median,
    # converthisto16to8, convertImage16to8bit,
    # minAndMax,
    # rotate90c, rotate90cc,
    # normalize, normalize_back,
    # separate_apply_mask,
    # draw_contours, draw_boxes, draw_boxes_filtered,
    # generate_vignettes,
    # mkdir,
    # resize,
    # rolling_ball_black_background,
)
from ZooProcess_lib.median import picheral_median
from ZooProcess_lib.to8bit import convertion
from tests.env_fixture import projects
from tests.projects_for_test import APERO, APERO2000


@pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
def test_picheral_median(projects):
    TP = ZooscanProject(projects, APERO2000)
    back_name = "20240112_1518_back_large_1.tif"

    back_file = Path(TP.back, back_name)
    print(f"file: {back_file}")
    back_image = loadimage(back_file.as_posix())

    median, mean = picheral_median(back_image)

    print(f"median: {median}, mean: {mean}")


def test_averaged_median(projects):
    TP = ZooscanProject(projects, APERO)
    source_bg_file = TP.getRawBackgroundFile("20241216_0926", 2)
    source_image = loadimage(source_bg_file)
    (marc_median, marc_mean) = averaged_median_mean(source_image)
    assert (float(marc_median), round(float(marc_mean), 4)) == (
        41331.45,
        40792.4142,
    )  # Values from ImageJ run


def debug_picheral_median(file):
    # project_folder = APERO2000
    # TP = ProjectClass(project_folder)

    # # back_name = "20240112_1518_back_large_1.tif"
    # # back_file = Path(TP.back, back_name)
    # back_file = Path(TP.back, filename)
    # print(f"file: {back_file}")
    # back_image = loadimage(back_file.as_posix())
    back_image = loadimage(file.as_posix())
    median, mean = averaged_median_mean(back_image)
    print(f"median: {median}, mean: {mean}")


def debug_picheral_median_local(file):
    project_folder = APERO2000
    TP = ZooscanProject(project_folder)

    # # back_name = "20240112_1518_back_large_1.tif"
    # # back_file = Path(TP.back, back_name)
    # back_file = Path(TP.back, filename)
    # print(f"file: {back_file}")
    # back_image = loadimage(back_file.as_posix())
    back_image = loadimage(file.as_posix())

    median, mean = picheral_median(back_image)
    # median, mean = median(back_image)

    print(f"median: {median}, mean: {mean}")


@pytest.mark.skip(reason="Skipping this test for now because of XY reason.")
def test_median(projects, tmp_path):
    TP = ZooscanProject(projects, APERO2000)

    back_name = "20240112_1518_back_large_1.tif"
    # back_file = Path(TP.back, back_name)

    # output_path = tmp_path
    # # saveimage(scan_unbordered, sample, "unbordered", ext="tiff", path=output_path)
    back_file = Path(getPath(back_name, "unbordered", ext="tiff", path=tmp_path))

    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(TP.rawscan, sample + "_raw" + "_1" + ".tif")
    # image = loadimage(rawscan_file.as_posix())

    rawscan_file = Path(getPath(sample, "unbordered", ext="tiff", path=tmp_path))
    # image = loadimage(rawscan_file.as_posix())

    debug_picheral_median(back_file)
    debug_picheral_median_local(back_file)

    debug_picheral_median(rawscan_file)
    debug_picheral_median_local(rawscan_file)


@pytest.mark.skip(reason="Skipping this test for now because of XY reason.")
def test_en_8bit(projects, tmp_path):
    TP = ZooscanProject(projects, APERO2000)

    back_name = "20240112_1518_back_large_1.tif"
    back_file = Path(getPath(back_name, "resized", ext="tiff", path=tmp_path))
    back_image = loadimage(back_file.as_posix())

    image_back_8bit = convertion(back_image, back_name, TP=TP)
    saveimage(image_back_8bit, back_name, "8bit", ext="jpg", path=tmp_path)

    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(getPath(sample , "unbordered", ext="tiff", path=tmp_path))
    # image = loadimage(rawscan_file.as_posix())
    # image_sample_8bit = convertion(image, sample)
    # saveimage(image_sample_8bit, sample, "8bit", ext="jpg", path=tmp_path)
    rawscan_file = Path(getPath(sample, "treated", ext="tiff", path=tmp_path))
    image_sample_8bit = loadimage(rawscan_file.as_posix())

    image_substracted = np.subtract(image_sample_8bit, image_back_8bit)
    saveimage(image_substracted, sample, "substracted", ext="tiff", path=tmp_path)

    image_substracted2 = np.subtract(image_back_8bit, image_sample_8bit)
    saveimage(image_substracted2, sample, "substracted2", ext="tiff", path=tmp_path)
    print("Done")


if __name__ == "__main__":
    # test_en_8bit()

    project_folder = APERO2000
    TP = ZooscanProject(project_folder)

    back_name = "20240112_1518_back_large_1.tif"
    back_file = Path(getPath(back_name, "resized", ext="tiff", path=tmp_path))
    back_image = loadimage(back_file.as_posix())

    image_back_8bit = convertion(back_image, back_name, TP=TP)
    saveimage(image_back_8bit, back_name, "8bit", ext="jpg", path=tmp_path)

    sample = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
    # rawscan_file = Path(getPath(sample , "unbordered", ext="tiff", path=tmp_path))
    # image = loadimage(rawscan_file.as_posix())
    # image_sample_8bit = convertion(image, sample)
    # saveimage(image_sample_8bit, sample, "8bit", ext="jpg", path=tmp_path)
    rawscan_file = Path(getPath(sample, "treated", ext="tiff", path=tmp_path))
    image_sample_8bit = loadimage(rawscan_file.as_posix())

    image_substracted = np.subtract(image_sample_8bit, image_back_8bit)
    saveimage(image_substracted, sample, "substracted", ext="tiff", path=tmp_path)

    image_substracted2 = np.subtract(image_back_8bit, image_sample_8bit)
    saveimage(image_substracted2, sample, "substracted2", ext="tiff", path=tmp_path)
    print("Done")
