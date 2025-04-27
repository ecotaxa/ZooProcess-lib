import os
from pathlib import Path

import gdown  # TODO: Need dependencies here

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
SEGMENTER_DIR = IMAGES_DIR / "segmenter"
MEASURES_DIR = IMAGES_DIR / "measures"
FEATURES_DIR = DATA_DIR / "features"
PROJECT_DIR = DATA_DIR / "project"
BACKGROUND_DIR = PROJECT_DIR / "background"
SAMPLE_DIR = PROJECT_DIR / "scan"
WORK_DIR = PROJECT_DIR / "work"

BACK_TIME = "20240529_0946"

big_files = [
    (
        "work/apero2023_tha_bioness_017_st66_d_n1_d3_1_vis1.zip",
        "https://drive.google.com/file/d/1RUgX5TbFJnXwAI5FXaqsKKM5t_N9jFqF/view?usp=drive_link",
    ),
    (
        "scan/apero2023_tha_bioness_017_st66_d_n1_d3_1.tif",
        "https://drive.google.com/file/d/1NMYCDRgE7VPiFx-6VTrelT-Zll5BPXX5/view?usp=drive_link",
    ),
    (
        "raw/apero2023_tha_bioness_017_st66_d_n1_d3_raw_1.tif",
        "https://drive.google.com/file/d/1F1mUKcpYWwvQh-rElBgpd23zhDqyCsm2/view?usp=drive_link",
    ),
]

for file, url in big_files:
    big_proj_file = PROJECT_DIR / file
    file_ok = True
    try:
        big_file_stat = os.stat(big_proj_file)
        file_ok = big_file_stat.st_size > 1
    except FileNotFoundError:
        file_ok = False
    if not file_ok:
        gdown.download(url=url, output=big_proj_file.as_posix(), fuzzy=True)
