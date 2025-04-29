#
# Reverse engineering of macro Zooscan_convert.txt from Zooprocess legacy
#
from pathlib import Path

import cv2

from .ImageJLike import convert_16bit_image_to_8bit_min_max
from .Legacy import averaged_median_mean
from .img_tools import loadimage, rotate90c, Lut, minAndMax, image_info, saveimage, load_tiff_image_and_info

CV2_VERTICAL_FLIP_CODE = 0
CV2_NO_TIFF_COMPRESSION = 1


def Zooscan_convert(from_file: Path, to_file: Path, lut: Lut):
    """
    :param from_file: e.g. 20241216_0926_back_large_raw_2.tif
    :param to_file: e.g. 20241216_0926_back_large_2.tif in another directory
    :param lut: processing configuration
    """
    is_a_background = from_file.name[13:19] == "_back_"
    # do_median_filter = not is_a_background  # TODO: Questionable if better as a param

    raw_info, raw_image = load_tiff_image_and_info(from_file)

    if (
        raw_info.width > 14000
        and raw_info.resolution == 2400
        and lut.resolutionreduct == 1200
    ):
        raise Exception("Reduction to 1200dpi not implemented")

    (marc_median, marc_mean) = averaged_median_mean(raw_image)
    (min_rec, max_rec) = minAndMax(marc_median, lut)

    if lut.sens == "before" and lut.gamma != 1:
        raise Exception("Gamma not implemented")

    if lut.medianchoice != "yes" or is_a_background:
        image8bit = convert_16bit_image_to_8bit_min_max(raw_image, min_rec, max_rec)
    else:
        if raw_info.resolution < 2400:
            raise Exception("medianchoice yes and res < 2400 not implemented")
        elif raw_info.resolution == 2400:
            if raw_info.width <= 14000:
                # Mimic legacy up to 5.12
                raise Exception(
                    "medianchoice yes, res = 2400 and width <= 14000 not implemented"
                )
            else:
                raise Exception(
                    "medianchoice yes, res = 2400 and width > 14000 not implemented"
                )
        else:
            assert False, f"Unexpected image resolution {raw_info.resolution} for {from_file}"

    image_rotated = rotate90c(image8bit)
    image_flipped = cv2.flip(image_rotated, CV2_VERTICAL_FLIP_CODE)

    dest_file = to_file.as_posix()
    saveimage(image_flipped, dest_file, dpi=raw_info.resolutions)
