import cv2

from ZooProcess_lib.Zooscan_convert import CV2_VERTICAL_FLIP_CODE
from ZooProcess_lib.img_tools import loadimage, rotate90cc


def reverse_convert(raw_sample_file):
    """Inverse transformation as convert to 8 bits, to have comparable pixels in debug"""
    raw_sample_image = loadimage(raw_sample_file, type=cv2.IMREAD_UNCHANGED)
    raw_sample_image = cv2.flip(raw_sample_image, CV2_VERTICAL_FLIP_CODE)
    return rotate90cc(raw_sample_image)


def diff_actual_with_ref_and_source(expected_image, actual_image, raw_sample_image):
    e_cum = 0
    for l in range(expected_image.shape[0]):
        e_cum += print_diff(expected_image, actual_image, l, raw_sample_image)
        if e_cum > 100:
            break


def print_line(array, index):
    line = array[index]
    print("-------------")
    print(",".join([str(int(elem)) for elem in line]))


def print_diff(array_e, array_a, line_num, comp_img) -> int:
    ret = 0
    line_e = array_e[line_num]
    line_a = array_a[line_num]
    for ndx, elem in enumerate(line_e):
        if line_a[ndx] != elem:
            print(
                f"lin {line_num} diff @{ndx}: seen {line_a[ndx]} vs exp {elem}, src {comp_img[line_num][ndx]}"
            )
            ret += 1
    return ret
