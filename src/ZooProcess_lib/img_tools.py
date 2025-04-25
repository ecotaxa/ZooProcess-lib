import math
import mmap
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Callable, Union
from zipfile import ZipFile

import PIL
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base, TAGS

from .Lut import Lut
from .tools import timeit

PIL.Image.MAX_IMAGE_PIXELS = 375000000

debug = True  # False


def print_image_info(img, title=None) -> None:
    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    # channels = img.shape[2]
    if title:
        print(title)
    print("Image Dimension    : ", dimensions)
    # print('Image Height       : ',height)
    # print('Image Width        : ',width)
    # print('Number of Channels : ',channels)


# draw a grayscale histogram
def histogram(image, normalize=True, title=None, show=True):
    from matplotlib import pyplot as plt

    # compute a grayscale histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # normalize the histogram
    if normalize:
        hist /= hist.sum()
        _title = "Grayscale Histogram (Normalized)"
        _xLabel = "Bins"
        _ylabel = "% of Pixels"
    else:
        _title = "Grayscale Histogram"
        _xLabel = "Bins"
        _ylabel = "# of Pixels"

    if title:
        _title = title

    # matplotlib expects RGB images so convert and then display the image
    # with matplotlib
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

    # plot the histogram
    plt.figure()
    plt.title(_title)
    plt.xlabel(_xLabel)
    plt.ylabel(_ylabel)
    plt.plot(hist)
    plt.xlim([0, 256])
    if show:
        plt.show()
    # plt.fig
    return plt


def convert_image_to_8bit(img) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0 / 256.0, beta=-0.49999)


def rotate90c(img) -> np.ndarray:
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return image


def rotate90cc(img) -> np.ndarray:
    image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def rename(filename, extraname, ext=None) -> str:
    """
    rename : define a new filename.
    The function insert the string extraname in the filename and its extension.
    also it could change the extension with ext parameter. Don't use dot ie: ext="jpg", ext="tiff", etc...
    """
    if extraname == None:
        raise ValueError("extraname could not eqal to None")
    import os

    decompose = os.path.splitext(filename)
    if debug:
        print(f"decompose {decompose}")
    extension = decompose[1]
    if ext:
        extension = "." + ext
    new_filename = decompose[0] + "_" + extraname + extension
    return new_filename


def getPath(filename, extraname=None, ext=None, path=None) -> str:
    # todo test extension in file, add one if missing
    if extraname:
        if filename != None:
            new_filename = rename(filename, extraname, ext)
        else:
            # new_filename = extraname
            raise ValueError("filename is None, please set a value")
        if debug:
            print(f"new name: {new_filename}")
    else:
        new_filename = filename
    if path:
        if debug:
            print(f"path: {path}")
        new_path = Path(path)
        # if path[-1] != "/": new_path + "/"
        # new_path += new_filename
        new_path = Path(new_path, new_filename)
        new_filename = new_path.absolute().as_posix()
    return new_filename


def saveimage(
    image: np.ndarray, filename, extraname=None, ext="jpg", path=None, dpi=None
) -> str:
    if debug:
        print(f"image shape: {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        return None

    new_filename = getPath(filename, extraname=extraname, ext=ext, path=path)
    # if extraname:
    #     new_filename = rename(filename,extraname,ext)
    # else:
    #     new_filename = filename
    # if path: new_filename = path+"/"+new_filename
    if debug:
        print(f"Saving {new_filename}")

    if dpi:
        from PIL import Image

        pil_image = Image.fromarray(image)
        if debug:
            print(f"** Save file ** (pil) {new_filename}")
        # TODO: Fails silently?
        pil_image.save(new_filename, dpi=dpi)
    else:
        path_as_str = Path(new_filename).absolute().as_posix()
        if debug:
            print(f"** Save file ** (cv2) {path_as_str}")
        # TODO: Fails silently?
        cv2.imwrite(path_as_str, image)
    return new_filename


# def saveimage2(image, filename, extraname=None, ext=None, path=None) -> str:
#     import cv2
#     new_filename = getPath(filename, extraname=extraname, ext=ext, path=path)
#     # if extraname:
#     #     new_filename = rename(filename,extraname,ext)
#     # else:
#     #     new_filename = filename
#     # if path: new_filename = path+"/"+new_filename
#     print(f"Saving {new_filename}")
#     cv2.imwrite(new_filename, image)
#     return new_filename


def loadimage(
    filename, extraname=None, ext=None, path=None, type=cv2.COLOR_BGR2GRAY
) -> np.ndarray:
    new_filename = getPath(filename, extraname=extraname, ext=ext, path=path)
    # if extraname:
    #     new_filename = rename(filename,extraname,ext)
    # else:
    #     new_filename = filename
    # if path: new_filename = path+"/"+new_filename
    print(f"Loading {new_filename}")
    if new_filename.name.endswith(".gif"):
        # patent issue? opencv cannot read GIF
        pil_image = PIL.Image.open(new_filename)
        image = np.array(pil_image)
    else:
        image = cv2.imread(new_filename, type)
    # assert image is not None, f"file {new_filename} could not be read, check with os.path.exists()"
    # assert image is not None
    if image is None:  # TODO: wrong diag if format issue or any
        raise Exception(f"file: {filename} doesn't exist\nat path {new_filename}")
    return image


def load_zipped_image(file_path: Path) -> np.ndarray:
    assert file_path.name.lower().endswith(".zip")
    print(f"Loading {file_path}")
    with ZipFile(file_path, "r") as img_zip:
        inside = img_zip.filelist
        assert len(inside) == 1
        the_file = inside[0]
        file_content = np.frombuffer(img_zip.read(the_file), np.uint8)
        return cv2.imdecode(file_content, flags=cv2.IMREAD_UNCHANGED)


# Is not needed in python 3.13, https://github.com/python/cpython/issues/111835
class SeekableMmap(mmap.mmap):
    def seekable(self):
        return False


def load_zipped_image_using_mmap(file_path: Path) -> np.ndarray:
    assert file_path.name.lower().endswith(".zip")
    print(f"Loading {file_path}")
    with open(file_path, mode="rb") as file_obj:
        with SeekableMmap(
            file_obj.fileno(), length=0, access=mmap.ACCESS_READ
        ) as mmap_obj:
            with ZipFile(mmap_obj, "r") as img_zip:
                inside = img_zip.filelist
                assert len(inside) == 1
                the_file = inside[0]
                file_content = np.frombuffer(img_zip.read(the_file), np.uint8)
                return cv2.imdecode(file_content, flags=cv2.IMREAD_UNCHANGED)


def properties(image, title=None, showMatrix=False) -> None:
    """
    print image properties
    change showMatrix to True if your are mad
    """
    if title:
        print(title)
    print("Type:", type(image))
    print("Shape of Image:", image.shape)
    print("Total Number of pixels:", image.size)
    print("Image data type:", image.dtype)
    print("Dimension:", image.ndim)
    if showMatrix:
        print("Pixel Values:\n", image)


def normalize(image: np.ndarray) -> np.ndarray:
    image8bit = convert_image_to_8bit(image)
    image_rotated = rotate90c(image8bit)
    return image_rotated


def normalize_filename(filename, path=None) -> str:
    image = loadimage(filename)
    normed = normalize(image)
    saved = saveimage(normed, filename, "normed", path=path)
    return saved


def normalize_back(image, size) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    normed = normalize(resized)
    return normed


def normalize_back_filename(filename, size, path=None) -> str:
    image = loadimage(filename)
    normed = normalize_back(image, size)
    new_filename = saveimage(normed, filename, "normed", path=path)
    return new_filename


@timeit
def canny(image, low_threshold, high_threshold, aperture_size) -> np.ndarray:
    """
    Aperture size should be odd between 3 and 7 in function 'Canny'
    """

    edges = cv2.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size)
    return edges


@timeit
def canny2(image, low_threshold, high_threshold, aperture_size) -> np.ndarray:
    from skimage.feature import canny

    edges = canny(
        image, sigma=aperture_size, low_threshold=low_threshold, high_threshold=80
    )
    return edges


def wait(purge=True) -> None:
    cv2.waitKey(0)
    if purge:
        cv2.destroyAllWindows()


def rolling_ball_black_background(
    image, filename=None, path=None
):  # -> Tuple(np.ndarray, np.ndarray):
    from skimage import restoration, util

    image_inverted = util.invert(image)
    saveimage(image_inverted, filename, "invert", path=path)

    background_inverted = restoration.rolling_ball(image_inverted, radius=45)
    saveimage(background_inverted, filename, "background_inverted", path=path)
    filtered_image_inverted = image_inverted - background_inverted
    saveimage(filtered_image_inverted, filename, "filtered_image_inverted", path=path)
    filtered_image = util.invert(filtered_image_inverted)
    saveimage(filtered_image, filename, "filtered_image", path=path)
    background = util.invert(background_inverted)
    saveimage(background, filename, "background_rollingball_final", path=path)
    return filtered_image, background


def label(image, mask):
    from skimage.morphology import label
    from skimage.measure import regionprops
    import matplotlib.patches as mpatches
    from matplotlib import pyplot as plt

    # label_image = label(cv2_edges)
    label_image = label(mask)
    fig, ax0 = plt.subplots(ncols=1, nrows=1)
    # ax0 = axes[0] #.flat
    ax0.imshow(image, cmap=plt.cm.gray)
    ax0.set_title("Labeled items", fontsize=24)
    ax0.axis("off")
    # props = regionprops(label_image,connectivity=2)
    props = regionprops(label_image)
    print(f"regionprops:{props.count}")
    # for region in regionprops(label_image,connectivity=2):
    for region in props:
        # Draw rectangle around segmented coins.
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax0.add_patch(rect)
    # ax[0].show()
    plt.tight_layout()
    plt.show()


def crop(
    image: np.ndarray, top: float, left: float, bottom: float, right: float
) -> np.ndarray:
    """Crop a numpy array image.

    Args:
        image: Input image as numpy array
        top: Top coordinate (row start)
        left: Left coordinate (column start)
        bottom: Bottom coordinate (row end)
        right: Right coordinate (column end)

    Returns:
        Cropped image as numpy array

    Raises:
        IndexError: If crop coordinates are outside image boundaries
        ValueError: If coordinates result in invalid crop dimensions
    """
    height, width = image.shape[:2]

    # Convert coordinates to int
    top_i = int(top)
    left_i = int(left)
    bottom_i = int(bottom)
    right_i = int(right)

    # Validate coordinates
    if top_i < 0 or left_i < 0 or bottom_i > height or right_i > width:
        raise IndexError("Crop coordinates out of image bounds")
    if bottom_i <= top_i or right_i <= left_i:
        raise ValueError(
            "Invalid crop dimensions: ensure bottom > top and right > left"
        )

    cropped_image = image[top_i:bottom_i, left_i:right_i]
    return cropped_image


def crop_right(image: np.ndarray, right: Union[float, int]) -> np.ndarray:
    cropped_image = image[0:, 0 : int(right)]
    return cropped_image


def clear_outside(
    image: np.ndarray, right: int, top: int, width: int, height: int
) -> None:
    """In-place modify the input image so that any pixel outside the input rectangle in set to 255 AKA white."""
    image[0:top, :] = 255
    image[top + height :, :] = 255
    image[top : top + height :, 0:right] = 255
    image[top : top + height :, right + width :] = 255


def draw_outside_lines(
    image: np.array,
    sample_dims: Tuple[int, int],
    right_limit,
    left_limit,
    top_limit,
    bottom_limit,
    limitod,
):
    from ZooProcess_lib.ImageJLike import draw_line

    height, width = sample_dims
    if limitod < width:
        width = limitod
    if right_limit != width and right_limit < limitod:
        draw_line(image, (right_limit, 0), (right_limit, height / 4), 0, 1)
        draw_line(image, (right_limit, height / 4 + 4), (right_limit, height), 0, 1)
    if left_limit != 0:
        draw_line(image, (left_limit, 0), (left_limit, height / 4), 0, 1)
        draw_line(image, (left_limit, height / 4 + 4), (left_limit, height), 0, 1)
    if bottom_limit != height:
        draw_line(image, (0, bottom_limit), (width / 4, bottom_limit), 0, 1)
        draw_line(image, (width / 4 + 4, bottom_limit), (width, bottom_limit), 0, 1)
    if top_limit != 0:
        draw_line(image, (0, top_limit), (width / 4, top_limit), 0, 1)
        draw_line(image, (width / 4 + 4, top_limit), (width, top_limit), 0, 1)


def cropped_if_larger(image: np.ndarray, right: int, bottom: int) -> np.ndarray:
    img_bottom, img_right = image.shape
    if img_bottom > bottom or img_right > right:
        return image[0 : min(bottom, img_bottom), 0 : min(right, img_right)]
    return image


def cropnp(
    image: np.ndarray,
    top: Optional[int] = None,
    left: Optional[int] = None,
    bottom: Optional[int] = None,
    right: Optional[int] = None,
) -> np.ndarray:
    """Return a sub-image at given dimensions/coordinates
    !!! The sub-image is not a copy, it shares the memory with original image (fast but risky)
    """
    assert (
        top is not None or left is not None or bottom is not None or right is not None
    )
    # cropped_image = img[80:280, 150:330]
    # start_row:end_row, start_col:end_col
    # print(top,left,bottom,right)
    # if type(top) != int : print("top not int")
    # if type(left) != int : print("left not int")
    # if type(bottom) != int : print("bottom not int")
    # if type(right) != int : print("right not int")
    # print(f"L:R {left}:{right}, T:B {top}:{bottom}")
    # try:
    # cropped_image = image[int(left):int(right), int(top):int(bottom)]
    # cropped_image = image[int(right):int(left), int(bottom):int(top)]
    # symetrie à 45°
    cropped_image = image[top:bottom, left:right]
    # except OSError as error:
    #   raise error
    return cropped_image


def crophw(
    image: np.ndarray, f_left: float, f_top: float, f_width: float, f_height: float
) -> np.ndarray:
    top, left, height, width = int(f_top), int(f_left), int(f_height), int(f_width)
    if top < 0 or left < 0:
        raise ValueError("Top and left must be positive")
    if height < 0 or width < 0:
        raise ValueError("Height and width must be positive")
    if height == 0:
        raise ValueError("Cannot crop with height of 0")
    if width == 0:
        raise ValueError("Cannot crop with width of 0")

    bottom = top + height
    right = left + width

    if top > bottom or left > right:
        raise ValueError("Cropped region is not within image boundaries")

    return crop(image, top=top, left=left, bottom=bottom, right=right)

    # cropped_image = crop(image,top=top,left=left,bottom=top+height,right=left+width)
    # # if cropped_image is None:
    #     # raise ValueError("Cropped image is None")
    # # cropped_image = crop(image, top=top, left=left, right=top+height, bottom=left+width)
    # print(f"shape: {cropped_image.shape}")
    # return cropped_image


def crop_scan(image) -> np.ndarray:
    start_point = (800, 100)
    end_point = (image.shape[1] - 1400, image.shape[0] - 300)
    top = start_point[0]
    left = start_point[1]
    bottom = end_point[0]
    right = end_point[1]

    image = crop(image, top=top, left=left, bottom=bottom, right=right)
    return image


def draw_contours(
    image, contours, color=(0, 255, 0), thickness=3, index=-1
) -> np.ndarray:
    """
    index = -1 : mean draw all contours
    index = x : mean draw the i th contour
    """
    image_3channels = cv2.merge([image, image, image])
    cv2.drawContours(image_3channels, contours, index, color, thickness)
    # saveimage(image_3channels, file_scan, "draw_contours2dilate", path=zooscan_test_folder)
    return image_3channels


def append_contours(
    image_3channels: np.ndarray, contours, color=(0, 255, 0), thickness=3, index=-1
) -> np.ndarray:
    """
    append a contour on a 3 channels image
    index = -1 : mean draw all contours
    index = x : mean draw the i th contour
    """
    cv2.drawContours(image_3channels, contours, index, color, thickness)
    return image_3channels


def draw_box(image_3channels, x, y, w, h, color=(0, 0, 255), thickness=3) -> np.ndarray:
    image = cv2.rectangle(
        image_3channels, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness
    )
    return image


class FontPref:
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10,500)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 1
    lineType = 2


def write_text(x, y, w, h, text, image):
    point = (x + w // 2, y + h // 2)
    print(f"writing {text} at {point}")
    font = FontPref()
    cv2.putText(
        img=image,
        text=text,
        org=point,
        bottomLeftOrigin=True,  # ((x+w)/2,(y+h)/2),
        # font=font,
        fontScale=font.fontScale,
        # fontColor=fontColor,
        color=font.fontColor,
        thickness=font.thickness,
        lineType=font.lineType,
        fontFace=font.fontType,
    )


def draw_boxes(image, contours, add_number=False, font=None) -> np.ndarray:
    image_3channels = cv2.merge([image, image, image])

    # if add_number and not font:
    # font                   = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10,500)
    # fontScale              = 1
    # fontColor              = (255,255,255)
    # thickness              = 1
    # lineType               = 2

    for i in range(0, len(contours)):
        # mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
        x, y, w, h = cv2.boundingRect(contours[i])
        image_3channels = draw_box(image_3channels, x, y, w, h)

        if add_number:
            write_text(x, y, w, h, str(i), image_3channels)
            # point = (x+w//2, y+h//2)
            # print(f"add_number {i} at {point}")
            # cv2.putText(
            #     img=image_3channels,
            #     text=str(i),
            #     org= point,
            #     bottomLeftOrigin= True, #((x+w)/2,(y+h)/2),
            #     # font=font,
            #     fontScale=fontScale,
            #     # fontColor=fontColor,
            #     color = fontColor,
            #     thickness=thickness,
            #     lineType=lineType,
            #     fontFace=font,
            # )

    return image_3channels


def filterContours(contours, h, w):
    # contours_filtered = contours.filter(lambda contour: h < 100 or w < 100)
    # contours_filtered = contours.filter(cv2.boundingRect (contours[i])[2] > h and cv2.boundingRect(contours[i])[3] > w)
    def size(contour):
        w = 50
        h = 50
        x, y, width, height = cv2.boundingRect(contour)
        if width < w or height < h:
            return False
        else:
            return True

    contours_filtered = list(filter(lambda c: size(c, h, w), contours))
    return contours_filtered


def draw_boxes_filtered(
    image, contours, filter: Callable[[int], int], add_number=False, font=None
) -> np.ndarray:
    image_3channels = cv2.merge([image, image, image])

    # if add_number and not font:
    # font                   = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10,500)
    # fontScale              = 1
    # fontColor              = (255,255,255)
    # thickness              = 1
    # lineType               = 2

    for i in range(0, len(contours)):
        # mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
        # x,y,w,h = cv2.boundingRect(contours[i])
        # if h < 100 or w < 100: continue
        # if filter(h,w):
        if filter(contours[i]):
            x, y, w, h = cv2.boundingRect(contours[i])

            image_3channels = draw_box(image_3channels, x, y, w, h)

            if add_number:
                write_text(x, y, w, h, str(i), image_3channels)
                # cv2.putText(image_3channels,text=str(i),
                # bottomLeftOrigin=((x+w)/2,(y+h)/2),
                # font=font,
                # fontScale=fontScale,
                # fontColor=fontColor,
                # thickness=thickness,
                # lineType=lineType)

    return image_3channels


def translate_contour(contour, x, y):
    for p in contour:
        print(p)


def generate_vignettes(image, contours, filter, path):
    image_3channels = cv2.merge([image, image, image])
    filelist = []
    for i in range(0, len(contours)):
        # mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
        # x,y,w,h = cv2.boundingRect(contours[i])
        # if h < 100 or w < 100: continue
        # if filter(h,w) :
        if filter(contours[i]):
            # image_3channels = draw_box(image_3channels,x,y,w,h)
            x, y, w, h = cv2.boundingRect(contours[i])
            print(f"x: {x} - y: {y} - h: {h} - w: {w}")
            if h != 0 and w != 0:
                cropped_image = crophw(image_3channels, y, x, h, w)
                if True:
                    filename = f"vignette_{i}.tif"
                    print(f"vignette filename: {filename}")
                    filepath = saveimage(cropped_image, filename, path=path)
                    filelist.append(filepath)

    return filelist


def generate_vignettes3(image: np.ndarray, contours, filter, path):
    # image_3channels = cv2.merge([image, image, image])
    filelist = []
    for i in range(0, len(contours)):
        # mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
        # x,y,w,h = cv2.boundingRect(contours[i])
        # if h < 100 or w < 100: continue
        # if filter(h,w) :
        if filter(contours[i]):
            # image_3channels = draw_box(image_3channels,x,y,w,h)
            x, y, w, h = cv2.boundingRect(contours[i])
            cropped_image = crophw(image, y, x, h, w)
            filename = f"vignette_{i}.tif"
            filepath = saveimage(cropped_image, filename, path=path)
            filelist.append(filepath)

    return filelist


def inside_a_bit(largeur, hauteur):
    BX = largeur * 0.03
    BY = hauteur * 0.05
    W = largeur * 0.94
    H = hauteur * 0.93
    from math import ceil, floor

    return [floor(BX), floor(BY), ceil(W), ceil(H)]


# min,max,odrange = read_lut()


def minAndMax(median, lut: Lut = None) -> Tuple[int, int]:
    """
    Return: (min, smax)
    """
    if not lut:
        lut = Lut()
    MINREC = lut.min
    MAXREC = lut.max
    if lut.adjust == "yes":
        MAXREC = math.floor(median * lut.ratio)
        MAXREC = min(65536, MAXREC)
        fact = pow(10, lut.odrange)
        MINREC = max(0, median / (fact * lut.ratio))
        MINREC = math.floor(MINREC)
    return MINREC, MAXREC


# setMinAndMax(MINREC, MAXREC); // setMinAndMax : Sets the minimum and maximum displayed pixel values (display range)


def picheral_median_background(image: np.ndarray):
    """
    Return: (median, mean)
    """
    # from 16to8bit import resize
    height = image.shape[0]
    width = image.shape[1]
    print(f"size {width}x{height}")

    BX, BY, W, H = inside_a_bit(width, height)

    median = np.median(image, axis=None)
    return median


def picheral_median(image: np.ndarray):
    """
    Return: (median, mean)
    """
    import math

    # from 16to8bit import resize
    height = image.shape[0]
    width = image.shape[1]
    print(f"size {width}x{height}")

    # BX = width*0.03
    # BY = height*0.05
    # W = width *0.94
    # H = height*0.93

    BX, BY, W, H = inside_a_bit(width, height)
    print(f"BX,BY,W,H = {BX},{BY},{W},{H}")
    step = math.floor(H / 20)
    print(f"step: {step}")
    By = BY
    k = 0
    mediansum = 0.0
    meansum = 0.0
    while By < H + step:
        BX = int(BX)
        By = int(By)
        W = int(W)
        step = int(step)
        # print(f"crop ({BX},{By})x({W},{step})")
        img = crophw(image, By, BX, step, W)

        # median,mean = mesure(img)

        median = np.median(img, axis=None)
        mediansum = mediansum + median

        # mean = getResult("Mean",k);
        mean = np.mean(img, axis=None)
        meansum = meansum + mean
        print(f"median: {median}, mean: {mean}")
        k = k + 1
        By = By + step

    median = mediansum / k
    mean = meansum / k

    print(f"median: {median}, mean: {mean}")

    return median, mean


def separate_apply_mask(filename_image, filename_mask) -> np.ndarray:
    img = loadimage(filename_image)
    mask = loadimage(filename_mask)  # ,type=cv2.COLOR_)

    mask = cv2.bitwise_not(mask)
    fg = cv2.bitwise_or(img, img, mask=mask)

    background = np.full(img.shape, 255, dtype=np.uint8)
    mask = cv2.bitwise_not(mask)
    bk = cv2.bitwise_or(background, background, mask=mask)

    final = cv2.bitwise_or(fg, bk)

    return final


# def minmax():
#     from math import floor
#     # if (adjust == "yes" ) {
# 	# // floor(n) : Returns the largest value that is not greater than n and is equal to an integer.
#     MAXREC = 	floor(median * ratio)
#     # //	MAXREC = 	floor(mean * ratio);
#     MAXREC =	min(65536,MAXREC)
#     fact = 		pow(10,odrange)
#     MINREC = 	max(0,median/(fact*ratio))
#     MINREC = 	floor(MINREC)
#     # } // if
#     # setMinAndMax(MINREC, MAXREC)# setMinAndMax : Sets the minimum and maximum displayed pixel values (display range)


def dpi():
    """
    To calculate DPI,
    divide the horizontal pixels by the total width in inches,
    add this result to the result of the vertical pixels divided by the height,
    then finally, divide this result by 2
    """
    pass


class ImageInfo(Dict[str, Any]):
    @property
    def width(self) -> int:
        return self["Image Size"][0]

    @property
    def height(self) -> int:
        return self["Image Size"][1]

    @property
    def resolution(self) -> int:
        dpi_vert, dpi_horiz = self["dpi"]
        assert dpi_vert == dpi_horiz
        return int(dpi_vert)


def image_info(image_path: Path) -> ImageInfo:
    image = Image.open(image_path)
    info_dict = {
        "Filename": image.filename,
        "Image Size": image.size,
        "Image Height": image.height,
        "Image Width": image.width,
        "Image Format": image.format,
        "Image Mode": image.mode,
        "Image is Animated": getattr(image, "is_animated", False),
        "Frames in Image": getattr(image, "n_frames", 1),
    }

    for info in image.info:
        info_dict[info] = image.info[info]

    exif_data = image.getexif()
    # iterating over all EXIF data fields
    for tag_id in exif_data:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        if tag == "StripOffsets" or tag == "StripByteCounts":
            continue
        data = exif_data.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        info_dict[tag] = data

    if "ExifOffset" in info_dict:
        for tag_id, tag_value in exif_data.get_ifd(Base.ExifOffset).items():
            tag = TAGS.get(tag_id, tag_id)
            info_dict[tag] = tag_value

    return ImageInfo(info_dict)


def get_date_time_digitized(img_info: Dict[str, str]) -> Optional[datetime]:
    val = img_info.get("DateTimeDigitized")  # e.g. '2024:01:16 10:46:21'
    if val is None:
        return None
    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")


def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    """
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    """
    if not (0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError('"lower_bound" must be in the range [0, 65535]')
    if not (0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError('"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError('"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate(
        [
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255,
        ]
    )
    return lut[img].astype(np.uint8)


def generate_cropped_image(
    image: np.ndarray, x, y, w, h, name=None, extraname=None, path=None
):
    crop = crophw(image, y, x, h, w)
    if name:
        s = f"_{x}_{y}_{w}_{h}"
        # print(s)
        _extraname = s
        if extraname:
            _extraname = extraname + _extraname
        saveimage(crop, name, _extraname, path=path)

    return crop


def save_cv2_image_with_dpi(opencv_image, output_path, dpi=(300, 300)) -> str:
    # from PIL import Image
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    color_coverted.save(
        output_path, optimize=True, quality=50, jfif_unit=1, dpi=dpi, jfif_density=dpi
    )
    return output_path


def set_image_dpi_resize(image):
    """
    Rescaling image to 300dpi while resizing
    :param image: An image
    :return: A rescaled image
    """
    import tempfile
    from PIL import Image

    length_x, width_y = image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image_resize = image.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="1.png")
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename


def set_image_dpi(image):
    """
    Rescaling image to 300dpi without resizing
    :param image: An image
    :return: A rescaled image
    """
    import tempfile

    image_resize = image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_filename = temp_file.name
    image_resize.save(temp_filename, dpi=(300, 300))
    return temp_filename


def close_image(image):
    """
    Closes opened image
    :param image: An image
    :return: None
    """
    image.close()


# def removing_file(path):
#     """
#     Removing file by path
#     :param path: The path of a file
#     :return: None
#     """
#     # Try/except library import
#     try:
#         import os  # Miscellaneous OS library
#     except ImportError as ImportException:
#         raise ImportError("Cannot import needed libraries. Please contact administrator and give the code FN0001")
#     os.remove(path)


def plot_histogram(image, title, mask=None):
    import matplotlib.pyplot as plt
    import cv2

    # split the image into its respective channels, then initialize
    # the tuple of channel names along with our figure for plotting
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    # loop over the image channels
    for chan, color in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()


def rotateAndFlip(image):
    """
    Transform image to look like the operator POV
    """
    image_rotated = rotate90c(image)
    image_flipped = cv2.flip(image_rotated, 0)
    return image_flipped


def convertShortToByte(image: np.ndarray, min, max) -> np.ndarray:
    """
    JImage code
    ByteProcessor convertShortToByte() {
        int size = width*height;
        short[] pixels16 = (short[])ip.getPixels();
        byte[] pixels8 = new byte[size];
        if (doScaling) {
            int value, min=(int)ip.getMin(), max=(int)ip.getMax();
            double scale = 256.0/(max-min+1);
            for (int i=0; i<size; i++) {
                value = (pixels16[i]&0xffff)-min;
                if (value<0) value = 0;
                value = (int)(value*scale+0.5);
                if (value>255) value = 255;
                pixels8[i] = (byte)value;
            }

        too slow, use converthisto16to8 follow by convertImage16to8bit to increase speed
    """
    # pixels8 = np.zeros(image.shape[:2], np.uint8)
    pixels8 = np.zeros(image.shape, np.uint8)
    scale = 256.0 / (max - min + 1)
    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[0]):
            # print(f"{i*100/image.shape[1]}/100")
            # pixel = (image[j][i][0] & 0xffff) - min
            pixel = (image[j][i] & 0xFFFF) - min
            if pixel < 0:
                pixel = 0
            else:
                pixel = int(pixel * scale + 0.5)
                if pixel > 255:
                    pixel = 255
            # pixels8[j][i][0] = pixel
            # pixels8[j][i][1] = pixel
            # pixels8[j][i][2] = pixel
            pixels8[j][i] = pixel

    return pixels8


def converthisto16to8(min, max):  #:_ShapeLike):
    # hist = [] # np.zeros(65536,np.int8)
    # for i in range(0,65536):
    #     level = i & 0xffff
    #     # hist[i]=level
    #     hist.append(level)
    print(f"converthisto16to8({min},{max})")

    scale = 256.0 / (max - min + 1)
    # np.arange()
    hist = np.arange(0, 65536)
    for i in hist:
        hist[i] = hist[i] - min
        if hist[i] < 0:
            hist[i] = 0
        else:
            # hist[i]=int(hist[i]*scale + 0.5)
            hist[i] = int(hist[i] * scale)

            # pourquoi j'avais enlevé le filtrage sur max ?????????
            if hist[i] > max:
                hist[i] = 254
            # if hist[i]>255: hist[i]=255

    return hist


@timeit
def convertImage16to8bit(image: np.ndarray, histolut, log=False) -> np.ndarray:
    print("image shape:", image.shape)
    pixels8 = np.zeros(image.shape, np.uint8)
    for i in range(0, image.shape[1]):
        if log:
            if (i % 1000) == 0:
                log = True
                print("")
            else:
                log = False
        for j in range(0, image.shape[0]):
            if log:
                if (j % 1000) == 0:
                    print(".", end="")
            pixel = histolut[image[j][i]]
            pixels8[j][i] = pixel

    return pixels8


@timeit
def convertImage16to8bitWithNumpy(image: np.ndarray, histolut) -> np.ndarray:
    """
    very slow
    """
    print("image shape:", image.shape)
    pixels8 = np.zeros(image.shape, np.uint8)
    for i in range(0, 65536):  # len(histolut)
        if (i % 1000) == 0:
            print(".", end="")
        pixels8[image == i] = histolut[i]

    return pixels8


def find_res(filename):
    """
    Find the resolution of a jpeg image
    """
    with open(filename, "rb") as img_file:  # open image in binary mode
        # height of image is at 164th position
        img_file.seek(163)
        # read the two bytes
        a = img_file.read(2)
        # calculate height
        height = (a[0] << 8) + a[1]
        # read next two bytes which stores the width
        a = img_file.read(2)
        # calculate width
        width = (a[0] << 8) + a[1]
    print("IMAGE RESOLUTION IS : ", width, "X", height)
