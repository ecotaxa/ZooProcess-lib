import cv2

from ZooProcess_lib.img_tools import crop
from ZooProcess_lib.img_tools import minAndMax, converthisto16to8


class pieceImage:

    # def __init__(self, image, pos, top, left, width, height):
    def __init__(self, image, top, left, width, height, histolut):
        # self.image = image
        # self.pos = pos # an id to known the part position
        self.histolut = histolut
        self.top = top
        self.left = left
        self.bottom = top + height
        self.right = left + width
        print(f"pieceImage {self.top}:{self.left} - {self.bottom}:{self.right}")
        self.image = self.crop(image)

    def crop(self, image):
        print(f"crop {self.top}:{self.left} - {self.bottom}:{self.right}")
        return crop(image, self.top, self.left, self.bottom, self.right)

    def get_image(self):
        return self.image


from ZooProcess_lib.tools import timeit
import numpy as np


@timeit
def convertImage16to8bit(image: np.ndarray, histolut, log=False) -> np.ndarray:
    print("image shape:", image.shape)
    pixels8 = np.zeros(image.shape, np.uint8)
    for i in range(0, image.shape[1]):
        if log:
            if (i % 1000) == 0:
                log = True; print("")
            else:
                log = False
        for j in range(0, image.shape[0]):
            if log:
                if (j % 1000) == 0:
                    print(".", end="")
            pixel = histolut[image[j][i]]
            pixels8[j][i] = pixel

    return pixels8


def convertPieceOfImage16to8bit(piece: pieceImage) -> np.ndarray:
    return convertImage16to8bit(piece.image, piece.histolut)


NB_THREADS = 4
thread_level = 243


def splitimage(image, histolut) -> list:
    shape = image.shape
    print("shape: ", shape)

    cropped = []

    mult = 4

    cropped.append(pieceImage(image, 0, 0, image.shape[0] // mult, image.shape[1] // 2, histolut))
    cropped.append(pieceImage(image, 0, image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2, histolut))
    cropped.append(pieceImage(image, 0, 2 * image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2, histolut))
    cropped.append(pieceImage(image, 0, 3 * image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2, histolut))
    cropped.append(pieceImage(image, image.shape[1] // 2, 0, image.shape[0] // mult, image.shape[1] // 2, histolut))
    cropped.append(
        pieceImage(image, image.shape[1] // 2, image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2,
                   histolut))
    cropped.append(
        pieceImage(image, image.shape[1] // 2, 2 * image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2,
                   histolut))
    cropped.append(
        pieceImage(image, image.shape[1] // 2, 3 * image.shape[0] // 4, image.shape[0] // mult, image.shape[1] // 2,
                   histolut))

    return cropped


import concurrent.futures
from ZooProcess_lib.img_tools import saveimage
from ZooProcess_lib.ZooscanProject import ZooscanProject


def convertion(image: np.ndarray, sample: str, TP: ZooscanProject):
    from img_tools import picheral_median

    median, mean = picheral_median(image)
    min, max = minAndMax(median)
    histolut = converthisto16to8(min, max)
    output_path = TP.testfolder

    splitted = splitimage(image, histolut)

    print("splitted: ", splitted)

    new_image = np.zeros(image.shape, np.uint8)
    new = saveimage(new_image, sample, extraname="empty", ext="tiff", path=output_path)

    with concurrent.futures.ThreadPoolExecutor(NB_THREADS, "thread_split") as executor:
        for split, newimage in zip(splitted, executor.map(convertPieceOfImage16to8bit, splitted)):
            name = f"{split.top}-{split.left}-{split.bottom}-{split.right}"
            print(f"name: {name}")
            new = saveimage(newimage, "result.jpg", extraname=sample + "_" + name, ext="jpg", path=output_path)
            # new = saveimage(newimage,"result.jpg", extraname=name, ext="tiff", path=output_path)
            print(f"new picture => {new}")
            new_image[split.left:split.right, split.top:split.bottom] = newimage

    # new = saveimage(new_image,sample, extraname="treated", ext="jpg", path=output_path)
    new = saveimage(new_image, sample, extraname="treated", ext="tiff", path=output_path)
    print(f"result picture => {new}")
    return new_image


def convertion2(image: np.ndarray, sample: str, TP: ZooscanProject):
    import cv2

    print("convertion2")

    from img_tools import picheral_median

    median, mean = picheral_median(image)
    min, max = minAndMax(median)

    output_path = TP.testfolder

    img_rescaled = 255 * (image - image.min()) / (image.max() - image.min())

    img_col = cv2.applyColorMap(img_rescaled.astype(np.uint8), cv2.COLORMAP_INFERNO)

    new = saveimage(img_col, sample, extraname="img_rescaled", ext="tiff", path=output_path)
    return img_rescaled


def convert(img, target_type_min, target_type_max, target_type):
    """
    sample: imgu8 = convert(img16u, 0, 255, np.uint8)
    """
    imin = img.min()
    imax = img.max()
    print(f"imin: {imin}, imax: {imax}")

    # target_type_max = imax * 255 / 65536
    # target_type_min = imin * 255 / 65553
    # target_type_max = 254

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    # new_img = new_img > 255 
    # new_img = new_img.astype(target_type)

    return new_img


def convert_forced(img, target_type_min, target_type_max, target_type):
    """
    sample: imgu8 = convert(img16u, 0, 255, np.uint8)
    """
    imin = img.min()
    imax = img.max()
    print(f"imin: {imin}, imax: {imax}")

    target_type_max = imax * 255 / 65536
    target_type_min = imin * 255 / 65553

    target_type_max = 254

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    print(f"a: {a}, b: {b}")
    new_img = (a * img + b).astype(target_type)
    # new_img = new_img > 255 

    return new_img


def filters(img) -> tuple:
    imin = img.min()
    imax = img.max()
    # target_type_max = imax * 255 / 65536
    target_type_min = imin * 255 / 65553

    target_type_max = 254

    return (imin, imax, target_type_min, target_type_max)


def convert_mm(img, target_type_min, target_type_max, min, max, target_type):
    """
    sample: imgu8 = convert(img16u, 0, 255, np.uint8)
    """
    imin = min
    imax = max
    print(f"imin: {imin}, imax: {imax}")

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def convert_s(img, target_type_min, target_type_max, target_type):
    """
    sample: imgu8 = convert(img16u, 0, 255, np.uint8)
    """
    imin = img.min()
    imax = img.max()
    print(f"imin: {imin}, imax: {imax}")

    # a = (target_type_max - target_type_min) / (imax - imin)
    a = (255 - 0) / (target_type_max - target_type_min)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def convert2(image_data, display_min, display_max):
    """
    bug
    > np.multiply(255. / (display_max - display_min), image_data, out=datab)
     numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('uint16') with casting rule 'same_kind'
    """
    np.clip(image_data, display_min, display_max, out=image_data)
    image_data -= display_min
    datab = np.empty_like(image_data)
    np.multiply(255. / (display_max - display_min), image_data, out=datab)
    return datab.astype(np.uint8)


def resized_like(image: np.ndarray, model: np.ndarray) -> np.ndarray:
    """ Return image resized to same size as model """
    height = model.shape[0]
    width = model.shape[1]
    interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, dsize=(width, height), interpolation=interpolation)
