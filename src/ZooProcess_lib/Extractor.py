from pathlib import Path
from typing import List

import cv2
import numpy as np

from .ImageJLike import parseInt
from .ROI import ROI
from .bitmaps import one_mm_img
from .img_tools import (
    cropnp,
    save_lossless_small_image,
)


class Extractor(object):
    """ Thumbnail extractor, from an image and ROI."""
    X1 = 10
    FOOTER = 31

    def __init__(
        self,
        longline_mm: float,
        threshold: int,
    ):
        self.longline_mm = longline_mm
        self.threshold = threshold

    def extract_all_from_image(
        self,
        image: np.ndarray,
        resolution: int,
        rois: List[ROI],
        destination_dir: Path,
        naming_prefix: str,
    ):
        longline = self.longline_mm * resolution / 25.4
        assert destination_dir.exists()
        for index, a_roi in enumerate(rois, 1):
            img = self.extract_one(image, a_roi)
            resized_img = self.add_border_and_legend(img, longline)
            img_filename = naming_prefix + "_" + str(index) + ".png"
            img_path = destination_dir / img_filename
            save_lossless_small_image(
                resized_img,
                resolution,
                img_path,
            )

    def add_border_and_legend(self, image: np.ndarray, longline: float):
        height, width = image.shape
        # Resized image is plain one...
        # ...+ 20% border and footer in height
        final_height = round(height * 1.4 + self.FOOTER)
        # ...and +20% border with enough space for line in width
        final_width = round(max(width * 1.4, longline + 2 * self.X1))
        # draw a new frame big enough and paste
        resized = np.full((final_height, final_width), 255, dtype=np.uint8)
        y_offset = (final_height - self.FOOTER - height) // 2
        x_offset = (final_width - width) // 2
        resized[y_offset : y_offset + height, x_offset : x_offset + width] = image
        # Paint the scale
        y1 = final_height - 5
        x2 = parseInt(self.X1 + longline)
        cv2.line(
            img=resized,
            pt1=(self.X1 - 1, y1 - 1),
            pt2=(x2, y1 - 1),
            color=(85,),
            thickness=1,
        )
        cv2.line(
            img=resized,
            pt1=(self.X1 - 1, y1),
            pt2=(x2, y1),
            color=(85,),
            thickness=1,
        )
        assert self.longline_mm == 1.0
        one_mm_height, one_mm_width = one_mm_img.shape[:2]
        resized[
            final_height - 8 - one_mm_height : final_height - 8 - 1,
            self.X1 : self.X1 + one_mm_width - 1,
        ] = one_mm_img[1:, 1:]
        # text = str(int(self.longline_mm)) + "mm"
        # cv2.putText(
        #     img=resized,  # image on which to draw text
        #     text=text,
        #     org=(self.X1, final_height - 8),  # bottom left corner of text
        #     fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,  # font to use
        #     fontScale=1,  # font scale
        #     color=(0,),  # color
        #     thickness=1,  # line thickness
        # )
        # print(f"height: {height}, width: {width} => {final_height}x{final_width} px xoffset: {x_offset} yoffset: {y_offset}")
        return resized

    def extract_one(self, image: np.ndarray, a_roi: ROI) -> np.ndarray:
        height, width = a_roi.mask.shape[:2]
        crop = cropnp(
            image,
            top=a_roi.y,
            left=a_roi.x,
            bottom=a_roi.y + height,
            right=a_roi.x + width,
        )
        # Whiten background -> push to 255 as min is black
        thumbnail = np.bitwise_or(crop, 255 - a_roi.mask * 255)
        # Whiten holes
        thumbnail[thumbnail > self.threshold] = 255
        return thumbnail
