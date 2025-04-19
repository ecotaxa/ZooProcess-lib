import cv2
import numpy as np

from ZooProcess_lib.img_tools import cropnp


class Vignette(object):
    def __init__(self, image: np.ndarray, x: int, y: int, mask: np.ndarray):
        self.image = image
        self.x, self.y = x, y  # Position in image
        self.mask = mask
        self.height, self.width = mask.shape

    def enlarged_by_20_percent(self) -> np.ndarray:
        enlarge_x, enlarge_y = self.width // 10, self.height // 10
        resized = cv2.copyMakeBorder(
            self.image,
            enlarge_y,
            enlarge_y,
            enlarge_x,
            enlarge_x,
            cv2.BORDER_CONSTANT,
            value=(255,),
        )
        return resized

    def symmetrical_vignette_added(self) -> np.ndarray:
        crop = self.particle_on_white_background()
        v_sym_crop = cv2.flip(crop, 1).astype(np.uint16)
        h_sym_crop = cv2.flip(crop, 0).astype(np.uint16)
        return ((v_sym_crop + h_sym_crop) / 2).astype(np.uint8)

    def particle_on_white_background(self):
        crop = cropnp(
            self.image,
            top=self.y,
            left=self.x,
            bottom=self.y + self.height,
            right=self.x + self.width,
        )
        crop |= 255 - self.mask * 255
        return crop
