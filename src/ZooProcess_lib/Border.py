import os

import numpy as np

from .ImageJLike import parseInt
from .img_tools import crophw, saveimage, mean_with_4_decimals


# TODO: Separate code from tests and debug
class Border:
    def __init__(self, image: np.ndarray) -> None:
        assert image.dtype == np.uint8
        self.image = image

        self.height = image.shape[0]
        self.width = image.shape[1]

        self.resolution = 2400
        self.step = parseInt(self.resolution / 240.0)

        self.greytaux = 0.9  # greytaux if greytaux is not None else 0.9
        self.output_path = None

        self._draw_image = None
        self.crop_output_path = None
        # crop_output_path = os.path.join(self.output_path, "crops")
        # if not os.path.exists(crop_output_path):
        #     os.makedirs(crop_output_path)

    @property
    def draw_image(self):
        # return self.image
        return self._draw_image

    @draw_image.setter
    def draw_image(self, draw_image):
        return
        # if draw_image:
        self.crop_output_path = os.path.join(self.output_path, "crops")
        print(f"Saving border image to {self.crop_output_path}")
        if not os.path.exists(self.crop_output_path):
            os.makedirs(self.crop_output_path)
        self._draw_image = draw_image

    def detect(self) -> tuple[int, int, int, int]:
        left = self.left_limit()
        right = self.right_limit()
        bottom = self.bottom_limit()
        top = self.top_limit()
        if self.draw_image is not None:
            # crop_output_path = os.path.join(self.output_path, "crops")
            # if not os.path.exists(crop_output_path):
            #     os.makedirs(crop_output_path)
            saveimage(
                self.draw_image, self.name, "border", ext="tiff", path=self.output_path
            )
        return top, bottom, left, right

    def left_limit(self) -> int:
        limit = self.width
        # limitgauche = limit

        k = self.width * 0.05
        print(f"k: {k}")
        img = crophw(
            self.image,
            f_top=k,
            f_left=self.height / 2 - self.height * 0.125,
            f_width=self.height * 0.25,
            f_height=10 * self.step,
        )
        # print(f"shape of cropped image: {img.shape}")
        img_mean = mean_with_4_decimals(img)
        # print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_left" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
        # self.draw_image = draw_box(self.draw_image,x=k,y=self.height / 2 - self.height * 0.125,h=self.height * 0.25,w=self.step,color=(255, 0, 0),)

        greyvalg: int = parseInt(img_mean * self.greytaux)
        print(f"greyvalg: {greyvalg}")

        while k >= 0:
            img = crophw(
                self.image,
                f_top=k,
                f_left=self.height / 2 - self.height * 0.125,
                f_width=self.height * 0.25,
                f_height=self.step,
            )
            # print(f"shape of cropped image: {img.shape}")
            # print(
            #     f"dim: {k}, {self.height / 2 - self.height * 0.125}, {self.step}, {self.height * 0.25} mean:{mean} <? {greyvalb}"
            # )
            mean = mean_with_4_decimals(img)
            # saveimage(img, self.name, "crop_left" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
            # self.draw_image = draw_box(self.draw_image,x=k,y=self.height / 2 - self.height * 0.125,h=self.height * 0.25,w=self.step,color=(255, 0, 0),)

            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalg:
                # limit = k
                limit = k + self.step
                # limitgauche = int(limit)
                # k = 0
                # print(f"Limit left: {limitgauche}")
                # print(f"Limit left: {int(limit)}")
                # return limitgauche
                # return int(limit)
                break
            # a += 1
            k -= self.step

        # print(f"Limit left: {limitgauche}")
        print(f"Left Limit: {int(limit)}")
        return parseInt(limit)

    def right_limit(self) -> int:
        limit = self.width

        k = max(self.width * 0.75, self.width - self.resolution / 2)
        # print(f"k: {k}")
        img = crophw(
            self.image,
            f_top=k,
            f_left=self.height / 4,
            f_width=self.height * 0.25,
            f_height=10 * self.step,
        )
        # print(f"shape of cropped image: {img.shape}")
        mean = mean_with_4_decimals(img)
        print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_right" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
        # self.draw_image = draw_box(self.draw_image,x=k,y=self.height / 4,h=self.height * 0.25,w=10 * self.step,color=(255, 255, 0),)

        greyvald: int = parseInt(mean * self.greytaux)

        while k <= self.width:
            img = crophw(
                self.image,
                f_top=k,
                f_left=self.height / 4,
                f_width=self.height * 0.25,
                f_height=self.step,
            )
            # print(f"shape of cropped image: {img.shape}")
            mean = mean_with_4_decimals(img)
            # saveimage(img, self.name, "crop_right" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
            # self.draw_image = draw_box(self.draw_image,x=k,y=self.height / 4,h=self.height * 0.25,w=self.step,color=(255, 255, 0),)
            # print(f"Mean of cropped image: {mean}")
            if mean < greyvald:
                limit = k
                limit -= self.step
                # print(f"Limit left: {int(limit)}")
                # return int(limit)
                break
            k += self.step

        print(f"Right Limit: {int(limit)}")
        return parseInt(limit)

    def bottom_limit(self):
        limit = self.height
        k = self.height * 0.95
        print(f"k: {k}")
        print(
            f"left={self.width / 6}, top={k}, width={self.width * 0.15}, height={self.step}"
        )
        print(f"image shape : {self.image.shape}")
        img = crophw(
            self.image,
            f_top=self.width / 6,
            f_left=k,
            f_height=self.width * 0.15,
            f_width=10 * self.step,
        )
        # print(f"shape of cropped image: {img.shape}")
        mean = mean_with_4_decimals(img)
        # print(f"Mean of cropped image: {mean}")
        # if mean is not int : print ("NNNNNNNNNNNaaaaaaaaaaNNNNNNNNNNN") ; return

        # saveimage(img, self.name, "crop_bottom" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
        # self.draw_image = draw_box(self.draw_image,x=self.width / 6,y=k,w=self.width * 0.15,h=self.step,color=(255, 0, 0),)

        greyvalb: int = parseInt(mean * self.greytaux)

        while k <= self.height:
            # img = crophw(self.image, top=self.width / 6, left=k, width=self.width*0.15, height=self.step)
            img = crophw(
                self.image,
                f_top=self.width / 6,
                f_left=k,
                f_height=self.width * 0.15,
                f_width=self.step,
            )
            # print(f"shape of cropped image: {img.shape}")
            mean = mean_with_4_decimals(img)
            # saveimage(img, self.name, "crop_bottom" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
            # self.draw_image = draw_box(self.draw_image,x=self.width / 6,y=k,w=self.width * 0.15,h=self.step,color=(255, 0, 0),)
            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalb:  # on arrete
                limit = k
                limit -= self.step
                break
            k += self.step

        print(f"Bottom Limit: {limit} rounded as {parseInt(limit)}")
        return parseInt(limit)

    def top_limit(self) -> int:
        limit = self.height
        k = self.height * 0.05

        img = crophw(
            self.image,
            f_top=self.width / 2 - self.width * 0.25,
            f_left=k,
            f_height=self.width * 0.2,
            f_width=10 * self.step,
        )

        # print(f"shape of cropped image: {img.shape}")
        img_mean = mean_with_4_decimals(img)
        greyvalh: int = parseInt(img_mean * self.greytaux)
        # print(f"top_limit mean: {mean}")
        # print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_top" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
        # self.draw_image = draw_box(self.draw_image, x=self.width / 4, y=k, h=self.width*0.2, w=self.step, color=(255,0,0))
        # self.draw_image = draw_box(self.draw_image,y=k,x=self.width / 2 - self.width * 0.125,w=self.width * 0.25,h=self.step,color=(255, 0, 0),)

        while k > 0:
            img = crophw(
                self.image,
                f_top=self.width / 2 - self.width * 0.25,
                f_left=k,
                f_height=self.width * 0.2,
                f_width=self.step,
            )

            # print(f"shape of cropped image: {img.shape}")
            mean = mean_with_4_decimals(img)
            # print(f"top_limit mean: {mean}")
            # saveimage(img, self.name, "crop_top" + "_" + str(int(k)) + "_" + str(parseInt(mean)), ext="tiff", path=self.crop_output_path)
            # self.draw_image = draw_box(self.draw_image, x=self.width / 4, y=k, h=self.width*0.2, w=self.step, color=(255,0,0))
            # self.draw_image = draw_box(self.draw_image,y=k,x=self.width / 2 - self.width * 0.125,w=self.width * 0.125,h=self.step,color=(255, 0, 0),)

            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalh:  # on arrete
                limit = k
                limit += self.step
                break
            k -= self.step

        print(f"Top Limit: {limit} rounded as {parseInt(limit)}")
        return parseInt(limit)

    def right_limit_to_removeable_from_image(self) -> int:
        limit = self.width

        k = max(self.width * 0.7, self.width - 1200)

        img = crophw(
            self.image,
            f_top=k,
            f_left=self.height / 4,
            f_width=self.height * 0.25,
            f_height=10 * self.step,
        )

        mean = mean_with_4_decimals(img)
        greyval = parseInt(mean * self.greytaux)

        while k <= self.width:
            img = crophw(
                self.image,
                f_top=k,
                f_left=self.height / 4,
                f_width=self.height * 0.25,
                f_height=self.step,
            )
            mean = mean_with_4_decimals(img)
            if mean < greyval or mean == 255:  # On arrête
                limit = k
                limit += (self.width - limit) / 10
                break
            k += self.step

        return parseInt(limit)

    def right_limit_to_removeable_from_right_limit(self):
        right_limit = self.right_limit()
        return right_limit + (self.width - right_limit) / 10
