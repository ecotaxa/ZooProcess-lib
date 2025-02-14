import numpy as np

import cv2

from .img_tools import crophw, crop, saveimage, draw_box


class Background:
    def __init__(
        self, image, image_name, resolution=300, scan_resolution=2400, output_path=None
    ):
        self.image = image
        self.name = image_name

        self.output_path = output_path

        self.blancres = resolution
        self.frametypeback = None
        if image_name.find("large"):
            self.frametypeback = "large"
        if image_name.find("narrow"):
            self.frametypeback = "narrow"

        self.backratio(scan_resolution)

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.L = int(self.width * self.backratio)
        self.H = int(self.height * self.backratio)
        self.lf = self.width
        self.hf = self.height

        self.Lm = self.L * 0.95
        self.LM = self.L * 1.05
        self.Hm = self.H * 0.95
        self.HM = self.H * 1.05

    def __str__(self):
        # def __repr__(self):

        return f"""
            name: {self.name}
            blancres: {self.blancres}

            L : {self.L:8}   H : {self.H:8}
            lf: {self.lf:8}   hf: {self.hf:8}
            LM: {self.LM:8}   HM: {self.HM:8}
            Lm: {self.Lm:8}   Hm: {self.Hm:8}
        
            ratio: {self.backratio}
        """

    def mean(self):
        x1 = self.width / 10
        y1 = self.height / 10
        x2 = self.width - x1
        y2 = self.height - y1
        w = x2 - x1
        h = y2 - y1
        w = 10

        image_cropped = crop(self.image, x1, y1, x2, y2)

        image_drew = draw_box(self.image, x=x1, y=y1, w=w, h=h)
        saveimage(image_drew, self.name, "drew", path=self.output_path)

        image_median = cv2.medianBlur(image_cropped, 3)
        # image_median = image_drew

        mean = np.mean(image_median, axis=None)
        print(mean)
        return mean

    # def backratio(self, scan_resolution=2400):
    def backratio(self, scan_resolution=2400):
        self.backratio = scan_resolution / self.blancres
        return self.backratio

    def voxel(self, scan_image):
        # backratio = self.backratio(scan_image)
        pass

    def redim(self):
        # cropx = getWidth();
        # cropy = getHeight();
        cropx = self.width
        cropy = self.height

        larg = cropx / self.backratio
        haut = cropy / self.backratio
        fondx0 = self.lf - larg
        fondy0 = self.hf - haut

        print(f"fond: {fondx0},{fondy0} {haut},{larg}")

        # makeRectangle(fondx0,fondy0,larg,haut);
        image_cropped = crophw(self.image, fondx0, fondy0, haut, larg)
        # run("Crop");

        image_median = cv2.medianBlur(image_cropped, 3)

        fx = self.L / larg
        fy = self.H / haut
        interpolation = cv2.INTER_LINEAR
        image_resized = cv2.resize(
            image_median, (self.L, self.H), fx, fy, interpolation
        )

        saveimage(
            image_resized, self.name, "resized", ext="tiff", path=self.output_path
        )

        return image_resized
