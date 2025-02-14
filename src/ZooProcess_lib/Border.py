

from .img_tools import (
        crop, crop_scan, crophw,
        loadimage, saveimage, 
        picheral_median, 
        converthisto16to8, convertImage16to8bit, 
        minAndMax, 
        rotate90c, rotate90cc,
        normalize, normalize_back,
        separate_apply_mask,
        draw_contours, draw_boxes, draw_boxes_filtered,
        generate_vignettes,
        mkdir,
        resize,
        rolling_ball_black_background,
        draw_box
    )

import numpy as np
import os

class Border:

    def __init__(self, image):
        self.image = image
        
        self.height = image.shape[0]
        self.width = image.shape[1]

        self.resolution = 2400
        self.step = self.resolution / 240

        self.greytaux = 0.9 # greytaux if greytaux is not None else 0.9
        # np.mean(self.image)
        self.output_path = None

        # self.draw_image = None
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
        # if draw_image:
            self.crop_output_path = os.path.join(self.output_path, "crops")
            print(f"Saving border image to {self.crop_output_path}")
            if not os.path.exists(self.crop_output_path):
                os.makedirs(self.crop_output_path)
            self._draw_image = draw_image

    def detect(self) -> tuple:
        left   = self.left_limit()
        right  = self.right_limit()
        bottom = self.bottom_limit()
        top    = self.top_limit()
        if self.draw_image is not None :
            # crop_output_path = os.path.join(self.output_path, "crops")
            # if not os.path.exists(crop_output_path):
            #     os.makedirs(crop_output_path)
            saveimage(self.draw_image, self.name, "border", ext="tiff", path=self.output_path)
        return (top, bottom, left, right)



    def left_limit(self) -> int:
        limit = self.width
        # limitgauche = limit

        k = self.width * 0.05
        print(f"k: {k}")
        img = crophw(self.image, top=k, left=self.height/2 - self.height/8, width=self.height/4, height=self.step)
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_left" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
        self.draw_image = draw_box(self.draw_image, x=k, y=self.height/2 - self.height/8, h=self.height*0.25, w=self.step, color=(255,0,0)) 

        greyvalb = int(mean * self.greytaux)
        print(f"greyvalb: {greyvalb}")

        while k > 0:
            img = crophw(self.image, top=k, left=self.height/2 - self.height/8, width=self.height/4, height=self.step)
            print(f"shape of cropped image: {img.shape}")
            print(f"dim: {k}, {self.height/2 - self.height*0.125}, {self.step}, {self.height*0.25} mean:{mean} <? {greyvalb}")
            mean = np.mean(img, axis=None)
            # saveimage(img, self.name, "crop_left" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
            self.draw_image = draw_box(self.draw_image, x=k, y=self.height/2 - self.height*0.125, h=self.height*0.25, w=self.step, color=(255,0,0)) 

            print(f"Mean of cropped image: {mean}")
            if mean < greyvalb:
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
            k = k - self.step

        # print(f"Limit left: {limitgauche}")
        print(f"Left Limit: {int(limit)}")
        return int(limit)
        

    def right_limit(self) -> int:
        limit = self.width

        k = max( self.width * 0.75 , self.width - self.resolution/2 )
        # print(f"k: {k}")
        img = crophw(self.image, top=k, left=self.height/4, width=self.height/4, height=10 * self.step)
        # print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_right" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
        self.draw_image = draw_box(self.draw_image, x=k, y=self.height/4, h=self.height/4, w=10*self.step, color=(255,255,0)) 

        greyvalb = int(mean * self.greytaux)

        while k <= self.width:
            img = crophw(self.image, top=k, left=self.height/4, width=self.height/4, height=self.step)
            # print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            # saveimage(img, self.name, "crop_right" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
            self.draw_image = draw_box(self.draw_image, x=k, y=self.height/4, h=self.height/4, w=self.step, color=(255,255,0)) 
            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalb:
                limit = k
                limit = limit - self.step
                # print(f"Limit left: {int(limit)}")
                # return int(limit)
                break
            k = k + self.step

        print(f"Right Limit: {int(limit)}")
        return int(limit)


    def bottom_limit(self):

        limit = self.height
        k = self.height * 0.95
        print(f"k: {k}")
        print(f"left={self.width / 6}, top={k}, width={self.width*0.15}, height={self.step}")
        print(f"image shappe : {self.image.shape}")
        img = crophw(self.image, top=self.width / 6, left=k, height=self.width*0.15, width=self.step)
        print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        print(f"Mean of cropped image: {mean}")
        # if mean is not int : print ("NNNNNNNNNNNaaaaaaaaaaNNNNNNNNNNN") ; return

        # saveimage(img, self.name, "crop_bottom" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
        self.draw_image = draw_box(self.draw_image, x=self.width / 6, y=k, w=self.width*0.15, h=self.step, color=(255,0,0)) 

        greyvalb = int(mean * self.greytaux)

        while k <= self.height :
            # img = crophw(self.image, top=self.width / 6, left=k, width=self.width*0.15, height=self.step)
            img = crophw(self.image, top=self.width / 6, left=k, height=self.width*0.15, width=self.step)
            # print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            # saveimage(img, self.name, "crop_bottom" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
            self.draw_image = draw_box(self.draw_image, x=self.width / 6, y=k, w=self.width*0.15, h=self.step, color=(255,0,0)) 
            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalb: # on arrete
                limit = k
                limit = limit - self.step
                break
            k = k + self.step

        print(f"Bottom Limit: {int(limit)}")
        return (int(limit))

    def top_limit(self) -> int:

        limit = self.height
        limitbas = limit
        k = self.height * 0.05
        # print(f"k: {k}")
        # img = crophw(self.image, left=self.width / 2 - self.width/4, top=k, width=self.width * 0.2, height=self.step)
        img = crophw(self.image, top=self.width / 2 - self.width / 4, left=k, height=self.width * 0.2, width=self.step)

        # print(f"shape of cropped image: {img.shape}")
        mean = np.mean(img, axis=None)
        # print(f"Mean of cropped image: {mean}")
        # saveimage(img, self.name, "crop_top" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
        # self.draw_image = draw_box(self.draw_image, x=self.width / 4, y=k, h=self.width*0.2, w=self.step, color=(255,0,0)) 
        self.draw_image = draw_box(self.draw_image, y=k, x=self.width/2 - self.width/8, w=self.width/4, h=self.step, color=(255,0,0)) 

        greyvalb = int(mean * self.greytaux)

        while k > 0 :
            # img = crophw(self.image, left=self.width / 2 - self.width/4, top=k, width=self.width * 0.2, height=self.step)
            img = crophw(self.image, top=self.width / 2 - self.width / 4, left=k, height=self.width * 0.2, width=self.step)

            # print(f"shape of cropped image: {img.shape}")
            mean = np.mean(img, axis=None)
            # saveimage(img, self.name, "crop_top" + "_" + str(int(k)) + "_" + str(int(mean)), ext="tiff", path=self.crop_output_path)
            # self.draw_image = draw_box(self.draw_image, x=self.width / 4, y=k, h=self.width*0.2, w=self.step, color=(255,0,0)) 
            self.draw_image = draw_box(self.draw_image, y=k, x=self.width/2 - self.width/8, w=self.width/4, h=self.step, color=(255,0,0)) 

            # print(f"Mean of cropped image: {mean}")
            if mean < greyvalb: #or mean == 255: # on arrete
                limit = k
                limit = limit + self.step
                break
            k = k - self.step

        print(f"Limit top: {int(limit)}")
        return int(limit)

