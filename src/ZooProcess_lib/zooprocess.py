from datetime import date
from pathlib import Path

import numpy as np

from .img_tools import (
    loadimage, saveimage,
    rotateAndFlip,
    picheral_median, minAndMax, convertShortToByte,
    converthisto16to8, convertImage16to8bit,
)
from .tools import timeit
from .ZooscanFolder import Zooscan_Project, Zooscan_sample_scan


class ZooProcess():
    """
    Properties:
        name: the name of the project
        path: the folder containing the project
        project: the access to the project path subfolders
    """

    def __init__(self, project_path: Path, project_name: str, output_folder=None) -> None:
        """
        project_name: the path containing the project
        project_name: the name of the project
        """
        self.path = project_path
        self.name = project_name
        self.project = Zooscan_Project(project_name, project_path)

        self.output_folder = output_folder

    def check_folders(self) -> bool:
        if not self.project.zooscan.path.is_dir(): return False
        if not self.project.backfolder.path.is_dir(): return False
        if not self.project.rawfolder.path.is_dir(): return False
        if not self.project.workfolder.path.is_dir(): return False

        return True

    def normalize_rawscan(self, scan: Path):
        image = loadimage(scan.as_posix())
        image = self.method3(image, filename=scan.as_posix())
        image = rotateAndFlip(image)
        return image

    def load_normalize_rawscans(self):

        rawscans = self.project.getRawScan()
        for scan in rawscans:
            image = self.normalize_rawscan(scan)
            samplename, sampleid = self.rawfolder.extract_sample_name(scan)
            sample_names = Zooscan_sample_scan(samplename, sampleid, self.project.path)

            outputPath = self.project.workfolder.path
            if self.output_folder:
                outputPath = self.output_folder

            saveimage(image, sample_names.work, outputPath)

    @timeit
    def method4(self, image, name=None, filename=None) -> np.ndarray:
        """
        faster than method2
        """
        print('method 4')
        median, mean = picheral_median(image)
        min, max = minAndMax(median)

        histolut = converthisto16to8(min, max)
        eightbitimage = convertImage16to8bit(image, histolut)
        pov_image = rotateAndFlip(eightbitimage)

        extraname = "cropped_eightbitimage_method4"
        if name:
            extraname = extraname + "_" + name
        saveimage(eightbitimage, filename, extraname, path=self.output_folder)
        return eightbitimage

    @timeit
    def method3(self, image, name=None, filename=None) -> np.ndarray:
        """
        faster than method2
        """
        print('method 3')

        median, mean = picheral_median(image)
        min, max = minAndMax(median)

        histolut = converthisto16to8(min, max)
        eightbitimage = convertImage16to8bit(image, histolut)
        pov_image = rotateAndFlip(eightbitimage)

        extraname = "cropped_eightbitimage_method3"
        if name:
            extraname = extraname + "_" + name
        saveimage(eightbitimage, filename, extraname, path=self.output_folder)
        return eightbitimage

    @timeit
    def method2(self, image, name=None, filename=None) -> np.ndarray:
        median, mean = picheral_median(image)
        min, max = minAndMax(median)

        eightbitimage = convertShortToByte(image, min, max)
        extraname = "cropped_eightbitimage_method2"
        if name:
            extraname = extraname + "_" + name
        saveimage(eightbitimage, filename, extraname, path=self.output_folder)
        return eightbitimage

    # def test_min_max_median(self):

    #     # assert min == 280
    #     # assert max == 23424
    #     # assert median == 20369.1
    #     # assert mean == 19910.093307967432
    #     pass

    def getdate(_date: date = date.today()):
        # today = _date.today()
        textdate = _date.strftime("%y%m%d_%H%M")
        return textdate

    # def process_background(self):

    #     from background import background

    #     rawscans = self.project.backfolder.get_back_scans()

    #     backs = []
    #     i = 0
    #     for scan in rawscans:
    #         backs[i] = scan
    #         i+=1

    #     if len(backs)<2: raise Exception("missing back scan")

    #     b = background()
    #     sum_image = b.sum_background(backs[0],backs[1])

    #     filtre = lambda back: "_1.tif" in backs

    #     # import re

    #     date = "" 

    #     name = b.getname('manual', date)

    #     # output_path = self.project.output('background')
    #     output_path = self.output_folder

    #     filename = saveimage(sum_image,name,path=output_path)

    def process_background(self):

        dates = self.project.backfolder.dates()

        if len(dates) > 0:
            data = dates[0]
