

import unittest
import pytest

from pathlib import Path

from ProjectClass import ProjectClass
# from img_tools import mkdir
from ReadLog import ReadLog

class Test_ReadLog(unittest.TestCase):
    
    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_readfile(self):

        project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
        TP = ProjectClass(project_folder)

        scan_name = "test_01_tot"
        index = 1
        logfile = TP.getLogFile(scan_name, index)

        self.assertEqual(logfile , "/Users/sebastiengalvagno/piqv/plankton/zooscan_lov/Zooscan_sn001_rond_carre_zooprocess_separation_training/Zooscan_scan/_work/test_01_tot_1/test_01_tot_1_log.txt")

        self.assertTrue(Path(logfile).is_file())

    # @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_findBackgroundUsed(self):
        project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
        TP = ProjectClass(project_folder)
        scan_name = "test_01_tot"
        logfile = TP.getLogFile(scan_name, 1)

        log = ReadLog(logfile)

        # background = log._find_key_in_file("Background_correct_using")
        background = log.getBackgroundPattern()
        # background = log.getBackgroundPattern()

        # #  Background_correct_using=  20141003_1144_background_large_manual.tif
        # self.assertEqual(background, "20141003_1144_background_large_manual.tif")
        self.assertEqual(background, "20141003_1144")

        # self.assertTrue(getLogFile())