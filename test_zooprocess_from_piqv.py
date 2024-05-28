import unittest
import pytest

from pathlib import Path

from zooprocess2 import zooprocessv10
from ProjectClass import ProjectClass, buildProjectClass
from img_tools import mkdir


class test_zooprocess_from_piqv(unittest.TestCase):
        
    def test_on_remote_piqv_project(self):
        
        TP = buildProjectClass(  
            project_name = "Zooscan_sn001_rond_carre_zooprocess_separation_training", 
            remotePIQVHome = "/Volumes/sgalvagno/plankton/",
            projectDir = "zooscan_archives_lov"
            )
        
        sample = "test_01_tot"

        file1 = TP.getBackgroundFile(sample,1)
        self.assertEqual(file1.name, "20141003_1144_back_large_raw_1.tif")
        self.assertTrue(file1.exists())

        bg_pattern = TP.getBackgroundUsed(sample + "_1") 
        self.assertEqual(bg_pattern, "20141003_1144")

        bg_name = bg_pattern + "_back_large"

        z = zooprocessv10(TP, sample, bg_name)

        output = Path(TP.testfolder, sample)
        self.assertEqual(output.as_posix(), "/Users/sebastiengalvagno/piqv/plankton/zooscan_archives_lov/Zooscan_sn001_rond_carre_zooprocess_separation_training/Test/test_01_tot")
        mkdir(output)
        z.output_path = output
        z.process()

