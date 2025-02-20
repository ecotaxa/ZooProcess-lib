import unittest
import pytest

from pathlib import Path

from ZooProcess_lib.zooprocess2 import zooprocessv10
from ZooProcess_lib.ZooscanProject import ZooscanProject, buildZooscanProject
from ZooProcess_lib.img_tools import mkdir


class test_zooprocess_from_piqv(unittest.TestCase):
        
    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_on_remote_piqv_project_sn001_rond_carre_zooprocess_separation_training(self):
        
        TP = buildZooscanProject(  
            project_name = "Zooscan_sn001_rond_carre_zooprocess_separation_training", 
            remotePIQVHome = "/Volumes/sgalvagno/plankton/",
            projectDir = "zooscan_archives_lov"
            )
        
        sample = "test_01_tot"

        file1 = TP.getRawBackgroundFile(sample, 1)
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

    @pytest.mark.skip(reason="No data")
    def test_on_remote_piqv_project_Zooscan_Lebanon_1999_2014_wp2_52(self):
        
        TP = buildZooscanProject(  
            project_name = "Zooscan_Lebanon_1999_2014_wp2_52", 
            remotePIQVHome = "/Volumes/sgalvagno/plankton/",
            projectDir = "zooscan_lov"
            )
        
        sample = "52_b2_20040317_d2"

        # file1 = TP.getBackgroundFile(sample,1)
        # self.assertEqual(file1.name, "20141003_1144_back_large_raw_1.tif")
        # self.assertTrue(file1.exists())

        bg_pattern = TP.getBackgroundUsed(sample + "_1") 
        # self.assertEqual(bg_pattern, "20141003_1144")

        bg_name = bg_pattern + "_back_large"

        z = zooprocessv10(TP, sample, bg_name)

        output = Path(TP.testfolder, sample)
        self.assertEqual(output.as_posix(), "/Users/sebastiengalvagno/piqv/plankton/zooscan_lov/Zooscan_Lebanon_1999_2014_wp2_52/Test/52_b2_20040317_d2")
        mkdir(output)
        z.output_path = output
        z.process()
