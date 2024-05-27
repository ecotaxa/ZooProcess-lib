
import unittest
import pytest

from pathlib import Path

from zooprocess2 import zooprocessv10
from ProjectClass import ProjectClass
from img_tools import mkdir
class test_zooprocess2(unittest.TestCase):

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_process_on_test_project_scan_rectangle(self):

        project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
        TP = ProjectClass(project_folder)

        scan_name = "test_01_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(TP, scan_name, bg_name)
        
        output = Path(TP.testfolder, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_process_on_test_project_scan_circle(self):

        project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
        TP = ProjectClass(project_folder)

        scan_name = "test_14_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(TP, scan_name, bg_name)

        output = Path(TP.testfolder, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_process_on_test_project_scan_circle_2(self):

        project_folder = "Zooscan_sn001_rond_carre_zooprocess_separation_training"
        TP = ProjectClass(project_folder)

        scan_name = "test_15_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(TP, scan_name, bg_name)

        output = Path(TP.testfolder, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    # @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")  
    def test_process_on_test_project_scan_bioness(self):

        project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
        TP = ProjectClass(project_folder)

        scan_name = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        # bg_name = "20240112_1518_back_large"
        # bg_name = "20240116_0818_back_large"
        bg_name = "20240115_1433_back_large"
        z = zooprocessv10(TP, scan_name, bg_name)

        output = Path(TP.testfolder, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

