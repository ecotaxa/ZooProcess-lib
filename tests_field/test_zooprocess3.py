import unittest
from pathlib import Path

import pytest

from ZooProcess_lib.ZooscanProject import ZooscanProject
from ZooProcess_lib.img_tools import mkdir
from ZooProcess_lib.zooprocess3 import zooprocessv10, Analyze_sample
from env_fixture import projects
from projects_for_test import ROND_CARRE


class test_zooprocess2(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, projects):
        self.TP = ZooscanProject(projects, ROND_CARRE)

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_scan_rectangle(self):
        scan_name = "test_01_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)
        z.use_average = False

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_scan_rectangle_average(self, tmp_path):
        scan_name = "test_01_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)
        z.use_average = True

        output = Path(tmp_path, scan_name + "_average")
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_scan_circle(self):
        scan_name = "test_14_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_scan_circle_2(self):
        scan_name = "test_15_tot"
        bg_name = "20141003_1144_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_scan_bioness(self):
        project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1"
        # bg_name = "20240112_1518_back_large"
        # bg_name = "20240116_0818_back_large"
        bg_name = "20240115_1433_back_large"
        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_trace_noire_1(self):
        project_folder = "Zooscan_test_traces_noires_sn173"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "test_traces_noires_1_tot"
        bg_name = "20240315_0948_back_large"
        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_trace_noire_2(self):
        project_folder = "Zooscan_test_traces_noires_sn173"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "test_traces_noires_2_tot"
        bg_name = "20240315_1007_back_large"
        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_trace_noire_2(self):
        project_folder = "Zooscan_iado_wp2_2023_sn002"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "t_17_2_tot"
        # bg_name = "20240315_1007_back_large"
        bg_pattern = TP.getBackgroundUsed(scan_name + "_1")
        bg_name = bg_pattern + "_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_trace_noire_2(self):
        project_folder = "Zooscan_iado_wp2_2023_sn002"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "t_17_a_tot"
        # bg_name = "20240315_1007_back_large"
        bg_pattern = TP.getBackgroundUsed(scan_name + "_1")
        bg_name = bg_pattern + "_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_project_Zooscan_test_traces_noires_sn173(self):
        project_folder = "Zooscan_test_traces_noires_sn173"
        TP = ZooscanProject(projects, project_folder)

        samples = TP.listSamples()
        print(f"{len(samples)} to analyze")

        for sample in samples:
            Analyze_sample(self.TP, sample)

        print("Done")

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_project_Zooscan_iado_wp2_2023_sn00(self):
        project_folder = "Zooscan_iado_wp2_2023_sn002"
        TP = ZooscanProject(projects, project_folder)

        samples = TP.listSamples()
        print(f"{len(samples)} to analyze")

        for sample in samples:
            Analyze_sample(self.TP, sample)

        print("Done")

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_test_project_t_17_c_tot(self):
        project_folder = "Zooscan_iado_wp2_2023_sn002"
        TP = ZooscanProject(projects, project_folder)

        scan_name = "t_17_c_tot"
        # bg_name = "20240315_1007_back_large"
        bg_pattern = TP.getBackgroundUsed(scan_name + "_1")
        # Background_correct_using=  20230918_1410_background_large_manual.tif

        print(f"bg_pattern: {bg_pattern}")
        bg_name = bg_pattern + "_back_large"

        z = zooprocessv10(self.TP, scan_name, bg_name)

        output = Path(tmp_path, scan_name)
        mkdir(output)
        z.output_path = output

        z.process()

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_project_Zooscan_iado_wp2_2023_sn00(self):
        project_folder = "Zooscan_apero_tha_bioness_sup2000_sn033"
        TP = ZooscanProject(projects, project_folder)

        samples = TP.listSamples()
        print(f"{len(samples)} to analyze")

        for sample in samples:
            Analyze_sample(self.TP, sample)

        print("Done")
