import unittest
from pathlib import Path

import pytest

from ZooProcess_lib.ZooscanProject import ZooscanProject, buildZooscanProject, ArchivedZooscanProject
from ZooProcess_lib.img_tools import mkdir
from projects_for_test import ROND_CARRE
from env_fixture import projects

class ProjectClassFactory(): pass


class Test_ProjectClass(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def setup(self, projects):
        self.projects = projects

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_temp_ProjectClass(self):
        project_name = "Zooscan_iado_wp2_2020_sn001"
        TPtemp = ZooscanProject(project_name=project_name)
        TP = ZooscanProject(projects, ROND_CARRE)
        testFolder = TPtemp.testfolder

        self.assertEqual(testFolder.as_posix(),
                         "/Users/sebastiengalvagno/piqv/plankton/zooscan_lov/Zooscan_iado_wp2_2020_sn001/Test")

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_folder(self):
        factory = ProjectClassFactory()
        # TP = factory.build("Zooscan_iado_wp2_2020_sn001", remotePIQVHome = "/Volumes/sgalvagno/plankton/")
        # TP = factory.build("Zooscan_iado_wp2_2020_sn001",  "/Volumes/sgalvagno/plankton/",  None,  None)
        # TP = factory.build( # projectName=
        TP = buildZooscanProject(project_name=
                               "Zooscan_iado_wp2_2020_sn001",
                               remotePIQVHome=
                               "/Volumes/sgalvagno/plankton/",
                               #    projectDir =
                               #    None
                               )

        self.assertEqual(TP.piqv, "piqv/plankton/zooscan_lov")  # /Zooscan_iado_wp2")

        self.assertEqual(TP.piqvhome, "/Volumes/sgalvagno/plankton/zooscan_lov")  # /Zooscan_iado_wp2")
        self.assertEqual(TP.folder.as_posix(), "/Volumes/sgalvagno/plankton/zooscan_lov/Zooscan_iado_wp2_2020_sn001")
        # self.assertEqual(str(TP.folder), "/Volumes/sgalvagno/plankton/zooscan_lov/Zooscan_iado_wp2")

        self.assertTrue(Path(TP.piqvhome).exists())
        self.assertTrue(Path(TP.folder).exists())

        self.assertEqual(Path(TP.testfolder).as_posix(),
                         "/Users/sebastiengalvagno/piqv/plankton/zooscan_lov/Zooscan_iado_wp2_2020_sn001/Test")

        mkdir(TP.testfolder)
        self.assertTrue(Path(TP.testfolder).exists())

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_logpattern(self):
        TP = ZooscanProject(projects, ROND_CARRE)
        sample = "test_01_tot_1"

        background = TP.getBackgroundUsed(sample)
        self.assertEqual(background, "20141003_1144")
        # self.assertTrue(getLogFile())

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_BackgroundFile(self):
        TP = ZooscanProject(projects, ROND_CARRE)
        sample = "test_01_tot"

        file1 = TP.getRawBackgroundFile(sample, 1)
        self.assertEqual(file1.name, "20141003_1144_back_large_raw_1.tif")
        self.assertTrue(file1.exists())

        # file2 = TP.getBackgroundFile(sample,2)
        # self.assertTrue(file2.exists())

    @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_process_on_remote_piqv_project(self):
        TP = buildZooscanProject(
            project_name="Zooscan_sn001_rond_carre_zooprocess_separation_training",
            remotePIQVHome="/Volumes/sgalvagno/plankton/",
            projectDir="zooscan_archives_lov"
        )

        sample = "test_01_tot"

        file1 = TP.getRawBackgroundFile(sample, 1)
        self.assertEqual(file1.name, "20141003_1144_back_large_raw_1.tif")
        self.assertTrue(file1.exists())

    # @pytest.mark.skip(reason="Skipping this test for now because of XYZ reason.")
    def test_list_sample(self):
        # TP = buildProjectClass(
        #     project_name = "Zooscan_sn001_rond_carre_zooprocess_separation_training", 
        #     remotePIQVHome = "/Volumes/sgalvagno/piqv/plankton/",
        #     # projectDir = "zooscan_archives_lov"
        #     projectDir = "zooscan_lov"
        # )

        TP = ArchivedZooscanProject(self.projects, ROND_CARRE)

        samples = TP.listSamples()
        print(samples)
