
from tools import create_folder

from pathlib import Path

class ProjectClass():
    """
    Properties:
        project_name: the name of the project
        home: the folder containing the project
        folder: the project path
    """


    userhome = "/Users/sebastiengalvagno/"
    # piqv="piqv/plankton/zooscan_monitoring/"
    piqv="piqv/plankton/zooscan_lov/"
    piqvhome=userhome+piqv

    def __init__(self,project_name, testfolder=None) -> None:
        print("ProjectClass __init__")
        self.project_name = project_name
        # if project_name:
        self.folder = Path(self.piqvhome, project_name)

        self.home = Path(self.piqvhome)

        if testfolder:
            self.testfolder = testfolder
        else:
            self.testfolder = Path(self.piqvhome, project_name, "Test")

        create_folder(self.testfolder)

        self.back = Path(self.folder, "Zooscan_back")
        self.scan = Path(self.folder, "Zooscan_scan")
        self.rawscan = Path(self.folder, "Zooscan_scan", "_raw")
        self.workscan = Path(self.folder, "Zooscan_scan", "_work")

        print("__init__ done")
        

    def getRawPath(self, sample, index):
        path = Path(self.folder, "Zooscan_scan", "_raw", sample + "_raw" + "_" + str(index) + ".tif")
        return path.absolute().as_posix()

    def getBackPack(self, background_date, index):
        # path = Path(self.folder, "Zooscan_back", background_date + "_back_large_" + str(index) + ".tif")
        path = Path(self.back, background_date + "_back_large_raw_" + str(index) + ".tif")  
        return path.absolute().as_posix()
    

