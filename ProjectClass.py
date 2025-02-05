
from tools import create_folder

from pathlib import Path
from ReadLog import ReadLog

import os
from  pathlib import Path

class ProjectClass():
    """
    Properties:
        project_name: the name of the project
        home: the folder containing the project
        folder: the project path
    """


    userhome = "/Users/sebastiengalvagno/"
    # piqv="piqv/plankton/zooscan_monitoring/"
    categoriefolder="zooscan_lov"
    piqv=Path("piqv/plankton/" , categoriefolder).as_posix()
    piqvhome= Path(userhome,piqv).as_posix()

    def __init__(self, project_name, piqvhome= None, testfolder = None) -> None:
        print("ProjectClass __init__")

        if piqvhome:
            self.piqvhome = piqvhome

        self.project_name = project_name
        # if project_name:
        self.folder = Path(self.piqvhome, project_name)

        self.home = Path(self.piqvhome)

        if testfolder:
            self.testfolder = Path(testfolder)
        else:
            self.testfolder = Path(self.piqvhome, project_name, "Test")

        # create_folder(self.testfolder)

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
    

    def getWorkScanPath(self, sample):
        path = Path(self.workscan, sample)
        return path.absolute().as_posix()
    
    def getLogFile(self, sample, index=None ):
        indexstr = ""
        if (index): indexstr = "_" + str(index)
        print(f"getLogFile -> indexstr: {indexstr}")
        path = Path(self.workscan, sample + indexstr, sample + indexstr + "_log.txt")
        return path.absolute().as_posix()
    
    def getBackgroundUsed(self, sample):

        logfile = self.getLogFile(sample)
        print(f"getBackgroundUsed -> logfile: {logfile}")
        # key = "Background_correct_using"

        # log = ReadLog(self, sample)
        log = ReadLog(Path(logfile))
        return log.getBackgroundPattern()

    def getBackgroundFile(self, sample:str, index:int=None):
        indexstr = ""
        if index:
            indexstr = "_" + str(index)
        background = self.getBackgroundUsed(sample + indexstr)
        return Path(self.back, background + "_back_large_raw" + indexstr + ".tif")

    def listSamples(self) -> list[str]:

        print(f"scan: {self.scan.as_posix()}")

        full_list_of_dirs = os.listdir(self.scan.as_posix())

        list = []

        for file in full_list_of_dirs:
            # print(f"-> {file [4:]} - {file [-4:]}")
            if file[-4:] == ".tif":
                # print(f"++++ files: {file} ++++")
                s =  file.split("_")
                # print(f"split : {s}")
                # print(f"split -1 : {s[:-1]}")
                f = "_".join( s[:-1] )
                # print(f"join : {f}")
                # list.append(file[:-4])
                list.append(f)

        return list




# Factory
def buildProjectClass(project_name, remotePIQVHome = None, projectDir = "zooscan_lov" ) -> ProjectClass:

    localPiqvhome = "/Volumes/sgalvagno/plankton/"

    if remotePIQVHome:
        piqvhome = Path(remotePIQVHome, projectDir ).as_posix()
    else:
        piqvhome = Path(localPiqvhome, projectDir ).as_posix() # Path.home()

    print(f"piqVhome: {piqvhome}")

    # if projectDir == None : projectDir = "zooscan_lov"
    # projectName = "Zooscan_iado_wp2_2020_sn001"
    sampletName = "s_21_14_tot_1"

    TPtemp = ProjectClass(project_name=project_name)

    # testFolder = TPtemp.testfolder
    testFolder = Path(TPtemp.userhome,"piqv/plankton/",projectDir,project_name,"Test").as_posix()

    # if ( projectDir ):
    # piqvhome = piqvhome + projectDir + "/"
    # else

    # projectDir = "zooscan_lov/"
    # projectName = projectDir

    # localFolder = "/home/sebastiengalvagno/"
    # testFolder = Path(localFolder, projectDir, projectName, "test")
    # TPtemp = ProjectClass( project_name="" ) 
    # testFolder = TPtemp.testfolder.as_posix()

    TP = ProjectClass( project_name=project_name, piqvhome=piqvhome, testfolder=testFolder )
    # TPtemp = None

    return TP

