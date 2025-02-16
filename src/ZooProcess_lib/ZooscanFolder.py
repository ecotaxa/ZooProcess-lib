from pathlib import Path
from typing import Generator, List, Tuple, Union, Any, Dict


class ZooscanFolder:
    # zooscan_back_folder="Zooscan_back/"
    # zooscan_scan_folder="Zooscan_scan/"
    # zooscan_config_folder="Zooscan_config/"
    # zooscan_check_folder="Zooscan_check/"
    # zooscan_meta_folder="Zooscan_meta/"
    # zooscan_results_folder="Zooscan_results/"
    # zooscan_PID_process_folder="PID_process/"

    # def __init__(self, project:str, home:str=None, piqv:str=None) -> None:
    def __init__(self, home: Path, project: str) -> None:
        # self.home="/Users/sebastiengalvagno/"
        # self.piqv="piqv/plankton/zooscan_monitoring/"
        # if home:
        #     self.home=home
        # if piqv:
        #     self.piqv=piqv
        # self._absolute_home_project_path = Path(self.home,self.piqv)
        self.project = project
        self.path = Path(home, project)  # noqa: E501

        self._folders()
        self._debug_folders()

    def _folders(self):
        self.zooscan_scan = Zooscan_scan_Folder(self.path)
        self.zooscan_back = Zooscan_back_Folder(self.path)

    def _debug_folders(self):
        # folder to save my image processed
        # self.zooscan_test_folder = self.path + "test2" + "/"
        # self.vignettes_folder    = self.zooscan_test_folder + '/' + "vignettes"
        self.zooscan_test_folder = Path(self.path, "test2")
        self.vignettes_folder = Path(self.zooscan_test_folder, "vignettes")

    def pathold(self, file_scan, file_back):
        self.zooscan_back_file_path = self.path + self.zooscan_back_folder + file_back
        self.zooscan_scan_file_path = self.path + self.zooscan_scan_folder + file_scan

        return self.zooscan_back_file_path, self.zooscan_scan_file_path

    def path(self, folder, file) -> str:
        path = self._absolute_home_project_path + self.project_folder + folder + file
        return path

    def sample_path(sample: str, mesure: int):
        raw = sample + "_" + str(mesure)


class Zooscan_scan_Folder:
    _zooscan_path = "Zooscan_scan"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self._zooscan_path)
        self.raw = Zooscan_scan_raw_Folder(self.path)
        self.work = Zooscan_scan_work_Folder(self.path)


class Zooscan_back_Folder:
    _zooscan_path = "Zooscan_back"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self._zooscan_path)

    def get_back_scans(self) -> Generator:
        """
        arg bug - a tester avec plusierus date
        """

        try:
            if self.files:
                return self.files
        except:
            pass

        # raw_files = Path(self.path).glob('*_raw_*')
        raw_files = Path(self.path).glob("*")

        file_type = {
            "_back_large_raw_1": "raw_1",
            "_back_large_raw_2": "raw_2",
            "_back_large_1": "scan_1",
            "_back_large_2": "scan_2",
            "_background_": "background",
            "_back_large_manual_log": "log",
        }

        files = {}
        # names = []
        for file in raw_files:
            # files.append(file)
            # filename = file.name
            # names.append(filename)
            # print(filename)

            types = file_type

            for pattern in file_type:
                if pattern in file.name:
                    # print("pattern:",pattern,end=" - ")
                    # print("append:", filename)
                    files[file_type[pattern]] = file
                    del types[pattern]
                    break

        # return raw_files
        # return (files,names)
        self.files = files
        return self.files

    # def getNameFromRawFile(self,filename,key):

    def getbacks(self):
        try:
            if self.backs:
                return self.backs
        except:
            pass

        # files = Path(self.path).glob('*')

        backs = []
        for file in self.files:
            name = file.filename()
            backs.append(name)

        return backs

    def get_samples(self) -> Generator:  # -> Array(Path):
        try:
            if self.files:
                return self.files
        except:
            pass

        raw_files = Path(self.path).glob("*_raw_*")

        files = []
        # names = []
        for file in raw_files:
            files.append(file)
            # filename = file.name
            # name = filename.split("_raw")[0]
            # print(name)
            # # names.append(file)
            # files.append((file,name))

        # return raw_files
        # return (files,names)
        self.files = files
        return self.files

    def extract_date(file: Path):
        """
        Extract the date from the raw filename path
        """

        filename = file.name
        name = filename.split("_raw")[0]
        print(name)
        return name

    def get_date(self):
        """
        return the list of dates
        """
        if self.dates:
            return self.dates
        if not self.files:
            self.get_samples()

        dates = {}
        for file in self.files:
            date = self.extract_date(file)
            dates.append(date)

        sorted_dates = sorted(frozenset(dates), reverse=True)

        self.dates = sorted_dates
        return self.dates


class Zooscan_scan_work_Folder:
    _work = "_work"

    def __init__(self, zooscan_scan_raw_folder: Path) -> None:
        self.path = Path(zooscan_scan_raw_folder, self._work)

    def get_files(
        self, sample_name: str, index: int
    ) -> dict[str, Union[list[Path], Path]]:
        path = Path(self.path, sample_name + "_" + str(index))
        filelist = Path(path).glob("*")

        file_type: Dict[str, str] = {
            ".tsv": "tsv",
            "_sep.gif": "sep",
            "_out1.gif": "out1",
            "_msk1.gif": "msk1",
            "_meta.txt": "meta",
            "_meas.txt": "meas",
            "_log.txt": "log",
            "_dat1.pid": "pid",
            "_vis1.zip": "rawz",
        }
        files = {"jpg": []}
        for file in filelist:
            for pattern in file_type:
                if pattern in file.name:
                    files[file_type[pattern]] = file
                    del file_type[pattern]
                    break
            if ".jpg" in file.name:
                files["jpg"].append(file)
        if len(files["jpg"]) == 0:
            del files["jpg"]
        return files


class Zooscan_scan_raw_Folder:
    _raw = "_raw"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self._raw)

    def get_samples(self) -> List[Path]:
        raw_files = self.path.glob("*_raw_*")
        files = []
        for file in raw_files:
            files.append(file)
        return files

    @staticmethod
    def extract_sample_name(file: Path) -> Tuple[str, int]:
        filename = file.name
        split_name = filename.split("_raw_")
        name = split_name[0]
        id_ = int(split_name[1].split(".")[0])
        return name, id_

    def get_names(self) -> list[dict[str, Union[int, str]]]:
        names = []
        for file in self.get_samples():
            name, id_ = self.extract_sample_name(file)
            nameid = {"name": name, "id": id_}
            names.append(nameid)
        return names


class Zooscan_sample_scan:
    _log = "_log"
    _raw = "_raw"
    _meta = "_meta"
    _tif_ext = ".tif"
    _log_ext = ".log"
    _txt_ext = ".txt"

    def __init__(self, sample: str, scan_id: int, project_path) -> None:
        self.rawfile = sample + self._raw + "_" + str(scan_id) + self._tif_ext
        self.logfile = sample + "_" + str(scan_id) + self._log + self._log_ext
        self.metafile = sample + "_" + str(scan_id) + self._meta + self._txt_ext

        self.work = sample + "_" + str(scan_id) + self._tif_ext

        self.folder = sample + "_" + str(scan_id)


class Zooscan_sample:
    def __init__(self, sample, raw_path) -> None:
        self.rawfile = ""
        self.workfolder = ""
        self.logfile = ""
        self.metafile = ""


class Zooscan_Project:
    def __init__(self, projectname, home_path: Path) -> None:
        # self.scans : Array(Zooscan_sample_scan) = []
        self.scans = []

        self.project = projectname

        self.path = Path(home_path, projectname)
        # self.zooscan = ZooscanFolder(project_path.absolute(),home="",piqv="")
        # self.zooscan = ZooscanFolder(project_path.absolute(),home="",piqv="")
        self.zooscan = ZooscanFolder(home_path, projectname)
        # self.zooscan = ZooscanFolder(self.path)
        self.rawfolder = self.zooscan.zooscan_scan.raw
        # samples = self.zooscan.zooscan_scan.raw.get_samples()
        self.workfolder = self.zooscan.zooscan_scan.work

        self.backfolder = self.zooscan.zooscan_back

    def getRawScan(self):
        rawFiles = self.rawfolder.get_samples()
        return rawFiles

    def getBackScan(self):
        backFiles = []
        scans = self.backfolder.get_back_scans()
        return scans

    def getSampleNames(self):
        rawscans = self.rawfolder.get_samples()
        samplenames = []
        for scan in rawscans:
            samplename = self.rawfolder.extract_sample_name(scan)
            samplenames.append(samplename[0])

        return samplenames

    def getSample(self):
        samplename = self.rawfolder.get_names()
        return samplename

    def getFilesFromSample(self, sample_name: str):
        # path = Path (self.workfolder.path, sample_name)
        # return self.getFilesFromSampleFolder(path)
        files = self.workfolder.get_files(sample_name)
        return files

    def getFilesFromSampleFolder(self, sample_folder: Path):
        sample_name = sample_folder.name
        files = self.getFilesFromSample(sample_name)
        return files
