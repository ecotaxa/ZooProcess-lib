import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Dict, TypedDict, Optional, Generator

from ZooProcess_lib.LegacyConfig import Lut, ZooscanConfig


class ZooscanFolder:
    def __init__(self, home: Path, project: str) -> None:
        self.project = project
        self.path = Path(home, project)  # noqa: E501
        self.zooscan_scan = Zooscan_scan_Folder(self.path)
        self.zooscan_back = Zooscan_back_Folder(self.path)
        self.zooscan_config = Zooscan_config_Folder(self.path)
        self._debug_folders()

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



class Zooscan_config_Folder:
    SUDIR_PATH = "Zooscan_config"
    INSTALL_CONFIG = "process_install_both_config.txt"


    def __init__(self, zooscan_folder: Path) -> None:
        self.path = Path(zooscan_folder, self.SUDIR_PATH)
        self.read()

    def read(self) -> ZooscanConfig:
        install_conf = Path(self.path, self.INSTALL_CONFIG)
        return ZooscanConfig.read(install_conf)

    def read_lut(self) -> Lut:
        config_file = self.path / "lut.txt"
        return Lut.read(config_file)


class Zooscan_scan_Folder:
    _zooscan_path = "Zooscan_scan"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self._zooscan_path)
        self.raw = Zooscan_scan_raw_Folder(self.path)
        self.work = Zooscan_scan_work_Folder(self.path)

    def get_file_produced_from(self, raw_file_name: str) -> Path:
        assert "_raw" in raw_file_name
        return Path(self.path, raw_file_name.replace("_raw", ""))

    def get_8bit_file(self, sample_name: str, index: int) -> Path:
        return Path(self.path, sample_name + "_" + str(index) + ".tif")

    def list_samples(self) -> Generator[str, None, None]:
        for a_file in self.path.iterdir():
            if a_file.suffix == ".tif":
                yield a_file.name[:-6]


class BackgroundEntry(TypedDict):
    nb_scans: int
    raw_scans: List[Path]
    scans_8bit: List[Path]
    final_background: Optional[Path]
    log_file: Optional[Path]
    raw_background_1: Optional[Path]
    raw_background_2: Optional[Path]


date_re = re.compile(r"(\d{8}_\d{4})_.*")


class Zooscan_back_Folder:
    _zooscan_path = "Zooscan_back"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self._zooscan_path)
        self.content: Dict[str, BackgroundEntry] = {}
        self.read()

    def read(self):
        """ """
        all_files = sorted(Path(self.path).glob("*"))
        for a_file in all_files:
            date_match = date_re.match(a_file.name)
            if not date_match:
                continue
            date = date_match.group(1)
            date_entry = self.content.setdefault(
                date,
                BackgroundEntry(
                    nb_scans=0,
                    raw_scans=[],
                    scans_8bit=[],
                    final_background=None,
                    log_file=None,
                    raw_background_1=None,
                    raw_background_2=None,
                ),
            )
            if a_file.name.endswith("_background_large_manual.tif"):
                date_entry["final_background"] = a_file
            if a_file.name.endswith("_back_large_raw_1.tif"):
                date_entry["raw_background_1"] = a_file
            if a_file.name.endswith("_back_large_raw_2.tif"):
                date_entry["raw_background_2"] = a_file

        # file_type = {
        #     "_back_large_raw_1": "raw_1",
        #     "_back_large_raw_2": "raw_2",
        #     "_back_large_1": "scan_1",
        #     "_back_large_2": "scan_2",
        #     "_background_": "background",
        #     "_back_large_manual_log": "log",
        # }
        #
        # files = {}
        # for file in raw_files:
        #     types = file_type
        #     for pattern in file_type:
        #         if pattern in file.name:
        #             files[file_type[pattern]] = file
        #             del types[pattern]
        #             break
        #
        # return files

    def get_dates(self) -> List[str]:
        """
        return the list of dates
        """
        return list(self.content.keys())

    def get_last_background_before(self, max_date: datetime) -> Path:
        sorted_dates = sorted(
            [
                (datetime.strptime(a_date, "%Y%m%d_%H%M"), a_date)
                for a_date in self.get_dates()
            ]
        )
        not_after = list(filter(lambda d: d[0] < max_date, sorted_dates))
        last_date, last_date_str = not_after[-1]
        return self.content[last_date_str]["final_background"]

    def get_last_raw_backgrounds_before(self, max_date: datetime) -> List[Path]:
        sorted_dates = sorted(
            [
                (datetime.strptime(a_date, "%Y%m%d_%H%M"), a_date)
                for a_date in self.get_dates()
            ]
        )
        not_after = list(filter(lambda d: d[0] < max_date, sorted_dates))
        last_date, last_date_str = not_after[-1]
        return [
            self.content[last_date_str]["raw_background_1"],
            self.content[last_date_str]["raw_background_2"],
        ]

    def get_raw_background_file(self, scan_date: str, index: int = None) -> Path:
        """Return conventional file path for scanned background image"""
        assert scan_date in self.get_dates()
        index_str = ""
        if index:
            index_str = "_" + str(index)
        return Path(self.path, scan_date + "_back_large_raw" + index_str + ".tif")

    def get_processed_background_file(self, scan_date: str, index: int = None) -> Path:
        """Return conventional file path for scanned background image"""
        assert scan_date in self.get_dates()
        index_str = ""
        if index:
            index_str = "_" + str(index)
        return Path(self.path, scan_date + "_back_large" + index_str + ".tif")


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
            "_vis1.zip": "combz",
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
        # e.g. xxx _raw_1.tif
        split_name = filename.split("_raw_")
        name = split_name[0]
        # e.g. 1.tif
        id_ = int(split_name[1].split(".")[0][0])
        return name, id_

    def get_names(self) -> list[dict[str, Union[int, str]]]:
        names = []
        for file in self.get_samples():
            name, id_ = self.extract_sample_name(file)
            nameid = {"name": name, "id": id_}
            names.append(nameid)
        return names

    def get_file(self, name: str, index: int) -> Path:
        return Path(self.path, f"{name}_raw_{index}.tif")


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
