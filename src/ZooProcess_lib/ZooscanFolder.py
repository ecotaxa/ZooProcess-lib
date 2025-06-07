import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Dict, TypedDict, Optional, Generator

from .LegacyConfig import Lut, ZooscanConfig, ProjectMeta
from .tools import parse_csv


class ZooscanDrive:
    """A directory with several projects inside and some conventional ones"""

    def __init__(self, drive_path: Path) -> None:
        self.path = Path(drive_path)

    def list(self) -> Generator[Path, None, None]:
        # Get all subdirectories in the drive
        for item in self.path.iterdir():
            if not item.is_dir():
                continue
            if item.name in ("Zooprocess", "Zooscan", "Background"):
                continue
            yield item

    def get_project_folder(self, project_name: str) -> "ZooscanProjectFolder":
        ret = ZooscanProjectFolder(self.path, project_name)
        assert ret.path.is_dir()
        return ret


class ZooscanProjectFolder:
    """A directory with a project inside"""

    def __init__(self, drive_path: Path, project: str) -> None:
        self.project = project
        self.name = project
        self.path = Path(drive_path, project)  # noqa: E501
        self.zooscan_scan = ZooscanScanFolder(self.path)
        self.zooscan_back = ZooscanBackFolder(self.path)
        self.zooscan_config = ZooscanConfigFolder(self.path)
        self.zooscan_meta = ZooscanMetaFolder(self.path)

    def path_old(self, folder, file) -> str:
        path = self._absolute_home_project_path + self.project_folder + folder + file
        return path

    def list_samples_with_state(self) -> list[str]:
        """
        The samples are entries in metadata CSV.
        """
        try:
            return [
                a_line["sampleid"] for a_line in self.zooscan_meta.read_samples_table()
            ]
        except (FileNotFoundError, KeyError):
            return []

    def list_scans_with_state(self) -> Generator[str, None, None]:
        """TODO: Finish inventory using:
        zooscan_lov/Zooscan_apero_tha_bioness_2_sn033$ find . -name "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2*" | sort
        Directory:
            ./Zooscan_scan/_work/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1/
        CSV lines:
            ./Zooscan_meta/zooscan_scan_header_table.csv
            ./Zooscan_meta/zooscan_scan_header_table.bak
        Files:
            Images:
                ./Zooscan_scan/_raw/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_raw_1.tif
                ./Zooscan_scan/_work/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1_vis1.zip
            Image, not really mandatory:
                ./Zooscan_scan/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1.tif
            Text files:
                ./Zooscan_scan/_raw/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1_log.bak
                ./Zooscan_scan/_raw/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1_log.txt
                ./Zooscan_scan/_raw/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1_meta.txt
                Below file should be identical to content of the scan TSV file.
                ./Zooscan_scan/_work/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1/apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2_1_meta.txt
        """
        for an_entry in self.zooscan_scan.work.path.iterdir():
            if an_entry.is_dir():
                yield an_entry.name


class ZooscanConfigFolder:
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


class ZooscanMetaFolder:
    SUDIR_PATH = "Zooscan_meta"
    PROJECT_META = "metadata.txt"
    SAMPLE_HEADER = "zooscan_sample_header_table.csv"
    SCAN_HEADER = "zooscan_scan_header_table.csv"

    def __init__(self, zooscan_folder: Path) -> None:
        self.path = Path(zooscan_folder, self.SUDIR_PATH)

    def read_project_meta(self) -> ProjectMeta:
        """Project metadata would rather be named "current metadata".
        It's a mix of real _project_ meta, which doesn't change over time
         as e.g., the ship (that all physical samples come from) is constant,
         _but_ also current values for the physical sample being scanned."""
        return ProjectMeta.read(self.path / self.PROJECT_META)

    def read_samples_table(self) -> List[Dict[str, str]]:
        return parse_csv(self.samples_table_path)

    @property
    def samples_table_path(self) -> Path:
        return self.path / self.SAMPLE_HEADER

    def read_scans_table(self) -> List[Dict[str, str]]:
        return parse_csv(self.path / self.SCAN_HEADER)

    @property
    def scans_table_path(self) -> Path:
        return self.path / self.SCAN_HEADER


class ZooscanScanFolder:
    SUBDIR_PATH = "Zooscan_scan"

    def __init__(self, project_folder: Path) -> None:
        self.path = Path(project_folder, self.SUBDIR_PATH)
        self.raw = ZooscanScanRawFolder(self.path)
        self.work = ZooscanScanWorkFolder(self.path)

    def get_file_produced_from(self, raw_file_name: str) -> Path:
        assert "_raw" in raw_file_name
        return Path(self.path, raw_file_name.replace("_raw", ""))

    def get_8bit_file(self, sample_name: str, index: int) -> Path:
        return Path(self.path, sample_name + "_" + str(index) + ".tif")

    def list_samples(self) -> Generator[str, None, None]:
        """Not really samples, rather scans i.e. subsamples, but kept with
        same name until a rename TODO"""
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


class ZooscanBackFolder:
    SUBDIR_PATH = "Zooscan_back"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self.SUBDIR_PATH)
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
        """Return a conventional file path for a scanned background image"""
        assert scan_date in self.get_dates()
        index_str = ""
        if index:
            index_str = "_" + str(index)
        return Path(self.path, scan_date + "_back_large_raw" + index_str + ".tif")

    def get_processed_background_file(self, scan_date: str, index: int = None) -> Path:
        """Return a conventional file path for a scanned background image"""
        assert scan_date in self.get_dates()
        index_str = ""
        if index:
            index_str = "_" + str(index)
        return Path(self.path, scan_date + "_back_large" + index_str + ".tif")


class ZooscanScanWorkFolder:
    SUBDIR_PATH = "_work"

    def __init__(self, zooscan_scan_raw_folder: Path) -> None:
        self.path = Path(zooscan_scan_raw_folder, self.SUBDIR_PATH)

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


class ZooscanScanRawFolder:
    SUBDIR_PATH = "_raw"

    def __init__(self, zooscan_scan_folder: Path) -> None:
        self.path = Path(zooscan_scan_folder, self.SUBDIR_PATH)

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


class ZooscanSampleScan:
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
        self.metadata: List[Metadata] = []


class ZooscanProjectOld:
    def __init__(self, projectname, home_path: Path) -> None:
        # self.scans : Array(Zooscan_sample_scan) = []
        self.scans = []

        self.project = projectname

        self.path = Path(home_path, projectname)
        # self.zooscan = ZooscanFolder(project_path.absolute(),home="",piqv="")
        # self.zooscan = ZooscanFolder(project_path.absolute(),home="",piqv="")
        self.zooscan = ZooscanProjectFolder(home_path, projectname)
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
