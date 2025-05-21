# A .lut file contains processing directives for a whole project
import dataclasses
from configparser import ConfigParser

from pathlib import Path


class Lut:
    def __init__(self):
        self.min: int = 0
        self.max: int = 65536
        self.gamma: float = 1
        self.sens: str = "before"
        # Appeared in a later version
        self.adjust: str = "no"
        self.odrange: float = 1.8
        # Appeared in a later version
        self.ratio: float = 1.15
        # Appeared in a later version
        self.sizelimit: int = 800  # Unused?
        self.overlap: float = 0.07
        # Appeared in a later version
        self.medianchoice: str = "no"
        self.medianvalue: int = 1
        # Appeared in a later version
        self.resolutionreduct: int = 1200

    @staticmethod
    def read(path: Path) -> "Lut":
        ret = Lut()
        with open(path, "r") as f:
            lines = f.readlines()
            for a_var_name, a_line in zip(ret.__dict__.keys(), lines):
                a_var = getattr(ret, a_var_name)
                # Below for clarity and extensibility, as we could use directly the class, class(a_line)
                if isinstance(a_var, int):
                    setattr(ret, a_var_name, int(a_line))
                elif isinstance(a_var, float):
                    setattr(ret, a_var_name, float(a_line))
                elif isinstance(a_var, str):
                    setattr(ret, a_var_name, a_line.strip())
        # From legacy macro, there is a typo "od_g_range" so the code doesn't do what it should I guess
        # if ret.odrange >= 3:
        #     ret.odgrange = 1.15
        return ret


# Full dump of a config file, for future use:
# background_process= last
# enhance_thumbnail= no
# calibration= created_20241212_1022
# jpeg= 100
# zip= 0
# greycor= 4
# greytaux= 0.9
# yminref= 0
# doyofset= 150
# doxpos= 2
# xdimref_inch= 0.025
# ydimref_inch= 0.25
# dostd= 2.0
# doecart= 20.0
# subimgx= 0
# method= neutral
# upper= 243
# greyref= 174
# voxelwidth= 1
# voxelheigth= 1
# voveldepth= 1
# voxelunit= pixel
# backval= 100.0
# doxabspos_inch= 0.34
# doyabspos_inch= 4.04
# bleft= 16.0
# broll= 8
# bright= 4.0
# contrast_pourcent= 1.3
# doubloonxy_inch= 0.05
# doubloonarea_pourcent= 0.1
# greylimit= 10
# frame= both


@dataclasses.dataclass(frozen=True)
class ZooscanConfig:
    background_process: str
    minsizeesd_mm: float
    maxsizeesd_mm: float
    upper: int
    resolution: int
    longline_mm: float

    @classmethod
    def read(cls, path: Path) -> "ZooscanConfig":
        parser = ConfigParser()
        with open(path, "r") as strm:
            parser.read_string("[conf]\n" + strm.read())
            args = []
            for a_field in dataclasses.fields(cls):
                got_val = parser.get("conf", a_field.name)
                try:
                    value = a_field.type(got_val)
                except ValueError:
                    if a_field.type == int and got_val.endswith(".0"):
                        value = int(got_val[:-2])
                    else:
                        raise
                args.append(value)
            return ZooscanConfig(*args)


@dataclasses.dataclass(frozen=False)
class ProjectMeta:
    """
    Class to read and store metadata from a metadata.txt file.
    All fields are explicitly defined with proper type annotations.
    """

    # String fields
    SampleId: str = ""  # e.g. apero2023_tha_bioness_sup2000_017_st66_d_n1
    Scanop: str = ""  # e.g. adelaide_perruchon
    Ship: str = ""  # e.g. thalassa
    Scientificprog: str = ""  # e.g. apero
    Date: str = ""  # e.g. 20230704-0503
    CTDref: str = ""  # e.g. apero_bio_ctd_017
    Otherref: str = ""  # e.g. apero_bio_uvp6_017u
    Nettype: str = ""  # e.g. bioness
    Observation: str = ""  # e.g. no
    SubMethod: str = ""  # e.g. motoda
    Sample_comment: str = ""  # e.g. vol_zooprocess_saisi
    barcode: str = ""  # e.g. ape000000147
    FracId: str = ""  # e.g. d2_4_sur_4

    # Integer fields
    StationId: int = 0  # e.g. 66
    Depth: int = 0  # e.g. 99999
    Townb: int = 0  # e.g. 1
    Towtype: int = 0  # e.g. 1
    Netmesh: int = 0  # e.g. 2000
    Netsurf: int = 0  # e.g. 1
    Zmax: int = 0  # e.g. 1008
    Zmin: int = 0  # e.g. 820
    Vol: int = 0  # e.g. 357
    Fracmin: int = 0  # e.g. 2000
    Fracsup: int = 0  # e.g. 999999
    Fracnb: int = 0  # e.g. 4
    Code: int = 0  # e.g. 1
    CellPart: int = 0  # e.g. 1
    Replicates: int = 0  # e.g. 1
    VolIni: int = 0  # e.g. 1
    VolPrec: int = 0  # e.g. 1
    vol_qc: int = 0  # e.g. 1
    depth_qc: int = 0  # e.g. 1
    sample_qc: int = 0  # e.g. 1111
    net_duration: int = 0  # e.g. 20
    cable_length: int = 0  # e.g. 9999
    cable_angle: int = 0  # e.g. 99999
    cable_speed: int = 0  # e.g. 0
    nb_jar: int = 0  # e.g. 1

    # Float fields
    Latitude: float = 0.0  # e.g. 51.4322000
    Longitude: float = 0.0  # e.g. 18.3108000
    latitude_end: float = 0.0  # e.g. 51.4421000
    longitude_end: float = 0.0  # e.g. 18.3417000
    ship_speed_knots: float = 0.0  # e.g. 2

    @classmethod
    def read(cls, path: Path) -> "ProjectMeta":
        """
        Read a metadata.txt file and return a ProjectMeta instance with fields
        populated from the file.
        """
        meta = ProjectMeta()
        with open(path, "r") as strm:
            for line in strm:
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if hasattr(meta, key):
                        field_type = type(getattr(meta, key))
                        try:
                            if field_type == int:
                                setattr(meta, key, int(value))
                            elif field_type == float:
                                setattr(meta, key, float(value))
                            else:
                                setattr(meta, key, value)
                        except ValueError:
                            # If conversion fails, keep as string
                            setattr(meta, key, value)
                    else:
                        # For any fields not explicitly defined, add them as strings
                        setattr(meta, key, value)

        return meta
