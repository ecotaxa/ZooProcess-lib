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
                value = a_field.type(parser.get("conf", a_field.name))
                args.append(value)
            return ZooscanConfig(*args)
