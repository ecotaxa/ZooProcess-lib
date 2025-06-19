import dataclasses
from configparser import ConfigParser

from pathlib import Path


# Full dump of a config file, for future use:
# _Used_:
# background_process= last
# minsizeesd_mm= 0.001
# maxsizeesd_mm= 0.001
# longline_mm= 0.001
# resolution= 1000
# upper= 243
#
# _Unused_:
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
