import dataclasses
from typing import TypedDict, Optional

from numpy import ndarray


class Features(TypedDict):
    # Enclosing rectangle
    BX: int
    BY: int
    Width: int
    Height: int
    # Number of pixels inside the blob
    Area: int
    # Quoting some forum: (XStart,YStart) are the coordinates of the first boundary point
    # of particles found by the particle analyzer.
    XStart: int
    YStart: int
    # Area fraction : For thresholded images is the percentage of pixels in the image or selection
    # that have been highlighted in red using Image▷Adjust▷Threshold… [T]↑.
    # For non-thresholded images is the percentage of non-zero pixels. Uses the heading %Area.
    # %Area: int
    Major: float
    Minor: float
    Angle: float


feature_unq = lambda f: (f["BX"], f["BY"], f["Width"], f["Height"])


@dataclasses.dataclass(
    frozen=False
)  # TODO: Should be 'True', temp until ROI merge is clean
class ROI(object):
    features: Features
    mask: ndarray  # convention: 0=not in particle, 1=particle and inside
    contour: Optional[ndarray] = None


def features_are_at_same_coord(features: Features, another_blob: Features):
    return features["BX"] == another_blob["BX"] and features["BY"] == another_blob["BY"]
