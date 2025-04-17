import dataclasses

from numpy import ndarray

feature_unq = lambda f: (f["BX"], f["BY"], f["Width"], f["Height"])


@dataclasses.dataclass(frozen=True)
class ROI(object):
    """ Region of interest (ROI) """
    # Position of the ROI inside its image
    x: int
    y: int
    mask: ndarray  # convention: 0=not in particle, 1=particle and inside
