import dataclasses

from numpy import ndarray

feature_unq = lambda f: (f["BX"], f["BY"], f["Width"], f["Height"])
ecotaxa_tsv_unq = lambda f: (f["object_bx"], f["object_by"], f["object_width"], f["object_height"])
roi_unq = lambda r: (r.x, r.y, r.mask.shape[1], r.mask.shape[0])


@dataclasses.dataclass(frozen=False)
class ROI(object):
    """ Region of interest (ROI) """
    # Position of the ROI inside its image
    x: int
    y: int
    mask: ndarray  # convention: 0=not in particle, 1=particle and eventual holes inside
