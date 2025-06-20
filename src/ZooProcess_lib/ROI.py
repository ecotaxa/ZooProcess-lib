import dataclasses

from numpy import ndarray

ecotaxa_tsv_unq = lambda f: (
    f["object_bx"],
    f["object_by"],
    f["object_width"],
    f["object_height"],
)
roi_unq = lambda r: (r.x, r.y, r.mask.shape[1], r.mask.shape[0])


@dataclasses.dataclass(frozen=False)
class ROI(object):
    """Region of interest (ROI)"""

    # Position of the ROI inside its image
    x: int
    y: int
    mask: ndarray  # convention: 0=not in particle, 1=particle and eventual holes inside


def unique_visible_key(roi: ROI) -> str:
    height, width = roi.mask.shape[:2]
    keys = [hex(m)[2:].upper() for m in [roi.x, roi.y, width, height]]
    return f"x{keys[0]}y{keys[1]}w{keys[2]}h{keys[3]}"
