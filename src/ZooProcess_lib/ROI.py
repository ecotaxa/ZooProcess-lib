import dataclasses
import re

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


def box_from_key(key: str) -> tuple[int, int, int, int]:
    """
    Inverse of unique_visible_key. Extracts the bounding box coordinates from a key string.

    Args:
        key: A string in the format "x{hex_x}y{hex_y}w{hex_width}h{hex_height}"

    Returns:
        A tuple of (x, y, width, height) as integers
    """
    pattern = r"x([0-9A-F]+)y([0-9A-F]+)w([0-9A-F]+)h([0-9A-F]+)"
    match = re.match(pattern, key)
    if not match:
        raise ValueError(f"Invalid key format: {key}")
    x_hex, y_hex, width_hex, height_hex = match.groups()
    return int(x_hex, 16), int(y_hex, 16), int(width_hex, 16), int(height_hex, 16)
