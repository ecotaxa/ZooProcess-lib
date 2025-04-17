from __future__ import annotations

from decimal import Decimal
from typing import Dict, TypedDict, List, Callable, Any

import numpy as np
from numpy import ndarray

from ZooProcess_lib.EllipseFitter import EllipseFitter
from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.img_tools import cropnp

TO_LEGACY = {}


def legacy(name: str) -> Callable[[Any], Callable[[tuple[Any, ...]], None]]:
    def wrap(f):
        assert name not in TO_LEGACY
        TO_LEGACY[name] = f

        def wrapped_f(*args):
            return f(*args)

        return wrapped_f

    return wrap


class Features_old(TypedDict):
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


class Features(object):
    """
    A set of computations made on a crop (AKA vignette) extracted from an image.
    """

    def __init__(self, image: ndarray, roi: ROI, threshold: int):
        self.mask = roi.mask
        self.image = image
        self.bx = roi.x
        self.by = roi.y
        # params
        self.threshold = threshold
        # cache
        self._ellipse_fitter = None

    def as_legacy(self) -> Dict[str, int | float]:
        """Return present object as a legacy dictionary, for comparison & other needs"""
        ret = {}
        for leg, fct in TO_LEGACY.items():
            ret[leg] = fct(self)
        return ret

    @legacy("BX")
    def bx(self):
        """X coordinate of the top left point of the image in the smallest rectangle enclosing the object."""
        return self.bx

    @legacy("BY")
    def by(self):
        """Y coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.by

    @property
    @legacy("XStart")
    def x_start(self):
        """X coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.bx + int(np.argmax(self.mask != 0))

    @property
    @legacy("YStart")
    def y_start(self):
        """Y coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.by

    @property
    @legacy("Width")
    def width(self):
        """Width of the smallest rectangle enclosing the object"""
        return self.mask.shape[1]

    @property
    @legacy("Height")
    def height(self):
        """Height of the smallest rectangle enclosing the object"""
        return self.mask.shape[0]

    @property
    @legacy("Area")
    def area(self):
        """Surface area of the object in square pixels"""
        return np.count_nonzero(self.mask)

    @property
    @legacy("%Area")
    def pct_area(self):
        """Percentage of object’s surface area that is comprised of holes, defined as the background grey level"""
        crop = cropnp(
            self.image,
            top=self.by,
            left=self.bx,
            bottom=self.by + self.height,
            right=self.bx + self.width,
        )
        crop = np.bitwise_or(crop, 255 - self.mask * 255)
        nb_holes = np.count_nonzero(crop <= self.threshold)
        ret = (
            100 - Decimal(nb_holes * 100) / self.area
        )  # Need exact arithmetic due to some Java<->python rounding diff
        return float(ret)

    @property
    @legacy("Major")
    def major(self):
        """Primary axis of the best fitting ellipse for the object"""
        return self.ellipse_fitter().major

    @property
    @legacy("Minor")
    def minor(self):
        """Primary axis of the best fitting ellipse for the object"""
        return self.ellipse_fitter().minor

    @property
    @legacy("Angle")
    def angle(self):
        """Angle between the primary axis and a line parallel to the x-axis of the image"""
        return self.ellipse_fitter().angle

    def ellipse_fitter(self):
        if self._ellipse_fitter is None:
            self._ellipse_fitter = EllipseFitter()
            self._ellipse_fitter.fit(self.mask)
        return self._ellipse_fitter


FeaturesListT = List[Features]


def legacy_features_list_from_roi_list(image: ndarray, roi_list: List[ROI], threshold: int) -> list[
    dict[str, int | float]]:
    return [Features(image, p, threshold).as_legacy() for p in roi_list]
