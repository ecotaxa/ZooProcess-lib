from __future__ import annotations

from decimal import Decimal
from typing import Dict, TypedDict, List, Callable, Any, Set, Optional

import cv2
import numpy as np
from numpy import ndarray
from scipy import stats

from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.calculators.Calculater import Calculater
from ZooProcess_lib.calculators.Custom import (
    fractal_mp,
    ij_perimeter,
)
from ZooProcess_lib.calculators.EllipseFitter import EllipseFitter
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
        self._crop = None  # AKA vignette
        self._mask_with_holes = None
        self._histogram = None

    def as_legacy(self, only: Optional[Set[str]] = None) -> Dict[str, int | float]:
        """Return present object as a legacy dictionary, for comparison & other needs"""
        ret = {}
        for leg, fct in TO_LEGACY.items():
            if only and leg not in only:
                continue
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
        nb_holes = np.count_nonzero(self.crop() <= self.threshold)
        ret = (
            100 - Decimal(nb_holes * 100) / self.area
        )  # Need exact arithmetic due to some Java<->python rounding method diff
        return float(round(ret, 3))

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

    @property
    @legacy("Feret")
    def feret(self):
        """Maximum feret diameter, i.e., the longest distance between any two points along the object boundary"""
        feret_calc = Calculater(self.mask, True)
        feret_calc.calculate_minferet()
        feret_calc.calculate_maxferet()
        return float(feret_calc.maxf)

    @property
    # @legacy("Fractal")
    def fractal(self):
        """Fractal dimension of object boundary (Berube and Jebrak 1999), calculated using the ‘Sausage’ method and the Minkowski dimension"""
        ret, _ = fractal_mp(self.mask_with_holes())
        return round(ret, 1)  # TODO, see test_calculators.test_ij_like_EDM

    @property
    @legacy("Perim.")
    def perim(self):
        """The length of the outside boundary of the object [pixel]"""
        ret = ij_perimeter(self.mask)
        return ret

    @property
    @legacy("Min")
    def min(self):
        """Minimum grey value within the object (0 = black)"""
        object_values_only = self.crop()[self.mask_with_holes() > 0]
        return int(np.min(object_values_only))

    @property
    # @legacy("Max")
    def max(self):
        """Maximum grey value within the object (255 = white)"""
        # TODO: is not the ordinary max, it's computed inside Legacy
        object_values_only = self.bug_stats_basis()
        return int(np.max(object_values_only))

    @property
    # @legacy("Median")
    def median(self):
        """Median grey value within the object"""
        return round(float(np.median(self.stats_basis()))+0.5)

    @property
    @legacy("Mean")
    def mean(self):
        """Average grey value within the object; sum of the grey values of all pixels in the object divided by the number of pixels"""
        return float(np.mean(self.stats_basis()))

    @property
    @legacy("Mode")
    def mode(self):
        """Modal grey value within the object"""
        mode = stats.mode(self.stats_basis(), axis=None)
        return int(mode.mode)

    @property
    # @legacy("Skew")
    def skew(self):
        """Skewness of the histogram of grey level values"""
        return float(stats.skew(self.bug_stats_basis()))

    @property
    # @legacy("Kurt")
    def kurtosis(self):
        """Kurtosis of the histogram of grey level values"""
        object_values_only = self.crop()[self.mask > 0]
        return float(stats.kurtosis(object_values_only))

    def ellipse_fitter(self):
        if self._ellipse_fitter is None:
            self._ellipse_fitter = EllipseFitter()
            self._ellipse_fitter.fit(self.mask)
        return self._ellipse_fitter

    def crop(self):
        if self._crop is None:
            self._crop = cropnp(
                self.image,
                top=self.by,
                left=self.bx,
                bottom=self.by + self.height,
                right=self.bx + self.width,
            )
            self._crop = np.bitwise_or(self._crop, 255 - self.mask * 255)
        return self._crop

    def mask_with_holes(self):
        if self._mask_with_holes is None:
            self._mask_with_holes = 1 - (self.crop() > self.threshold).astype(np.uint8)
        return self._mask_with_holes

    def histogram(self) -> np.ndarray:
        if self._histogram is None:
            self._histogram = np.histogram(self.mask, bins=256)
        return self._histogram

    def bug_stats_basis(self):
        """There is a strong doubt on this being the OK dataset for stats on grey level features"""
        # vign = Vignette(self.image, self.bx, self.by, self.mask)
        # sym_sum = vign.symmetrical_vignette_added()
        v_flipped_mask = cv2.flip(self.mask, 1)
        # saveimage(sym_sum, Path("/tmp/sym_sum.png"))
        object_values_only = self.crop()[v_flipped_mask > 0]
        return object_values_only

    def stats_basis(self):
        """Dataset used for statistical functions"""
        vals = self.crop()[self.mask > 0]
        return vals


FeaturesListT = List[Features]


def legacy_features_list_from_roi_list(
    image: ndarray, roi_list: List[ROI], threshold: int, only: Optional[Set[str]] = None
) -> list[dict[str, int | float]]:
    return [Features(image, p, threshold).as_legacy(only) for p in roi_list]
