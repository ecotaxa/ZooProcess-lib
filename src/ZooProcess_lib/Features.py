from __future__ import annotations

import math
from functools import cached_property
from typing import Dict, List, Callable, Any, Set, Optional, get_type_hints, Type

import cv2
import numpy as np
from numpy import ndarray
from scipy import stats

from .ROI import ROI
from .calculators.Calculater import Calculater
from .calculators.Custom import (
    fractal_mp,
    ij_perimeter,
)
from .calculators.EllipseFitter import EllipseFitter
from .calculators.symmetry import imagej_like_symmetry
from .img_tools import cropnp

TO_LEGACY: Dict[
    str, Callable
] = {}  # Key=legacy property name, Value=bound method to call
TYPE_BY_LEGACY: Dict[
    str, Type
] = {}  # Key=legacy property name, Value=a type e.g. int of np.float64
TO_ECOTAXA: Dict[
    str, Callable
] = {}  # Key=ecotaxa feature name, Value=bound method to call
TYPE_BY_ECOTAXA: Dict[
    str, Type
] = {}  # Key=ecotaxa feature name, Value=a type e.g. int of np.float64

# Unicity inside a list of measures/features
feature_unq = lambda f: (f["BX"], f["BY"], f["Width"], f["Height"])


def legacy(name: str) -> Callable[[Any], Callable[[tuple[Any, ...]], None]]:
    def wrap(f):
        assert name not in TO_LEGACY
        ret_type = get_type_hints(f).get("return")
        assert ret_type is not None
        TO_LEGACY[name] = f
        TYPE_BY_LEGACY[name] = ret_type
        ecotaxa_name = "object_" + name.lower()
        TO_ECOTAXA[ecotaxa_name] = f
        TYPE_BY_ECOTAXA[ecotaxa_name] = ret_type

        def wrapped_f(*args):
            return f(*args)

        return wrapped_f

    return wrap


class Features(object):
    """
    A set of computations made on a crop (AKA vignette) extracted from an image.
    """

    def __init__(self, image: ndarray, resolution: int, roi: ROI, threshold: int):
        self.mask = roi.mask
        self.image = image
        self.resolution = resolution
        self.bx = roi.x
        self.by = roi.y
        # params
        self.threshold = threshold

    def as_measures(self, only: Optional[Set[str]] = None) -> Dict[str, int | float]:
        """Return present object as a dictionary, for comparison & other needs.
        Only measures, i.e. not derived features, are returned."""
        ret = {}
        for leg, fct in TO_LEGACY.items():
            if only and leg not in only:
                continue
            if leg == leg.lower():
                continue
            ret[leg] = fct(self)
        return ret

    def as_ecotaxa(self, only: Optional[Set[str]] = None) -> Dict[str, int | float]:
        """Return present object as (part of) an Ecotaxa import line"""
        ret = {}
        for nam, fct in TO_ECOTAXA.items():
            if only and nam not in only:
                continue
            val = fct(self)
            val = float(val) if isinstance(val, np.float64) else val
            ret[nam] = val
        return ret

    @legacy("BX")
    def bx(self) -> int:
        """X coordinate of the top left point of the image in the smallest rectangle enclosing the object."""
        return self.bx

    @legacy("BY")
    def by(self) -> int:
        """Y coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.by

    @cached_property
    @legacy("X")
    def x_centroid(self) -> np.float64:
        """X position of the center of gravity of the object in the smallest rectangle enclosing the object"""
        fg_coords_y, fg_coords_x = np.nonzero(self.mask)
        return np.sum(fg_coords_x) / fg_coords_x.shape[0] + 0.5

    @cached_property
    @legacy("Y")
    def y_centroid(self) -> np.float64:
        """Y position of the center of gravity of the object in the smallest rectangle enclosing the object"""
        fg_coords_y, fg_coords_x = np.nonzero(self.mask)
        return np.sum(fg_coords_y) / fg_coords_y.shape[0] + 0.5

    @cached_property
    @legacy("XStart")
    def x_start(self) -> int:
        """X coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.bx + int(np.argmax(self.mask != 0))

    @property
    @legacy("YStart")
    def y_start(self) -> int:
        """Y coordinate of the top left point of the image in the smallest rectangle enclosing the object"""
        return self.by

    @cached_property
    @legacy("Width")
    def width(self) -> int:
        """Width of the smallest rectangle enclosing the object"""
        return int(self.mask.shape[1])

    @cached_property
    @legacy("Height")
    def height(self) -> int:
        """Height of the smallest rectangle enclosing the object"""
        return int(self.mask.shape[0])

    @cached_property
    @legacy("Area")
    def area(self) -> int:
        """Surface area of the object in square pixels"""
        return int(np.count_nonzero(self.mask))

    @cached_property
    @legacy("Area_exc")
    def area_exc(self) -> int:
        """Zooscan, FlowCam and Generic : Surface area of the object excluding holes, in square pixels (=Area*(1-(%area/100))
        UVP5 and UVP6 : Surface area of the holes in the object, in square pixels (=Area*(1-(%area/100))
        """
        return int(np.count_nonzero(self._mask_with_holes))

    @cached_property
    @legacy("%Area")
    def pct_area(self) -> float:
        """Percentage of object’s surface area that is comprised of holes, defined as the background grey level"""
        nb_holes = np.count_nonzero(self._crop <= self.threshold)
        ret = 100 - nb_holes * 100 / self.area
        return ret

    @property
    @legacy("Major")
    def major(self) -> float:
        """Primary axis of the best fitting ellipse for the object"""
        return self._ellipse_fitter.major

    @property
    @legacy("Minor")
    def minor(self) -> float:
        """Primary axis of the best fitting ellipse for the object"""
        return self._ellipse_fitter.minor

    @property
    @legacy("Angle")
    def angle(self) -> float:
        """Angle between the primary axis and a line parallel to the x-axis of the image"""
        return self._ellipse_fitter.angle

    @cached_property
    @legacy("Feret")
    def feret(self) -> np.float64:
        """Maximum feret diameter, i.e., the longest distance between any two points along the object boundary"""
        feret_calc = Calculater(self.mask, True)
        feret_calc.calculate_maxferet()
        return feret_calc.maxf

    @cached_property
    @legacy("Fractal")
    def fractal(self) -> float:
        """Fractal dimension of object boundary (Berube and Jebrak 1999), calculated using the ‘Sausage’ method and the Minkowski dimension"""
        ret, _ = fractal_mp(self._mask_with_holes)
        return ret

    @cached_property
    @legacy("Perim.")
    def perim(self) -> float:
        """The length of the outside boundary of the object [pixel]"""
        ret = ij_perimeter(self.mask)
        return ret

    @cached_property
    @legacy("Min")
    def min(self) -> int:
        """Minimum grey value within the object (0 = black)"""
        return int(np.min(self._stats_basis))

    @cached_property
    @legacy("Max")
    def max(self) -> int:
        """Maximum grey value within the object (255 = white)"""
        return int(np.max(self._stats_basis))

    @cached_property
    @legacy("Median")
    def median(self) -> int:
        """Median grey value within the object"""
        # Legacy computes an 'integer median' which is never a xxx.5
        hist = self._histogram
        sums = np.cumsum(hist)
        half = np.sum(hist) / 2
        ret = np.argmax(sums > half)
        return int(ret)

    @cached_property
    @legacy("Mean")
    def mean(self) -> np.float64:
        """Average grey value within the object; sum of the grey values of all pixels in the object divided by the number of pixels"""
        ret = np.mean(self._stats_basis)
        return ret

    @cached_property
    @legacy("Mode")
    def mode(self) -> int:
        """Modal grey value within the object"""
        mode = stats.mode(self._stats_basis, axis=None)
        return int(mode.mode)

    @cached_property
    @legacy("Skew")
    def skew(self) -> np.float64:
        """Skewness of the histogram of grey level values"""
        return stats.skew(self._stats_basis)

    @cached_property
    @legacy("Kurt")
    def kurtosis(self) -> np.float64:
        """Kurtosis of the histogram of grey level values"""
        return stats.kurtosis(self._stats_basis)

    @cached_property
    @legacy("StdDev")
    def stddev(self) -> np.float64:
        """Standard deviation of the grey value used to generate the mean grey value"""
        return np.std(self._stats_basis, ddof=1)

    @cached_property
    @legacy("IntDen")
    def intden(self) -> int:
        """Integrated density. This is the sum of the grey values of the pixels in the object (i.e. = Area*Mean)"""
        return int(np.sum(self._stats_basis, axis=0))

    @cached_property
    # @legacy("ThickR")
    def thickr(self) -> int:  # pragma: no cover
        """Thickness ratio : relation between the maximum thickness of an object and the average thickness of the object excluding the maximum.
        TODO: Fix with new specs"""
        return self._symmetry[2]

    @cached_property
    # @legacy("Symetrieh")
    def symmetry_h(self) -> int:  # pragma: no cover
        """Bilateral horizontal symmetry index.
        TODO: Fix with new specs"""
        return self._symmetry[1]

    @cached_property
    # @legacy("Symetriev")
    def symmetry_v(self) -> int:  # pragma: no cover
        """Bilateral vertical symmetry index.
        TODO: Fix with new specs"""
        return self._symmetry[0]

    @cached_property
    @legacy("Convarea")
    def convarea(self) -> int:
        """The area of the smallest polygon within which all points in the object fit"""
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        assert len(contours) == 1
        hull = cv2.convexHull(contours[0])
        ret = cv2.contourArea(hull) + cv2.arcLength(hull, True)
        return int(ret)

    @cached_property
    def _crop(self):
        crop = cropnp(
            self.image,
            top=self.by,
            left=self.bx,
            bottom=self.by + self.height,
            right=self.bx + self.width,
        )
        return np.bitwise_or(crop, 255 - self.mask * 255)

    @cached_property
    def _mask_with_holes(self):
        return 1 - (self._crop > self.threshold).astype(np.uint8)

    @cached_property
    def _ellipse_fitter(self):
        ret = EllipseFitter()
        ret.fit(self.mask)
        return ret

    @cached_property
    def _stats_basis(self):
        """Dataset used for statistical functions"""
        vals = self._crop[self.mask > 0]
        return vals

    @cached_property
    def _histogram(self):
        (hist, _) = np.histogram(self._stats_basis, 256, range=(0, 255))
        return hist

    @cached_property
    def _symmetry(self):  # pragma: no cover
        """TODO: Fix with new specs"""
        symmetry_h, symmetry_v, thick_ratio = imagej_like_symmetry(
            mask=self.mask * 255,
            x_centroid=self.x_centroid,
            y_centroid=self.y_centroid,
            angle=self.angle,
            area=self.area,
            pixel_size=25.4 / self.resolution,
            coords=(self.bx, self.by),
        )
        return symmetry_h, symmetry_v, thick_ratio

    #
    # Derived (from others) features, no need to cache them
    #
    @property
    @legacy("esd")
    def esd(self) -> float:
        """Equivalent Spherical Diameter = 2 * SQR(Area / Pi)"""
        return 2 * math.sqrt(self.area / math.pi)

    @property
    @legacy("elongation")
    def elongation(self) -> float:
        """major / minor (‘ellipse' elongation)"""
        return self.major / self.minor

    @property
    @legacy("range")
    def range(self) -> int:
        """max - min"""
        return self.max - self.min

    @property
    @legacy("meanpos")
    def meanpos(self) -> float:
        """(mean-max) / (mean-min)"""
        return (self.mean - self.max) / (self.mean - self.min)

    @property
    @legacy("cv")
    def cv(self) -> np.float64:
        """100*(stddev/mean)"""
        return 100 * (self.stddev) / self.mean

    @property
    @legacy("sr")
    def sr(self) -> float:
        """100*(stddev/(max-min))"""
        if self.max - self.min != 0:
            return 100 * (self.stddev / (self.max - self.min))
        else:
            return np.nan

    @property
    @legacy("perimareaexc")
    def perimareaexc(self) -> float:
        """perim/(sqrt(area_exc))"""
        return self.perim / math.sqrt(self.area_exc)

    @property
    @legacy("feretareaexc")
    def feretareaexc(self) -> float:
        """feret/(sqrt(area_exc))"""
        return self.feret / math.sqrt(self.area_exc)

    @property
    @legacy("perimmajor")
    def perimmajor(self) -> float:
        """perim/major"""
        return self.perim / self.major

    @property
    @legacy("perimferet")
    def perimferet(self) -> float:
        """perim/feret"""
        return self.perim / self.feret

    @property
    @legacy("circex")
    def circex(self) -> float:
        """(4*PI*area_exc)/(pow(perim,2))"""
        return (4 * math.pi * self.area_exc) / (self.perim**2)

    @property
    @legacy("Circ.")
    def circ(self) -> float:
        """Circularity = (4 * Pi * Area) / Perim^2) ; a value of 1 indicates a perfect circle, a value approaching 0 indicates an increasingly elongated polygon"""
        return (4 * math.pi * self.area) / (self.perim**2)


FeaturesListT = List[Features]


class FeaturesCalculator(object):
    """Features calculator, just to encapsulate a few primitives on several ROIs."""

    def __init__(self, threshold: int):
        self.threshold = threshold

    def legacy_measures_list_from_roi_list(
        self,
        image: ndarray,
        resolution: int,
        roi_list: List[ROI],
        only: Optional[Set[str]] = None,
    ) -> list[dict[str, int | float]]:
        return [
            Features(image, resolution, p, self.threshold).as_measures(only)
            for p in roi_list
        ]
