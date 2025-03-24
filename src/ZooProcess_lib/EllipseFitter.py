"""
This module provides functionality for fitting ellipses to ROIs.
Original implementation by Bob Rodieck, Department of Ophthalmology,
University of Washington, Seattle, WA, 98195

Translated from Java to Python with additional error handling and type safety.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray


class EllipseFitter:
    """
    This class fits an ellipse to an ROI.
    Port of ImageJ EllipseFitter.java
    """

    HALFPI = 1.5707963267949

    def __init__(self):
        """Initialize the EllipseFitter with default values."""
        # Public attributes
        self.x_center: float = 0.0  # X centroid
        self.y_center: float = 0.0  # Y centroid
        self.major: float = 0.0  # Length of major axis
        self.minor: float = 0.0  # Length of minor axis
        self.angle: float = 0.0  # Angle in degrees
        self.theta: float = 0.0  # Angle in radians

        # Private attributes
        self._bit_count: int = 0
        self._xsum: float = 0.0
        self._ysum: float = 0.0
        self._x2sum: float = 0.0
        self._y2sum: float = 0.0
        self._xysum: float = 0.0
        self._mask: Optional[ndarray] = None
        self._left: int = 0
        self._top: int = 0
        self._width: int = 0
        self._height: int = 0
        self._n: float = 0.0
        self._xm: float = 0.0  # mean x value
        self._ym: float = 0.0  # mean y value
        self._u20: float = 0.0  # central moment u20
        self._u02: float = 0.0  # central moment u02
        self._u11: float = 0.0  # central moment u11

    def fit(self, mask: ndarray) -> None:
        """
        Fits an ellipse to the given mask image.
        Args:
            mask: np image
        """
        self._mask = mask  # ip.get_mask_array()
        self._left = 0
        self._top = 0
        self._height, self._width = mask.shape[:2]
        self._get_ellipse_param()

    def _get_ellipse_param(self) -> None:
        """
        Calculate ellipse parameters from the ROI data.

        This method computes the major and minor axes, center coordinates,
        and orientation of the best-fitting ellipse.
        """
        # sqrt_pi = 1.772453851
        # a11 = a12 = a22 = m4 = z = scale = tmp = xoffset = yoffset = 0.0

        # if self._mask is None:
        #     self.major = (self._width * 2) / sqrt_pi
        #     self.minor = (self._height * 2) / sqrt_pi
        #     self.angle = 0.0
        #     self.theta = 0.0
        #     if self.major < self.minor:
        #         self.major, self.minor = self.minor, self.major
        #         self.angle = 90.0
        #         self.theta = math.pi / 2.0
        #     self.x_center = self._left + self._width / 2.0
        #     self.y_center = self._top + self._height / 2.0
        #     return

        self._compute_sums()
        self._get_moments()
        m4 = 4.0 * abs(self._u02 * self._u20 - self._u11 * self._u11)
        if m4 < 0.000001:
            m4 = 0.000001
        a11 = self._u02 / m4
        a12 = self._u11 / m4
        a22 = self._u20 / m4
        xoffset = self._xm
        yoffset = self._ym

        tmp = a11 - a22
        if tmp == 0.0:
            tmp = 0.000001
        self.theta = 0.5 * math.atan(2.0 * a12 / tmp)
        if self.theta < 0.0:
            self.theta += self.HALFPI
        if a12 > 0.0:
            self.theta += self.HALFPI
        elif a12 == 0.0:
            if a22 > a11:
                self.theta = 0.0
                a22, a11 = a11, a22
            elif a11 != a22:
                self.theta = self.HALFPI

        tmp = math.sin(self.theta)
        if tmp == 0.0:
            tmp = 0.000001
        z = a12 * math.cos(self.theta) / tmp
        self.major = math.sqrt(1.0 / abs(a22 + z))
        self.minor = math.sqrt(1.0 / abs(a11 - z))
        scale = math.sqrt(
            self._bit_count / (math.pi * self.major * self.minor)
        )  # equalize areas
        self.major *= scale * 2.0
        self.minor *= scale * 2.0
        self.angle = 180.0 * self.theta / math.pi
        if self.angle == 180.0:
            self.angle = 0.0
        if self.major < self.minor:
            self.major, self.minor = self.minor, self.major
        self.x_center = self._left + xoffset + 0.5
        self.y_center = self._top + yoffset + 0.5

    def _compute_sums(self) -> None:
        """
        Compute the sums needed for moment calculations.

        This method calculates various sums from the mask data that are
        used to compute the central moments of the ROI.
        """
        _xsum = _ysum = _x2sum = _y2sum = _xysum = _bit_count = 0

        for y in range(self._height):
            nz_line = np.nonzero(self._mask[y])[0]
            _x2sum += np.sum(nz_line * nz_line)
            x_sum_of_line = np.sum(nz_line)
            bit_count_of_line = nz_line.shape[0]

            _xsum += x_sum_of_line
            _ysum += bit_count_of_line * y
            _xysum += x_sum_of_line * y
            _y2sum += y * y * bit_count_of_line
            _bit_count += bit_count_of_line

        (
            self._xsum,
            self._ysum,
            self._x2sum,
            self._y2sum,
            self._xysum,
            self._bit_count,
        ) = (
            float(_xsum),
            float(_ysum),
            float(_x2sum),
            float(_y2sum),
            float(_xysum),
            _bit_count,
        )

    def _get_moments(self) -> None:
        """
        Calculate central moments of the ROI.

        This method computes the central moments that are used to determine
        the best-fitting ellipse parameters.
        """
        if self._bit_count == 0:
            return

        self._x2sum += 0.08333333 * self._bit_count
        self._y2sum += 0.08333333 * self._bit_count
        self._n = float(self._bit_count)
        x1 = self._xsum / self._n
        y1 = self._ysum / self._n
        x2 = self._x2sum / self._n
        y2 = self._y2sum / self._n
        xy = self._xysum / self._n
        self._xm = x1
        self._ym = y1
        self._u20 = x2 - (x1 * x1)
        self._u02 = y2 - (y1 * y1)
        self._u11 = xy - x1 * y1

    def draw_ellipse(self, img: ndarray) -> None:
        """
        Draws the ellipse on the specified image.
        Args:
            img: image to draw on
        Raises:
            ValueError: If the image processor is invalid or if the ellipse parameters are invalid
        """
        if img is None:
            raise ValueError("img cannot be None")
        if self.major == 0.0 and self.minor == 0.0:
            return

        def _sqr(x: float) -> float:
            return x * x

        xc = round(self.x_center)
        yc = round(self.y_center)
        max_y = img.shape[0]
        sint = math.sin(self.theta)
        cost = math.cos(self.theta)
        rmajor2 = 1.0 / _sqr(self.major / 2)
        rminor2 = 1.0 / _sqr(self.minor / 2)
        g11 = rmajor2 * _sqr(cost) + rminor2 * _sqr(sint)
        g12 = (rmajor2 - rminor2) * sint * cost
        g22 = rmajor2 * _sqr(sint) + rminor2 * _sqr(cost)
        k1 = -g12 / g11
        k2 = (_sqr(g12) - g11 * g22) / _sqr(g11)
        k3 = 1.0 / g11
        ymax = int(math.floor(math.sqrt(abs(k3 / k2))))

        if ymax > max_y:
            ymax = max_y
        if ymax < 1:
            ymax = 1
        ymin = -ymax

        txmin = [0] * max_y
        txmax = [0] * max_y

        lines = []

        # Precalculation and use of symmetry speeds things up
        for y in range(ymax + 1):
            j2 = math.sqrt(k2 * _sqr(y) + k3)
            j1 = k1 * y
            txmin[y] = round(j1 - j2)
            txmax[y] = round(j1 + j2)

        lines.append((xc + txmin[ymax - 1], yc + ymin))

        for y in range(ymin, ymax):
            x = txmax[-y] if y < 0 else -txmin[y]
            lines.append((xc + x, yc + y))

        for y in range(ymax, ymin, -1):
            x = txmin[-y] if y < 0 else -txmax[y]
            lines.append((xc + x, yc + y))

        cv2.polylines(img, [np.array(lines)], True, (0,), 2)
