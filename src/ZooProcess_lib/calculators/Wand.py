from enum import Enum

import numpy as np


class Dir(Enum):
    UP = 0
    DOWN = 1
    UP_OR_DOWN = 2
    LEFT = 3
    RIGHT = 4
    LEFT_OR_RIGHT = 5
    NA = 6


class Wand:
    """This class implements ImageJ's wand (tracing) tool."""

    def __init__(self, image: np.ndarray):
        """Constructs a Wand object from a numpy image."""
        self.npoints: int = 0
        self.max_points: int = 1000  # Will be increased if necessary
        self.xpoints: np.ndarray = np.zeros(self.max_points, dtype=int)
        self.ypoints: np.ndarray = np.zeros(self.max_points, dtype=int)
        self.image: np.ndarray = image

        self.height, self.width = image.shape
        self.lower_threshold: float = 0.0
        self.upper_threshold: float = 6

    def _get_byte_pixel(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.image[y, x])
        else:
            return float("inf")

    def _inside(self, x: int, y: int) -> bool:
        value = self._get_byte_pixel(x, y)
        return self.lower_threshold <= value <= self.upper_threshold

    def is_line(self, xs: int, ys: int) -> bool:
        """Are we tracing a one pixel wide line?"""
        return False
        r = 5
        xmin = xs
        xmax = min(xs + 2 * r, self.width - 1)
        ymin = max(ys - r, 0)
        ymax = min(ys + r, self.height - 1)
        area = 0
        inside_count = 0
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                area += 1
                if self._inside(x, y):
                    inside_count += 1
        if False:  # Equivalent to IJ.debugMode
            print(
                f"{'line ' if (inside_count / area >= 0.75) else 'blob '} {inside_count} {area} {inside_count / area:.2f}"
            )
        return inside_count / area >= 0.75

    def auto_outline(
        self,
        startX: int,
        startY: int,
    ):
        """
        Traces the boundary of an area of uniform color
        starting at (startX, startY).
        Args:
            startX: The starting x-coordinate inside the area.
            startY: The starting y-coordinate inside the area.
        """
        self.npoints = 0
        x = startX
        y = startY
        direction: Dir

        self.lower_threshold = self.upper_threshold = self._get_byte_pixel(x, y)
        x += 1
        while self._inside(x, y):
            x += 1
        if self.is_line(x, y):
            self.lower_threshold = self.upper_threshold = self._get_byte_pixel(x, y)
            direction = Dir.UP
        else:
            if not self._inside(x - 1, y - 1):
                direction = Dir.RIGHT
            elif self._inside(x, y - 1):
                direction = Dir.LEFT
            else:
                direction = Dir.DOWN
        self._trace_edge(x, y, direction)

    def _trace_edge(self, xstart: int, ystart: int, starting_direction: Dir):
        table = [  # 1234, 1=upper left pixel,  2=upper right, 3=lower left, 4=lower right
            Dir.NA,  # 0000, should never happen
            Dir.RIGHT,  # 000X,
            Dir.DOWN,  # 00X0
            Dir.RIGHT,  # 00XX
            Dir.UP,  # 0X00
            Dir.UP,  # 0X0X
            Dir.UP_OR_DOWN,  # 0XX0 Go up or down depending on current direction
            Dir.UP,  # 0XXX
            Dir.LEFT,  # X000
            Dir.LEFT_OR_RIGHT,  # X00X  Go left or right depending on current direction
            Dir.DOWN,  # X0X0
            Dir.RIGHT,  # X0XX
            Dir.LEFT,  # XX00
            Dir.LEFT,  # XX0X
            Dir.DOWN,  # XXX0
            Dir.NA,  # XXXX Should never happen
        ]
        index: int
        new_direction: Dir
        x = xstart
        y = ystart
        direction = starting_direction
        count = 0

        ul = self._inside(x - 1, y - 1)  # upper left
        ur = self._inside(x, y - 1)  # upper right
        ll = self._inside(x - 1, y)  # lower left
        lr = self._inside(x, y)  # lower right

        while True:
            index = 0
            if lr:
                index |= 1
            if ll:
                index |= 2
            if ur:
                index |= 4
            if ul:
                index |= 8
            new_direction = table[index]

            if new_direction == Dir.UP_OR_DOWN:
                new_direction = Dir.UP if direction == Dir.RIGHT else Dir.DOWN
            if new_direction == Dir.LEFT_OR_RIGHT:
                new_direction = Dir.LEFT if direction == Dir.UP else Dir.RIGHT

            if new_direction != direction:
                self.xpoints[count] = x
                self.ypoints[count] = y
                count += 1
                if count == self.xpoints.size:
                    xtemp = np.zeros(self.max_points * 2, dtype=int)
                    ytemp = np.zeros(self.max_points * 2, dtype=int)
                    np.copyto(xtemp[: self.max_points], self.xpoints)
                    np.copyto(ytemp[: self.max_points], self.ypoints)
                    self.xpoints = xtemp
                    self.ypoints = ytemp
                    self.max_points *= 2

            if new_direction == Dir.UP:
                y -= 1
                ll = ul
                lr = ur
                ul = self._inside(x - 1, y - 1)
                ur = self._inside(x, y - 1)
            elif new_direction == Dir.DOWN:
                y += 1
                ul = ll
                ur = lr
                ll = self._inside(x - 1, y)
                lr = self._inside(x, y)
            elif new_direction == Dir.LEFT:
                x -= 1
                ur = ul
                lr = ll
                ul = self._inside(x - 1, y - 1)
                ll = self._inside(x - 1, y)
            elif new_direction == Dir.RIGHT:
                x += 1
                ul = ur
                ll = lr
                ur = self._inside(x, y - 1)
                lr = self._inside(x, y)

            direction = new_direction
            if x == xstart and y == ystart and direction == starting_direction:
                break

        self.npoints = count
