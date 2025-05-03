from typing import List, Tuple, Sequence, Generator, Set, Optional

import cv2
import numpy as np
from numpy import ndarray

from .ConnectedComponents import ConnectedComponentsSegmenter
from ..ROI import ROI
from ..img_tools import cropnp, saveimage
from ..tools import graph_connected_components, ColumnTemporarilySet


# TODO:
#  If while joining left&right a touching object is seen, don't join the bitmaps?
#  A code profiler better than IntelliJ one
#  draw_contours: if split shape, don't FLOOD_FILL before another FLOOD_FILL same place
#  RectOfI constructor is weird


class RectOfI:
    """Another rectangular region with interesting information"""

    __slots__ = ("contours", "x", "y", "w", "h")
    contours: List[ndarray]  # from openCV
    x: int
    y: int
    w: int
    h: int

    def __init__(
        self, contour: Optional[ndarray] = None, dimension: Optional[List[int]] = None
    ):
        if contour is not None:
            self.contours = [contour]
            self.x, self.y, self.w, self.h = (
                dimension if dimension is not None else cv2.boundingRect(contour)
            )
        else:
            self.contours = []
            self.x, self.y, self.w, self.h = 0, 0, 0, 0

    def touching(self, height: int, width: int):
        return (
            self.x == 0
            or self.y == 0
            or self.x + self.w == width
            or self.y + self.h == height
        )

    @property
    def area(self):
        return self.w * self.h

    def is_inside(self, other: "RectOfI") -> bool:
        return (
            self.x > other.x
            and self.y > other.y
            and self.x + self.w < other.x + other.w
            and self.y + self.h < other.y + other.h
        )

    def __repr__(self):
        ret = f"{self.__class__.__name__}({self.x}, {self.y}, {self.w}, {self.h}):"
        ret += str([cv2.boundingRect(a_c) for a_c in self.contours])
        return ret

    @classmethod
    def from_contours_and_dimensions(
        cls, contours_and_dims: List[Tuple[ndarray, List[int]]]
    ):
        return [RectOfI(contour, dimension) for contour, dimension in contours_and_dims]


PotentialROI_T = Tuple[RectOfI, ndarray, int]


class ExternalContoursSegmenter:
    single_point_contour_shape = (1, 1, 2)
    single_line_contour_shape = (2, 1, 2)

    @classmethod
    def find_particles_contours(
        cls,
        inv_mask: np.ndarray,
        s_p_min: int,
        s_p_max: int,
        max_w_to_h_ratio: float,
        split,
    ) -> Tuple[List[ROI],List[int]]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        height, width = inv_mask.shape[:2]

        filtering_stats = [0] * 8
        contours = cls.find_contours(inv_mask, split, filtering_stats)
        if not split and len(contours) <= 1:
            print("0 or 1 contour!")
            return ConnectedComponentsSegmenter().find_particles_via_cc(
                inv_mask,
                s_p_min,
                s_p_max,
                max_w_to_h_ratio,
                with_split=False,
            )

        print(
            "Number of RETR_EXTERNAL Contours found = ",
            len(contours) + filtering_stats[0],
        )
        potential_ROIs: List[PotentialROI_T] = []
        split_rects = set()
        for a_rect in contours:
            # Basic numpy shape-based filtering was done in find_contours
            if a_rect.touching(height, width):
                filtering_stats[1] += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            # -> don't bother drawing the contour, which is expensive
            if a_rect.area < s_p_min:
                filtering_stats[2] += 1
                continue
            contour_mask = cls.draw_contour(a_rect)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                filtering_stats[6] += 1
                continue
            ratiobxby = a_rect.w / a_rect.h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[7] += 1
                continue

            # With 'split' method, small particles completely embedded in split ones
            # can appear wrongly in the output, as they are at the same external level
            # if considering split sub-images, but are ignored if considering the whole one.
            if split and len(a_rect.contours) > 1:
                split_rects.add(a_rect)

            potential_ROIs.append((a_rect, contour_mask, area))

        # Remove particles inside split ones (by side effect of split)
        final_ROIs = cls.remove_if_embedded(potential_ROIs, split_rects)
        ret = [
            ROI(mask=contour_mask, x=contour.x, y=contour.y)
            for (contour, contour_mask, area) in final_ROIs
        ]
        print(
            "Initial contours",
            len(contours) + filtering_stats[0],
            "filter stats",
            filtering_stats,
            "left",
            len(ret),
        )
        # image_3channels = draw_contours(self.image, self.contours)
        # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret, filtering_stats

    @classmethod
    def find_contours(
        cls, inv_mask: ndarray, split: bool, stats: List[int]
    ) -> List[RectOfI]:
        height, width = inv_mask.shape
        if not split:
            raw = cls.find_contours_in_stripe(inv_mask, 0, width)
            found = cls.filter_and_objectify(raw, -2, stats, 0)
            return [contour for _, contour in found]

        # Note: Column split_w is conventionally included in _left_ part
        split_w = width * 50 // 100

        # findContours has an indeterminate behaviour around borders, arrange it's not the case
        with ColumnTemporarilySet(inv_mask, split_w + 1, 0):
            raw = cls.find_contours_in_stripe(inv_mask, 0, split_w + 1)
        left_contours = cls.filter_and_objectify(raw, split_w + 1, stats, 0)
        left_touching_right = [
            (idx, contour)
            for (idx, contour) in left_contours
            if contour.x + contour.w == split_w + 1
        ]

        with ColumnTemporarilySet(inv_mask, split_w, 0):
            raw = cls.find_contours_in_stripe(inv_mask, split_w, width)
        right_contours = cls.filter_and_objectify(
            raw, split_w + 1, stats, len(left_contours)
        )
        right_touching_left = [
            (idx, contour)
            for (idx, contour) in right_contours
            if contour.x == split_w + 1
        ]

        contours = [contour for (idx, contour) in left_contours + right_contours]

        center_contours = cls.find_common_contours(
            left_touching_right, right_touching_left, split_w, height
        )

        # lst = []
        # for a_center_contour in center_contours:
        #     lst.extend(a_center_contour)
        # assert len(lst) == len(right_touching_left) + len(left_touching_right)
        # cls.dbg_contours(inv_mask, [contours[c] for c in lst])
        # cls.dbg_contours(inv_mask, contours)

        useless_idxs = set()
        for a_center_contour in center_contours:
            common = sorted(a_center_contour, reverse=False)
            cover = cls.compose_contour([contours[c] for c in common])
            # print("__>", theo)
            contours[common[0]] = cover
            useless_idxs.update(common[1:])

        for idx in sorted(useless_idxs, reverse=True):
            del contours[idx]

        return contours

    @classmethod
    def filter_and_objectify(cls, raw, border_x, stats, idx_start):
        """Basically just join contours with their most useful
        calculated information AKA bounding rectangle"""
        dimensions = [cv2.boundingRect(a_contour) for a_contour in raw]
        found, saved = cls.filter_not_a_particle(raw, dimensions, border_x)
        found = RectOfI.from_contours_and_dimensions(found)
        stats[0] += saved
        ret = list(enumerate(found, start=idx_start))
        return ret

    @classmethod
    def find_contours_in_stripe(
        cls, inv_mask: ndarray, from_x: int, to_x: int
    ) -> List[ndarray]:
        """Find the contours in the stripe defined by from_x:to_x.
        All returned contours are in inv_mask coordinate system."""
        height, width = inv_mask.shape
        sub_mask = cropnp(image=inv_mask, top=0, left=from_x, bottom=height, right=to_x)
        contours, _ = cv2.findContours(
            sub_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
            offset=(from_x, 0),
        )
        return contours

    @staticmethod
    def draw_contour(contour: RectOfI) -> np.ndarray:
        contour_canvas = np.zeros((contour.h, contour.w), np.uint8)
        for idx, a_contour in enumerate(contour.contours):
            # contour_canvas[:,:] = 0
            # Contours overlap so sending the whole gives wrong results, see note in opencv doc
            cv2.drawContours(
                image=contour_canvas,
                contours=[a_contour],
                contourIdx=-1,
                color=(1,),
                thickness=cv2.FILLED,
                offset=(-contour.x, -contour.y),
            )
            # saveimage(
            #     contour_canvas * 255,
            #     f"/tmp/zooprocess/contour_{contour.x}_{contour.y}_p{idx}.png",
            # )
        if len(contour.contours) > 1:
            # There can be holes at the frontier of the split contour, they
            # appear inside the whole shape but can be outside of split ones
            # incase they are on the separation line
            (whole, _) = cv2.findContours(
                contour_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            assert len(whole) == 1
            cv2.drawContours(
                image=contour_canvas,
                contours=whole,
                contourIdx=-1,
                color=(1,),
                thickness=cv2.FILLED,
            )
        # saveimage(
        #     contour_canvas * 255,
        #     f"/tmp/zooprocess/contour_{contour.x}_{contour.y}_p{idx}.png",
        # )
        return contour_canvas

    @classmethod
    def filter_not_a_particle(
        cls, contours: Sequence[ndarray], dimensions: List[List[int]], border: int
    ) -> tuple[list[tuple[ndarray, list[int]]], int]:
        length_before = len(contours)
        ret = [
            (a_contour, a_dimension)
            for a_contour, a_dimension in zip(contours, dimensions)
            if a_contour.shape
            not in (cls.single_point_contour_shape, cls.single_line_contour_shape)
            or a_dimension[0] == border  # x
            or a_dimension[0] + a_dimension[2] == border  # x + w
        ]
        return ret, length_before - len(ret)

    @classmethod
    def find_common_contours(
        cls,
        l_contours: List[Tuple[int, RectOfI]],
        r_contours: List[Tuple[int, RectOfI]],
        frontier: int,
        height: int,
    ) -> List[Set[int]]:
        if len(l_contours) == 0 or len(r_contours) == 0:
            return []
        # TODO: dup code of ConnectedComponents.find_common_cc_regions
        left_zone = cls.contact_zone(l_contours, frontier, height)
        right_zone = cls.contact_zone(r_contours, frontier + 1, height)
        (l_dc,) = np.nonzero(left_zone)
        (r_dc,) = np.nonzero(right_zone)
        assert len(l_dc) != 0 and len(r_dc) != 0
        same_contours = set()
        for offs in (-1, 0, 1):  # 8 connectivity
            in_contact = np.intersect1d(l_dc, r_dc + offs)
            contact_idxs_left = left_zone[in_contact]
            contact_idxs_right = right_zone[in_contact - offs]
            same_contours_offs = (
                np.unique(
                    np.column_stack((contact_idxs_left, contact_idxs_right)), axis=0
                )
                - 1  # Rollback the "+1" in cls.contact_zone
            )
            same_contours.update(list(map(tuple, same_contours_offs)))
        same_contours = sorted(list(same_contours))
        # Do connected components on the graph of neighbour contours to group them
        connections = ConnectedComponentsSegmenter.prepare_graph(same_contours)
        return graph_connected_components(connections)

    @classmethod
    def contact_zone(cls, rects: List[RectOfI], limit: int, height: int):
        zone = np.zeros(height, np.uint32)
        for idx, a_contour in rects:
            for from_y, to_y in cls.segments_touching(a_contour.contours[0], limit):
                if from_y == to_y:
                    zone[to_y] = idx + 1
                else:
                    zone[from_y : to_y + 1] = idx + 1
        return zone

    @classmethod
    def segments_touching(
        cls, contour: ndarray, x_limit: int
    ) -> Generator[Tuple[int, int], None, None]:
        from_x, from_y = -1, -1
        contour = contour.tolist()
        for a_point in contour + contour[:1]:
            to_x, to_y = a_point[0]
            if to_x == x_limit:
                if from_x == x_limit:
                    if to_y > from_y:
                        yield from_y, to_y
                    else:
                        yield to_y, from_y
                else:
                    yield to_y, to_y
            from_x, from_y = to_x, to_y

    @classmethod
    def compose_contour(cls, contours: List[RectOfI]) -> RectOfI:
        """Return a pseudo-contour enclosing all given ones"""
        min_x, min_y = 1000000, 1000000
        max_x, max_y = -1, -1
        all_contours = []
        for a_contour in contours:
            min_x = min(min_x, a_contour.x)
            min_y = min(min_y, a_contour.y)
            max_x = max(max_x, a_contour.x + a_contour.w)
            max_y = max(max_y, a_contour.y + a_contour.h)
            all_contours.extend(a_contour.contours)
        contours_cp = []
        for a_contour in all_contours:
            cp = np.copy(a_contour)
            contours_cp.append(cp)
        ret = RectOfI()
        ret.contours = contours_cp
        ret.x, ret.y, ret.w, ret.h = min_x, min_y, max_x - min_x, max_y - min_y
        return ret

    @classmethod
    def remove_if_embedded(
        cls, potentialROIs: List[PotentialROI_T], split_rects: Set
    ) -> List[PotentialROI_T]:
        """Remove from potentialROIs the elements included in split_rects."""
        # Optimize by sorting by area, bigger area cannot be included in smaller one
        p_rois = sorted(potentialROIs, key=lambda p_roi: p_roi[2], reverse=True)
        to_remove = set()
        for i, mbder in enumerate(p_rois):
            rect_of_i_big, mask_big, _ = mbder
            if rect_of_i_big not in split_rects:
                continue
            for j in range(i, len(p_rois)):
                rect_of_i_small, mask_small, _ = p_rois[j]
                if not rect_of_i_small.is_inside(rect_of_i_big):
                    continue
                x_start = int(np.argmax(mask_small != 0))
                small_in_big_x, small_in_big_y = (
                    rect_of_i_small.x - rect_of_i_big.x,
                    rect_of_i_small.y - rect_of_i_big.y,
                )
                if mask_big[small_in_big_y, small_in_big_x + x_start] != 0:
                    to_remove.add(rect_of_i_small)
        return [a_p_roi for a_p_roi in potentialROIs if a_p_roi[0] not in to_remove]

    @classmethod
    def debug_contours(cls, inv_mask, contours):
        dbg = np.zeros_like(inv_mask, dtype=np.uint8)
        for a_contour in contours:
            for a_ocv in a_contour.contours:
                cv2.drawContours(
                    image=dbg,
                    contours=[a_ocv],
                    contourIdx=0,
                    color=(1,),
                    thickness=1,
                )
        saveimage(255 - dbg * 255, f"/tmp/zooprocess/contours{len(contours)}.jpg")
