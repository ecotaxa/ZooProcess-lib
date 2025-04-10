import math
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import cv2
import numpy as np
from numpy import ndarray

from ..ROI import ROI
from ..img_tools import cropnp, saveimage
from ..tools import timeit, graph_connected_components

# Left TODO:
# Choix + rapide entre les 2 sous-cas "one contour"
# rename cropnp to reflect it's returning pointer on data e.g. np_cropped
# Benchmark choice b/w region finders
# estim hnoise: essai de perf option bouclage sur les bincount np.for_each
# rajouter cc_id dans classe CC

R_MARKER = 1000000


class CC:
    __slots__ = "x", "y", "w", "h", "touching", "entire"
    x: int
    y: int
    w: int
    h: int
    touching: bool  # The CC touches one border at least
    entire: bool  # The CC touches all borders

    def __init__(self, x: int, y: int, w: int, h: int, width: int, height: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.touching = x == 0 or y == 0 or x + w == width or y + h == height
        self.entire = w == width and h == height

    def is_at(self, x: int, y: int, w: int, h: int) -> bool:
        return self.x == x and self.y == y and self.w == w and self.h == h


class ConnectedComponentsSegmenter:
    def __init__(self, image):
        self.image = image

    def find_particles_via_cc(
        self, inv_mask: ndarray, s_p_min: int, s_p_max: int, max_w_to_h_ratio: float
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        hnoise = self.horizontal_noise_ratio(inv_mask)  # Takes a few tens of ms
        denoised, before = False, None
        if hnoise >= 10:
            # For noisy images (~10M potential particles), retrieving contours can jump to minutes
            before = self.denoise(inv_mask, s_p_min)
            denoised = True

        labels, retval, stats = self.extract_ccs_vertical_split(inv_mask)
        if denoised:
            print("Noisy! number of initial CCs:", before, "then found: ", retval)
        else:
            print("Number of cc found: ", retval)
        filtering_stats = [0] * 8
        maybe_kept, filtering_stats[0:3] = self.prefilter(stats, s_p_min, not denoised)
        ret = []
        # Note: The labels matrix is used for marking exclusion zones as well
        for x, y, w, h, area_excl_holes, cc_id in maybe_kept:
            # Proceed to more expensive filtering

            cc_id_present = np.any(labels[y, x : x + w] == cc_id)
            if not cc_id_present:
                # Shape was erased, i.e. excluded
                filtering_stats[3] += 1
                continue

            cc = CC(x, y, w, h, width, height)

            if cc.entire:
                self.prevent_entire_cc_inclusion(inv_mask, labels, cc_id, cc)
                filtering_stats[4] += 1
                continue

            holes, obj_mask = self.get_regions(labels, cc_id, cc, area_excl_holes)
            area = area_excl_holes + np.count_nonzero(holes)
            # Eliminate if touching any border ('entire' case treated above)
            if cc.touching:
                self.prevent_inclusion(labels, holes, cc)
                filtering_stats[4] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                # print("Excluded region: ", w, h, area_excl_holes, area, s_p_min)
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                self.prevent_inclusion(labels, holes, cc)
                filtering_stats[6] += 1
                continue
            # Horizontal stripes from scanner carriage movement
            ratiobxby = w / h
            if ratiobxby > max_w_to_h_ratio:
                filtering_stats[7] += 1
                continue

            self.prevent_inclusion(labels, holes, cc)

            ret.append(
                ROI(
                    features={
                        "BX": int(x),
                        "BY": int(y),
                        "Width": int(w),
                        "Height": int(h),
                        "Area": int(area),
                    },
                    mask=obj_mask + holes,
                    contour=None,
                )
            )
        print("Initial CCs", retval, "filter stats", filtering_stats, "left", len(ret))
        return ret

    @classmethod
    def prevent_inclusion(cls, labels: ndarray, mask: ndarray, cc: CC):
        """
        Mark exclusion zone in shape. "0" in shape means allowed, so "1" is OK, as we never exclude anything
        before the first CC.
        """
        sub_labels = labels[cc.y : cc.y + cc.h, cc.x : cc.x + cc.w]
        sub_labels[mask != 0] = 1

    @classmethod
    def prefilter(
        cls, cc_stats: ndarray, s_p_min: float, do_square: bool
    ) -> Tuple[ndarray, Tuple[int, int, int]]:
        assert (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ) == (0, 1, 2, 3, 4)
        # Add index after stats, starting at 1, as the first 'component' is the whole image.
        offs = 1
        indices = np.reshape(
            np.arange(start=offs, stop=len(cc_stats), dtype=np.uint32),
            (len(cc_stats) - offs, 1),
        )
        ret = np.concatenate((cc_stats[offs:], indices), axis=1)
        # return ret, (0, 0, 0)
        # Even if all pixels formed a 1-pixel-wide square, adding the hole inside would not make enough
        if do_square:
            min_pixels = int(math.sqrt(s_p_min))
            by_holes_area = ret[:, cv2.CC_STAT_AREA] > min_pixels
            holes_area_flt = len(ret)
            ret = ret[by_holes_area]
            holes_area_flt -= len(ret)
        else:
            holes_area_flt = 0
        # Even if contour was around a filled rectangle it would not meet min criterion
        by_area = ret[:, cv2.CC_STAT_WIDTH] * ret[:, cv2.CC_STAT_HEIGHT] > int(s_p_min)
        area_flt = len(ret)
        ret = ret[by_area]
        area_flt -= len(ret)
        # 1-pixel lines
        # TODO: a OR here (np.where or np.select?)
        by_size_1 = ret[:, cv2.CC_STAT_WIDTH] > 1
        size_flt = len(ret)
        ret = ret[by_size_1]
        by_size_2 = ret[:, cv2.CC_STAT_HEIGHT] > 1
        ret = ret[by_size_2]
        size_flt -= len(ret)
        return ret, (holes_area_flt, area_flt, size_flt)

    @classmethod
    def extract_ccs(cls, inv_mask: ndarray):
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        return labels, retval, stats

    @classmethod
    def extract_ccs_vertical_split(cls, inv_mask: ndarray):
        """
        Extract connected components from the mask, but in 2 parts, in order to avoid
        the 'entire image' problem described elsewhere. Basically mimic-ing the historical
        implementation but fixing its problems.
        """
        height, width = inv_mask.shape
        split_w = width * 75 // 100
        (
            l_half,
            l_labels,
            l_retval,
            l_stats,
            r_half,
            r_labels,
            r_retval,
            r_stats,
        ) = cls.extract_ccs_2_bands(inv_mask, height, width, split_w)
        # Join results

        groups = cls.find_common_cc_regions(
            l_half, l_labels, l_stats, r_half, r_labels, r_stats
        )

        # Shift right labels matrix
        cls.shift_labels(r_labels, l_retval - 1)

        cls.apply_common_cc_regions(
            groups=groups,
            l_labels=l_labels,
            l_stats=l_stats,
            r_labels=r_labels,
            r_stats=r_stats,
            r_label_offs=l_retval - 1,
        )

        # Shift right stats which are 0 based
        r_stats[:, cv2.CC_STAT_LEFT] += split_w

        labels = cls.paste_cc_parts(l_labels, r_labels)

        # Summary is in stats[0]
        area_excl_holes = l_stats[0, cv2.CC_STAT_AREA] + r_stats[0, cv2.CC_STAT_AREA]
        stats = np.concatenate(
            (
                np.array([[0, 0, width, height, area_excl_holes]]),
                l_stats[1:],
                r_stats[1:],
            )
        )
        # Count of CCs
        retval = l_retval + r_retval - 1  # first element stats[0] was removed
        assert retval == len(stats)
        return labels, retval, stats

    @classmethod
    def paste_cc_parts(cls, l_labels: ndarray, r_labels: ndarray):
        labels = np.concatenate((l_labels, r_labels), axis=1)
        return labels

    @classmethod
    def shift_labels(cls, labels: ndarray, offset: int):
        """Offset all valid labels inside given matrix"""
        labels[np.nonzero(labels)] += offset

    @classmethod
    def extract_ccs_2_bands(cls, inv_mask, height, width, split_w):
        l_half = cropnp(image=inv_mask, top=0, left=0, bottom=height, right=split_w)
        l_height, l_width = l_half.shape
        (
            l_retval,
            l_labels,
            l_stats,
            l_centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=l_half, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        r_half = cropnp(image=inv_mask, top=0, left=split_w, bottom=height, right=width)
        r_height, r_width = r_half.shape
        (
            r_retval,
            r_labels,
            r_stats,
            r_centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=r_half, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        assert l_width + r_width == width
        return l_half, l_labels, l_retval, l_stats, r_half, r_labels, r_retval, r_stats

    @classmethod
    def find_common_cc_regions(
        cls, l_half, l_labels, l_stats, r_half, r_labels, r_stats
    ) -> List[Set]:
        """Find common CCs, i.e. the ones in left part touching ones in right part
        Return a list of connected CCs, left ones having their usual index, right ones shifted by
        100M
        """
        zone_column_left = l_half[:, -1]
        zone_labels_left = l_labels[:, -1]
        (l_dc,) = np.nonzero(zone_column_left)
        zone_column_right = r_half[:, 0]
        zone_labels_right = r_labels[:, 0] + R_MARKER
        (r_dc,) = np.nonzero(zone_column_right)
        same_ccs = set()
        for offs in (-1, 0, 1):  # 8 connectivity
            in_contact = np.intersect1d(l_dc, r_dc + offs)
            contact_labels_left = zone_labels_left[in_contact]
            contact_labels_right = zone_labels_right[in_contact - offs]
            same_ccs_offs = np.unique(
                np.column_stack((contact_labels_left, contact_labels_right)), axis=0
            )
            same_ccs.update(list(map(tuple, same_ccs_offs)))
        same_ccs = sorted(list(same_ccs))
        # Do connected components (on the graph of neighbour CCs, that's confusing) to group CCs
        connections = {}
        for l_cc, r_cc in same_ccs:
            if r_cc not in connections:
                connections[r_cc] = {l_cc}
            else:
                connections[r_cc].add(l_cc)
            if l_cc not in connections:
                connections[l_cc] = {r_cc}
            else:
                connections[l_cc].add(r_cc)
        groups = graph_connected_components(connections)
        return groups

    @classmethod
    def apply_common_cc_regions(
        cls, groups: List[Set], l_labels, l_stats, r_labels, r_stats, r_label_offs
    ):
        """:param r_label_offs: how to warp values in right labels space"""
        for a_group in groups:
            assert len(a_group) > 1
            a_group = sorted(list(a_group))
            [
                cls.merge_ccs(a_group[0], a_cc, l_labels, l_stats, r_labels, r_label_offs, r_stats)
                for a_cc in a_group[1:]
            ]

    @classmethod
    def merge_ccs(cls, cc1, cc2, l_labels, l_stats, r_labels, r_label_offs, r_stats):
        right_offs = l_labels.shape[1]  # width
        if cc1 < R_MARKER:
            cc1_stats, cc1_ndx = l_stats, cc1
            l1_offs = 0
            cc1_offs = 0
        else:
            cc1_stats, cc1_ndx = r_stats, cc1 - R_MARKER
            l1_offs = right_offs
            cc1_offs = R_MARKER
        l1, t1, w1, h1, a1 = cc1_stats[cc1_ndx]
        l1 += l1_offs
        if cc2 < R_MARKER:
            cc2_labels, cc2_stats, cc2_ndx = l_labels, l_stats, cc2
            l2_offs = 0
            cc2_offs = 0
            cc2_labels_offs = 0
        else:
            cc2_labels, cc2_stats, cc2_ndx = r_labels, r_stats, cc2 - R_MARKER
            l2_offs = right_offs
            cc2_offs = R_MARKER
            cc2_labels_offs = r_label_offs
        l2, t2, w2, h2, a2 = cc2_stats[cc2_ndx]
        l2 += l2_offs
        if w1 == 0 or w2 == 0:
            assert False
        t_fin = min(t1, t2)
        h_fin = max(t1 + h1, t2 + h2) - t_fin
        l_fin = min(l1, l2)
        w_fin = max(l1 + w1, l2 + w2) - l_fin
        cc1_stats[cc1_ndx][cv2.CC_STAT_LEFT] = l_fin - l1_offs
        cc1_stats[cc1_ndx][cv2.CC_STAT_TOP] = t_fin
        cc1_stats[cc1_ndx][cv2.CC_STAT_WIDTH] = w_fin
        cc1_stats[cc1_ndx][cv2.CC_STAT_HEIGHT] = h_fin
        cc1_stats[cc1_ndx][cv2.CC_STAT_AREA] += a2
        cc2_labels_to_change = cc2_labels[
            t2 : t2 + h2, l2 - l2_offs : l2 - l2_offs + w2
        ]
        src_cc_id = cc2 - cc2_offs + cc2_labels_offs
        to_change = np.nonzero(cc2_labels_to_change == src_cc_id)
        cc2_labels_to_change[to_change] = cc1 - cc1_offs
        # Nullify cc2
        cc2_stats[cc2_ndx][cv2.CC_STAT_WIDTH] = 1
        cc2_stats[cc2_ndx][cv2.CC_STAT_HEIGHT] = 1
        cc2_stats[cc2_ndx][cv2.CC_STAT_AREA] = 1

    @classmethod
    @timeit
    def extract_holes_ccs(cls, inv_mask: ndarray):
        retval, labels = cv2.connectedComponents(
            image=1 - inv_mask, connectivity=4, ltype=cv2.CV_32S
        )
        return labels, retval

    @staticmethod
    def get_regions(
        labels: ndarray, cc_id: int, cc: CC, area_excl: int
    ) -> Tuple[ndarray, ndarray]:
        assert not cc.entire
        # before = time.time()
        empty_ratio = cc.h * cc.w // area_excl
        # if np.count_nonzero(labels == cc_id) != area_excl:
        #     pass
        # Compute filled area
        if empty_ratio > 200:
            # It's faster to draw the hole shapes inside very sparse shapes
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_contours(
                labels, cc_id, cc
            )
        else:
            holes, sub_mask = ConnectedComponentsSegmenter.get_regions_using_floodfill(
                labels, cc_id, cc
            )
        # elapsed = int((time.time() - before) * 10000)
        # if elapsed > 10:
        #     print(
        #         "get_regions:",
        #         elapsed,
        #         " ratio ",
        #         empty_ratio,
        #         (w, h),
        #         " img ",
        #         (width, height),
        #     )
        # if not area_excl == np.count_nonzero(sub_mask):
        #     sub_labels = np.count_nonzero(
        #         cropnp(
        #             image=labels,
        #             top=cc.y,
        #             left=cc.x,
        #             bottom=cc.y + cc.h,
        #             right=cc.x + cc.w,
        #         )
        #         == cc_id
        #     )
        #     nb_holes = np.count_nonzero(holes)
        #     nb_mask = np.count_nonzero(sub_mask)
        #     raise "pb here"
        #     pass
        return holes, sub_mask

    @staticmethod
    def get_mask_framed_by_1(labels: ndarray, cc: CC, cc_id: int):
        if cc.touching:
            # The shape touches an image border
            sub_labels = cropnp(
                image=labels, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
            )
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                np.uint8
            )  # 0=not in shape (either around shape or inside), 1=shape
            # Enlarge with 1 line around
            obj_mask = cv2.copyMakeBorder(
                obj_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
            )
        else:
            # Wa can save the copyMakeBorder above by picking around the existing lines
            sub_labels = cropnp(
                image=labels,
                top=cc.y - 1,
                left=cc.x - 1,
                bottom=cc.y + cc.h + 2,
                right=cc.x + cc.w + 2,
            )
            # noinspection PyUnresolvedReferences
            obj_mask = (sub_labels == cc_id).astype(
                np.uint8
            )  # 0=not in shape (either around shape or inside), 1=shape
        return obj_mask

    @staticmethod
    def get_regions_using_floodfill(
        labels: ndarray, cc_id: int, cc: CC
    ) -> Tuple[ndarray, ndarray]:
        obj_mask = ConnectedComponentsSegmenter.get_mask_framed_by_1(labels, cc, cc_id)
        holes = 1 - obj_mask  # 1=around 0=shape 1=holes
        cv2.floodFill(
            holes, mask=None, seedPoint=(0, 0), newVal=(0,)
        )  # 0=around 0=shape 1=holes
        sub_mask = cropnp(
            image=obj_mask, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1
        )
        holes = cropnp(image=holes, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1)
        return holes, sub_mask

    @staticmethod
    def get_regions_using_contours(
        labels: ndarray,
        cc_id: int,
        cc: CC,
    ) -> Tuple[ndarray, ndarray]:
        sub_labels = cropnp(
            image=labels, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
        )
        # noinspection PyUnresolvedReferences
        obj_mask = (sub_labels == cc_id).astype(
            np.uint8
        )  # 0=not in shape (either around shape or inside), 1=shape
        contours, _ = cv2.findContours(
            obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )  # Holes, if any, are in second level of RETR_CCOMP method output
        holes = np.zeros_like(obj_mask)
        if len(contours) > 1:
            cv2.drawContours(
                image=holes,
                contours=contours[1:],
                contourIdx=-1,
                color=(1,),
                thickness=cv2.FILLED,  # FILLED → inside + contour
            )  # 0=not in hole, 1=hole
            # Above 'holes' is not pixel-exact as the edges b/w particle and holes is drawn, eliminate them
            # holes = holes - holes & obj_mask # Remove common pixels, not working, see test_holes_62388
            cv2.drawContours(
                image=holes,
                contours=contours[1:],
                contourIdx=-1,
                color=(0,),
                thickness=1,  # Remove contours borders
            )  # 0=not in hole, 1=hole
        return holes, obj_mask

    @staticmethod
    def get_regions_using_cc(
        labels: ndarray,
        cc_id: int,
        cc: CC,
    ) -> Tuple[ndarray, ndarray]:
        obj_mask = ConnectedComponentsSegmenter.get_mask_framed_by_1(labels, cc, cc_id)
        retval, labels = cv2.connectedComponents(
            image=1 - obj_mask, connectivity=4, ltype=cv2.CV_32S
        )
        labels = cropnp(image=labels, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1)
        # 0=object 1=background the rest is holes
        holes = (labels > 1).astype(np.uint8)
        obj_mask = cropnp(
            image=obj_mask, top=1, left=1, bottom=cc.h + 1, right=cc.w + 1
        )
        return holes, obj_mask

    @staticmethod
    def prevent_entire_cc_inclusion(
        inv_mask: ndarray, labels: ndarray, cc_id: int, cc: CC
    ):
        # Here we don't build a mask from labels like elsewhere, because it's the full image to clone.
        sub_image = cropnp(
            image=inv_mask, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
        )
        ext_contours, _ = cv2.findContours(
            sub_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Note: the returned contours _might_ include [truncated] neighbours as we look
        # in original mask, not in the masked-by-cc_id one like in get_regions_using_floodfill
        # TODO: The returned contours might as well include all interesting particles! both CC and
        #  top-level contours info is fetched using openCV at this point.
        contour_4_cc = ConnectedComponentsSegmenter.find_top_contour(ext_contours, cc)
        print("nb contour_4_cc:", len(ext_contours))
        assert contour_4_cc is not None, "Touching shape not found"
        if cc.entire and cv2.contourArea(contour_4_cc) > 0.9 * cc.w * cc.h:
            # Despite all the efforts made to avoid it (e.g. small holes in border lines so contour detection algo
            # can sneak inside particles area), sometimes the first contour is an _outer_ one. It can be fitting perfectly
            # around the border lines to image borders, or, when there are tiny holes in border lines, all holes could be covered
            # in by some particle or noise.
            # Historical algorithm is immune to this problem, because when taking 2 slices, one vertical side is completely opened
            # and contour detection will find everything. But it raises other issues like lost big particles.
            #
            # OTOH, sometimes there could be only 3 full borders, resulting in same contour, but an inner
            # one as usually. This is sorted out by the criterion using contourArea() above.
            #
            # First case manifest itself as an inner contour filling nearly all the image. It's geometrically OK as it's a
            # hole inside the shape, but we need to get rid of it in decent time.
            print("4 borders closed")
            # Open a gap so the 4-borders doesn't have the annoying property anymore
            cv2.line(inv_mask, (cc.w // 2, 0), (cc.w // 2, cc.h // 2), (0,), 1)
            ext_contours, _ = cv2.findContours(
                inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_4_cc = ConnectedComponentsSegmenter.find_top_contour(
                ext_contours, cc
            )
            print("nb contour_4_cc after cut:", len(ext_contours))
        else:
            # We have an entire shape but its interior is not the full image.
            # Imagine a giant "U" covering 3 borders but not the top one,
            #      or a giant "C" covering 3 borders but not the right one.
            # We can paint the whole CC as "forbidden".
            print("4 borders not closed")
            # Disable the full shape, we're done

        # Note: It's a bit border-line as we use a drawing primitive on a non-image.
        cv2.drawContours(
            image=labels,
            contours=[contour_4_cc],
            contourIdx=-1,
            color=(1,),
            thickness=cv2.FILLED,  # FILLED → inside + contour
        )

    @staticmethod
    def find_matching_contour(contours: Sequence[ndarray], cc: CC) -> Optional[ndarray]:
        """Match a contour in given list with a cc shape"""
        for a_contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(a_contour)
            x_c, y_c = x_c + cc.x, y_c + cc.y  # Translate as we're in a sub-rect
            if cc.is_at(x_c, y_c, w_c, h_c):
                return a_contour
        return None

    @staticmethod
    def find_top_contour(contours: Sequence[ndarray], cc: CC) -> Optional[ndarray]:
        """Find top-level contour, the one enclosing all the rest"""
        w, h = cc.w, cc.h
        for a_contour in reversed(contours):  # Enclosing ones are at the end
            _x_c, _y_c, w_c, h_c = cv2.boundingRect(a_contour)
            if w == w_c and h_c == h_c:
                return a_contour
        return None

    @staticmethod
    def remove_unwanted_inside_contour(
        contours: Sequence[ndarray], big_area_threshold: int
    ) -> List[ndarray]:
        for contour_ndx in range(1, len(contours)):
            if cv2.contourArea(contours[contour_ndx]) > big_area_threshold:
                to_remove = contour_ndx
                break
        else:
            to_remove = None
        assert to_remove is not None, "Failed to locate inside contour"
        contours = list(contours)
        del contours[to_remove]
        return contours

    @staticmethod
    def build_mask_from_labels(labels, cc, cc_id):
        sub_labels = cropnp(
            image=labels, top=cc.y, left=cc.x, bottom=cc.y + cc.h, right=cc.x + cc.w
        )
        # noinspection PyUnresolvedReferences
        obj_mask = (sub_labels == cc_id).astype(
            np.uint8
        )  # 0=not in shape (either around shape or inside), 1=shape
        return obj_mask

    @classmethod
    def horizontal_noise_ratio(cls, inv_mask: ndarray) -> int:
        """Per 1000 number of != pixels from one line to another, in 90% central region of the image"""
        height, width = inv_mask.shape[:2]
        excluded = int(height * 0.9) // 2
        orig = cropnp(image=inv_mask, top=excluded, bottom=-excluded)
        below = cropnp(image=inv_mask, top=excluded + 1, bottom=-excluded + 1)
        diff = orig ^ below
        ret = int(np.sum(diff) * 1000 / ((height - excluded) * width))
        return ret

    @classmethod
    def denoise(cls, inv_mask: ndarray, s_p_min: float) -> int:
        """Erase connected components which cannot end up in final result"""
        min_pixels = int(math.sqrt(s_p_min))
        retval, labels = cv2.connectedComponents(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S
        )

        _unique, counts = np.unique(labels, return_counts=True)
        to_erase = np.nonzero(counts <= min_pixels)
        labels2 = np.isin(labels, to_erase)
        inv_mask[labels2] = 0
        return int(retval)

    @classmethod
    def debug_save_cc_image(cls, image, cc: CC, cc_id: int):
        cc_image = cropnp(image, cc.y, cc.x, cc.y + cc.h, cc.x + cc.w)
        saveimage(cc_image, Path(f"/tmp/zooprocess/cc_{cc_id}.png"))

    @staticmethod
    def debug_cc_comp_contour(obj_mask, contours, thickness):
        dbg_img = np.zeros_like(obj_mask)
        dbg_img_3chan = cv2.merge([dbg_img, dbg_img, dbg_img])
        cv2.drawContours(dbg_img_3chan, contours[0:1], -1, (255, 0, 0), thickness)
        cv2.drawContours(dbg_img_3chan, contours[1:], -1, (255, 255, 0), thickness)
        saveimage(dbg_img_3chan, Path("/tmp/zooprocess/contours.tif"))
