import math
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

import cv2
import numpy as np
from numpy import ndarray

from ..ROI import ROI
from ..img_tools import cropnp, saveimage


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
        hnoise = self.horizontal_noise_ratio(inv_mask)  # takes a few tens of ms
        denoised = False
        if hnoise >= 10:
            before = self.denoise(inv_mask, s_p_min)
            denoised = True
            labels, retval, stats = self.extract_ccs(inv_mask)
            print("Noisy! number of initial CCs:", before, "then found: ", retval)
        else:
            labels, retval, stats = self.extract_ccs(inv_mask)
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

            # if cc_id == 62388:
            #     ConnectedComponentsSegmenter.debug_save_cc_image(self.image, cc, cc_id)

            if x == 6170 and y == 41:
                pass
            if cc.entire:
                self.prevent_entire_cc_inclusion(labels, cc_id, cc)
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
            # Horizontal stripes from scanner bed movement
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
    def prevent_entire_cc_inclusion(labels: ndarray, cc_id: int, cc: CC):
        obj_mask = ConnectedComponentsSegmenter.build_mask_from_labels(
            labels, cc, cc_id
        )
        contours, _ = cv2.findContours(
            obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ext_contour = contours[0]
        big_area_threshold = cc.h * cc.w * 8 // 10
        if cv2.contourArea(ext_contour) > big_area_threshold:
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
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = ConnectedComponentsSegmenter.remove_unwanted_inside_contour(
                contours, big_area_threshold
            )
            contours = contours[1:]
        else:
            # We have an entire shape but its interior is not the full image.
            # Imagine a giant "U" covering 3 borders but not the top one,
            #      or a giant "C" covering 3 borders but not the right one.
            # We can paint the interior holes as "forbidden".
            print("4 borders not closed")
            # Disable the full shape, we're done

        # Note: It's a bit border-line as we use a drawing primitive on a non-image.
        cv2.drawContours(
            image=labels,
            contours=contours,
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
        """ Per 1000 number of != pixels from one line to another, in 90% central region of the image"""
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
