import dataclasses
import math
from decimal import Decimal
from typing import List, TypedDict, Tuple, Optional, Set

import cv2
import numpy as np
from numpy import ndarray

from .EllipseFitter import EllipseFitter
from .img_tools import cropnp


class Features(TypedDict):
    # Enclosing rectangle
    BX: int
    BY: int
    Width: int
    Height: int
    # Number of pixels inside the blob
    Area: int
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


feature_unq = lambda f: (f["BX"], f["BY"], f["Width"], f["Height"])


@dataclasses.dataclass(
    frozen=False
)  # TODO: Should be 'True', temp until ROI merge is clean
class ROI(object):
    features: Features
    mask: ndarray
    contour: Optional[ndarray] = None


def features_are_at_same_coord(features: Features, another_blob: Features):
    return features["BX"] == another_blob["BX"] and features["BY"] == another_blob["BY"]


class Segmenter(object):
    """
    Divide an image into segments and store the result sub-images.
    """

    THRESH_MAX = 243
    RESOLUTION = 2400
    # Constants for 2-image processing. Historical.
    Wlimit = 20000
    Hlimit = 6500
    overlap = 0.6
    # For filtering horizontal lines
    max_w_to_h_ratio = 40

    METH_CONTOUR = 1
    METH_CONNECTED_COMPONENTS = 2
    METH_RETR_TREE = 4
    LEGACY_COMPATIBLE = 16

    def __init__(self, image: ndarray, minsize: float, maxsize: float):
        assert image.dtype == np.uint8
        self.image = image
        self.height, self.width = image.shape[:2]
        pixel = 25.4 / self.RESOLUTION
        sm_min = (3.1416 / 4) * pow(minsize, 2)
        sm_max = (3.1416 / 4) * pow(maxsize, 2)
        # s_p_* are in pixel^2
        self.s_p_min = round(sm_min / (pow(pixel, 2)))
        self.s_p_max = round(sm_max / (pow(pixel, 2)))

    def find_blobs(self, method: int = METH_CONTOUR) -> List[ROI]:
        # Threshold the source image to have a b&w mask
        thresh_max = self.THRESH_MAX
        # mask is white objects on black background
        _th, inv_mask = cv2.threshold(
            self.image, thresh_max, 255, cv2.THRESH_BINARY_INV
        )
        # saveimage(inv_mask, "/tmp/inv_mask.tif")
        self.sanity_check(inv_mask)
        if method & self.LEGACY_COMPATIBLE:
            # Process image in 2 overlapping parts, split vertically.
            # There is a side effect in that objects in the middle 20% band can appear if embedded or disappear if too large.
            if self.width > self.Wlimit and self.height > self.Hlimit:
                overlap_size = int(self.width * self.overlap)
                left_mask = cropnp(
                    inv_mask, top=0, left=0, bottom=self.height, right=overlap_size
                )
                left_rois = self.find_particles_with_method(method, left_mask)
                right_mask = cropnp(
                    inv_mask,
                    top=0,
                    left=self.width - overlap_size,
                    bottom=self.height,
                    right=self.width,
                )
                right_rois = self.find_particles_with_method(method, right_mask)
                # Fix coordinates from right pane
                for right_roi in right_rois:
                    right_roi.features["BX"] += self.width - overlap_size
                # Merge ROI lists
                key_func = lambda r: feature_unq(r.features)
                right_by_key = {key_func(ri): ri for ri in right_rois}
                left_by_key = {key_func(le): le for le in left_rois}
                assert len(left_by_key) == len(left_rois)
                assert len(right_by_key) == len(right_rois)
                for right_key, right_roi in right_by_key.items():
                    if right_key in left_by_key:
                        _left_roi = left_by_key[right_key]  # Unused so far
                # Add different
                for left_key, left_roi in left_by_key.items():
                    if left_key not in right_by_key:
                        right_by_key[left_key] = left_roi
                return list(right_by_key.values())
        return self.find_particles_with_method(method, inv_mask)

    def find_particles_with_method(self, method: int, inv_mask: ndarray) -> List[ROI]:
        # Required measurements:
        #       area bounding area_fraction limit decimal=2
        # Result:
        #       Area	BX	BY	Width	Height	%Area	XStart	YStart
        if method & self.METH_CONNECTED_COMPONENTS:
            # Most reliable from execution/complexity points of view
            return self.find_particles_via_cc(inv_mask, self.s_p_min, self.s_p_max)
        elif method & self.METH_RETR_TREE:
            # Universal, but very slow on noisy images, collapses from second to ten of minutes
            # saveimage(inv_mask, "/tmp/bef_denoised.tif")
            #
            # self.denoise_particles_via_cc(inv_mask, self.s_p_min)
            # saveimage(inv_mask, "/tmp/aft_denoised.tif")
            return self.find_particles_contour_tree(
                inv_mask, self.s_p_min, self.s_p_max
            )
        else:
            return self.find_particles(inv_mask, self.s_p_min, self.s_p_max)

    @staticmethod
    def find_particles(inv_mask: ndarray, s_p_min: int, s_p_max: int) -> List[ROI]:
        # ImageJ calls args are similar to:
        # analysis1 = "minimum=" + Spmin + " maximum=" + Spmax + " circularity=0.00-1.00 bins=20 show=Outlines include exclude flood record";
        # 'include' is 'Include holes'
        # 'exclude' is 'Exclude on hedges'
        # -> circularity is never used as a filter
        height, width = inv_mask.shape[:2]
        contours, _ = cv2.findContours(
            inv_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,  # same as cv2.RETR_EXTERNAL but returns less data
        )
        if len(contours) == 1:
            print("1 contour!")
            return Segmenter.find_particles_via_cc(inv_mask, s_p_min, s_p_max)

        print("Number of RETR_EXTERNAL Contours found = " + str(len(contours)))
        ret: List[ROI] = []
        single_point_contour_shape = (1, 1, 2)
        for a_contour in contours:
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                continue
            x, y, w, h = cv2.boundingRect(a_contour)
            # Eliminate if touching the border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            # -> don't bother drawing the contour, which is expensive
            if w * h < s_p_min:
                continue
            contour_mask = Segmenter.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                continue
            if area > s_p_max:
                continue
            ratiobxby = w / h
            if ratiobxby > Segmenter.max_w_to_h_ratio:
                continue
            ret.append(
                ROI(
                    features={
                        "BX": x,
                        "BY": y,
                        "Width": w,
                        "Height": h,
                        "Area": area,
                    },
                    mask=contour_mask,
                    contour=a_contour,
                )
            )
            # image_3channels = draw_contours(self.image, self.contours)
            # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret

    @staticmethod
    def find_particles_contour_tree(
        inv_mask: ndarray, s_p_min: int, s_p_max: int
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        # In some cases and despite previous steps, the border of the scan goes fully round the image, so
        # there is a single contour!
        # image_3channels = draw_contours(inv_mask, contours, thickness=1)
        # saveimage(image_3channels, Path("/tmp/contours.tif"))
        # Fix by removing it.
        # first_pixel = np.argmax(inv_mask[0] == 255)
        # saveimage(inv_mask, "/tmp/bef_flood.tif")
        # cv2.floodFill(
        #     image=inv_mask,
        #     mask=None,
        #     seedPoint=(first_pixel, 0),
        #     newVal=(0,),
        #     flags=8,  # 8-pixel connectivity, like contour detection does
        # )
        # Segmenter.undo_border_lines(inv_mask)
        # cv2.drawContours(
        #     image=inv_mask,
        #     contours=contours,
        #     contourIdx=0,
        #     color=(0,),
        #     thickness=4
        # )
        # Breach the border
        # cv2.line(inv_mask, (0, 0), (300, 300), (0,), thickness=2)
        # inv_mask = cv2.copyMakeBorder(
        #     inv_mask, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(0,)
        # )
        # saveimage(inv_mask, "/tmp/aft_flood.tif")
        approx = cv2.CHAIN_APPROX_NONE
        contours, (hierarchy,) = cv2.findContours(inv_mask, cv2.RETR_TREE, approx)
        # return []
        # root_children_contours = []
        # for a_contour, its_hierarchy in zip(contours, hierarchy):
        #     (
        #         next_contour,
        #         previous_contour,
        #         child_contour,
        #         parent_contour,
        #     ) = its_hierarchy
        #     # In RETR_CCOMP mode we have 2 hierarchies, -1 is enclosing one, other is holes on
        #     if parent_contour == -1:
        #         root_children_contours.append(a_contour)
        # contours = root_children_contours
        # contours, (hierarchy,) = cv2.findContoursLinkRuns(
        #     inv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        # )
        # image_3channels = draw_contours(inv_mask, contours, thickness=3)
        # saveimage(image_3channels, Path("/tmp/contours2.tif"))
        # image_3channels2 = draw_contours(inv_mask, root_children_contours, thickness=3)
        # saveimage(image_3channels2, Path("/tmp/contours3.tif"))
        # contours = root_children_contours

        print("Number of RETR_TREE Contours found = " + str(len(contours)))
        ret: List[ROI] = []
        accepted_parents = {-1: 0}  # contour_id, level
        single_point_contour_shape = (1, 1, 2)
        # Optimization
        accepted_parents_get = accepted_parents.get
        for contour_id, (a_contour, its_hierarchy) in enumerate(
            zip(contours, hierarchy)
        ):
            parent_contour = int(its_hierarchy[3])
            # assert parent_contour < contour_id  # Ensure we've seen the parent before
            level = accepted_parents_get(parent_contour)
            if level is None:
                continue
            # More frequent exclusion reasons first
            if a_contour.shape == single_point_contour_shape:  # Single-point "contour"
                continue  # Too small, really
            x, y, w, h = cv2.boundingRect(a_contour)
            # Eliminate if touching the border
            if x == 0 or y == 0 or x + w == width or y + h == height:
                # Keep descending, maybe an embedded shape fits
                accepted_parents[contour_id] = accepted_parents[parent_contour] + 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < s_p_min:
                continue
            # Compute filled area
            contour_mask = Segmenter.draw_contour(a_contour, x, y, w, h)
            area = np.count_nonzero(contour_mask)
            if area < s_p_min:
                continue
            elif area > s_p_max:
                # Keep descending, maybe an embedded shape fits
                accepted_parents[contour_id] = accepted_parents[parent_contour] + 1
                continue
            if level % 2 != 0:
                # Is a contour around a hole
                continue
            ratiobxby = w / h
            if ratiobxby > Segmenter.max_w_to_h_ratio:
                continue
            roi = ROI(
                features={
                    "BX": x,
                    "BY": y,
                    "Width": w,
                    "Height": h,
                    "Area": area,
                },
                mask=contour_mask,
                contour=a_contour,
            )
            ret.append(roi)
            # image_3channels = draw_contours(self.image, self.contours)
            # saveimage(image_3channels, Path("/tmp/contours.tif"))
        return ret

    @staticmethod
    def undo_border_lines(inv_mask: ndarray) -> Tuple[int, int, int, int]:
        height, width = inv_mask.shape[:2]
        # Find the 2 border dots in each dimension
        (left, right) = np.where(inv_mask[0] == 255)[0]
        (top, bottom) = np.where(inv_mask[:, 0] == 255)[0]
        # Clear them
        cv2.line(
            img=inv_mask,
            pt1=(0, top),
            pt2=(width - 1, top),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(0, bottom),
            pt2=(width - 1, bottom),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(left, 0),
            pt2=(left, height - 1),
            color=(0,),
            thickness=1,
        )
        cv2.line(
            img=inv_mask,
            pt1=(right, 0),
            pt2=(right, height - 1),
            color=(0,),
            thickness=1,
        )
        return top, left, bottom, right

    @staticmethod
    def sanity_check(inv_mask: ndarray):
        min_bwratio = 25
        nb_black = np.count_nonzero(inv_mask)
        nb_white = inv_mask.shape[0] * inv_mask.shape[1] - nb_black
        bwratiomeas = nb_black / nb_white
        print(f"bwratiomeas: {bwratiomeas}")
        if bwratiomeas > min_bwratio / 100:
            # Note: below message refers to non-inverted mask
            print(
                f"########### WARNING : More than {min_bwratio}% of the segmented image is black ! \nThe associated background image maybe NOK."
            )

    @staticmethod
    def draw_contour(contour, x, y, w, h) -> ndarray:
        contour_canvas = np.zeros([h, w], np.uint8)
        cv2.drawContours(
            image=contour_canvas,
            contours=[contour],
            contourIdx=0,
            color=(255,),
            thickness=cv2.FILLED,
            offset=(-x, -y),
        )
        return contour_canvas

    @staticmethod
    def denoise_particles_via_cc(inv_mask: ndarray, s_p_min: int):
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        assert (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ) == (0, 1, 2, 3, 4)
        ret = []
        print("cc filter, initial: ", retval)
        eliminated_1 = 0
        eliminated_l = 0
        eliminated_r = 0
        for cc_id in range(retval):
            x, y, w, h, area_excl_holes = [int(m) for m in stats[cc_id]]
            # Single point
            if area_excl_holes == 1:
                inv_mask[y, x] = 0
                eliminated_1 += 1
                continue
            # More frequent exclusion reasons first
            if w == 1 or h == 1:
                cv2.line(
                    img=inv_mask, pt1=(x, y), pt2=(x + w - 1, y + h - 1), color=(0,)
                )
                eliminated_l += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < s_p_min:
                sub_labels = cropnp(
                    image=labels, top=y, left=x, bottom=y + h, right=x + w
                )
                sub_mask = (sub_labels == cc_id).astype(
                    dtype=np.uint8
                ) * 255  # 255=shape, 0=not in shape
                inv_mask[y : y + h, x : x + w] = np.bitwise_xor(
                    inv_mask[y : y + h, x : x + w], sub_mask
                )
                eliminated_r += 1
                continue
        left = retval - (eliminated_1 + eliminated_l + eliminated_r)
        print(
            "cc filter, eliminated: ",
            eliminated_1,
            eliminated_l,
            eliminated_r,
            " left: ",
            left,
        )
        return ret

    @staticmethod
    def find_particles_via_cc(
        inv_mask: ndarray, s_p_min: int, s_p_max: int
    ) -> List[ROI]:
        height, width = inv_mask.shape[:2]
        (
            retval,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStatsWithAlgorithm(
            image=inv_mask, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_GRANA
        )
        assert (
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ) == (0, 1, 2, 3, 4)
        ret = []
        print("Number of cc found: ", retval)
        embedded = set()
        filtering_stats = [0] * 8
        for cc_id in range(1, retval):  # First 'component' is the whole image
            if cc_id in embedded:
                filtering_stats[0] += 1
                continue
            x, y, w, h, area_excl_holes = [int(m) for m in stats[cc_id]]
            # More frequent exclusion reasons first
            if w == 1 or h == 1:
                filtering_stats[1] += 1
                continue
            # Even if contour was around a filled rectangle it would not meet min criterion
            if w * h < s_p_min:
                filtering_stats[2] += 1
                continue
            # The cc bounding rect is the whole image minus borders, filter it but don't exclude inside
            # TODO: 2 sub-cases: either the whole area is included,
            #  or there is a band around the whole image
            if (
                x == 0
                and y == 0
                and x + w == width
                and y + h == height
                # and area_excl_holes > w * h / 2  # TODO: Not accurate
            ):
                filtering_stats[3] += 1
                continue
            # Proceed to more expensive filtering
            sub_labels, holes, filled_mask = Segmenter.get_regions(
                labels, cc_id, x, y, w, h
            )
            area = area_excl_holes + holes.sum()
            # Eliminate if touching some border (but not all of them)
            if x == 0 or y == 0 or x + w == width or y + h == height:
                Segmenter.forbid_inside_objects(sub_labels * holes, cc_id, embedded)
                filtering_stats[4] += 1
                continue
            # Criteria from parameters
            if area < s_p_min:
                filtering_stats[5] += 1
                continue
            if area > s_p_max:
                Segmenter.forbid_inside_objects(sub_labels * holes, cc_id, embedded)
                filtering_stats[6] += 1
                continue

            ratiobxby = w / h
            if ratiobxby > Segmenter.max_w_to_h_ratio:
                filtering_stats[7] += 1
                continue

            Segmenter.forbid_inside_objects(sub_labels * holes, cc_id, embedded)

            ret.append(
                ROI(
                    features={
                        "BX": x,
                        "BY": y,
                        "Width": w,
                        "Height": h,
                        "Area": int(area),
                    },
                    mask=filled_mask,
                    contour=None,
                )
            )
        print("Initial", retval, "filter stats", filtering_stats, "left", len(ret))
        return ret

    @staticmethod
    def get_regions(
        labels: ndarray,
        cc_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        # Compute filled area
        sub_labels = cropnp(image=labels, top=y, left=x, bottom=y + h, right=x + w)
        obj_mask = (sub_labels == cc_id).astype(
            dtype=np.uint8
        ) * 255  # 0=not in shape (either around shape or inside), 255=shape
        if True:
            contours, (hierarchy,) = cv2.findContours(
                obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            # contours, (hierarchy,) = cv2.findContoursLinkRuns(obj_mask)
            sub_mask3 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=sub_mask3,
                contours=contours,
                contourIdx=0,
                color=(255,),
                thickness=cv2.FILLED,
            )  # 0=not in filled shape, 255=filled shape
            holes3 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=holes3,
                contours=contours[1:],
                contourIdx=-1,
                color=(255,),
                thickness=cv2.FILLED,  # filled -> inside + contour
            )  # 0=not in filled shape, 255=filled shape
            cv2.drawContours(
                image=holes3,
                contours=contours[1:],
                contourIdx=-1,
                color=(0,),
                thickness=1,  # fix the "contour' part of cv2.FILLED above
            )  # 0=not in filled shape, 255=filled shape
            holes3 = holes3 == 255
            # if x == 0 and y == 0:
            #     saveimage(sub_mask3, "/tmp/contour_sub.tif")
            # holes2 = np.bitwise_xor(sub_mask, sub_mask3) == 255
            # holes_id = np.unique(np.where(sub_labels > cc_id)).tolist()
            return sub_labels, holes3, sub_mask3  # , holes_id
        if False:
            contours, _ = cv2.findContours(
                obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            sub_mask2 = np.zeros_like(obj_mask)
            cv2.drawContours(
                image=sub_mask2,
                contours=contours,
                contourIdx=0,
                color=(255,),
                thickness=cv2.FILLED,
            )  # 0=not in filled shape, 255=filled shape
            holes2 = np.bitwise_xor(obj_mask, sub_mask2) == 255
            return sub_labels, holes2, sub_mask2  # , holes_id
        if False:
            obj_mask2 = cv2.copyMakeBorder(
                obj_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0,)
            )
            cv2.floodFill(
                image=obj_mask2,
                mask=None,
                seedPoint=(0, 0),
                newVal=(128,),
                flags=4,  # 4-pixel connectivity, don't cross a cc border
            )  # 0=not part of shape but inside it i.e. holes, 255=shape, 128=outside border
            sub_mask = cropnp(image=obj_mask2, top=1, left=1, bottom=h + 1, right=w + 1)
            holes = sub_mask == 0  # False:non-hole True:hole
            sub_mask[holes] = 255
            sub_mask[sub_mask == 128] = 0

        return sub_labels, holes, orig_mask
        return sub_labels, holes, sub_mask

    @staticmethod
    def forbid_inside_objects(other: ndarray, contour_id: int, embedded: Set[int]):
        # Check for embedded objects
        in_holes = np.unique(other).tolist()
        if len(in_holes) > 1:
            in_holes.remove(0)
            # for a_hole in in_holes:
            #     assert a_hole > contour_id
            embedded.update(in_holes)

    def split_by_blobs(self, rois: List[ROI]):
        assert rois, "No ROIs"
        for ndx, a_roi in enumerate(rois):
            features = a_roi.features
            width = features["Width"]
            height = features["Height"]
            bx = features["BX"]
            by = features["BY"]
            vignette = cropnp(
                self.image, top=by, left=bx, bottom=by + height, right=bx + width
            )
            vignette = np.bitwise_or(vignette, 255 - a_roi.mask)
            # Compute more features
            # Xstart
            # First white pixel in first line of shape seems OK for this measurement
            x_start = features["BX"] + int(np.argmax(a_roi.mask == 255))
            features["XStart"] = x_start
            features["YStart"] = features["BY"]
            # %Area
            nb_holes = np.count_nonzero(vignette <= self.THRESH_MAX)
            pct_area = (
                100 - Decimal(nb_holes * 100) / features["Area"]
            )  # Need exact arithmetic due to some Java<->python rounding diff
            features["%Area"] = float(round(pct_area, 3))
            # major, minor, angle
            if False:
                # Very different from ref. and drawing them gives strange results sometimes
                # From some docs, we must not expect anything reasonable if the shape is not close from an ellipse
                vignette_contour = a_roi.contour - (features["BX"], features["BY"])
                min_ellipse = cv2.fitEllipse(vignette_contour)
                cv2.ellipse(vignette, min_ellipse, 0)
                cv2.drawContours(vignette, [vignette_contour], 0, 0)
            if False:
                # Angle matches with ref, but for some reason the axis are sometimes very different, and
                # visual check shows sometimes the ellipse extremely shifted. Maybe a centroid issue.
                import skimage

                props = skimage.measure.regionprops(a_roi.mask, cache=False)[0]
                features["Major"] = round(props.axis_major_length, 3)
                features["Minor"] = round(props.axis_minor_length, 3)
                features["Angle"] = round(math.degrees(props.orientation) + 90, 3)
                vignette = cv2.ellipse(
                    img=vignette,
                    center=(int(props.centroid[0]), int(props.centroid[1])),
                    axes=(
                        int(props.axis_minor_length / 2),
                        int(props.axis_major_length / 2),
                    ),
                    angle=90 - (math.degrees(props.orientation) + 90),
                    startAngle=0,
                    endAngle=360,
                    color=(0,),
                    thickness=1,
                )
            # Port of ImageJ algo
            fitter = EllipseFitter()
            fitter.fit(a_roi.mask)
            features["Major"] = round(fitter.major, 3)
            features["Minor"] = round(fitter.minor, 3)
            features["Angle"] = round(fitter.angle, 3)
            # fitter.draw_ellipse(vignette)
            # vignette = cv2.ellipse(
            #     img=vignette,
            #     center=(int(fitter.x_center), int(fitter.y_center)),
            #     axes=(int(fitter.minor / 2), int(fitter.major / 2)),
            #     angle=90 - fitter.angle,
            #     startAngle=0,
            #     endAngle=360,
            #     color=(0,),
            #     thickness=2,
            # )

            # saveimage(vignette, "/tmp/vignette_%s.png" % ndx)
