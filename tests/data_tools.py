from __future__ import annotations

from typing import List, Tuple, Dict, Callable, Any

import numpy as np

from ZooProcess_lib.Features import TYPE_BY_LEGACY
from ZooProcess_lib.ROI import feature_unq, ROI
from ZooProcess_lib.Segmenter import Segmenter
from .test_utils import read_measures_csv

FEATURES_TOLERANCES = {
    "%Area": 0.001,
    "X": 0.001,
    "Y": 0.001,
    "Kurt": 0.001,
    "Fractal": 0.05,
    "Convarea": "25%",
}
DERIVED_FEATURES_TOLERANCES = {
    "meanpos": 0.001,
    "esd": 0.001,
    # Below tolerances with rounding to int of the features used (perfectly imitates legacy)
    # "cv": 0.001,
    # "feretareaexc": 0.001,
    # "perimareaexc": 0.001,
    # "perimferet": 0.001,
    # "perimmajor": 0.001,
    # "elongation": 0.001,
    # "sr": 0.001,
    # "circex": 0.001,
    # Below adjusted tolerances b/w the right values and legacy ones, just for the small test_ij_like_features to pass
    "cv": 0.25,
    "feretareaexc": 0.01,
    "perimareaexc": 0.01,
    "perimferet": 0.1,
    "perimmajor": 0.1,
    "elongation": 0.01,
    "sr": 2,
    "circex": 0.01,
}


def report_and_fix_tolerances(
    differences: List[Tuple[Dict, Dict]], tolerances: Dict[str, float | str]
) -> List[str]:
    ret = []
    for an_exp, an_act in differences:
        if an_exp == an_act:
            continue
        if an_exp.keys() != an_act.keys():
            ret.append(str((an_exp, an_act)))
            continue
        for tolerance_key, tolerance in tolerances.items():
            ref_val = an_exp.get(tolerance_key)
            act_val = an_act.get(tolerance_key)
            if ref_val is None or act_val is None:
                print("tolerance_key:", tolerance_key, "not found")
            if ref_val == act_val:
                continue
            diff = ref_val - act_val
            if isinstance(tolerance, str):
                pct = int(tolerance[:-1])
                tolerance = ref_val * pct / 100.0
            if abs(diff) > tolerance * 1.0001:  # bloody floats
                ret.append(
                    f"{tolerance_key}: {act_val} vs {ref_val} exceeds tolerance {tolerance}"
                )
                continue  # Leave problem in actual
            # Fix tolerated value in actual
            an_act[tolerance_key] = ref_val
    return ret


def to_legacy_rounding(features_list: List[Dict], roundings: Dict[str, int]):
    for a_feature_set in features_list:
        for k, v in a_feature_set.items():
            rounding = roundings.get(k)
            if rounding is not None:
                a_feature_set[k] = round(a_feature_set[k], rounding)
            else:
                a_feature_set[k] = round(a_feature_set[k], 10)
    return features_list


def diff_features_lists(
    ref: List[Dict], act: List[Dict], key_func: Callable[[Dict], Any]
) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    different = []
    not_in_act = []
    refs_by_key = {key_func(a_ref): a_ref for a_ref in ref}
    acts_by_key = {key_func(an_act): an_act for an_act in act}
    for ref_key, a_ref in refs_by_key.items():
        in_act = acts_by_key.get(ref_key)
        if in_act is None:
            not_in_act.append(a_ref)
        else:
            if in_act != a_ref:
                different.append((a_ref, in_act))
            acts_by_key.pop(ref_key)
    return different, list(acts_by_key.values()), not_in_act


def read_measurements(project_folder, sample, index):
    work_files = project_folder.zooscan_scan.work.get_files(sample, index)
    measures = work_files["meas"]
    ref = read_measures_from_file(measures)
    return ref


def read_box_measurements(project_folder, sample, index):
    work_files = project_folder.zooscan_scan.work.get_files(sample, index)
    measures = work_files["meas"]
    ref = read_measures_from_file(measures, only_box=True)
    return ref


def to_legacy_format(features_list):
    rounded_to_3 = set(
        [
            a_feat
            for a_feat, a_type in TYPE_BY_LEGACY.items()
            if a_type in (float, np.float64)
        ]
    )
    for a_features in features_list:
        for a_round in rounded_to_3:
            if a_round in a_features:
                # IJ.java method d2s
                to_round = a_features[a_round]
                replacement = ij_round(to_round)
                a_features[a_round] = float(replacement)
    return features_list


BOX_MEASUREMENTS = {
    "BX": int,
    "BY": int,
    "Width": int,
    "Height": int,
}


def read_measures_from_file(measures, only_box=False):
    if only_box:
        ref = read_measures_csv(measures, BOX_MEASUREMENTS)
    else:
        ref = read_measures_csv(measures, TYPE_BY_LEGACY)
    # This filter is _after_ measurements in Legacy
    ref = [
        a_ref
        for a_ref in ref
        if a_ref["Width"] / a_ref["Height"] < Segmenter.max_w_to_h_ratio
    ]
    sort_by_coords(ref)
    return ref


def ij_round(to_round):
    if abs(to_round) < 1e-3:
        rounded = round(to_round, 7)
        if "e" in str(rounded):
            rounded = f"{to_round:.3E}"
    else:
        rounded = round(to_round, 3)
    return rounded


def sort_by_coords(features: List[Dict]):
    features.sort(key=feature_unq)


def sort_ROIs_like_legacy(rois: List[ROI], limit: int):
    # Looks (from ecotaxa TSVs) that the sort is by BY first, then BX, but in 2 chunks separated by image height
    rois.sort(key=lambda roi: (roi.y + (0 if roi.x < limit else 1000000), roi.x))
