#
# Various classification of existing projects, according to their characteristics, e.g. image quality
#
from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder
from .env_fixture import read_home
from .projects_for_test import (
    IADO,
    APERO2000,
    APERO,
    TRIATLAS,
    APERO1,
    APERO_REDUCED2,
    POINT_B_JB,
)


def all_samples_in(project: str, but_not=()) -> list[tuple[str, str]]:
    folder = ZooscanProjectFolder(read_home(), project)
    scans = folder.zooscan_scan.list_samples()
    return [(project, a_scan) for a_scan in sorted(scans) if a_scan not in but_not]


tested_samples = (
    all_samples_in(IADO)
    + all_samples_in(
        APERO2000,
        [
            "apero2023_tha_bioness_sup2000_013_st46_d_n4_d1_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_sup2000_013_st46_d_n4_d2_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_sup2000_016_st55_d_n9_d2_1_sur_1",  # Corrupted ZIP
        ],
    )
    + all_samples_in(
        APERO,
        [
            "apero2023_tha_bioness_014_st46_n_n5_d1_1_sur_1",  # Corrupted ZIP
            "apero2023_tha_bioness_013_st46_d_n5_d1_1_sur_1",  # Corrupted ZIP
        ],
    )
    + all_samples_in(TRIATLAS)
    + all_samples_in(APERO1)
)

APERO_tested_samples = all_samples_in(APERO1)
APERO_tested_samples_raw_to_work = all_samples_in(
    APERO1,
    [
        "apero2023_tha_bioness_005_st20_d_n7_d1_1_sur_1",  # output diff
        "apero2023_tha_bioness_005_st20_d_n7_d2_3_sur_4",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d1_1_sur_2",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d1_2_sur_2",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d2_1_sur_1",  # output diff
        "apero2023_tha_bioness_013_st46_d_n1_d3",  # output diff
        "apero2023_tha_bioness_017_st66_d_n1_d2_1_sur_4",  # tiff problem?
        "apero2023_tha_bioness_018_st66_n_n1_d1_1_sur_2",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n1_d1_2_sur_2",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n3_d2_1_sur_1",  # AttributeError
        "apero2023_tha_bioness_018_st66_n_n3_d3",  # AttributeError
    ],
)

extra_big = [
    (APERO, "apero2023_tha_bioness_014_st46_n_n6_d1_1_sur_1"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d1_2_sur_4"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n2_d1_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_006_st20_n_n7_d1_2_sur_2"),
    (TRIATLAS, "m158_mn18_n2_d1_1_sur_4"),
]

missingd = [
    (APERO, "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2"),
    (APERO, "apero2023_tha_bioness_014_st46_n_n7_d2_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d1_1_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n1_d1_2_sur_2"),
    (APERO1, "apero2023_tha_bioness_005_st20_d_n3_d1_3_sur_4"),
    (APERO1, "apero2023_tha_bioness_017_st66_d_n3_d2_1_sur_1"),
]

more_than25_is_black_4_borders_closed = [
    (APERO1, "apero2023_tha_bioness_018_st66_n_n8_d3"),
]

more_than25_is_black_4_borders_unclosed = [
    (APERO1, "apero2023_tha_bioness_017_st66_d_n7_d2_2_sur_2"),
]

more_than25_is_black = (
    more_than25_is_black_4_borders_closed + more_than25_is_black_4_borders_unclosed
)

stripes_in_thresholded = (
    more_than25_is_black_4_borders_closed
    + more_than25_is_black_4_borders_unclosed
    + [
        (APERO1, "apero2023_tha_bioness_018_st66_n_n6_d1_1_sur_1"),
        (APERO1, "apero2023_tha_bioness_017_st66_d_n4_d2_2_sur_2"),
        (APERO1, "apero2023_tha_bioness_013_st46_d_n1_d1_1_sur_2"),
    ]
)

very_big = [(APERO1, "apero2023_tha_bioness_006_st20_n_n7_d1_2_sur_2")]

wrong_mask_maybe_gives_no_roi_when_legacy_has = [
    (APERO1, "apero2023_tha_bioness_013_st46_d_n1_d2_1_sur_1"),
]

different_algo_diff_outputs = sorted(extra_big + missingd + more_than25_is_black)

closed_statuses = {
    (IADO, "s_17_3_tot"): "all_borders_unclosed",
    (IADO, "s_17_6_tot"): "all_borders_unclosed",
    (IADO, "t_22_6_tot"): "all_borders_unclosed",
    (IADO, "t_22_8_tot"): "all_borders_unclosed",
    (
        APERO2000,
        "apero2023_tha_bioness_sup2000_017_st66_d_n1_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO2000,
        "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO2000,
        "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO2000,
        "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_3_sur_4",
    ): "all_borders_closed",
    (
        APERO2000,
        "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n3_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n3_d2_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n3_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n4_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n4_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n4_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n5_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n5_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n5_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n6_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n6_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n6_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n6_d3",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n7_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n7_d3",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n8_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n8_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n8_d3",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n9_d2_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_013_st46_d_n9_d3",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n1_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n1_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n1_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n1_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n2_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n2_d1_2_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n2_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n2_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n2_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n3_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n3_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n3_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n3_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n4_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n4_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n4_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n4_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n5_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n5_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n5_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n6_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n6_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n6_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n6_d3",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d1_1_sur_4",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d1_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d1_4_sur_4",
    ): "all_borders_closed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n7_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_1_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_2_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_3_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_4_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_5_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_6_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_7_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d1_8_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_1_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_2_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_3_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_4_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_5_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_6_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_7_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d2_8_sur_8",
    ): "all_borders_unclosed",
    (
        APERO,
        "apero2023_tha_bioness_014_st46_n_n9_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn03_n2_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn03_n5_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n2_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n4_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n4_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n5_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n5_d1_3_sur_4",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n5_d1_4_sur_4",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n5_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn04_n5_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn05_n1_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn05_n1_d2",
    ): "all_borders_closed",
    (
        TRIATLAS,
        "m158_mn05_n2_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn05_n3_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn05_n3_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn05_n4_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn06_n4_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn11_n1_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn15_n2_d2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn18_n1_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn18_n4_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n1_d1",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n1_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n2_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n2_d3",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n5_d1_1_sur_5",
    ): "all_borders_unclosed",
    (
        TRIATLAS,
        "m158_mn19_n5_d1_2_sur_5",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n1_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n1_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n1_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n1_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n1_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n2_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d1_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d1_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d1_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n3_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n4_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_1_sur_8",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_2_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_3_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_4_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_5_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_6_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_7_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d2_8_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n5_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d1_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d1_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d1_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_1_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_2_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_4_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_5_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_6_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_7_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d2_8_sur_8",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n6_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d2_1_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n7_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n8_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n8_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n8_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_005_st20_d_n9_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n1_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n1_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n1_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n1_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d2_3_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n2_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n3_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n3_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n3_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n3_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n4_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n4_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n4_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n4_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n5_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d1_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d1_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d1_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d1_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n6_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d1_2_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d2_4_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d2_3_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d2_4_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n8_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d2_1_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d2_2_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d2_4_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n9_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_013_st46_d_n1_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_013_st46_d_n1_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_013_st46_d_n1_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_013_st46_d_n1_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d1_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d2_1_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d2_2_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d2_3_sur_4",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d2_4_sur_4",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n1_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n2_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n2_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n2_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n2_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n3_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n3_d2_1_sur_1",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n3_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n4_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n4_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n4_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n4_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n5_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n5_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n5_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n6_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n6_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n6_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n6_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n7_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n7_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n8_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n8_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n8_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n9_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n9_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n9_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_017_st66_d_n9_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n1_d1_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n1_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n1_d2_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n1_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n2_d1_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n2_d1_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n2_d2_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n2_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n3_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n3_d2_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n3_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n4_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n4_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n4_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n4_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n5_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n5_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n5_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n5_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n6_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n6_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n6_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n7_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n7_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n7_d2_2_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n7_d3",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n8_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n8_d2_1_sur_2",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n8_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n8_d3",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n9_d1_1_sur_1",
    ): "all_borders_unclosed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n9_d2_1_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n9_d2_2_sur_2",
    ): "all_borders_closed",
    (
        APERO1,
        "apero2023_tha_bioness_018_st66_n_n9_d3",
    ): "all_borders_closed",
}

all_borders_unclosed = [
    k for (k, v) in closed_statuses.items() if v == "all_borders_unclosed"
]

all_borders_closed = [
    k for (k, v) in closed_statuses.items() if v == "all_borders_closed"
]

lost_shrimp = [
    (  # In this image, a big shrimp touches an image border on the right, we lose it
        APERO1,
        "apero2023_tha_bioness_006_st20_n_n7_d1_1_sur_2",
    ),
]

embedded_in_border_objects = (
    [  # In these, there are small particles inside bigger ones, middle of the image
        (
            APERO,
            "apero2023_tha_bioness_014_st46_n_n9_d1_2_sur_8",
        ),
        (
            APERO,
            "apero2023_tha_bioness_014_st46_n_n9_d1_5_sur_8",
        ),
        (TRIATLAS, "m158_mn04_n5_d1_4_sur_4"),
        (
            APERO1,
            "apero2023_tha_bioness_005_st20_d_n5_d2_2_sur_8",
        ),
        (
            APERO1,
            "apero2023_tha_bioness_006_st20_n_n1_d1_1_sur_1",
        ),
        (
            APERO1,
            "apero2023_tha_bioness_018_st66_n_n3_d1_1_sur_1",
        ),
        (APERO1, "apero2023_tha_bioness_018_st66_n_n8_d3"),
    ]
)

parallel_with_ij = [
    (APERO_REDUCED2, "apero2023_tha_bioness_013_st46_d_n4_d2_2_sur_2")
]  # LS: Coded in ImageJ locally

point_b_jb = all_samples_in(POINT_B_JB)
