import numpy as np


CAMERA_METADATA = {
    "datlcam1": {
        "leftlane_coords": np.array(
            [(200, 520), (703, 521), (72, 197), (10, 228), (10, 330)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "middlelane_coords": np.array(
            [(703, 521), (872, 472), (129, 165), (72, 197)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "rightlane_coords": np.array(
            [(872, 472), (950, 316), (223, 110), (129, 165)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "leftlane_ref": (731, 535),
        "middlelane_ref": (878, 533),
        "rightlane_ref": (924, 467),
        "mid_ref": 250,
        "adaptive_countintervals": {
            "3t,4t,5t,6t,lgv,tractr,2t,bus,mb": [300, 675],
            "ml,car,auto": [275, 650],
            "tw": [275, 550],
        },
        "initial_maxdistances": {
            "4t,5t,6t,tractr,bus,mb": 120,
            "ml,auto,2t,3t,lgv": 140,
            "tw,car": 150,
        },
        "lane_angles": {
            "1": 2.772,
            "2": 2.624,
        },
    },
    "datlcam2": {
        "leftlane_coords": np.array(
            [(20, 185), (20, 335), (690, 134), (505, 91)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "middlelane_coords": np.array(
            [(20, 335), (20, 509), (806, 158), (690, 134)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "rightlane_coords": np.array(
            [(20, 509), (20, 525), (462, 525), (934, 187), (806, 158)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "leftlane_ref": (5, 300),
        "middlelane_ref": (4, 417),
        "rightlane_ref": (5, 535),
        "mid_ref": 250,
        "adaptive_countintervals": {
            "3t,4t,5t,6t,lgv,tractr,2t,bus,mb": [300, 675],
            "ml,car,auto": [275, 650],
            "tw": [275, 550],
        },
        "initial_maxdistances": {
            "4t,5t,6t,tractr,bus,mb": 120,
            "ml,auto,2t,3t,lgv": 140,
            "tw,car": 150,
        },
        "lane_angles": {
            "1": 2.772,
            "2": 2.624,
        },
    },
}


if __name__ == "__main__":
    from pprint import pprint
    pprint(CAMERA_METADATA)
