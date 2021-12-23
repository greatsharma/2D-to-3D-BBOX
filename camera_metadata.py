import numpy as np


CAMERA_METADATA = {
    "datlcam1": {
        "leftlane_coords": np.array(
            [(318, 513), (628, 482), (72, 197), (10, 228), (11, 270)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "middlelane_coords": np.array(
            [(628, 482), (783, 434), (143, 170), (72, 197)], dtype=np.int32
        ).reshape((-1, 1, 2)),
        "rightlane_coords": np.array(
            [(783, 434), (885, 300), (245, 116), (143, 170)], dtype=np.int32
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
            "4t,5t,6t,tractr,bus,mb": 80,
            "ml,auto,2t,3t,lgv": 100,
            "tw,car": 110,
        },
        "lane_angles": {
            "1": 2.789,
            "2": 2.722,
            "3": 2.616
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
            "4t,5t,6t,tractr,bus,mb": 110,
            "ml,auto,2t,3t,lgv": 130,
            "tw,car": 140,
        },
        "lane_angles": {
            "1": 0.482,
            "2": 0.331,
            "3": 0.235
        },
    },
}


if __name__ == "__main__":
    from pprint import pprint
    pprint(CAMERA_METADATA)
