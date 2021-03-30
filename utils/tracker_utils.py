from typing import Callable
from scipy.spatial import distance


def init_within_interval(camera_meta: dict) -> Callable:
    ref1, ref2 = camera_meta["adaptive_countintervals"][
        "3t,4t,5t,6t,lgv,tractr,2t,bus,mb"
    ]
    ref3, ref4 = camera_meta["adaptive_countintervals"]["ml,car,auto"]
    ref5, ref6 = camera_meta["adaptive_countintervals"]["tw"]

    def within_interval(pt, cls=None):
        if cls in "3t,4t,5t,6t,lgv,tractr,2t,bus,mb":
            return (ref1 + 100) < pt[0] < ref2
        elif cls in "ml,car,auto":
            return ref3 < pt[0] < ref4
        else:
            return ref5 < pt[0] < ref6

    return within_interval


def init_direction_detector(camera_meta: dict) -> Callable:
    leftlane_ref = camera_meta["leftlane_ref"]
    rightlane_ref = camera_meta["rightlane_ref"]

    def direction_detector(lane, pt1, pt2):
        if lane == "2":
            return distance.euclidean(pt1, leftlane_ref) > distance.euclidean(
                pt2, leftlane_ref
            )
        else:
            return distance.euclidean(pt1, rightlane_ref) > distance.euclidean(
                pt2, rightlane_ref
            )

    return direction_detector
