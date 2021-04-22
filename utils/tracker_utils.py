from typing import Callable
from scipy.spatial import distance
from utils.detector_utils import intersection_over_rect


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


def _get_axleconfig(axles):
    axle_dists = []

    for i in range(len(axles) - 1):
        dist = distance.euclidean(axles[i], axles[i + 1])
        axle_dists.append(dist)

    num_axles = len(axles)

    if num_axles == 3:
        axleconfig = "12"
    elif num_axles == 4:
        if axle_dists[0] > axle_dists[1] and axle_dists[0] > axle_dists[2]:
            axleconfig = "13"
        else:
            axleconfig = "112"
    elif num_axles == 5:
        if axle_dists[1] > axle_dists[3]:
            axleconfig = "113"
        else:
            axleconfig = "122"
    elif num_axles == 6:
        axleconfig = "123"
    else:
        axleconfig = "11"

    return axleconfig


def axle_assignments(tracked_objs, axles, sort_order):
    if sort_order not in ["asce", "desc"]:
        raise ValueError("Invalid sort_order, choose one among [asce, desc]")

    _axles = axles.copy()

    for obj in tracked_objs.values():
        if obj.obj_class[0] != "tw":
            obj_ax = []

            temp = []
            for ax in _axles:
                if intersection_over_rect(obj.rect, ax) > 0.9:
                    obj_ax.append(ax)
                else:
                    temp.append(ax)

            if sort_order == "asce":
                obj_ax = sorted(obj_ax, key=lambda x: x[0])
            else:
                obj_ax = sorted(obj_ax, key=lambda x: x[0], reverse=True)

            obj.axle_track.append(obj_ax)

            if len(obj_ax) >= 1:
                largest_axle = sorted(obj_ax, key=lambda x: x[3]-x[1])[0]
                if not obj.lastdetected_axle or (largest_axle[3] - largest_axle[1]) >= (obj.lastdetected_axle[3] - obj.lastdetected_axle[1]):
                    obj.lastdetected_axle = largest_axle
    
            if len(obj_ax) > len(obj.max_axles_detected):
                obj.max_axles_detected = obj_ax

                if (obj.obj_class[0] in ["3t", "4t", "5t", "6t"]):
                    obj.obj_class[0][0] = str(len(obj.max_axles_detected))
                    obj.axle_config = _get_axleconfig(obj.max_axles_detected)

            _axles = temp
            if len(_axles) == 0:
                break
