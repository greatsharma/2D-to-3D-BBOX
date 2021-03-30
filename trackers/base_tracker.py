import math
import numpy as np
from typing import Callable
from scipy.spatial import distance
from collections import deque, OrderedDict


class EllipseofSearch(object):
    def __init__(self, centre, semi_majoraxis, semi_minoraxis, angle):
        self.centre = centre
        self.semi_majoraxis = semi_majoraxis
        self.semi_minoraxis = semi_minoraxis
        self.angle = angle
        self.last_d = 0


class VehicleObject(object):
    def __init__(
        self, objid, obj_bottom, rect, lane, direction, path, absent_count, obj_class
    ) -> None:

        self.objid = objid
        self.obj_bottom = obj_bottom
        self.rect = rect
        self.lane = lane
        self.direction = direction
        self.path = path
        self.absent_count = absent_count
        self.obj_class = obj_class

        self.starttime = None
        self.endtime = None

        self.axles = []  # only for trucks
        self.axle_config = None
        self.axle_track = []

        # self.state_list = []
        self.state = [0] * 4  # this attribute is for kalman tracker only

        # state uncertainity covariance matrix, this attribute is for kalman tracker only
        self.P = np.matrix(
            [
                [1000.0, 0.0, 0.0, 0.0],
                [0.0, 1000.0, 0.0, 0.0],
                [0.0, 0.0, 1000.0, 0.0],
                [0.0, 0.0, 0.0, 1000.0],
            ]
        )


class BaseTracker(object):
    def __init__(
        self,
        direction_detector: Callable,
        initial_maxdistances: dict,
        within_interval: Callable,
        lane_angles: dict,
        velocity_regression: dict,
        max_absent: int,
    ) -> None:

        self.direction_detector = direction_detector
        self.initial_maxdistances = initial_maxdistances
        self.within_interval = within_interval
        self.lane_angles = lane_angles
        self.velocity_regression = velocity_regression
        self.max_absent = max_absent

        self.next_objid = 0
        self.objects = OrderedDict()
        self.trackpath_filewriter = None

    def _register_object(self, detection):
        self.next_objid += 1

        self.objects[self.next_objid] = VehicleObject(
            self.next_objid,
            detection["obj_bottom"],
            detection["rect"],
            detection["lane"],
            True,
            [],
            0,
            detection["obj_class"],
        )

        self.objects[self.next_objid].path.append(detection["obj_bottom"])

        for k, v in self.initial_maxdistances.items():
            if detection["obj_class"][0] in k:
                semi_majoraxis = v

        if detection["lane"] == "2":
            angle = 18
        else:
            angle = 10

        if detection["obj_class"][0] in "2t,3t,4t,5t,6t,lgv,tractr,bus,mb":
            semi_minoraxis = semi_majoraxis // 2
        else:
            semi_minoraxis = semi_majoraxis // 3

        self.objects[self.next_objid].eos = EllipseofSearch(
            detection["obj_bottom"], semi_majoraxis, semi_minoraxis, angle
        )

    def _update_eos(self, obj_id, lost=False) -> None:
        self.objects[obj_id].eos.centre = self.objects[obj_id].path[-1]

        if len(self.objects[obj_id].path) <= 2:
            return

        pt1, pt2 = self.objects[obj_id].path[-2], self.objects[obj_id].path[-1]
        dy, dx = (pt2[1] - pt1[1]), (pt2[0] - pt1[0])
        angle1 = math.degrees(math.atan2(dy, dx))

        pt3, pt4 = self.objects[obj_id].path[-3], self.objects[obj_id].path[-2]
        dy, dx = (pt4[1] - pt3[1]), (pt4[0] - pt3[0])
        angle2 = math.degrees(math.atan2(dy, dx))

        self.objects[obj_id].eos.angle = (angle1 + angle2) / 2

        self.objects[obj_id].eos.last_d = distance.euclidean(pt1, pt2)

        if not lost:
            self.objects[obj_id].eos.semi_majoraxis = max(
                int(2.25 * self.objects[obj_id].eos.last_d), 30
            )

            if self.objects[obj_id].obj_class[0] in "2t,3t,4t,5t,6t,lgv,tractr,bus,mb":
                self.objects[obj_id].eos.semi_minoraxis = max(
                    int(self.objects[obj_id].eos.semi_majoraxis / 2.5), 20
                )
            else:
                self.objects[obj_id].eos.semi_minoraxis = max(
                    int(self.objects[obj_id].eos.semi_majoraxis / 4), 15
                )
        else:
            self.objects[obj_id].eos.semi_majoraxis = max(
                int(2.2 * self.objects[obj_id].eos.last_d), 30
            )

            if self.objects[obj_id].obj_class[0] in "2t,3t,4t,5t,6t,lgv,tractr,bus,mb":
                self.objects[obj_id].eos.semi_minoraxis = max(
                    int(self.objects[obj_id].eos.semi_majoraxis / 2), 25
                )
            else:
                self.objects[obj_id].eos.semi_minoraxis = max(
                    int(self.objects[obj_id].eos.semi_majoraxis / 3), 22
                )

        n = self.objects[obj_id].eos.last_d * 2
        m = -self.objects[obj_id].eos.last_d * 1

        if self.objects[obj_id].eos.last_d != 0 and not lost:
            self.objects[obj_id].eos.centre = (
                int((m * pt1[0] + n * pt2[0]) / (m + n + 1e-6)),
                int((m * pt1[1] + n * pt2[1]) / (m + n + 1e-6)),
            )

    def _within_eos(self, obj_id, pt):
        a = self.objects[obj_id].eos.semi_majoraxis
        b = self.objects[obj_id].eos.semi_minoraxis
        h = self.objects[obj_id].eos.centre[0]
        k = self.objects[obj_id].eos.centre[1]
        x, y = pt
        angle = math.radians(self.objects[obj_id].eos.angle)

        cosa = math.cos(angle)
        sina = math.sin(angle)

        n1 = math.pow(cosa * (x - h) + sina * (y - k), 2)
        n2 = math.pow(sina * (x - h) - cosa * (y - k), 2)

        d1 = a * a
        d2 = b * b

        return (n1 / d1) + (n2 / d2) <= 1

    def _deregister_object(self, obj_id) -> None:
        txt = f"{obj_id} : {self.objects[obj_id].obj_class[0]}"

        if self.objects[obj_id].obj_class[0] in [
            "2t",
            "3t",
            "4t",
            "5t",
            "6t",
            "bus",
            "lgv",
        ]:
            if len(self.objects[obj_id].axle_track) >= 2:
                txt += f" : {self.objects[obj_id].axle_track}"
                self.trackpath_filewriter.write(txt + "\n")
        else:
            if len(self.objects[obj_id].path) >= 2:
                txt += f" : {self.objects[obj_id].path}"
                self.trackpath_filewriter.write(txt + "\n")

        del self.objects[obj_id]

    def update(self, detection_list: list):
        raise NotImplementedError("Function `update` is not implemented !")
