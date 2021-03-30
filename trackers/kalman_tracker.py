import math
import numpy as np
from scipy.spatial import distance
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment

from trackers import BaseTracker

# time interval
dt = 1.0

I = np.identity(4)

u = np.zeros((4, 1))

# state transition matrix assuming constant velocity model
# new_pos = old_pos + velocity * dt
F = np.matrix(
    [
        [1.0, dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, dt],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# measurement function (maps the states to measurements)
H = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

# measurement noise covariance (measurement is done only for position not velocity)
R = np.matrix([[5, 0.0], [0.0, 5]])

# Q is process covariance
Q_comp_mat = np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]])
Q = block_diag(Q_comp_mat, Q_comp_mat)


class KalmanTracker(BaseTracker):
    def _apply_kf(self, obj_id, z, lost=False):
        global dt, u, I, F, H, R, Q

        x = self.objects[obj_id].state
        x = np.array([x]).T
        x = x.astype(float)

        if z is None:

            # updating state, x' = x + v.dt + u
            x = F * x  # + u
            x = tuple(x.astype(int).ravel().tolist()[0])

            if lost:
                x = list(x)
                pt1 = (self.objects[obj_id].state[0], self.objects[obj_id].state[2])
                pt2 = (x[0], x[2])
                angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])

                objlane = self.objects[obj_id].lane
                angle = self.lane_angles[objlane] - angle

                x[0] = int(
                    pt1[0]
                    + math.cos(angle) * (pt2[0] - pt1[0])
                    - math.sin(angle) * (pt2[1] - pt1[1])
                )
                x[2] = int(
                    pt1[1]
                    + math.sin(angle) * (pt2[0] - pt1[0])
                    + math.cos(angle) * (pt2[1] - pt1[1])
                )

                x = tuple(x)

            self.objects[obj_id].P = F * self.objects[obj_id].P * F.T + Q
            self.objects[obj_id].state = x

        else:
            P_ = self.objects[obj_id].P

            z = np.array(z)
            z = z.reshape(1, 2)

            e = z.T - H * x

            PHT = P_ * H.T

            S = H * PHT + R

            K = PHT * np.linalg.inv(S)

            x += K * e

            IKH = I - K * H

            self.objects[obj_id].P = (IKH * P_ * IKH.T) + (K * R * K.T)

            x = tuple(x.astype(int).ravel().tolist())
            self.objects[obj_id].state = x

    def update(self, detection_list: list):
        if len(detection_list) == 0:
            to_deregister = []

            for obj_id, obj in self.objects.items():
                obj.absent_count += 1
                self._apply_kf(obj_id, None, lost=True)
                self.objects[obj_id].path.append(
                    (self.objects[obj_id].state[0], self.objects[obj_id].state[2])
                )

                self._update_eos(obj_id, lost=True)

                if obj.absent_count > self.max_absent:
                    to_deregister.append(obj_id)

            for obj_id in to_deregister:
                self._deregister_object(obj_id)

            return self.objects

        if len(self.objects) == 0:
            for det in detection_list:
                self._register_object(det)
                self._apply_kf(self.next_objid, det["obj_bottom"])
        else:
            obj_ids = list(self.objects.keys())
            obj_bottoms = [
                (self.objects[obj_id].state[0], self.objects[obj_id].state[2])
                for obj_id in obj_ids
            ]

            detected_rects = [det["rect"] for det in detection_list]
            detected_bottoms = [det["obj_bottom"] for det in detection_list]
            detected_classes = [det["obj_class"] for det in detection_list]

            D = distance.cdist(np.array(obj_bottoms), detected_bottoms)

            rows, cols = linear_sum_assignment(D)
            rows, cols = rows.tolist(), cols.tolist()

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if self._within_eos(obj_ids[row], detected_bottoms[col]):

                    obj_id = obj_ids[row]
                    self._apply_kf(obj_id, None)
                    self._apply_kf(obj_id, detected_bottoms[col])
                    self.objects[obj_id].rect = detected_rects[col]

                    if self.within_interval(
                        detected_bottoms[col], self.objects[obj_id].obj_class[0]
                    ):
                        if self.objects[obj_id].obj_class[1] < detected_classes[col][1]:
                            self.objects[obj_id].obj_class = detected_classes[col]

                    if len(self.objects[obj_id].path) > 10:
                        self.objects[obj_id].direction = self.direction_detector(
                            self.objects[obj_id].lane,
                            self.objects[obj_id].path[-1],
                            self.objects[obj_id].path[-10],
                        )

                    self.objects[obj_id].path.append(
                        (self.objects[obj_id].state[0], self.objects[obj_id].state[2])
                    )

                    self._update_eos(obj_id)

                    self.objects[obj_id].absent_count = 0

                    used_rows.add(row)
                    used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                to_deregister = []

                for row in unused_rows:
                    obj_id = obj_ids[row]
                    self.objects[obj_id].absent_count += 1
                    self._apply_kf(obj_id, None, lost=True)
                    self.objects[obj_id].path.append(
                        (self.objects[obj_id].state[0], self.objects[obj_id].state[2])
                    )

                    self._update_eos(obj_id, lost=True)

                    if self.objects[obj_id].absent_count > self.max_absent:
                        to_deregister.append(obj_id)

                for obj_id in to_deregister:
                    self._deregister_object(obj_id)

            else:
                for col in unused_cols:
                    self._register_object(detection_list[col])
                    self._apply_kf(self.next_objid, detected_bottoms[col])

        return self.objects
