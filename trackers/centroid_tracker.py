import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

from trackers import BaseTracker


class CentroidTracker(BaseTracker):
    def update(self, detection_list: list):
        if len(detection_list) == 0:
            to_deregister = []

            for obj_id, obj in self.objects.items():
                obj.absent_count += 1

                if len(self.objects[obj_id].path) > 2:
                    self._update_eos(obj_id, lost=True)

                if obj.absent_count > self.max_absent:
                    to_deregister.append(obj_id)

            for obj_id in to_deregister:
                self._deregister_object(obj_id)

            return self.objects

        if len(self.objects) == 0:
            for det in detection_list:
                self._register_object(det)
        else:
            obj_ids = list(self.objects.keys())
            obj_bottoms = [self.objects[obj_id].obj_bottom for obj_id in obj_ids]

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

                if D[row][col] <= self.adaptive_maxdistance(
                    detected_bottoms[col], detected_classes[col][0]
                ):

                    obj_id = obj_ids[row]
                    self.objects[obj_id].obj_bottom = tuple(detected_bottoms[col])
                    self.objects[obj_id].rect = detected_rects[col]

                    if self.within_interval(
                        detected_bottoms[col], self.objects[obj_id].obj_class[0]
                    ):
                        if self.objects[obj_id].obj_class[1] < detected_classes[col][1]:
                            self.objects[obj_id].obj_class = detected_classes[col]

                    if len(self.objects[obj_id].path) > 3:
                        self.objects[obj_id].direction = self.direction_detector(
                            self.objects[obj_id].lane,
                            self.objects[obj_id].path[-1],
                            self.objects[obj_id].path[-3],
                        )

                    self.objects[obj_id].path.append(detected_bottoms[col])

                    if len(self.objects[obj_id].path) > 2:
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

                    if len(self.objects[obj_id].path) > 2:
                        self._update_eos(obj_id, lost=True)

                    if self.objects[obj_id].absent_count > self.max_absent:
                        to_deregister.append(obj_id)

                for obj_id in to_deregister:
                    self._deregister_object(obj_id)

            else:
                for col in unused_cols:
                    self._register_object(detection_list[col])

        return self.objects
