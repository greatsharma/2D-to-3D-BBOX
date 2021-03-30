import cv2
from typing import Callable


def init_lane_detector(camera_meta: dict) -> Callable:
    leftlane_coords = camera_meta["leftlane_coords"]
    rightlane_coords = camera_meta["rightlane_coords"]

    def lane_detector(pt):
        if cv2.pointPolygonTest(leftlane_coords, pt, False) == 1:
            return "2"
        elif cv2.pointPolygonTest(rightlane_coords, pt, False) == 1:
            return "1"
        else:
            return None

    return lane_detector


def intersection_over_rect(rect1, rect2):
    if (rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) < (rect2[2] - rect2[0]) * (
        rect2[3] - rect2[1]
    ):
        rect1, rect2 = rect2, rect1

    # finds ratio of intersection and rect2 (which is smaller)

    if (
        (rect2[0] >= rect1[0])
        and (rect2[1] >= rect1[1])
        and (rect2[2] <= rect1[2])
        and (rect2[3] <= rect1[3])
    ):
        # if rect2 is completely inside rect1
        return 1
    else:
        intersection_x1 = max(rect1[0], rect2[0])
        intersection_y1 = max(rect1[1], rect2[1])
        intersection_x2 = min(rect1[2], rect2[2])
        intersection_y2 = min(rect1[3], rect2[3])

        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        intersection_area = max((intersection_x2 - intersection_x1), 0) * max(
            (intersection_y2 - intersection_y1), 0
        )

        return intersection_area / (rect2_area)


def intersection_over_union(rect1: list, rect2: list) -> float:
    intersection_x1 = max(rect1[0], rect2[0])
    intersection_y1 = max(rect1[1], rect2[1])
    intersection_x2 = min(rect1[2], rect2[2])
    intersection_y2 = min(rect1[3], rect2[3])

    intersection_area = max((intersection_x2 - intersection_x1), 0) * max(
        (intersection_y2 - intersection_y1), 0
    )

    if intersection_area == 0:
        return 0

    rect1_area = abs((rect1[2] - rect1[0]) * (rect1[3] - rect1[1]))
    rect2_area = abs((rect2[2] - rect2[0]) * (rect2[3] - rect2[1]))

    return intersection_area / (rect1_area + rect2_area - intersection_area + 1e-6)


def nonmax_suppression(detection_list: list, iou_thresh: float) -> list:
    nms_detection_list = []

    detection_list = sorted(
        detection_list, key=lambda x: x["obj_class"][1], reverse=True
    )

    while detection_list:
        detection = detection_list.pop(0)
        parent_rect = detection["rect"]

        keep = []
        for det in detection_list:
            rect = det["rect"]

            if intersection_over_union(parent_rect, rect) < iou_thresh:
                keep.append(det)

        detection_list = keep
        nms_detection_list.append(detection)

    return nms_detection_list
