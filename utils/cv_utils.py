import cv2
import math
from math import atan2, degrees
from scipy.spatial import distance


class_colors = {
    "tw": (255, 0, 0),
    "car": (0, 255, 0),
    "lgv": (100, 0, 100),
    "2t": (0, 255, 255),
    "3t": (255, 0, 255),
    "4t": (255, 255, 0),
    "5t": (138, 43, 226),
    "6t": (0, 0, 0),
    "bus": (255, 255, 255),
    "ml": (255, 215, 0),
    "auto": (128, 0, 128),
    "mb": (139, 69, 19),
    "tractr": (255, 153, 153),
    "axle": (100, 200, 100),
}


def draw_text_with_backgroud(
    img,
    text,
    x,
    y,
    font_scale,
    thickness=1,
    font=cv2.FONT_HERSHEY_COMPLEX,
    background=None,
    foreground=(10, 10, 10),
    box_coords_1=(-5, 5),
    box_coords_2=(5, -5),
):
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]

    box_coords = (
        (x + box_coords_1[0], y + box_coords_1[1]),
        (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]),
    )

    if background is not None:
        cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        fontScale=font_scale,
        color=foreground,
        thickness=thickness,
    )


def _checkpoint(h, k, x, y, a, b, angle):
    angle = math.radians(angle)

    cosa = math.cos(angle)
    sina = math.sin(angle)

    n1 = math.pow(cosa * (x - h) + sina * (y - k), 2)
    n2 = math.pow(sina * (x - h) - cosa * (y - k), 2)

    d1 = a * a
    d2 = b * b

    return (n1 / d1) + (n2 / d2)


def draw_tracked_objects(self, frame, tracked_objs):
    global class_colors

    to_deregister = []

    for obj in tracked_objs.values():
        obj_rect = obj.rect
        obj_centroid = (obj_rect[0] + obj_rect[2]) // 2, (
            obj_rect[1] + obj_rect[3]
        ) // 2
        obj_bottom = (
            obj.obj_bottom
            if self.tracker_type == "centroid"
            else (obj.state[0], obj.state[2])
        )

        if (
            self.lane_detector(obj_bottom) is None
            and obj_bottom[0] < self.camera_meta["adaptive_countintervals"]["tw"][0]
        ):
            to_deregister.append(
                (obj.objid, self.lane_detector(obj_bottom), obj.absent_count)
            )
            continue

        if self.mode == "debug":
            cv2.circle(frame, obj_bottom, radius=3, color=(0, 0, 0), thickness=-1)

        base_color = class_colors[obj.obj_class[0]]

        if obj.absent_count == 0:
            x, y = obj_centroid[0] - 10, obj_centroid[1]
            cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)
        else:
            x, y = obj_bottom[0] - 10, obj_bottom[1]

        if obj.direction:
            txt = obj.obj_class[0]
            if self.mode != "pretty":
                txt = str(obj.objid) + ": " + obj.obj_class[0]

            draw_text_with_backgroud(
                frame,
                txt,
                x,
                y,
                font_scale=0.6,
                thickness=2,
                background=(243, 227, 218),
                foreground=(0, 0, 0),
                box_coords_1=(-7, 7),
                box_coords_2=(10, -10),
            )
        else:
            base_color = [0, 0, 255]

            txt = "Wrong Direction"
            if self.mode != "pretty":
                txt = str(obj.objid) + ": " + "Wrong Direction"

            draw_text_with_backgroud(
                frame,
                txt,
                x,
                y,
                font_scale=0.5,
                thickness=1,
                foreground=(0, 0, 0),
                background=(0, 0, 255),
            )

        if len(obj.path) <= self.max_track_pts:
            path = obj.path
        else:
            path = obj.path[len(obj.path) - self.max_track_pts :]

        prev_point = None
        for pt in path:
            if not prev_point is None:
                cv2.line(
                    frame,
                    (prev_point[0], prev_point[1]),
                    (pt[0], pt[1]),
                    base_color,
                    thickness=2,
                    lineType=8,
                )
            prev_point = pt

        centre = obj.eos.centre
        semi_majoraxis = obj.eos.semi_majoraxis
        semi_minoraxis = obj.eos.semi_minoraxis
        angle = obj.eos.angle

        if len(path) > 2:
            cv2.arrowedLine(frame, path[-2], path[-1], (0, 0, 255), 2)

        if self.mode == "debug":
            cv2.circle(frame, centre, radius=4, color=(0, 255, 0), thickness=-1)
            cv2.circle(frame, obj_bottom, radius=4, color=(0, 0, 255), thickness=-1)
            cv2.ellipse(
                frame,
                center=centre,
                axes=(semi_majoraxis, semi_minoraxis),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=base_color,
                thickness=2,
            )

        v = _checkpoint(
            centre[0],
            centre[1],
            obj_bottom[0],
            obj_bottom[1],
            semi_majoraxis,
            semi_minoraxis,
            angle,
        )
        if v > 1:
            print(f"objid: {obj.objid}, v: {v}, out\n\n")
        elif v == 1:
            print(f"objid: {obj.objid}, v: {v}, on\n\n")

    for obj_id, _, _ in to_deregister:
        self.tracker._deregister_object(obj_id)


def draw_axles(frame, axles):
    for axle in axles:
        cv2.rectangle(frame, axle[:2], axle[2:], (255, 255, 0), 2)


def draw_3dbox(frame, pts, boxcolor=(0,255,0)):
    pt1, pt2, pt3, pt4, pt5, pt6, pt7 = pts
    cv2.line(frame, pt1, pt2, boxcolor, 2)
    cv2.line(frame, pt2, pt3, boxcolor, 2)
    cv2.line(frame, pt1, pt4, boxcolor, 2)
    cv2.line(frame, pt3, pt4, boxcolor, 2)
    cv2.line(frame, pt4, pt5, boxcolor, 2)
    cv2.line(frame, pt5, pt6, boxcolor, 2)
    cv2.line(frame, pt3, pt6, boxcolor, 2)
    cv2.line(frame, pt5, pt7, boxcolor, 2)
    cv2.line(frame, pt1, pt7, boxcolor, 2)