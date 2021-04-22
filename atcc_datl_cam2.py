import os
import cv2
import time
import datetime
import subprocess
import numpy as np

from utils import axle_assignments
from trackers import KalmanTracker
from camera_metadata import CAMERA_METADATA
from detectors.trt_detector import TrtYoloDetector
from utils import init_lane_detector, init_direction_detector, init_within_interval
from utils import draw_text_with_backgroud, draw_tracked_objects, draw_3dbox


WRITE_VIDEO = False

vidcap1 = cv2.VideoCapture("inputs/datlcam1_clip1.mp4")
width1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2 = cv2.VideoCapture("inputs/datlcam2_clip1.mp4")
width2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2.set(cv2.CAP_PROP_POS_FRAMES, 505)

_, initial_frame1 = vidcap1.read()
_, initial_frame2 = vidcap2.read()

initial_frame1 = cv2.resize(initial_frame1, dsize=(width1//2, height1//2))

camera_meta = CAMERA_METADATA["datlcam2"]

detector = TrtYoloDetector(
    initial_frame1,
    init_lane_detector(camera_meta),
    detection_thresh=0.5,
    bottom_type="bottom-left"
)

direction_detector = init_direction_detector(camera_meta)
within_interval = init_within_interval(camera_meta)
initial_maxdistances = camera_meta["initial_maxdistances"]
lane_angles = camera_meta["lane_angles"]
velocity_regression = None

tracker = KalmanTracker(
    direction_detector,
    initial_maxdistances,
    within_interval,
    lane_angles,
    velocity_regression,
    max_absent=1
)

date = datetime.datetime.now()
date = date.strftime("%d_%m_%Y_%H:%M:%S")
date = date.replace(":", "")

curr_folder = "outputs/datl/" + date

if not os.path.exists(curr_folder):
    os.mkdir(curr_folder)

outvideo_path = curr_folder + "/linear_mapping.avi"

if WRITE_VIDEO:
    videowriter = cv2.VideoWriter(
        outvideo_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        50.0,
        (960, 540),
    )


def line_intersect(A1, A2, B1, B2):
    Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2 = A1[0], A1[1], A2[0], A2[1], B1[0], B1[1], B2[0], B2[1]
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        return
    if not(0 <= uA <= 1 and 0 <= uB <= 1):
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
 
    return int(x), int(y)


def twoD_2_threeD_secondarycam(obj):
    rect = obj.rect
    rect = list(rect)
    objcls = obj.obj_class[0]

    if objcls in ["tw", "auto", "car", "ml"]:
        height_ratio = 0.05
        width_ratio =  0.00039 * rect[2] + 0.396
    elif obj.lane == "3":
        height_ratio = 0.06
        width_ratio = -0.000304 * rect[0] + 0.388
    else:
        height_ratio = 0.08
        width_ratio = 0.000582 * rect[0] + 0.0442

    height = (rect[3] - rect[1])

    if objcls in ["tw", "auto", "car", "ml"]:
        rect[3] += int(height * 0.08)
    else:
        if len(obj.axles) > 0:
            last_axle = obj.axles[0]
            lastaxle_btm_midpt = int((last_axle[0] + last_axle[2])/2), last_axle[3]
        elif obj.lane == "1":
            rect[3] += int(height * 0.12)
        else:
            rect[3] += int(height * 0.1)

    pt1 = rect[2], int(rect[1] + height * height_ratio)

    width = (rect[2] - rect[0])
    pt2 = int(rect[2] - width * width_ratio), rect[1]

    c1, c2 = pt2, (1227, -35)
    cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)

    pt3 = line_intersect(pt2, (cx,cy), (rect[0], rect[1]), (rect[0], rect[3]))

    c1, c2 = pt1, (1227, -35)
    cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)

    pt4_temp = line_intersect(pt1, (cx,cy), (rect[0], rect[1]), (rect[0], rect[3]))
    if pt4_temp is None:
        pt4_temp = line_intersect(pt1, (cx,cy), (rect[0], rect[3]), (rect[2], rect[3]))

    m = -(pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
    if objcls == "tw":
        m += m
    elif objcls in ["auto", "car", "ml"]:
        m += 4*m
    c = -pt3[1] - m * pt3[0]
    pt_temp = int((-540-c)/m), 540
    pt4 = line_intersect(pt1, pt4_temp, pt3, pt_temp)

    pt_temp2 = pt4[0], pt4[1] + 2*height

    try:
        c1, c2 = lastaxle_btm_midpt, (1227, -35)
        cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
        cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)
        pt5 = line_intersect(pt4, pt_temp2, lastaxle_btm_midpt, (cx,cy))
        if pt5 is None:
            raise UnboundLocalError
    except UnboundLocalError:
        if objcls in ["tw", "auto", "car", "ml"]:
            c1, c2 = pt2, (1227, -35)
            cx = int(c2[0] + (c1[0]-c2[0]) * -3.8)
            cy = int(c2[1] + (c1[1]-c2[1]) * -3.8)

            c = -pt1[1] - m * pt1[0]
            pt_temp = 0, int(-c)
            
            pt2 = line_intersect(pt2, (cx,cy), pt1, pt_temp)
        
        pt5 = line_intersect(pt4, pt_temp2, (rect[0], rect[3]), (rect[2], rect[3]))

    m = -(pt3[1] - pt4[1]) / (pt3[0] - pt4[0])
    if objcls in ["tw", "auto", "car", "ml"]:
        m += 0.2 * m
    else:
        m += 0.5 * m
    c = -pt5[1] - m * pt5[0]
    pt_temp = int((0-c)/m), 0

    pt6 = line_intersect(pt5, pt_temp, (rect[0], rect[1]), (rect[0], rect[3]+height))
    pt7 = line_intersect(pt5, (1227, -35), (rect[2], rect[1]), (rect[2], rect[3]))

    if obj.lane == "1" and objcls not in ["tw", "auto", "car", "ml"]:
        btm_pt = int(0.4*pt5[0] + 0.6*pt6[0]), int(0.4*pt5[1] + 0.6*pt6[1])
    else:
        btm_pt = int(0.5*pt5[0] + 0.5*pt6[0]), int(0.5*pt5[1] + 0.5*pt6[1])

    return btm_pt, [pt1, pt2, pt3, pt4, pt5, pt6, pt7]


frame_count = 0

while vidcap2.isOpened():
    start_time = time.time()

    status2, frame2 = vidcap2.read()

    if not status2:
        print("status2 false")
        break

    frame2 = cv2.resize(frame2, dsize=(width2//2, height2//2))

    # for l in ["leftlane", "middlelane", "rightlane"]:
    #     cv2.polylines(frame2, [camera_meta[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

    frame_count += 1

    detection_list, axles = detector.detect(frame2)

    tracked_objects = tracker.update(detection_list)

    axle_assignments(tracked_objects, axles, sort_order="asce")

    for obj in tracked_objects.values():
        obj.obj_bottom, obj.threed_box = twoD_2_threeD_secondarycam(obj)
        
        btm = obj.obj_bottom
        lane = obj.lane

        # if obj.obj_class[0] not in ["car", "ml", "auto", "tw"]:
        #     rect = obj.rect
        #     cv2.rectangle(frame2, rect[:2], rect[2:], (255,0,0), 1)

        draw_3dbox(frame2, obj.threed_box)
        cv2.circle(frame2, btm, 3, (0,0,255), -1)

        try:
            for ax in obj.axle_track[-1]:
                cv2.rectangle(frame2, ax[:2], ax[2:], (255,0,255), 3)
        except IndexError:
            pass

    if WRITE_VIDEO:
        videowriter.write(frame2)

    cv2.imshow("video", frame2)

    key = cv2.waitKey(1)
    if key == ord('q'):
       break
    elif key == ord('p'):
        cv2.waitKey(-1)
    
    print(f"frame_count: {frame_count}, fps: {1.0 / (time.time()-start_time)}")

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()

if WRITE_VIDEO:
    status = subprocess.call(
        [
            "ffmpeg",
            "-i",
            outvideo_path,
            "-vcodec",
            "libx264",
            "-crf",
            "28",
            curr_folder + "/linear_mapping_comp.avi",
        ]
    )

    if status:
        print(f"\n unable to compress {outvideo_path} ! \n")
    else:
        os.remove(outvideo_path)
