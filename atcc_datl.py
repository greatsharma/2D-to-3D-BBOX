import os
import cv2
import time
import datetime
import subprocess
import numpy as np

from utils import init_lane_detector
from camera_metadata import CAMERA_METADATA
from detectors.trt_detector import TrtYoloDetector


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

camera_meta = CAMERA_METADATA["datlcam1"]

detector = TrtYoloDetector(
    initial_frame1,
    init_lane_detector(camera_meta),
    detection_thresh=0.4,
)

date = datetime.datetime.now()
date = date.strftime("%d_%m_%Y_%H:%M:%S")
date = date.replace(":", "")

curr_folder = "outputs/datl/" + date

if not os.path.exists(curr_folder):
    os.mkdir(curr_folder)

outvideo_path = curr_folder + "/linear_mapping.avi"

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


def twod_2_threed(frame1, det, boxcolor=(0,255,0)):
    rect = det["rect"]

    if det["obj_class"][0] in ["tw", "auto", "car", "ml"]:
        height_ratio = 0.15
        width_ratio = -0.000353 * rect[2] + 0.595
    elif det["lane"] == "1":
        height_ratio = 0.06
        width_ratio = -0.000304 * rect[2] + 0.388
    else:
        height_ratio = 0.06
        width_ratio = -0.000244 * rect[2] + 0.4314

    height = (rect[3] - rect[1])
    pt1 = rect[0], int(rect[1] + height * height_ratio)

    width = (rect[2] - rect[0])
    pt2 = int(rect[0] + width * width_ratio), rect[1]

    cv2.line(frame1, pt1, pt2, boxcolor, 2)

    c1, c2 = pt2, (-411, -54)
    cx = int(c2[0] + (c1[0]-c2[0]) * 2.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 2.8)

    pt3 = line_intersect(pt2, (cx,cy), (rect[2], rect[1]), (rect[2], rect[3]))
    cv2.line(frame1, pt2, pt3, boxcolor, 2)

    c1, c2 = pt1, (-411, -54)
    cx = int(c2[0] + (c1[0]-c2[0]) * 2.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 2.8)

    pt4_temp = line_intersect(pt1, (cx,cy), (rect[2], rect[1]), (rect[2], rect[3]))
    if pt4_temp is None:
        pt4_temp = line_intersect(pt1, (cx,cy), (rect[0], rect[3]), (rect[2], rect[3]))

    m = -(pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
    c = -pt3[1] - m * pt3[0]
    pt_temp = int((-540-c)/m), 540
    pt4 = line_intersect(pt1, pt4_temp, pt3, pt_temp)

    pt_temp = pt4[0], pt4[1] + height
    pt5 = line_intersect(pt4, pt_temp, (rect[0], rect[3]), (rect[2], rect[3]))
    cv2.line(frame1, pt4, pt5, boxcolor, 2)

    cv2.line(frame1, pt1, pt4, boxcolor, 2)
    cv2.line(frame1, pt3, pt4, boxcolor, 2)

    m = -(pt3[1] - pt4[1]) / (pt3[0] - pt4[0])
    c = -pt5[1] - m * pt5[0]
    pt_temp = int((0-c)/m), 0
    pt6 = line_intersect(pt5, pt_temp, (rect[2], rect[1]), (rect[2], rect[3]))
    cv2.line(frame1, pt5, pt6, boxcolor, 2)
    cv2.line(frame1, pt3, pt6, boxcolor, 2)

    pt7 = line_intersect(pt5, (-411, -54), (rect[0], rect[1]), (rect[0], rect[3]))
    cv2.line(frame1, pt5, pt7, boxcolor, 2)
    cv2.line(frame1, pt1, pt7, boxcolor, 2)

    if det["obj_class"][0] not in ["tw", "auto", "car", "ml"]:
        cv2.circle(frame1, pt1, 3, (0,0,0), -1)
        cv2.circle(frame1, pt2, 3, (0,0,0), -1)
        cv2.circle(frame1, pt3, 3, (0,0,0), -1)
        cv2.circle(frame1, pt4, 3, (0,0,0), -1)
        cv2.circle(frame1, pt5, 3, (0,0,0), -1)
        cv2.circle(frame1, pt6, 3, (0,0,0), -1)
        cv2.circle(frame1, pt7, 3, (0,0,0), -1)

    btm_pt = (pt5[0] + pt6[0]) // 2, (pt5[1] + pt6[1]) // 2
    cv2.circle(frame1, btm_pt, 3, (0,0,255), -1)


frame_count = 0

while vidcap1.isOpened() and vidcap2.isOpened():
    start_time = time.time()

    status1, frame1 = vidcap1.read()
    status2, frame2 = vidcap2.read()

    if not status1:
        print("status1 false")
        break

    if not status2:
        print("status2 false")
        break

    frame1 = cv2.resize(frame1, dsize=(width1//2, height1//2))
    frame2 = cv2.resize(frame2, dsize=(width2//2, height2//2))

    # for l in ["leftlane", "middlelane", "rightlane"]:
    #     cv2.polylines(frame1, [camera_meta[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

    detection_list, axles = detector.detect(frame1)

    frame_count += 1

    for det in detection_list:
        rect = det["rect"]
        btm = det["obj_bottom"]

        # if det["obj_class"][0] not in ["car", "ml", "auto", "tw"]:
        cv2.rectangle(frame1, rect[:2], rect[2:], (255,0,0), 1)
    
        twod_2_threed(frame1, det)

        if det['lane'] == "1":
            btmx_tf = int((0.65523379 * btm[0]) + (-3.67679969 * btm[1]) + 1349.9740589597031)
            btmy_tf = int((0.41925539 * btm[0]) + (-0.04235352 * btm[1]) + 99.19415894167156)
        elif det['lane'] == "2":
            btmx_tf = int((0.55305366 * btm[0]) + (-3.3535726 * btm[1]) + 1301.5774568947409)
            btmy_tf = int((0.37036275  * btm[0]) + (-0.02834603 * btm[1]) + 113.54696288217643)
        else:
            btmx_tf = int((0.39657267 * btm[0]) + (-2.85288489 * btm[1]) + 1195.282143258536)
            btmy_tf = int((0.30479565 * btm[0]) + (0.04620164 * btm[1]) + 108.36117943778677)

        cv2.circle(frame2, (int(btmx_tf), int(btmy_tf)), 3, (0,0,255), -1)

    final_frame = np.hstack((frame1, frame2))
    videowriter.write(frame1)

    cv2.imshow("video", frame1)

    key = cv2.waitKey(1)
    if key == ord('q'):
       break
    elif key == ord('p'):
        cv2.waitKey(-1)
    
    print(f"frame_count: {frame_count}, fps: {1.0 / (time.time()-start_time)}")

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()

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
