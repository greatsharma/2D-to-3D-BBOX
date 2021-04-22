import os
import cv2
import time
import datetime
import subprocess
import numpy as np
from scipy.spatial import distance
from multiprocessing import Process, Queue, Value

from utils import axle_assignments
from trackers import KalmanTracker
from camera_metadata import CAMERA_METADATA
from detectors.trt_detector import TrtYoloDetector
from utils import init_lane_detector, init_direction_detector, init_within_interval
from utils import draw_text_with_backgroud, draw_tracked_objects, draw_3dbox


vidcap1 = cv2.VideoCapture("inputs/datlcam1_clip1.mp4")
width1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

vidcap2 = cv2.VideoCapture("inputs/datlcam2_clip1.mp4")
width2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

if int(vidcap1.get(cv2.CAP_PROP_FPS)) != int(vidcap2.get(cv2.CAP_PROP_FPS)):
    raise ValueError("Both the videos has different fps")

vidcap2.set(cv2.CAP_PROP_POS_FRAMES, 505)

_, initial_frame1 = vidcap1.read()
_, initial_frame2 = vidcap2.read()

initial_frame1 = cv2.resize(initial_frame1, dsize=(width1//2, height1//2))
initial_frame2 = cv2.resize(initial_frame2, dsize=(width2//2, height2//2))

camera_meta1 = CAMERA_METADATA["datlcam1"]
camera_meta2 = CAMERA_METADATA["datlcam2"]

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
    int(vidcap1.get(cv2.CAP_PROP_FPS)),
    (1920, 540),
)


def compress_video():
    global curr_folder, outvideo_path
    status = subprocess.call(
        [
            "ffmpeg",
            "-i",
            outvideo_path,
            "-vcodec",
            "libx264",
            "-crf",
            "30",
            curr_folder + "/linear_mapping_comp.avi",
        ]
    )

    if status:
        print("unable to compress")
    else:
        os.remove(outvideo_path)


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


def twoD_2_threeD_primarycam(obj):
    rect = obj.rect
    rect = list(rect)
    objcls = obj.obj_class[0]

    if objcls in ["tw", "auto", "car", "ml"]:
        height_ratio = 0.1
        width_ratio = -0.000353 * rect[2] + 0.595
    elif obj.lane == "1":
        height_ratio = 0.06
        width_ratio = -0.000304 * rect[2] + 0.388
    else:
        height_ratio = 0.06
        width_ratio = -0.000244 * rect[2] + 0.4314

    height = (rect[3] - rect[1])

    if objcls in ["tw", "auto", "car", "ml"]:
        rect[3] += int(height * 0.08)
    else:
        if len(obj.axles) > 0:
            last_axle = obj.axles[-1]
            lastaxle_btm_midpt = int((last_axle[0] + last_axle[2])/2), last_axle[3]
        else:
            rect[3] += int(height * 0.1)

    pt1 = rect[0], int(rect[1] + height * height_ratio)

    width = (rect[2] - rect[0])
    pt2 = int(rect[0] + width * width_ratio), rect[1]

    c1, c2 = pt2, (-411, -54)
    cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)

    pt3 = line_intersect(pt2, (cx,cy), (rect[2], rect[1]), (rect[2], rect[3]))

    c1, c2 = pt1, (-411, -54)
    cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
    cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)

    pt4_temp = line_intersect(pt1, (cx,cy), (rect[2], rect[1]), (rect[2], rect[3]))
    if pt4_temp is None:
        pt4_temp = line_intersect(pt1, (cx,cy), (rect[0], rect[3]), (rect[2], rect[3]))

    m = -(pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
    if objcls == "tw":
        m = 0.75 * m
    c = -pt3[1] - m * pt3[0]
    pt_temp = int((-540-c)/m), 540
    pt4 = line_intersect(pt1, pt4_temp, pt3, pt_temp)

    pt_temp = pt4[0], pt4[1] + 2*height

    try:
        c1, c2 = lastaxle_btm_midpt, (-411, -54)
        cx = int(c2[0] + (c1[0]-c2[0]) * 3.8)
        cy = int(c2[1] + (c1[1]-c2[1]) * 3.8)
        pt5 = line_intersect(pt4, pt_temp, lastaxle_btm_midpt, (cx,cy))
        if pt5 is None:
            raise UnboundLocalError
    except UnboundLocalError:
        pt5 = line_intersect(pt4, pt_temp, (rect[0], rect[3]), (rect[2], rect[3]))

    m = -(pt3[1] - pt4[1]) / (pt3[0] - pt4[0])
    c = -pt5[1] - m * pt5[0]
    pt_temp = int((0-c)/m), 0
    
    pt6 = line_intersect(pt5, pt_temp, (rect[2], rect[1]), (rect[2], rect[3]+height))
    pt7 = line_intersect(pt5, (-411, -54), (rect[0], rect[1]), (rect[0], rect[3]))

    btm_pt = (pt5[0] + pt6[0]) // 2, (pt5[1] + pt6[1]) // 2

    if obj.lane == "3":
        btm_pt = int(0.5*pt5[0] + 0.5*pt6[0]), int(0.5*pt5[1] + 0.5*pt6[1])
    else:
        btm_pt = int(0.4*pt5[0] + 0.6*pt6[0]), int(0.4*pt5[1] + 0.6*pt6[1])
    
    return btm_pt, [pt1, pt2, pt3, pt4, pt5, pt6, pt7]


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


def postprocess_detections(preprocessedframe1_queue, preprocessedframe2_queue, tilldetection1_queue, tilldetection2_queue, vidcap_status):
    global videowriter

    tik1 = time.time()
    while True:
        if tilldetection1_queue.qsize() > 0 and tilldetection2_queue.qsize() > 0:
            tik2 = time.time()

            trackedobjs_list1, frame1, frame_count1, fps_list1 = tilldetection1_queue.get()
            trackedobjs_list2, frame2, frame_count2, fps_list2 = tilldetection2_queue.get()     

            if frame_count1 != frame_count2:
                print("frames out of sync !")
                vidcap_status.value = 0

            # for l in ["leftlane", "middlelane", "rightlane"]:
            #     cv2.polylines(frame1, [camera_meta1[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

            # for l in ["leftlane", "middlelane", "rightlane"]:
            #     cv2.polylines(frame2, [camera_meta2[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

            for obj in trackedobjs_list1.values():
                # obj_centroid = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
                # x,y = obj_centroid[0] - 10, obj_centroid[1]
                # draw_text_with_backgroud(frame1,det["obj_class"][0],x,y,font_scale=0.4,thickness=1,background=(0, 0, 0),
                #                         foreground=(255,255,255), box_coords_1=(-8, 8), box_coords_2=(6, -6),)

                # if det["obj_class"][0] not in ["car", "ml", "auto", "tw"]:
                #     cv2.rectangle(frame1, rect[:2], rect[2:], (255,0,0), 1)

                btm = obj.obj_bottom
                lane = obj.lane

                draw_3dbox(frame1, obj.threed_box)
                cv2.circle(frame1, btm, 3, (0,0,255), -1)

                if lane == "1":
                    btmx_tf = int((0.65523379 * btm[0]) + (-3.67679969 * btm[1]) + 1349.9740589597031)
                    btmy_tf = int((0.41925539 * btm[0]) + (-0.04235352 * btm[1]) + 99.19415894167156)
                elif lane == "2":
                    btmx_tf = int((0.55305366 * btm[0]) + (-3.3535726 * btm[1]) + 1301.5774568947409)
                    btmy_tf = int((0.37036275  * btm[0]) + (-0.02834603 * btm[1]) + 113.54696288217643)
                else:
                    btmx_tf = int((0.39657267 * btm[0]) + (-2.85288489 * btm[1]) + 1195.282143258536)
                    btmy_tf = int((0.30479565 * btm[0]) + (0.04620164 * btm[1]) + 108.36117943778677)

                cv2.circle(frame2, (int(btmx_tf), int(btmy_tf)), 3, (0,0,255), -1)

                try:
                    for ax in obj.axle_track[-1]:
                        cv2.rectangle(frame1, ax[:2], ax[2:], (255,0,255), 3)
                except IndexError:
                    pass

            for obj in trackedobjs_list2.values():
                # obj_centroid = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
                # x,y = obj_centroid[0] - 10, obj_centroid[1]
                # draw_text_with_backgroud(frame2,det["obj_class"][0],x,y,font_scale=0.4,thickness=1,background=(0, 0, 0),
                #                         foreground=(255,255,255), box_coords_1=(-8, 8), box_coords_2=(6, -6),)

                # if det["obj_class"][0] not in ["car", "ml", "auto", "tw"]:
                #     cv2.rectangle(frame2, rect[:2], rect[2:], (255,0,0), 1)

                btm = obj.obj_bottom
                lane = obj.lane

                draw_3dbox(frame2, obj.threed_box)
                cv2.circle(frame2, btm, 3, (0,0,255), -1)

                try:
                    for ax in obj.axle_track[-1]:
                        cv2.rectangle(frame2, ax[:2], ax[2:], (255,0,255), 3)
                except IndexError:
                    pass

            final_frame = np.hstack((frame1, frame2))
            videowriter.write(final_frame)

            cv2.imshow("video", final_frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                vidcap_status.value = 0

            if (
                not vidcap_status.value
                and preprocessedframe1_queue.qsize() == 0 and tilldetection1_queue.qsize() == 0
                and preprocessedframe2_queue.qsize() == 0 and tilldetection2_queue.qsize() == 0
            ):
                break

            fps_list1.append(fps_list2[-1])

            tok = time.time()
            avg_fps = round(frame_count1 / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)
            fps_list1.append((inst_fps, avg_fps))

            print(frame_count1, frame_count2, end=" ; ")
            for n, v in zip(["inp", "det1", "det2", "pp"], fps_list1):
                print(f"{n}-fps: {v[0]}, {v[1]}", end=" ; ")
            print(
                f"qsize: {preprocessedframe1_queue.qsize()}, {preprocessedframe2_queue.qsize()}, {tilldetection1_queue.qsize()}, {tilldetection2_queue.qsize()}"
            )

        if (
            not vidcap_status.value
            and preprocessedframe1_queue.qsize() == 0 and tilldetection1_queue.qsize() == 0
            and preprocessedframe2_queue.qsize() == 0 and tilldetection2_queue.qsize() == 0
        ):
            break
    
    compress_video()


def detection_primarycam(preprocessedframes1_queue, tilldetection1_queue):
    from detectors.trt_detector import TrtYoloDetector

    global initial_frame1, camera_meta1

    detector = TrtYoloDetector(
        initial_frame1,
        init_lane_detector(camera_meta1),
        detection_thresh=0.5,
        bottom_type="bottom-right"
    )

    direction_detector = init_direction_detector(camera_meta1)
    within_interval = init_within_interval(camera_meta1)
    initial_maxdistances = camera_meta1["initial_maxdistances"]
    lane_angles = camera_meta1["lane_angles"]
    velocity_regression = None

    tracker = KalmanTracker(
        direction_detector,
        initial_maxdistances,
        within_interval,
        lane_angles,
        velocity_regression,
        max_absent=1
    )

    tik1 = time.time()
    while True:
        if preprocessedframes1_queue.qsize() > 0:
            tik2 = time.time()

            frame1, frame_count, fps_list = preprocessedframes1_queue.get()

            detection_list, axles = detector.detect(frame1)

            tracked_objects = tracker.update(detection_list)

            axle_assignments(tracked_objects, axles, sort_order="asce")

            for obj in tracked_objects.values():
                obj.obj_bottom, obj.threed_box = twoD_2_threeD_primarycam(obj)

            tok = time.time()
            avg_fps = round(frame_count / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)
            fps_list.append((inst_fps, avg_fps))

            tilldetection1_queue.put((tracked_objects, frame1, frame_count, fps_list))


def detection_secondarycam(preprocessedframes2_queue, tilldetection2_queue):
    from detectors.trt_detector import TrtYoloDetector

    global initial_frame2, camera_meta2

    detector = TrtYoloDetector(
        initial_frame2,
        init_lane_detector(camera_meta2),
        detection_thresh=0.5,
        bottom_type="bottom-left"
    )

    direction_detector = init_direction_detector(camera_meta2)
    within_interval = init_within_interval(camera_meta2)
    initial_maxdistances = camera_meta2["initial_maxdistances"]
    lane_angles = camera_meta2["lane_angles"]
    velocity_regression = None

    tracker = KalmanTracker(
        direction_detector,
        initial_maxdistances,
        within_interval,
        lane_angles,
        velocity_regression,
        max_absent=1
    )

    tik1 = time.time()
    while True:
        if preprocessedframes2_queue.qsize() > 0:
            tik2 = time.time()

            frame2, frame_count, fps_list = preprocessedframes2_queue.get()

            detection_list, axles = detector.detect(frame2)

            tracked_objects = tracker.update(detection_list)

            axle_assignments(tracked_objects, axles, sort_order="asce")

            for obj in tracked_objects.values():
                obj.obj_bottom, obj.threed_box = twoD_2_threeD_secondarycam(obj)

            tok = time.time()
            avg_fps = round(frame_count / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)
            fps_list.append((inst_fps, avg_fps))

            tilldetection2_queue.put((tracked_objects, frame2, frame_count, fps_list))


preprocessedframe1_queue = Queue()
preprocessedframe2_queue = Queue()
tilldetection1_queue = Queue()
tilldetection2_queue = Queue()

vidcap_status = Value("i", 1)

process1 = Process(target=detection_primarycam, args=(preprocessedframe1_queue, tilldetection1_queue))
process1.start()

process2 = Process(target=detection_secondarycam, args=(preprocessedframe2_queue, tilldetection2_queue))
process2.start()

process3 = Process(target=postprocess_detections, args=(preprocessedframe1_queue, preprocessedframe2_queue, tilldetection1_queue, tilldetection2_queue, vidcap_status))
process3.start()

with open(f"pid_datl.txt", "w") as f:
    f.write(
        f"main-{os.getpid()} ; process1-{process1.pid} ; process2-{process2.pid} ; process3-{process3.pid}"
    )

frame_count1 = 0
frame_count2 = 0
tik1 = time.time()

while vidcap1.isOpened() and vidcap2.isOpened():
    tik2 = time.time()

    vidcap_status.value, frame1 = vidcap1.read()
    vidcap_status.value, frame2 = vidcap2.read()

    if not vidcap_status.value:
        if not process3.is_alive():
            process1.terminate()
            process2.terminate()

        if not process1.is_alive() and not process2.is_alive():
            vidcap1.release()
            vidcap2.release()
            cv2.destroyAllWindows()
            sys.exit()

    frame_count1 += 1
    frame_count2 += 1

    frame1 = cv2.resize(frame1, dsize=(width1//2, height1//2))
    frame2 = cv2.resize(frame2, dsize=(width2//2, height2//2))

    key = cv2.waitKey(35)

    tok = time.time()
    avg_fps = round(frame_count1 / (tok - tik1), 2)
    inst_fps = round(1.0 / (tok - tik2), 1)

    preprocessedframe1_queue.put((frame1, frame_count1, [(inst_fps, avg_fps)]))
    preprocessedframe2_queue.put((frame2, frame_count2, [(inst_fps, avg_fps)]))

    if (
        preprocessedframe1_queue.qsize() > 500
        or preprocessedframe2_queue.qsize() > 500
        or tilldetection1_queue.qsize() > 500
        or tilldetection2_queue.qsize() > 500
    ):
        print(
            f"queue overflow ! , qsize: {preprocessedframe1_queue.qsize()}, {tilldetection1_queue.qsize()}"
        )
        print(
            f"queue overflow ! , qsize: {preprocessedframe2_queue.qsize()}, {tilldetection2_queue.qsize()}"
        )

        process1.terminate()
        process2.terminate()
        process3.terminate()

        vidcap1.release()
        vidcap2.release()

        cv2.destroyAllWindows()
        print("Now press Ctrl+C to exit...")

    if not vidcap_status.value and process3.is_alive():
        while process3.is_alive():
            continue

        process1.terminate()
        print("Terminated Process-1")

        process2.terminate()
        print("Terminated Process-2")

    if not process1.is_alive() and not process2.is_alive():
        break

vidcap1.release()
vidcap2.release()
cv2.destroyAllWindows()
