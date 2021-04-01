import os
import cv2
import time
import datetime
import subprocess
import numpy as np
from multiprocessing import Process, Queue, Value

from utils import init_lane_detector
from camera_metadata import CAMERA_METADATA


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
    30.0,
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


def postprocess_detections(preprocessedframe1_queue, preprocessedframe2_queue, tilldetection1_queue, tilldetection2_queue, vidcap_status):
    global videowriter

    tik1 = time.time()
    while True:
        if tilldetection1_queue.qsize() > 0 and tilldetection2_queue.qsize() > 0:
            tik2 = time.time()

            detection_list1, axles1, frame1, frame_count1, fps_list1 = tilldetection1_queue.get()
            detection_list2, axles2, frame2, frame_count2, fps_list2 = tilldetection2_queue.get()                

            for l in ["leftlane", "rightlane"]:
                cv2.polylines(frame1, [camera_meta1[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

            for l in ["leftlane", "rightlane"]:
                cv2.polylines(frame2, [camera_meta2[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

            for det in detection_list1:
                rect = det["rect"]
                btm = det["obj_bottom"]
                cv2.rectangle(frame1, rect[:2], rect[2:], (225,0,0), 2)
                cv2.circle(frame1, btm, 4, (0,0,255), -1)

                btmx_tf = (0.44958882 * btm[0]) + (-3.23493889 * btm[1]) + 1299.903505513684
                btmy_tf = (0.41243032 * btm[0]) + (-0.07925315 * btm[1]) + 116.09248088227169

                # btmx_tf = (0.39795221 * btm[0]) + (-3.06747396 * btm[1]) + 1269.8043491220583
                # btmy_tf = (0.38150148 * btm[0]) + (-0.01415993 * btm[1]) + 109.26557888289531

                cv2.circle(frame2, (int(btmx_tf), int(btmy_tf)), 4, (0,0,255), -1)

            for det in detection_list2:
                rect = det["rect"]
                btm = det["obj_bottom"]
                cv2.rectangle(frame2, rect[:2], rect[2:], (225,0,0), 2)
                # cv2.circle(frame2, btm, 3, (0,255,0), -1)

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

    tik1 = time.time()
    while True:
        if preprocessedframes1_queue.qsize() > 0:
            tik2 = time.time()

            frame1, frame_count, fps_list = preprocessedframes1_queue.get()
            detection_list, axles = detector.detect(frame1)

            tok = time.time()
            avg_fps = round(frame_count / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)
            fps_list.append((inst_fps, avg_fps))

            tilldetection1_queue.put((detection_list, axles, frame1, frame_count, fps_list))


def detection_secondarycam(preprocessedframes2_queue, tilldetection2_queue):
    from detectors.trt_detector import TrtYoloDetector

    global initial_frame2, camera_meta2

    detector = TrtYoloDetector(
        initial_frame2,
        init_lane_detector(camera_meta2),
        detection_thresh=0.25,
        bottom_type="bottom-left"
    )

    tik1 = time.time()
    while True:
        if preprocessedframes2_queue.qsize() > 0:
            tik2 = time.time()

            frame2, frame_count, fps_list = preprocessedframes2_queue.get()
            detection_list, axles = detector.detect(frame2)

            tok = time.time()
            avg_fps = round(frame_count / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)
            fps_list.append((inst_fps, avg_fps))

            tilldetection2_queue.put((detection_list, axles, frame2, frame_count, fps_list))


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

    key = cv2.waitKey(40)

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
