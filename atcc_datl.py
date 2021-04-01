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
    detection_thresh=0.5,
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
    30.0,
    (1920, 540),
)

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

    for l in ["leftlane", "rightlane"]:
        cv2.polylines(frame1, [camera_meta[f"{l}_coords"]], isClosed=True, color=(0, 0, 0), thickness=2)

    detection_list, axles = detector.detect(frame1)

    for det in detection_list:
        rect = det["rect"]
        btm = det["obj_bottom"]
        cv2.rectangle(frame1, rect[:2], rect[2:], (225,0,0), 2)
        cv2.circle(frame1, btm, 4, (0,0,255), -1)

        btmx_tf = (0.44958882 * btm[0]) + (-3.23493889 * btm[1]) + 1299.903505513684
        btmy_tf = (0.41243032 * btm[0]) + (-0.07925315 * btm[1]) + 116.09248088227169

        # btmx_tf = (0.39795221 * btm[0]) + (-3.06747396 * btm[1]) + 1269.8043491220583
        # btmy_tf = (0.38150148 * btm[0]) + (-0.01415993 * btm[1]) + 109.26557888289531

        cv2.circle(frame2, (int(btmx_tf), int(btmy_tf)), 4, (0,0,255), -1)

    final_frame = np.hstack((frame1, frame2))
    videowriter.write(final_frame)

    cv2.imshow("video", final_frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
       break
    elif key == ord('p'):
        cv2.waitKey(-1)
    
    print(f"fps: {1.0 / (time.time()-start_time)}")

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
        "30",
        curr_folder + "/linear_mapping_comp.avi",
    ]
)

if status:
    print(f"\n unable to compress {outvideo_path} ! \n")
else:
    os.remove(outvideo_path)
