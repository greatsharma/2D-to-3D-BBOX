import os
import cv2
import sys
import time
import pyodbc
import argparse
import datetime
import threading
import subprocess
import numpy as np
from collections import deque
from scipy.spatial import distance
from multiprocessing import Process, Queue, Value

from camera_metadata import CAMERA_METADATA
from detectors import VanillaYoloDetector
from trackers import CentroidTracker, KalmanTracker
from utils import init_lane_detector, init_direction_detector
from utils import init_within_interval, intersection_over_rect
from utils import draw_text_with_backgroud, draw_tracked_objects, draw_axles


Abbrevation_Mapper = {
    "tw": "Two wheeler",
    "car": "Car",
    "lgv": "LGV",
    "2t": "2-axle Truck",
    "3t": "3-axle Truck",
    "4t": "4-axle Truck",
    "5t": "5-axle Truck",
    "6t": "6-axle Truck",
    "bus": "Bus",
    "ml": "Mini LGV",
    "auto": "Auto",
    "mb": "Mini Bus",
    "tractr": "Tractor",
    "axle": "Axle",
}


class VehicleTracking(object):
    def __init__(
        self,
        input_path,
        inference_type,
        write_db,
        output,
        output_fps,
        resize,
        detection_thresh,
        tracker_type,
        max_track_pts,
        max_absent,
        mode,
    ):

        if input_path.startswith("inputs"):
            self.camera_id = input_path.split("/")[1].split("_")[0]
            self.input_path = input_path
        elif input_path == "hyderabad":
            self.camera_id = input_path
            self.input_path = "rtsp://admin:admin@1234@192.168.10.65/1"
        elif input_path == "kurnul":
            self.camera_id = input_path
            self.input_path = "rtsp://admin:admin@1234@192.168.10.64/1"
        else:
            raise ValueError("Invalid input !")

        self.inference_type = inference_type

        self.write_db = write_db
        if self.write_db:
            if self.camera_id == "hyderabad":
                self.sql_insert = """INSERT INTO TowardsHyderabad(AVC_ID,AVC_Class,Date,Time,Lane,Axle_Configuration) VALUES(?,?,?,?,?,?)"""
            elif self.camera_id == "kurnul":
                self.sql_insert = """INSERT INTO TowardsKurnool(AVC_ID,AVC_Class,Date,Time,Lane,Axle_Configuration) VALUES(?,?,?,?,?,?)"""

            try:
                self.conn = pyodbc.connect(
                    "Driver={ODBC Driver 17 for SQL Server};Server=yolo-desktop; Database=APEL; uid=SA ; pwd=Vrt@1234;"
                )
                self.cursor = self.conn.cursor()
                print("\nConnected to DB ...\n")
            except Exception:
                print("\nUnable to connect to DB !\n")
                sys.exit()

        self.output = output
        self.output_fps = output_fps

        self.camera_meta = CAMERA_METADATA[self.camera_id]

        self.resize = resize
        self.detection_thresh = detection_thresh
        self.tracker_type = tracker_type
        self.max_track_pts = max_track_pts
        self.max_absent = max_absent
        self.mode = mode

        self.logged_ids = []

        self.wrongdir_count = 0

        self.class_counts = {
            "1": {
                "tw": 0,
                "car": 0,
                "lgv": 0,
                "2t": 0,
                "3t": 0,
                "4t": 0,
                "5t": 0,
                "6t": 0,
                "bus": 0,
                "ml": 0,
                "auto": 0,
                "mb": 0,
                "tractr": 0,
            },
            "2": {
                "tw": 0,
                "car": 0,
                "lgv": 0,
                "2t": 0,
                "3t": 0,
                "4t": 0,
                "5t": 0,
                "6t": 0,
                "bus": 0,
                "ml": 0,
                "auto": 0,
                "mb": 0,
                "tractr": 0,
            },
        }

        self.countintervals = self.camera_meta["adaptive_countintervals"]

        self.vidcap = cv2.VideoCapture(self.input_path)

        self.preprocessedframes_queue = Queue()
        self.tilldetection_queue = Queue()

        _, initial_frame = self.vidcap.read()

        if initial_frame is None:
            if "rtsp" in self.input_path:
                raise RuntimeError("RTSP Error : Unable to capture frames !")
            else:
                raise FileNotFoundError("Unable to capture frames !")

        if self.resize[0] <= 1:
            self.frame_h, self.frame_w = tuple(
                int(d * s) for d, s in zip(initial_frame.shape[:2], self.resize)
            )
        else:
            self.frame_h, self.frame_w = self.resize

        self.initial_frame = cv2.resize(
            initial_frame,
            dsize=(self.frame_w, self.frame_h),
            interpolation=cv2.INTER_LINEAR,
        )

        self.img_for_text = cv2.imread("right_image.jpg")

        self.img_for_log = np.zeros(
            (self.frame_h, int(self.frame_w // 2.5), 3), dtype=np.uint8
        )
        self.img_for_log[:, :, 0:3] = (243, 227, 218)

        self.log_buffer = deque([])

        self.lane_detector = init_lane_detector(self.camera_meta)

        direction_detector = init_direction_detector(self.camera_meta)
        within_interval = init_within_interval(self.camera_meta)
        initial_maxdistances = self.camera_meta["initial_maxdistances"]
        lane_angles = self.camera_meta["lane_angles"]
        velocity_regression = self.camera_meta["velocity_regression"]

        if self.tracker_type == "centroid":
            self.tracker = CentroidTracker(
                direction_detector,
                initial_maxdistances,
                within_interval,
                lane_angles,
                velocity_regression,
                self.max_absent,
            )
        else:
            self.tracker = KalmanTracker(
                direction_detector,
                initial_maxdistances,
                within_interval,
                lane_angles,
                velocity_regression,
                self.max_absent,
            )

    def _count_vehicles(self, tracked_objs):
        for obj in tracked_objs.values():
            obj_bottom = (
                obj.obj_bottom if self.tracker_type == "centroid" else obj.state[:2]
            )

            for k, v in self.countintervals.items():
                if obj.obj_class[0] in k:
                    ref1, ref2 = v
                    break

            if ref1 < obj_bottom[0] < ref2 and obj.starttime is None:
                obj.starttime = datetime.datetime.now()

            elif (
                (obj_bottom[0] < ref1 or obj_bottom[0] > ref2)
                and obj.endtime is None
                and obj.starttime is not None
            ):
                obj.endtime = datetime.datetime.now()
                if obj.direction:
                    self.class_counts[obj.lane][obj.obj_class[0]] += 1
                else:
                    self.wrongdir_count += 1

            elif (
                (not obj.direction)
                and (obj.starttime is not None)
                and (obj.obj_class[0] in ["tw", "car", "auto"])
                and (obj.lane == "2")
                and (obj.objid not in self.logged_ids)
            ):
                obj.endtime = "not_none"
                self.wrongdir_count += 1

    def _get_axleconfig(self, axles):
        axle_dists = []

        for i in range(len(axles) - 1):
            dist = distance.euclidean(axles[i], axles[i + 1])
            axle_dists.append(dist)

        num_axles = len(axles)

        if num_axles == 3:
            axleconfig = "12"
        elif num_axles == 4:
            if axle_dists[0] > axle_dists[1] and axle_dists[0] > axle_dists[2]:
                axleconfig = "13"
            else:
                axleconfig = "112"
        elif num_axles == 5:
            if axle_dists[1] > axle_dists[3]:
                axleconfig = "113"
            else:
                axleconfig = "122"
        elif num_axles == 6:
            axleconfig = "123"
        else:
            axleconfig = "11"

        return axleconfig

    def _axle_assignments(self, tracked_objs, axles):
        _axles = axles.copy()

        for obj in tracked_objs.values():
            if obj.obj_class[0] in ["2t", "3t", "4t", "5t", "6t", "bus", "lgv"]:
                obj_ax = []

                temp = []
                for ax in _axles:
                    if intersection_over_rect(obj.rect, ax) > 0.9:
                        obj_ax.append(ax)
                    else:
                        temp.append(ax)

                if len(obj_ax) > 0:
                    obj.axle_track.append(
                        sorted(obj_ax, key=lambda x: x[0])[-1][2:]
                    )  # adding last axle

                if obj.obj_class[0] in ["3t", "4t", "5t", "6t"]:

                    if len(obj_ax) > len(obj.axles):
                        obj.axles = sorted(obj_ax, key=lambda x: x[0])

                    if (
                        len(obj.axles) == int(obj.obj_class[0][0])
                        and obj.axle_config is None
                    ):
                        obj.axle_config = self._get_axleconfig(obj.axles)

                _axles = temp
                if len(_axles) == 0:
                    break

    def _log(self, tracked_objs):
        for obj in tracked_objs.values():
            if (
                (obj.starttime is not None)
                and (obj.endtime is not None)
                and (obj.objid not in self.logged_ids)
            ):

                obj_id = str(obj.objid)
                obj_class = obj.obj_class[0]
                obj_date = obj.starttime.strftime("%Y:%m:%d:%H:%M:%S")[:10]
                obj_time = obj.starttime.strftime("%Y:%m:%d:%H:%M:%S")[11:]
                obj_lane = f"lane {obj.lane}"

                obj_axconifg = "_"
                if (
                    obj_class in ["3t", "4t", "5t", "6t"]
                    and obj.axle_config is not None
                ):
                    obj_axconifg = obj.axle_config

                logbuffer_length = 14
                if self.mode != "debug":
                    logbuffer_length = 18

                if len(self.log_buffer) >= logbuffer_length:
                    self.log_buffer.rotate(-1)
                    if self.mode != "debug":
                        self.log_buffer[17] = (
                            obj_class,
                            obj_time,
                            obj_lane,
                            obj_axconifg,
                        )
                    else:
                        self.log_buffer[13] = (
                            obj_class,
                            obj_time,
                            obj_lane,
                            obj_axconifg,
                        )
                else:
                    self.log_buffer.append(
                        (obj_class, obj_time, obj_lane, obj_axconifg)
                    )

                if self.write_db:
                    try:
                        params = (
                            obj_id,
                            obj_class,
                            obj_date,
                            obj_time,
                            obj_lane,
                            obj_axconifg,
                        )
                        self.cursor.execute(self.sql_insert, params)
                        self.conn.commit()
                    except Exception:
                        msg = f"{obj_id},{obj_time},{obj_lane},DB_WRITING_FAILED"
                        self.error_filewriter.write(msg + "\n")
                        print(msg)

                txt = (
                    obj_id
                    + ","
                    + obj_class
                    + ","
                    + obj_date
                    + ","
                    + obj_time
                    + ","
                    + obj_lane
                    + ","
                    + obj_axconifg
                    + ","
                )

                if obj.direction:
                    txt += "_"
                else:
                    txt += "WRONG_DIRECTION"

                self.log_filewriter.write(txt + "\n")
                self.logged_ids.append(obj.objid)

        self.img_for_log[:, :, 0:3] = (243, 227, 218)

        for name, xcoord in zip(
            ["Class", "Time", "Lane", "Axle-conf"], [15, 90, 190, 275]
        ):
            draw_text_with_backgroud(
                self.img_for_log,
                name,
                x=xcoord,
                y=40,
                font_scale=0.6,
                thickness=2,
            )

        y = 70
        for row in self.log_buffer:
            for col, xcoord in zip(row, [15, 90, 190, 275]):
                draw_text_with_backgroud(
                    self.img_for_log,
                    col,
                    x=xcoord,
                    y=y,
                    font_scale=0.5,
                    thickness=1,
                )
            y += 20

    def _compress_video(self, input_path, output_path, del_prev_video):
        status = subprocess.call(
            [
                "ffmpeg",
                "-i",
                input_path,
                "-vcodec",
                "libx264",
                "-crf",
                "30",
                output_path,
                "-hide_banner",
                "-loglevel",
                "panic",
                "-y",
            ]
        )

        if status:
            msg = f"Compression_Error : {datetime.datetime.now()} : Unable to compress {input_path} !"
            self.error_filewriter.write(msg + "\n")
            print(msg)
        else:
            if del_prev_video:
                os.remove(input_path)

    def _cc_writer(self):
        tt_vehicles = 0
        for (k1, v1), (_, v2) in zip(
            self.class_counts["1"].items(), self.class_counts["2"].items()
        ):
            self.cc_filewriter.write(k1 + " : " + str(v1 + v2) + "\n")
            tt_vehicles += v1 + v2
            self.class_counts["1"][k1] = 0
            self.class_counts["2"][k1] = 0

        self.cc_filewriter.write(f"Total : {tt_vehicles}" + "\n")
        self.cc_filewriter.write(f"Wrong direction : {self.wrongdir_count}" + "\n")

    def _daily_filewriter_plotter(self, output_path):
        self.cc_filewriter.close()
        self.log_filewriter.close()
        self.tracker.trackpath_filewriter.close()

        status = subprocess.call(
            [
                "python",
                "daily_filewriter_plotter.py",
                "-ip",
                output_path,
                "-cp",
                self.camera_id,
            ]
        )

        if status:
            msg = f"DailyFW_Error : {datetime.datetime.now()} : Error in function _daily_filewriter_plotter !"
            self.error_filewriter.write(msg + "\n")
            print(msg)

    def _hourly_plotter(self, output_path):
        self.cc_filewriter.close()
        self.log_filewriter.close()

        status = subprocess.call(
            [
                "python",
                "hourly_plotter.py",
                "-ip",
                output_path,
            ]
        )

        if status:
            msg = f"HourlyPlotter_Error : {datetime.datetime.now()} : Error in function _hourly_plotter !"
            self.error_filewriter.write(msg + "\n")
            print(msg)

    def _delete_oneday_videos(self, output_path):
        status = subprocess.call(
            [
                "python",
                "delete_oneday_videos.py",
                "-ip",
                output_path,
            ]
        )

        if status:
            msg = f"VideoDeletion_Error : {datetime.datetime.now()} : Error in function _delete_oneday_videos !"
            self.error_filewriter.write(msg + "\n")
            print(msg)

    def _clean_exit(self, currenthour_dir, currentday_dir):
        self._cc_writer()
        self._hourly_plotter(currenthour_dir)
        self._daily_filewriter_plotter(currentday_dir)

        print(
            "\nExecuting clean exit, this may take few minutes depending on compression...\n"
        )

        if self.output:
            self.videowriter.release()
            compressed_filename = self.video_filename.split(".")[0] + "_comp.avi"
            self._compress_video(self.video_filename, compressed_filename, True)

        self.dev_cvutil_filewriter.close()
        self.error_filewriter.close()

    def _postprocess_detections(self, vidcap_status):
        date = datetime.datetime.now()
        videodeletion_day = date

        videodeletion_initialization_day = videodeletion_day + datetime.timedelta(
            days=4
        )
        videodeletion_initialization_day = videodeletion_initialization_day.strftime(
            "%d_%m_%Y"
        )
        flag_videodeletion = False

        date = date.strftime("%d_%m_%Y_%H:%M:%S")

        currentday_dir = f"outputs/{self.camera_id}/{date[:10]}/"
        if not os.path.exists(currentday_dir):
            os.mkdir(currentday_dir)

        error_filename = currentday_dir + date[:10] + f"_log.txt"
        self.error_filewriter = open(error_filename, "w")

        currenthour_dir = currentday_dir + date[11:13] + "/"
        if not os.path.exists(currenthour_dir):
            os.mkdir(currenthour_dir)

        currenthour_devdir = currenthour_dir + "dev/"
        if not os.path.exists(currenthour_devdir):
            os.mkdir(currenthour_devdir)

        dev_cvutil_filename = currenthour_devdir + "cv.txt"
        self.dev_cvutil_filewriter = open(dev_cvutil_filename, "w")

        log_filename = currenthour_dir + date[11:13] + ".txt"
        self.log_filewriter = open(log_filename, "w")

        trackpath_filename = currenthour_dir + date[11:13] + f"_trkpath.txt"
        self.tracker.trackpath_filewriter = open(trackpath_filename, "w")

        cc_filename = currenthour_dir + date[11:13] + "_finalcounts.txt"
        self.cc_filewriter = open(cc_filename, "w")

        if self.output:
            self.video_filename = currenthour_dir + date[11:13] + ".avi"
            self.videowriter = cv2.VideoWriter(
                self.video_filename,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                self.output_fps,
                (1784, 540),
            )

        flag1 = True
        flag2 = True

        tik1 = time.time()
        while True:
            tik2 = time.time()

            date = datetime.datetime.now()
            date = date.strftime("%d_%m_%Y_%H:%M:%S")

            if videodeletion_initialization_day == date[:10]:
                flag_videodeletion = True

            if date[11:16] == "00:00":
                if flag1:
                    self._cc_writer()

                    t1 = threading.Thread(
                        target=self._daily_filewriter_plotter,
                        kwargs={"output_path": currentday_dir},
                    )
                    t1.start()

                    if self.output and flag_videodeletion:
                        self.videowriter.release()

                        output_path = f"outputs/{self.camera_id}/{videodeletion_day.strftime('%d_%m_%Y')}/"
                        t2 = threading.Thread(
                            target=self._delete_oneday_videos,
                            kwargs={"output_path": output_path},
                        )
                        t2.start()

                        videodeletion_day += datetime.timedelta(days=1)

                    currentday_dir = f"outputs/{self.camera_id}/{date[:10]}/"
                    if not os.path.exists(currentday_dir):
                        os.mkdir(currentday_dir)

                    self.error_filewriter.close()
                    error_filename = currentday_dir + date[:10] + f"_log.txt"
                    self.error_filewriter = open(error_filename, "w")

                    self.tracker.next_objid = 0
                    self.logged_ids = []

                    flag1 = False

            else:
                flag1 = True

            if date[14:16] == "00":
                if flag2:
                    if flag1:
                        self._cc_writer()

                    t3 = threading.Thread(
                        target=self._hourly_plotter,
                        kwargs={"output_path": currenthour_dir},
                    )
                    t3.start()

                    self.wrongdir_count = 0
                    self.img_for_text = cv2.imread("right_image.jpg")

                    currenthour_dir = currentday_dir + date[11:13] + "/"
                    if not os.path.exists(currenthour_dir):
                        os.mkdir(currenthour_dir)

                    currenthour_devdir = currenthour_dir + "dev/"
                    if not os.path.exists(currenthour_devdir):
                        os.mkdir(currenthour_devdir)

                    self.dev_cvutil_filewriter.close()
                    dev_cvutil_filename = currenthour_devdir + "cv.txt"
                    self.dev_cvutil_filewriter = open(dev_cvutil_filename, "w")

                    log_filename = currenthour_dir + date[11:13] + ".txt"
                    self.log_filewriter = open(log_filename, "w")

                    trackpath_filename = currenthour_dir + date[11:13] + f"_trkpath.txt"
                    self.tracker.trackpath_filewriter = open(trackpath_filename, "w")

                    cc_filename = currenthour_dir + date[11:13] + "_finalcounts.txt"
                    self.cc_filewriter = open(cc_filename, "w")

                    if self.output:
                        compressed_file_name = (
                            self.video_filename.split(".")[0] + "_comp.avi"
                        )
                        t4 = threading.Thread(
                            target=self._compress_video,
                            args=(self.video_filename, compressed_file_name, True),
                        )
                        t4.start()

                        self.video_filename = currenthour_dir + date[11:13] + ".avi"
                        self.videowriter = cv2.VideoWriter(
                            self.video_filename,
                            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                            self.output_fps,
                            (1784, 540),
                        )

                    msg = f"\n-------initialized new files for the hour-------{date}\n"
                    self.error_filewriter.write(msg)
                    print(msg)

                    flag2 = False
            else:
                flag2 = True

            if self.tilldetection_queue.qsize() > 0:
                (
                    detection_list,
                    axles,
                    frame,
                    frame_count,
                    fps_list,
                ) = self.tilldetection_queue.get()

                tracked_objects = self.tracker.update(detection_list)

                self._count_vehicles(tracked_objects)
                self._axle_assignments(tracked_objects, axles)

                self._log(tracked_objects)

                draw_tracked_objects(self, frame, tracked_objects)

                if self.mode == "debug":
                    for l in ["leftlane", "rightlane"]:
                        cv2.polylines(
                            frame,
                            [self.camera_meta[f"{l}_coords"]],
                            isClosed=True,
                            color=(0, 0, 0),
                            thickness=2,
                        )
                        cv2.circle(
                            frame,
                            self.camera_meta[f"{l}_ref"],
                            radius=4,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

                    pt = self.camera_meta["mid_ref"]
                    cv2.line(frame, (pt, frame.shape[0]), (pt, 0), (0, 0, 255), 2)

                    pt = self.camera_meta["adaptive_countintervals"][
                        "3t,4t,5t,6t,lgv,tractr,2t,bus,mb"
                    ]
                    cv2.line(
                        frame, (pt[0], frame.shape[0]), (pt[0], 0), (255, 255, 255), 2
                    )
                    cv2.line(
                        frame, (pt[1], frame.shape[0]), (pt[1], 0), (255, 255, 255), 2
                    )

                    pt = self.camera_meta["adaptive_countintervals"]["ml,car,auto"]
                    cv2.line(frame, (pt[0], frame.shape[0]), (pt[0], 0), (0, 255, 0), 2)
                    cv2.line(frame, (pt[1], frame.shape[0]), (pt[1], 0), (0, 255, 0), 2)

                    pt = self.camera_meta["adaptive_countintervals"]["tw"]
                    cv2.line(frame, (pt[0], frame.shape[0]), (pt[0], 0), (255, 0, 0), 2)
                    cv2.line(frame, (pt[1], frame.shape[0]), (pt[1], 0), (255, 0, 0), 2)

                    draw_axles(frame, axles)

                for name, xcoord in zip(
                    ["Class", "Lane-1", "Lane-2", "Total"], [15, 150, 250, 350]
                ):
                    draw_text_with_backgroud(
                        self.img_for_text,
                        name,
                        x=xcoord,
                        y=150,
                        font_scale=0.6,
                        thickness=2,
                    )

                y = 180
                vehicles_lane1 = 0
                vehicles_lane2 = 0
                for (k1, v1), (_, v2) in zip(
                    self.class_counts["1"].items(), self.class_counts["2"].items()
                ):
                    vehicles_lane1 += v1
                    vehicles_lane2 += v2

                    for name, xcoord, bg in zip(
                        [Abbrevation_Mapper[k1], str(v1), str(v2), str(v1 + v2)],
                        [15, 175, 275, 375],
                        (None, (246, 231, 215), (242, 226, 209), (241, 222, 201)),
                    ):

                        draw_text_with_backgroud(
                            self.img_for_text,
                            name,
                            x=xcoord,
                            y=y,
                            font_scale=0.5,
                            thickness=1,
                            background=bg,
                        )

                    y += 20

                y += 20
                for name, xcoord, bg in zip(
                    [
                        "Total",
                        str(vehicles_lane1),
                        str(vehicles_lane2),
                        str(vehicles_lane1 + vehicles_lane2),
                    ],
                    [15, 175, 275, 375],
                    (None, (246, 231, 215), (242, 226, 209), (241, 222, 201)),
                ):

                    draw_text_with_backgroud(
                        self.img_for_text,
                        name,
                        x=xcoord,
                        y=y,
                        font_scale=0.55,
                        thickness=2,
                        background=bg,
                    )

                draw_text_with_backgroud(
                    self.img_for_text,
                    f"WD : {self.wrongdir_count}",
                    x=15,
                    y=500,
                    font_scale=0.6,
                    thickness=2,
                    background=(242, 226, 209),
                )

                if self.mode != "debug":
                    for name in ["Lane 1", "Lane 2"]:
                        k = name + " annotation_data"
                        txt = name + " : "

                        if name == "Lane 1":
                            txt += str(vehicles_lane1)
                        else:
                            txt += str(vehicles_lane2)

                        cv2.line(
                            frame,
                            self.camera_meta[k][0],
                            self.camera_meta[k][1],
                            (128, 0, 128),
                            2,
                        )
                        cv2.line(
                            frame,
                            self.camera_meta[k][1],
                            self.camera_meta[k][2],
                            (128, 0, 128),
                            2,
                        )
                        draw_text_with_backgroud(
                            frame,
                            txt,
                            x=self.camera_meta[k][3],
                            y=self.camera_meta[k][4],
                            font_scale=0.7,
                            thickness=1,
                            background=(128, 0, 128),
                            foreground=(255, 255, 255),
                            box_coords_1=(-7, 7),
                            box_coords_2=(10, -10),
                        )

                draw_text_with_backgroud(
                    self.img_for_log,
                    f"Frame count: {frame_count}",
                    x=15,
                    y=460,
                    font_scale=0.5,
                    thickness=1,
                )

                draw_text_with_backgroud(
                    self.img_for_log,
                    f"Curr FPS: {fps_list[0][0]}",
                    x=15,
                    y=480,
                    font_scale=0.5,
                    thickness=1,
                )

                draw_text_with_backgroud(
                    self.img_for_log,
                    f"Avg FPS: {fps_list[0][1]}",
                    x=15,
                    y=500,
                    font_scale=0.5,
                    thickness=1,
                )

                out_frame = np.hstack((self.img_for_log, frame, self.img_for_text))
                cv2.imshow(f"ATCC-APEL, Towards {self.camera_id}", out_frame)

                if self.output:
                    self.videowriter.write(out_frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    vidcap_status.value = 0

                if (
                    not vidcap_status.value
                    and self.tilldetection_queue.qsize() == 0
                    and self.preprocessedframes_queue.qsize() == 0
                ):
                    break

                tok = time.time()
                avg_fps = round(frame_count / (tok - tik1), 2)
                inst_fps = round(1.0 / (tok - tik2), 1)
                fps_list.append((inst_fps, avg_fps))

                print(frame_count, end=" ; ")
                for n, v in zip(["inp", "det", "pp"], fps_list):
                    print(f"{n}-fps: {v[0]}, {v[1]}", end=" ; ")
                print(
                    f"qsize: {self.tilldetection_queue.qsize()}, {self.preprocessedframes_queue.qsize()}"
                )

            if (
                not vidcap_status.value
                and self.input_path.startswith("inputs")
                and self.tilldetection_queue.qsize() == 0
                and self.preprocessedframes_queue.qsize() == 0
            ):
                break

        self._clean_exit(currenthour_dir, currentday_dir)
        print("Exiting Process-2 !")

    def _do_detection(self):
        if self.inference_type == "trt":
            from detectors.trt_detector import TrtYoloDetector

            self.detector = TrtYoloDetector(
                self.initial_frame,
                self.lane_detector,
                self.detection_thresh,
            )
        else:
            self.detector = VanillaYoloDetector(
                self.initial_frame,
                self.lane_detector,
                self.detection_thresh,
            )

        tik1 = time.time()
        while True:
            if self.preprocessedframes_queue.qsize() > 0:
                tik2 = time.time()
                frame, frame_count, fps_list = self.preprocessedframes_queue.get()
                detection_list, axles = self.detector.detect(frame)

                tok = time.time()
                avg_fps = round(frame_count / (tok - tik1), 2)
                inst_fps = round(1.0 / (tok - tik2), 1)
                fps_list.append((inst_fps, avg_fps))

                self.tilldetection_queue.put(
                    (detection_list, axles, frame, frame_count, fps_list)
                )

    def run(self):
        frame_count = 0
        vidcap_status = Value("i", 1)

        process1 = Process(target=self._do_detection)
        process1.start()

        process2 = Process(target=self._postprocess_detections, args=(vidcap_status,))
        process2.start()

        with open(f"pid_{self.camera_id}.txt", "w") as f:
            f.write(
                f"main-{os.getpid()} ; process1-{process1.pid} ; process2-{process2.pid}"
            )

        tik1 = time.time()

        while self.vidcap.isOpened():
            tik2 = time.time()
            vidcap_status.value, frame = self.vidcap.read()

            while not vidcap_status.value:
                if self.input_path.startswith("rtsp"):
                    msg = f"RTSP_Error : {datetime.datetime.now()} : Unable to capture frames !"
                    print(msg)

                    self.vidcap.release()
                    time.sleep(1)

                    self.vidcap = cv2.VideoCapture(self.input_path)
                    vidcap_status.value, frame = self.vidcap.read()
                else:
                    if not process2.is_alive():
                        process1.terminate()

                    if not process1.is_alive():
                        self.vidcap.release()
                        cv2.destroyAllWindows()
                        sys.exit()

            frame_count += 1

            frame = cv2.resize(
                frame,
                dsize=(self.frame_w, self.frame_h),
                interpolation=cv2.INTER_LINEAR,
            )

            if self.input_path.startswith("rtsp"):
                key = cv2.waitKey(1)
            else:
                key = cv2.waitKey(25)

            tok = time.time()
            avg_fps = round(frame_count / (tok - tik1), 2)
            inst_fps = round(1.0 / (tok - tik2), 1)

            self.preprocessedframes_queue.put(
                (frame, frame_count, [(inst_fps, avg_fps)])
            )

            if (
                self.preprocessedframes_queue.qsize() > 1500
                or self.tilldetection_queue.qsize() > 1500
            ):
                print(
                    f"queue overflow ! , qsize: {self.preprocessedframes_queue.qsize()}, {self.tilldetection_queue.qsize()}"
                )
                process1.terminate()
                process2.terminate()
                self.vidcap.release()
                cv2.destroyAllWindows()
                print("Now press Ctrl+C to exit...")

            if not vidcap_status.value and process2.is_alive():
                while process2.is_alive():
                    continue

                process1.terminate()
                print("Terminated Process-1")

            if not process1.is_alive():
                break

        self.vidcap.release()
        cv2.destroyAllWindows()
        print("Now press Ctrl+C to exit...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input",
        type=str,
        required=False,
        default="inputs/datl_clip1.mp4",
        help="path to input video",
    )

    ap.add_argument(
        "-if",
        "--inference",
        type=str,
        required=False,
        default="trt",
        choices=["vanilla", "trt"],
        help="type pf inference",
    )

    ap.add_argument(
        "-wdb", "--write_db", type=int, required=True, help="whether to write to DB"
    )

    ap.add_argument(
        "-o",
        "--output",
        type=int,
        required=False,
        default=0,
        help="whether to write output videos, default is 0",
    )

    ap.add_argument(
        "-ofps",
        "--output_fps",
        type=int,
        required=True,
        help="output fps",
    )

    ap.add_argument(
        "-r",
        "--resize",
        nargs="+",
        type=float,
        required=False,
        default=[0.5, 0.5],
        help="resize factor/shape of image",
    )

    ap.add_argument(
        "-dt",
        "--detection_thresh",
        type=float,
        required=False,
        default=0.5,
        help="detection threshold",
    )

    ap.add_argument(
        "-t",
        "--tracker",
        type=str,
        required=False,
        default="kalman",
        help="tracker to use",
        choices=["centroid", "kalman"],
    )

    ap.add_argument(
        "-mtp",
        "--max_track_points",
        type=int,
        required=False,
        default=35,
        help="maximum points to be tracked for a vehicle",
    )

    ap.add_argument(
        "-ma",
        "--max_absent",
        type=int,
        required=False,
        default=2,
        help="maximum frames a vehicle can be absent, after that it will be deregister",
    )

    ap.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="release",
        help="execution mode, either `debug`, `release`, `pretty`",
    )

    args = vars(ap.parse_args())

    vt_obj = VehicleTracking(
        args["input"],
        args["inference"],
        args["write_db"],
        args["output"],
        args["output_fps"],
        args["resize"],
        args["detection_thresh"],
        args["tracker"],
        args["max_track_points"],
        args["max_absent"],
        args["mode"],
    )

    print("\n")
    vt_obj.run()
