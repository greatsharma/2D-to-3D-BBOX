import os
import cv2
import img2pdf
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString, Point

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(style="dark")


ap = argparse.ArgumentParser()

ap.add_argument(
    "-ii",
    "--input_image",
    type=str,
    required=True,
    help="path to image",
)

ap.add_argument(
    "-if",
    "--input_file",
    type=str,
    required=True,
    help="path to track file",
)

ap.add_argument(
    "-rfl",
    "--reference_line",
    nargs="+",
    type=int,
    required=False,
    default=[250, 400, 575, 200],
    help="reference line coords",
)

args = vars(ap.parse_args())

date = args["input_file"].split("/")[2]
base_path = "/".join(args["input_file"].split("/")[:-1]) + "/"

class_colors = {
    "tw": (0, 0, 255),
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
}


def string2list(s):
    if s == "[]" or s is None:
        return []
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("(", "")
    s = s.replace(" ", "")

    l1 = s.split("),")
    l1[-1] = l1[-1].replace(")", "")
    l1

    l2 = []
    for e in l1:
        t1 = []
        for n in e.split(","):
            t1.append(int(n))
        t1 = tuple(t1)
        l2.append(t1)

    return l2


def position_of_point(Px, Py, Ax, Ay, Bx, By):
    Ay *= -1
    By *= -1
    Py *= -1
    cross_product = (Px - Ax) * (By - Ay) - (Py - Ay) * (Bx - Ax)
    return cross_product > 0


def get_intersections(Ax, Ay, Bx, By, lines):
    line1 = LineString([(Ax, -1 * Ay), (Bx, -1 * By)])

    inter_pts = []

    for l in lines:
        line2 = LineString([(l[0][0], -l[0][1]), (l[1][0], -l[1][1])])

        int_pt = line1.intersection(line2)
        try:
            poi = int_pt.x, int_pt.y
        except AttributeError:
            # empty LineString
            continue

        inter_pts.append(poi)

    inter_pts = [tuple((int(pt[0]), int(-pt[1]))) for pt in inter_pts]
    return inter_pts


def plot_tracks(img, df, cls, Ax, Ay, Bx, By):
    global base_path
    lines = []

    for cl, trk in zip(df.obj_class, df.trk):
        if cls != "all" and cl not in cls:
            continue

        flag1 = True
        prev_point = None

        pts = []

        for pt in trk:
            if flag1:
                if position_of_point(pt[0], pt[1], Ax, Ay, Bx, By):
                    try:
                        pts[0] = pt
                    except Exception:
                        pts.append(pt)
                else:
                    if len(pts) == 1:
                        pts.append(pt)
                    flag1 = False

            if not prev_point is None:
                cv2.line(
                    img,
                    (prev_point[0], prev_point[1]),
                    (pt[0], pt[1]),
                    class_colors[cl],
                    thickness=1,
                    lineType=4,
                )
            prev_point = pt

        if len(pts) == 2:
            lines.append(pts)

    cv2.imwrite(f"{base_path}tracks_{cls}.jpeg", img)

    return lines


def plot_hist(cls, inter_pts):
    global base_path

    ypts = [p[1] for p in inter_pts]
    ypts_inv = [540 - y for y in ypts]

    f, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(x=ypts_inv, stat="density")
    sns.kdeplot(x=ypts_inv)
    ax.set_title(cls)
    ax.set_xticks([])
    ax.set_xlabel("")
    plt.savefig(f"{base_path}hist_{cls}.jpeg", bbox_inches="tight")
    plt.clf()


def merge_images(cls):
    global base_path

    img1 = cv2.imread(f"{base_path}tracks_{cls}.jpeg")
    img1 = cv2.resize(img1, (500, 300))

    img2 = cv2.imread(f"{base_path}hist_{cls}.jpeg")
    img2 = cv2.resize(img2, (500, 300))

    img = np.hstack((img1, img2))

    cv2.imwrite(f"{base_path}tracks_hist_{cls}.jpeg", img)


df = pd.read_csv(
    args["input_file"],
    delimiter=" : ",
    names=["id", "obj_class", "trk"],
    engine="python",
)
df.trk = df.trk.apply(string2list)

Ax, Ay = args["reference_line"][:2]
Bx, By = args["reference_line"][2:]

interested_classes = [
    "2t",
    "3t",
    "4t",
    "5t",
    "6t",
    "lgv",
    ["2t", "3t", "4t", "5t", "6t", "lgv"],
]

pdf_img = None

for cls in interested_classes:
    img = cv2.imread(args["input_image"])

    cls = [cls] if type(cls) != list else cls

    lines = plot_tracks(img, df, cls, Ax, Ay, Bx, By)
    inter_pts = get_intersections(Ax, Ay, Bx, By, lines)

    plot_hist(cls, inter_pts)
    merge_images(cls)

    if pdf_img is None:
        pdf_img = cv2.imread(f"{base_path}tracks_hist_{cls}.jpeg")
    else:
        pdf_img = np.vstack((pdf_img, cv2.imread(f"{base_path}tracks_hist_{cls}.jpeg")))

    os.remove(f"{base_path}tracks_{cls}.jpeg")
    os.remove(f"{base_path}hist_{cls}.jpeg")

cv2.imwrite(f"{base_path}{date}_tracks.jpeg", pdf_img)

pdf_bytes = img2pdf.convert(f"{base_path}{date}_tracks.jpeg")
f = open(f"{base_path}{date}_tracks.pdf", "wb")
f.write(pdf_bytes)
f.close()
