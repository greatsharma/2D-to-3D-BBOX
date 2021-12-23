import os
import argparse
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


ap = argparse.ArgumentParser()

ap.add_argument(
    "-ip", "--input_path", type=str, required=True, help="path to input directory"
)

ap.add_argument(
    "-cp", "--camera_place", type=str, required=True, help="place of camera"
)

args = vars(ap.parse_args())


def parse_finalcounts_file(final_counts, file_path, hourly_finalcounts_filewriter):
    f = open(file_path, "r")

    for line in f.readlines():
        k, v = line.split(":")
        k, v = k.strip(), int(v.strip())
        final_counts[k] += v

        if k == "Total":
            fn = file_path.split("/")[-1]
            hr = fn.split("_")[0]
            hourly_finalcounts_filewriter.write(f"{hr} : {v}\n")

    return final_counts


if os.path.exists(args["input_path"]):
    log_filename = (
        args["input_path"]
        + args["input_path"][len(args["input_path"]) - 11 :].replace("/", "")
        + ".txt"
    )

    if not os.path.exists(args["input_path"] + "tracks/"):
        os.mkdir(args["input_path"] + "tracks/")

    trk_filename = (
        args["input_path"]
        + "tracks/"
        + args["input_path"][len(args["input_path"]) - 11 :].replace("/", "")
        + "_trkpath.txt"
    )

    hourly_finalcounts_filename = (
        args["input_path"]
        + args["input_path"][len(args["input_path"]) - 11 :].replace("/", "")
        + "_hourlytotalvehicles.txt"
    )

    if os.path.exists(log_filename):
        os.remove(log_filename)

    if os.path.exists(trk_filename):
        os.remove(trk_filename)

    if os.path.exists(hourly_finalcounts_filename):
        os.remove(hourly_finalcounts_filename)

    log_filewriter = open(log_filename, "a+")
    trk_filewriter = open(trk_filename, "a+")

    hourly_finalcounts_filewriter = open(hourly_finalcounts_filename, "a+")
    hourly_finalcounts_filewriter.write("Time : Total_Vehicles\n")

    final_counts = {
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
        "Total": 0,
        "Wrong direction": 0,
    }

    for dir in sorted(os.listdir(args["input_path"])):
        if not dir.endswith(".txt") and not dir.endswith(".jpeg") and dir != "tracks":

            for f in os.listdir(args["input_path"] + dir + "/"):
                fname = args["input_path"] + dir + "/" + f

                if f.endswith("finalcounts.txt"):
                    final_counts = parse_finalcounts_file(
                        final_counts, fname, hourly_finalcounts_filewriter
                    )

                elif f.endswith("trkpath.txt"):
                    f = open(fname, "r")
                    trk_filewriter.write(f.read())

                elif f.endswith(".txt"):
                    f = open(fname, "r")
                    log_filewriter.write(f.read())

    log_filewriter.close()
    trk_filewriter.close()
    hourly_finalcounts_filewriter.close()

    cc_filename = (
        args["input_path"]
        + args["input_path"][len(args["input_path"]) - 11 :].replace("/", "")
        + "_finalcounts.txt"
    )

    with open(cc_filename, "w") as cc_filewriter:
        for k, v in final_counts.items():
            cc_filewriter.write(k + " : " + str(v) + "\n")

    try:
        plot_dir = args["input_path"]
        plot_day = plot_dir.split("/")[-2]

        df1 = pd.read_csv(
            cc_filename, delimiter=" : ", names=["v", "c"], engine="python"
        )[:-2]
        fig = plt.figure(1, (14, 8))
        ax1 = plt.subplot(1, 1, 1)
        sns.barplot(df1.v, df1.c, ax=ax1)
        ax1.set_title(f"Vehicle count throughout {plot_day}")
        ax1.set_xlabel("vehicle type")
        ax1.set_ylabel("vehicle count")
        plt.savefig(f"{plot_dir}{plot_day}_vc.jpeg")
        plt.clf()

        df2 = pd.read_csv(
            log_filename,
            delimiter=",",
            names=["id", "name", "date", "time", "lane", "tconfig", "dir"],
        )
        fig = plt.figure(1, (14, 8))
        ax2 = plt.subplot(1, 1, 1)
        sns.countplot(df2.name, hue=df2.lane, ax=ax2)
        ax2.set_title(f"Vehicle count throughout {plot_day}, bifurcated by lane")
        ax2.set_xlabel("vehicle type")
        ax2.set_ylabel("vehicle count")
        plt.savefig(f"{plot_dir}{plot_day}_vcbylane.jpeg")
        plt.clf()

    except Exception:
        pass

    status = subprocess.call(
        [
            "python",
            "trackdensity_plotter.py",
            "-ii",
            f"inputs/{args['camera_place']}.jpeg",
            "-if",
            trk_filename,
        ]
    )

    if status:
        print(f"\n error in plotting tracks ! \n")

else:
    print(f"{args['input_path']} does not exist")
