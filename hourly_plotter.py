import os
import argparse
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

args = vars(ap.parse_args())


if os.path.exists(args["input_path"]):

    try:
        log_filename = args["input_path"] + args["input_path"].split("/")[-2] + ".txt"

        plot_dir = args["input_path"]
        plot_hour = plot_dir.split("/")[-2]

        df2 = pd.read_csv(
            log_filename,
            delimiter=",",
            names=["id", "name", "date", "time", "lane", "tconfig", "dir"],
        )
        fig = plt.figure(1, (14, 8))
        ax1 = plt.subplot(1, 1, 1)
        sns.countplot(df2.name, hue=df2.lane, ax=ax1)
        ax1.set_title(f"Vehicle count throughout {plot_hour}, bifurcated by lane")
        ax1.set_xlabel("vehicle type")
        ax1.set_ylabel("vehicle count")
        plt.savefig(f"{plot_dir}{plot_hour}_vcbylane.jpeg")
        plt.clf()

        cc_filename = (
            args["input_path"] + args["input_path"].split("/")[-2] + "_finalcounts.txt"
        )

        df1 = pd.read_csv(
            cc_filename, delimiter=" : ", names=["v", "c"], engine="python"
        )[:-2]
        fig = plt.figure(1, (14, 8))
        ax2 = plt.subplot(1, 1, 1)
        sns.barplot(df1.v, df1.c, ax=ax2)
        ax2.set_title(f"Vehicle count throughout {plot_hour}")
        ax2.set_xlabel("vehicle type")
        ax2.set_ylabel("vehicle count")
        plt.savefig(f"{plot_dir}{plot_hour}_vc.jpeg")
        plt.clf()

    except Exception:
        pass

else:
    print(f"{args['input_path']} does not exist")
