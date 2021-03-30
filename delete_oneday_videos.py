import os
import argparse


ap = argparse.ArgumentParser()

ap.add_argument(
    "-ip", "--input_path", type=str, required=True, help="path to input directory"
)

args = vars(ap.parse_args())

if os.path.exists(args["input_path"]):

    for dir in sorted(os.listdir(args["input_path"])):
        if not dir.endswith(".txt") and not dir.endswith(".jpeg") and dir != "tracks":

            for f in os.listdir(args["input_path"] + dir + "/"):
                if f.endswith(".avi"):
                    fname = args["input_path"] + dir + "/" + f
                    os.remove(fname)

else:
    print(f"{args['input_path']} does not exist")
