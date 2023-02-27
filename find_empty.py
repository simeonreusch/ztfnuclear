import glob
import os

import pandas as pd

folder = "/Users/simeon/ZTFDATA/nuclear_sample/BTS/data/"

files = []

for f in os.listdir(folder):
    if f[-3:] == "csv":
        files.append(os.path.join(folder, f))

for f in files:

    df = pd.read_csv(f, comment="#")
    passcol = df["pass"].values
    for i in passcol:
        if i == "pass":
            print(f)
