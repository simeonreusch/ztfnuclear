#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
import pandas as pd
from ztfnuclear import io
from ztfnuclear.sample import NuclearSample

s = NuclearSample()

ratings = {1: [], 2: [], 3: []}

for t in s.get_transients():
    rating = t.get_rating(username="tderate")
    if rating != 0:
        so_far = ratings[rating]
        so_far.append(t.ztfid)
        ratings.update({rating: so_far})

for score in [1, 2, 3]:
    print(f"{score}: {len(ratings[score])}")

total = len(ratings[1]) + len(ratings[2]) + len(ratings[3])
print(f"Total: {total}")

# for ztfid in ratings[2]:
#     print(ztfid)
# quit()

infile = Path("/Users/simeon/Desktop/evaluate_classifier_wrong.csv")
df = pd.read_csv(infile)
print(df.value_counts(subset=["rejection"]))
print(f"total: {len(df)}")
# infile = Path(io.LOCALSOURCE_plots) / "maghist" / "surviving_tde_nocut.csv"
# infile_test = Path("/Users/simeon/Desktop/df_test.csv")


# df = pd.read_csv(infile)
# df.rename(columns={"Unnamed: 0": "ztfid"}, inplace=True)


# df_test = pd.read_csv(infile_test)
# test_ztfids = df_test.ztfid.values

# has_not_been_seen = []
# for ztfid in df.ztfid.values:
#     if ztfid[0] in test_ztfids:
#         print(ztfid[0])
#     # if ztfid in test_ztfids:
#     # print(ztfid)
#     # printa = 1
