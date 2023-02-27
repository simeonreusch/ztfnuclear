#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from ztfnuclear import io
from ztfnuclear.sample import NuclearSample

resdict = {}
s = NuclearSample(sampletype="nuclear")
for t in tqdm(s.get_transients(), total=len(s.ztfids)):
    distnr = t.meta.get("crossmatch", {}).get("sgscore", -999.0)
    resdict.update({t.ztfid: {"distnr": distnr}})

df = pd.DataFrame.from_dict(resdict, orient="index")
df.to_csv(io.LOCALSOURCE_bts_distnr)

get_from_ampel = []
df = pd.read_csv(io.LOCALSOURCE_bts_distnr)

for i, entry in df.iterrows():
    ztfid = entry[0]
    distnr = entry[1]
    if distnr == -999:
        get_from_ampel.append(ztfid)

print(get_from_ampel)

failed = []
s = NuclearSample(sampletype="bts")
for ztfid in tqdm(get_from_ampel):
    t = s.transient(ztfid)
    try:
        t.crossmatch(crossmatch_types=["dist"])
    except:
        failed.append(t.ztfid)
print(failed)


def get_from_ampel():
    failed = []

    s = NuclearSample(sampletype="nuclear")
    for t in tqdm(s.get_transients(start=0), total=len(s.ztfids[0:])):
        try:
            t.crossmatch(crossmatch_types=["sgscore"])
        except:
            failed.append(t.ztfid)

    print(len(failed))
    print("-----")
    print(failed)
