#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

from pathlib import Path

import numpy as np
import pandas as pd
from ztfnuclear import io

infile = Path(io.LOCALSOURCE_plots) / "maghist" / "surviving_tde.csv"

df_raw = pd.read_csv(infile)
df = df_raw[
    [
        "Unnamed: 0",
        "ampel_z_z",
        "xgclass",
        "crossmatch_TNS_name",
        "peak_mags_g",
    ]
]
df.rename(columns={"Unnamed: 0": "ztfid", "ampel_z_z": "z"}, inplace=True)

z_type = []
fritz_class = []
for entry in df_raw.ampel_z_z_group.values:
    if np.isnan(entry):
        z_type.append(entry)
    elif entry < 4:
        z_type.append("spec.")
    else:
        z_type.append("phot.")

for entry in df_raw.fritz_class:
    if entry == "Tidal Disruption Event":
        fritz_class.append("TDE")
    else:
        fritz_class.append(entry)

df["fritz"] = fritz_class
df["z_type"] = z_type
df.sort_values(by="ztfid", inplace=True)

df.to_csv(Path(io.LOCALSOURCE_plots) / "maghist" / "surviving_tde_cleaned.csv")
