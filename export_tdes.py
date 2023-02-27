#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from ztfnuclear import crossmatch, io, utils
from ztfnuclear.sample import NuclearSample, Transient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.baseline").setLevel(logging.WARN)
logging.getLogger("ztfnuclear.crossmatch").setLevel(logging.DEBUG)


s = NuclearSample(sampletype="nuclear")

sample = s.meta.get_dataframe(["crossmatch", "RA", "Dec", "classif", "fritz_z"])
sample = sample.query("classif == 'tde'")

basedir = Path("/Users/simeon/ZTFDATA/nuclear_sample/NUCLEAR/baseline")
outdir = Path("/Users/simeon/Desktop/tdes")


for ztfid in sample.ztfid.values:
    t = s.transient(ztfid)
    xmatch = t.crossmatch_dict
    meta = t.meta
    fritz_z = meta.get("fritz_z")

    peakmag, peakfilt = t.peak_mag
    peakdate = t.peak_date
    if peakdate is not None:
        peakdate = peakdate + 2400000.5
        tns_z = xmatch.get("TNS", {}).get("z")
        distnr = meta.get("distnr", {}).get("distnr")

        if fritz_z is None:
            z = tns_z
        else:
            z = fritz_z

        header = io.get_ztfid_header(ztfid=ztfid, baseline=True)
        if header is not None:
            header.update(
                {
                    "bts_class": "Tidal Disruption Event",
                    "bts_z": z,
                    "bts_peak_jd": peakdate,
                    "bts_peak_mag": peakmag,
                    "bts_peak_absmag": "-",
                    "bts_peak_filter": peakfilt,
                    "bts_rise": "-",
                    "bts_fade": "-",
                    "bts_duration": "-",
                    "median_distnr": distnr,
                    "mean_distnr": "-",
                }
            )

            path = basedir / f"{ztfid}_bl.csv"
            df = pd.read_csv(path, comment="#", index_col=0)
            outpath = outdir / f"{ztfid}_bl.csv"
            io.to_csv(df=df, header=header, outpath=outpath)
