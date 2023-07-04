#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import argparse
import json
import logging
import multiprocessing
import os
import re

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

import matplotlib  # type: ignore
from ztfnuclear.database import MetadataDB
from ztfnuclear.sample import NuclearSample, Transient

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.baseline").setLevel(logging.WARN)
logging.getLogger("ztfnuclear.crossmatch").setLevel(logging.DEBUG)
logging.getLogger("ztfnuclear.sample").setLevel(logging.DEBUG)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

RECREATE_BASELINE = True
DEBUG = False
SINGLECORE = False
CORES = 16
SAMPLE = "train"

if not CORES:
    nprocess = int(multiprocessing.cpu_count() / 2)
else:
    nprocess = int(CORES)

s = NuclearSample(sampletype=SAMPLE)
meta = MetadataDB(sampletype=SAMPLE)


def _tde_fitter(ztfid):
    t = Transient(ztfid, sampletype=SAMPLE)
    if RECREATE_BASELINE:
        t.recreate_baseline()
    t.fit_tde(debug=DEBUG)
    if DEBUG:
        t.plot_tde(debug=DEBUG)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Fit TDE lightcurves")
    parser.add_argument(
        "-tde",
        "--tde",
        action="store_true",
        help="Fit certified TDEs only.",
    )

    if SINGLECORE:
        logger.info(f"Running TDE fits for in 1 thread.")
    else:
        logger.info(f"Running TDE fits for in {nprocess} threads.")

    commandline_args = parser.parse_args()
    tde_only = commandline_args.tde

    if tde_only:
        res = meta.read_parameters(
            params=[
                "_id",
                "fritz_class",
            ]
        )
        fritz_class_all = res["fritz_class"]
        tdes = []

        for i, entry in enumerate(fritz_class_all):
            if entry == "Tidal Disruption Event":
                tdes.append(res["_id"][i])

        ztfids = tdes
    else:
        ztfids = s.ztfids

    if SINGLECORE:
        for ztfid in tqdm(ztfids):
            t = Transient(ztfid, sampletype=SAMPLE)
            if RECREATE_BASELINE:
                t.recreate_baseline()
            t.fit_tde(debug=DEBUG)
            if DEBUG:
                t.plot_tde(debug=DEBUG)

    else:
        with multiprocessing.Pool(nprocess) as p:
            for result in tqdm(
                p.imap_unordered(_tde_fitter, ztfids),
                total=len(ztfids),
            ):
                a = 1

        fitres = meta.read_parameters(["_id", FIT_TYPE])
        outfile = f"{FIT_TYPE}.json"

        logger.info(f"Fitting for {FIT_TYPE} done, exporting result to {outfile}")

        with open(outfile, "w") as fp:
            json.dump(fitres, fp)
