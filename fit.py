#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging, re, json, multiprocessing
import pandas as pd
import matplotlib
from tqdm import tqdm

from ztfnuclear.database import MetadataDB
from ztfnuclear.sample import NuclearSample, Transient

s = NuclearSample()
meta = MetadataDB()

cores = multiprocessing.cpu_count()
nprocess = int(cores / 2)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

FIT_TYPE = "tde_fit_exp"
RECREATE_BASELINE = True


def _tde_fitter(ztfid):
    t = Transient(ztfid)
    if RECREATE_BASELINE:
        t.recreate_baseline()
    if FIT_TYPE == "tde_fit_exp":
        t.fit_tde()


if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    logger.info(f"Running fits for {FIT_TYPE} in {nprocess} threads.")

    startindex = 0

    with multiprocessing.Pool(nprocess) as p:
        for result in tqdm(
            p.imap_unordered(_tde_fitter, s.ztfids[startindex:]),
            total=len(s.ztfids[startindex:]),
        ):
            a = 1

    fitres = meta.read_parameters(["_id", FIT_TYPE])
    outfile = f"{FIT_TYPE}.json"

    logger.info(f"Fitting for {FIT_TYPE} done, exporting result to {outfile}")

    with open(outfile, "w") as fp:
        json.dump(fitres, fp)
