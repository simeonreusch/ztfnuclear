#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging, re, json, multiprocessing
import pandas as pd
import matplotlib
from tqdm import tqdm

from ztfnuclear.sample import NuclearSample, Transient

s = NuclearSample()
nprocess = 6

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _tde_fitter(ztfid):
    t = Transient(ztfid)
    t.recreate_baseline()
    t.fit_tde()
    # t.plot_tde()
    return ztfid


if __name__ == "__main__":
    startindex = 0

    with multiprocessing.Pool(nprocess) as p:
        for result in tqdm(
            p.imap_unordered(_tde_fitter, s.ztfids[startindex:]),
            total=len(s.ztfids[startindex:]),
        ):
            a = 1
