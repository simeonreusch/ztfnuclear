#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging
from typing import Tuple, List

import numpy as np
import matplotlib  # type: ignore
from tqdm import tqdm  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
from ztfnuclear.plot import (
    plot_tde_scatter,
    plot_tde_scatter_seaborn,
    get_tde_selection,
    plot_mag_hist,
    plot_dist_hist,
)
from ztfnuclear.sample import NuclearSample, Transient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.baseline").setLevel(logging.WARN)
matplotlib.pyplot.switch_backend("Agg")

SAMPLE = "nuclear"


def aggregate_cuts(
    plottype: str = "mag",
    sampletype: str = "nuclear",
    cuts: bool = None,
    xval: str = "rise",
    yval="decay",
    xlim: Tuple[float] = (0.1, 3),
    ylim: Tuple[float] = (0.1, 4),
    plotrange: List[float] | None = None,
    rerun: bool = False,
):
    if cuts is None:
        cuts_to_use = [
            "nocut",
            "coredist",
            "milliquas_noagn",
            "snia",
            "temp",
            "risedecay",
            "chisq",
            "bayes",
        ]
    else:
        cuts_to_use = cuts
    cuts_now = []

    for cut in cuts_to_use:
        cuts_now.append(cut)
        if plottype == "scatter":
            plot_tde_scatter(
                sampletype=sampletype,
                cuts=cuts_now,
                x_values=xval,
                y_values=yval,
                xlim=xlim,
                ylim=ylim,
            )
        if plottype == "mag":
            plot_mag_hist(cuts=cuts_now, logplot=True, plot_ext="pdf", rerun=rerun)


def iterate_classes(
    classes: List[str] = [
        "all",
        "snia",
        "tde",
        "sn_other",
        "unclass",
    ]
):
    for c in classes:
        plot_dist_hist(classif=c)


def plot_all_in_selection(
    sampletype: str = "nuclear", cuts: list = ["full"], rerun=False
):
    sample = get_tde_selection(cuts=cuts, sampletype=sampletype, rerun=rerun)

    for i in tqdm(sample.index.values):
        t = Transient(i)
        t.plot(magplot=True, plot_raw=False, snt_threshold=5, no_magrange=False)


def plot_bright(rerun=False, bl=False):
    sample = get_tde_selection(
        cuts=["milliquas_noagn"], sampletype="nuclear", rerun=rerun
    )
    sample["peak_mag"] = [
        k if not np.isnan(k) else sample.peak_mags_r.values[i]
        for i, k in enumerate(sample.peak_mags_g.values)
    ]
    sample.query(
        "16<peak_mag<17 and classif == 'unclass' and crossmatch_bts_class.isnull()",
        inplace=True,
    )
    if bl:
        plot_raw = False
    else:
        plot_raw = True
    # print(sample.query("index == 'ZTF17aaagpwv'"))
    for i in tqdm(sample.index.values):
        t = Transient(i)
        t.plot(magplot=True, plot_raw=plot_raw, snt_threshold=6, no_magrange=True)


def plot_single(name):
    t = Transient(name)
    t.plot(
        magplot=True, plot_raw=True, snt_threshold=6, no_magrange=True, plot_png=True
    )


# plot_single("ZTF19aafnogq")
plot_bright(bl=False)
# plot_mag_hist(cuts=["milliquas_noagn"], logplot=True, plot_ext="png", rerun=False)
# plot_all_in_selection()
# aggregate_cuts(rerun=False)
