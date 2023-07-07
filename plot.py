#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import argparse
import logging
import os
from typing import List, Tuple

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore

from ztfnuclear.plot import (
    get_tde_selection,
    plot_confusion,
    plot_dist_hist,
    plot_mag_hist,
    plot_mag_hist_2x2,
    plot_sgscore_hist,
    plot_tde_scatter,
    plot_tde_scatter_seaborn,
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
    plot_ext: str = "pdf",
    rerun: bool = False,
):
    if cuts is None:
        cuts_to_use = [
            "nocut",
            "coredist",
            "sgscore",
            "milliquas_noagn",
            "snia",
            "temp",
            "risedecay",
            "chisq",
            "bayes",
        ]
        if plottype == "mag2x2":
            cuts_to_use.remove("milliquas_noagn")
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
            plot_mag_hist(cuts=cuts_now, logplot=True, plot_ext=plot_ext, rerun=rerun)
        if plottype == "mag2x2":
            plot_mag_hist_2x2(
                cuts=cuts_now,
                logplot=True,
                plot_ext=plot_ext,
                rerun=rerun,
            )
        if plottype == "mag2x2xg":
            plot_mag_hist_2x2(
                cuts=cuts_now,
                logplot=True,
                plot_ext=plot_ext,
                rerun=rerun,
                compare="nuc_xg",
            )


def iterate_classes(
    plottype="dist",
    classes: List[str] = [
        "all",
        "snia",
        "tde",
        "sn_other",
        "unclass",
    ],
):
    for c in classes:
        if plottype == "dist":
            plot_dist_hist(classif=c)
        if plottype == "sgscore":
            plot_sgscore_hist(classif=c)


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

    for i in tqdm(sample.index.values):
        t = Transient(i)
        t.plot(magplot=True, plot_raw=plot_raw, snt_threshold=6, no_magrange=True)


def plot_single(name):
    t = Transient(name)
    t.plot(
        magplot=True, plot_raw=True, snt_threshold=6, no_magrange=True, plot_png=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the ZTF nuclear sample")
    parser.add_argument(
        "-type",
        "-rd",
        type=str,
        default="mag",
        help="Choose the plot type",
    )
    parser.add_argument(
        "-ext",
        "-e",
        type=str,
        default="pdf",
        help="Choose the plot file extension",
    )
    parser.add_argument("-rerun", "-r", action="store_true", help="Rerun cut ingestion")
    cl = parser.parse_args()

    if cl.type in ["mag", "mag2x2", "mag2x2xg", "scatter"]:
        aggregate_cuts(
            rerun=cl.rerun,
            plottype=cl.type,
            plot_ext=cl.ext,
            cuts=[
                "nocut",
                "coredist",
                "sgscore",
                "snia",
                "temp",
                "risedecay",
                "chisq",
                "bayes",
            ],
        )
    if cl.type in ["confusion"]:
        plot_confusion(
            cuts=["milliquas_noagn", "wise_noagn"],
            plot_ext=cl.ext,
            rerun=cl.rerun,
            norm=None,
            plot_misclass=True,
        )
        # plot_confusion(
        #     cuts=["nocut"],
        #     plot_ext=cl.ext,
        #     rerun=cl.rerun,
        #     norm="true",
        #     plot_misclass=False,
        # )
        # plot_confusion(
        #     cuts=["nocut"],
        #     plot_ext=cl.ext,
        #     rerun=cl.rerun,
        #     norm="pred",
        #     plot_misclass=False,
        # )

    if cl.type in ["dist", "sgscore"]:
        iterate_classes(plottype=cl.type, plot_ext=cl.ext)
