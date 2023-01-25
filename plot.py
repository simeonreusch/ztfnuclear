#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import os, logging
from typing import Tuple, List

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
    plottype: str = "scatter",
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
            plot_mag_hist(cuts=cuts_now, logplot=True, plot_ext="png", rerun=rerun)


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


# aggregate_cuts(plottype="mag", sampletype=SAMPLE, rerun=False)

aggregate_cuts(plottype="mag", sampletype=SAMPLE, cuts=["milliquas_noagn"], rerun=False)
# aggregate_cuts(sampletype=SAMPLE)
# iterate_classes()
