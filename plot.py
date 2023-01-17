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

SAMPLE = "bts"


def successive_cuts(
    plottype="scatter",
    sampletype="nuclear",
    cuts=None,
    xval="rise",
    yval="decay",
    xlim=(0.1, 3),
    ylim=(0.1, 4),
    plotrange: List[float] | None = None,
):
    if cuts is None:
        cuts_to_use = [
            "nocut",
            "wise",
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
        plot_tde_scatter(
            sampletype=sampletype,
            cuts=cuts_now,
            x_values=xval,
            y_values=yval,
            xlim=xlim,
            ylim=ylim,
        )


def successive_classes(
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


# successive_cuts(sampletype=SAMPLE)
successive_classes()
