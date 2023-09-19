#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Palatino",
}
matplotlib.rcParams.update(nice_fonts)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

GOLDEN_RATIO = 1.62

stages = [
    "No cut",
    "Milliquas match",
    "WISE colors",
    "Core distance",
    r"\texttt{sgscore}",
    "SN Ia diag. cut",
    "Temperature",
    "Rise/decay",
    r"$\chi^2$",
    "Only one flare",
]
efficiency = [
    100.0,
    100.0,
    98.4375,
    82.8125,
    81.25,
    71.875,
    64.0625,
    59.375,
    56.25,
    48.4375,
]
purity = [
    1.5632633121641426,
    1.5916438696841582,
    1.5877016129032258,
    9.981167608286253,
    12.839506172839506,
    31.292517006802722,
    58.57142857142858,
    67.85714285714286,
    76.59574468085107,
    79.48717948717949,
]

colordict = {
    "purity": "#39b54a",
    "efficiency": "#ec008c",
}

fig, ax = plt.subplots(figsize=(5, 5 / GOLDEN_RATIO), dpi=300)
ax.plot(stages, efficiency, label="Efficiency", c=colordict["efficiency"])
ax.plot(stages, purity, label="Purity", c=colordict["purity"], ls="-.")

ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_xlabel("Cut stage (aggregate)", fontsize=11)
ax.set_ylabel("Percent", fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()

outfile = (
    Path("/")
    / "Users"
    / "simeon"
    / "thesis"
    / "figures"
    / "nuclear"
    / "efficiency_purity.pdf"
)
plt.savefig(outfile)
