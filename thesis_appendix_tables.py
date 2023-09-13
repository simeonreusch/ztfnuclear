#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import logging

import numpy as np
from ztfnuclear import io
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.sample import NuclearSample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("ztfnuclear.database").setLevel(logging.INFO)
logging.getLogger("ztfnuclear.sample").setLevel(logging.INFO)


INCLUDE_PICS = True


def get_string_start(tex_str):
    tex_str += "\\begin{table*}[t!]\n"
    tex_str += "\\resizebox{1.5\\textwidth}{!}{%\n"
    if INCLUDE_PICS:
        tex_str += "\\begin{tabular}{c  c  c  c  c  c  c  c}\n"
    else:
        tex_str += "\\begin{tabular}{c  c  c  c  c  c  c}\n"
    tex_str += "\\hline\n"
    if INCLUDE_PICS:
        tex_str += "\\textbf{Light curve} & \\textbf{Transient} & \\textbf{z} & \\textbf{z} & \\textbf{Community} & \\textbf{IAU name} & \\textbf{Peak mag.} & \\textbf{Notes} \\\\\n & "
    else:
        tex_str += "\\textbf{Transient} & \\textbf{z} & \\textbf{z} & \\textbf{Community} & \\textbf{IAU name} & \\textbf{Peak mag.} & \\textbf{Notes} \\\\\n"
    tex_str += " & & \\textbf{type} & \\textbf{classification} & & \\textbf{(\\textit{g}-band)} & \\\\\n\\hline\n\\hline\n"

    return tex_str


def get_string_end(tex_str):
    tex_str += "\\end{tabular}}\n\\end{table*}\n\n\n"

    return tex_str


def create_table(df, classifstr="Community classification"):
    tex_str = "\n\n\n"

    for i, row in enumerate(df.iterrows()):
        if i % 9 == 0:
            tex_str = get_string_start(tex_str)

        ztfid = row[0]
        vals = row[1]
        if INCLUDE_PICS:
            tex_str += "\\parbox[c]{12em}{\\includegraphics[width=0.4\\textwidth]{/Users/simeon/ZTFDATA/nuclear_sample/plots/lightcurves/thumbnails/"
            tex_str += ztfid
            tex_str += ".pdf}} & "
        tex_str += "\\textit{\\href{https://ztfnuclear.simeonreusch.com/transient/"
        tex_str += ztfid
        tex_str += "}{"
        tex_str += ztfid
        tex_str += "}}"
        if not isinstance(vals["IAU name"], str):
            vals["IAU name"] = "~"
        if np.isnan(vals["z"]):
            vals["z"] = "~"
        if not isinstance(vals["z type"], str):
            vals["z type"] = "~"
        tex_str += f" & {vals['z']} & {vals['z type']} & {vals[classifstr]} & {vals['IAU name']} & {vals['peak mag. (g-band)']} & \\\\\n"
        if i % 9 == 8 or i == len(df) - 1:
            tex_str = get_string_end(tex_str)

    return tex_str


s = NuclearSample()

flaring = [t for t in s.get_flaring_transients()]

flaring_ztfids = [t.ztfid for t in flaring]

interesting = []
ratings = []
gold = []

for t in flaring:
    rating = t.get_rating(username="simeon2")
    rating_overview = t.get_rating_overview()

    if rating == 3:
        interesting.append(t.ztfid)
        ratings.append(rating_overview)

for i, entry in enumerate(ratings):
    if len(entry) != 1:
        gold.append(interesting[i])

df = s.get_thesis_dataframe(ztfids=gold)

table = create_table(df)

print(table)
