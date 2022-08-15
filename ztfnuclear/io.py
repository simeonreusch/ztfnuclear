#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re

import pandas as pd

_SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
LOCALSOURCE = os.path.join(os.getenv("ZTFDATA"), "nuclear_sample")
REMOTESOURCE = "https://syncandshare.desy.de/index.php/s/GHeGQYxgk5FeToY/download"

if not os.path.exists(LOCALSOURCE):
    os.makedirs(LOCALSOURCE)


def is_ztf_name(ztfid: str) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    is_match = re.match("^ZTF[1-2]\d[a-z]{7}$", name)
    if is_match:
        return True
    else:
        return False


def get_ztfid_dataframe(ztfid: str) -> pd.DataFrame:
    """
    Get the Pandas Dataframe of a single transient
    """
    filepath = os.path.join(LOCALSOURCE)
    return df
