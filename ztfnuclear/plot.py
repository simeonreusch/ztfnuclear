#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

import astropy
from astropy import units as u
from astropy.coordinates import Angle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ztfnuclear import io
from ztfnuclear.sample import NuclearSample
from ztfnuclear.database import MetadataDB


def plot_location(sample: NuclearSample):
    """Plot the sky location of all transients"""
    meta = MetadataDB()
    location = meta.read_parameters(params=["RA", "Dec"])

    ra = location["RA"]
    dec = location["Dec"]

    ra = Angle(np.asarray(ra) * u.degree)
    ra = ra.wrap_at(180 * u.degree)
    dec = Angle(np.asarray(dec) * u.degree)

    fig = plt.figure(figsize=(8, 6), dpi=300)

    fig.suptitle("ZTF Nuclear Sample", fontsize=14)

    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(ra.radian, dec.radian, s=0.05)
    ax.grid(True)
    outfile = os.path.join(io.LOCALSOURCE_plots, "sky_localization.pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# ra = coord.Angle(data['RA'].filled(np.nan)*u.degree)
# ra = ra.wrap_at(180*u.degree)
# dec = coord.Angle(data['Dec'].filled(np.nan)*u.degree)
