#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

import astropy  # type: ignore
from astropy import units as u  # type: ignore
from astropy.coordinates import Angle  # type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

from ztfnuclear import io
from ztfnuclear.database import MetadataDB

GOLDEN_RATIO = 1.62

logger = logging.getLogger(__name__)


def plot_location():
    """Plot the sky location of all transients"""
    meta = MetadataDB()
    location = meta.read_parameters(params=["RA", "Dec"])

    ra = location["RA"]
    dec = location["Dec"]

    ra = Angle(np.asarray(ra) * u.degree)
    ra = ra.wrap_at(180 * u.degree)
    dec = Angle(np.asarray(dec) * u.degree)

    fig = plt.figure(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)

    fig.suptitle(f"ZTF Nuclear Sample (n={len(ra)})", fontsize=14)

    ax = fig.add_subplot(111, projection="mollweide")
    ax.scatter(ra.radian, dec.radian, s=0.05)
    ax.grid(True)
    outfile = os.path.join(io.LOCALSOURCE_plots, "sky_localization.pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_salt():
    """Plot the salt fit results from the Mongo DB"""
    meta = MetadataDB()
    saltres = meta.read_parameters(params=["salt"])["salt"]

    red_chisq = []
    for entry in saltres:
        if entry:
            if entry != "failure":
                chisq = float(entry["chisq"])
                ndof = float(entry["ndof"])
                red_chisq.append(chisq / ndof)

    fig, ax = plt.subplots(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)
    fig.suptitle(
        f"SALT fit reduced chisquare distribution (n={len(red_chisq)})", fontsize=14
    )

    ax.hist(red_chisq, bins=100, range=[0, 60])

    outfile = os.path.join(io.LOCALSOURCE_plots, "salt_chisq_dist.pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_ampelz():
    """Plot the ampel z distribution"""
    meta = MetadataDB()
    ampelz = meta.read_parameters(params=["ampel_z"])["ampel_z"]

    z = []
    for entry in ampelz:
        if entry:
            z.append(entry["z"])

    fig, ax = plt.subplots(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)
    fig.suptitle(f"Ampel redshift distribution (n={len(z)})", fontsize=14)

    ax.hist(z, bins=100)

    outfile = os.path.join(io.LOCALSOURCE_plots, "ampel_z_dist.pdf")

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_lightcurve(
    df: pd.DataFrame,
    ztfid: str,
    magplot: bool = True,
    wise_df: pd.DataFrame = None,
    snt_threshold=3,
):
    """Plot a lightcurve"""
    if magplot:
        logger.debug("Plotting lightcurve (in magnitude space)")
    else:
        logger.debug("Plotting lightcurve (in flux space)")

    color_dict = {1: "green", 2: "red", 3: "orange"}

    if magplot:
        plot_dir = os.path.join(io.LOCALSOURCE_plots, "lightcurves", "mag")
    else:
        plot_dir = os.path.join(io.LOCALSOURCE_plots, "lightcurves", "flux")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)

    bl_correction = True if "ampl_corr" in df.keys() else False

    for filterid in df["filterid"].unique():
        _df = df.query("filterid == @filterid")

        if bl_correction:
            ampl_column = "ampl_corr"
            ampl_err_column = "ampl_err_corr"
            fig.suptitle(f"{ztfid} (baseline correction)", fontsize=14)
        else:
            ampl_column = "ampl"
            ampl_err_column = "ampl.err"
            fig.suptitle(f"{ztfid} (no baseline correction)", fontsize=14)

        obsmjd = _df.obsmjd.values

        if magplot:
            F0 = 10 ** (_df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * _df.magzpunc
            Fratio = _df[ampl_column] / F0
            Fratio_err = np.sqrt(
                (_df[ampl_err_column] / F0) ** 2
                + (_df[ampl_column] * F0_err / F0**2) ** 2
            )
            abmag = -2.5 * np.log10(Fratio)
            abmag_err = 2.5 / np.log(10) * Fratio_err / Fratio

            if snt_threshold:
                snt_limit = Fratio_err * snt_threshold
                abmag = np.where(Fratio > snt_limit, abmag, 99)
                abmag_err = np.where(Fratio > snt_limit, abmag_err, 0.0000000001)
                placeholder_obsmjd = obsmjd[np.argmin(abmag)]
                obsmjd = np.where(abmag < 99, obsmjd, placeholder_obsmjd)

            ax.errorbar(
                obsmjd,
                abmag,
                abmag_err,
                fmt="o",
                mec=color_dict[filterid],
                ecolor=color_dict[filterid],
                mfc="None",
                alpha=0.7,
                ms=2,
                elinewidth=0.8,
            )
            ax.set_ylim([23, 15])
            ax.set_ylabel("Mag (AB)")

            if wise_df is not None:
                if len(wise_df) > 0:
                    ax.errorbar(
                        wise_df.mean_mjd,
                        wise_df.W1_mean_mag_ab,
                        fmt="o",
                        mec="black",
                        ecolor="black",
                        alpha=1,
                        ms=3,
                        elinewidth=1,
                    )

        else:
            ax.errorbar(
                _df.obsmjd,
                _df[ampl_column],
                _df[ampl_err_column],
                fmt="o",
                mec=color_dict[filterid],
                ecolor=color_dict[filterid],
                mfc="None",
                alpha=0.7,
                ms=2,
                elinewidth=0.5,
            )
        ax.set_xlabel("Date (MJD)")
        ax.grid(b=True, alpha=0.8)  # (, axis="y")

    if bl_correction:
        outfile = os.path.join(plot_dir, ztfid + "_bl.pdf")
    else:
        outfile = os.path.join(plot_dir, ztfid + ".pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
