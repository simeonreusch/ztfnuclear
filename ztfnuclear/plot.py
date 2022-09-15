#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, warnings

import astropy  # type: ignore
from astropy import units as u  # type: ignore
from astropy.coordinates import Angle  # type: ignore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

from ztfnuclear import io, utils
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


def plot_salt_tde_chisq():
    """Plot the salt fit vs. TDE fit chisq"""

    meta = MetadataDB()
    metadata = meta.read_parameters(params=["_id", "tde_fit_loose_bl", "salt"])

    ztfids = metadata["_id"]
    tde_res = metadata["tde_fit_loose_bl"]
    salt_res = metadata["salt"]

    salt_red_chisq = []
    tde_red_chisq = []

    for i, entry in enumerate(salt_res):
        if entry:
            if entry != "failure":
                if tde_res[i]:
                    if tde_res[i] != "failure":
                        salt_chisq = float(entry["chisq"])
                        salt_ndof = float(entry["ndof"])

                        tde_chisq = float(tde_res[i]["chisq"])
                        tde_ndof = float(tde_res[i]["ndof"])

                        salt_red_chisq.append(salt_chisq / salt_ndof)
                        tde_red_chisq.append(tde_chisq / tde_ndof)

    fig, ax = plt.subplots(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)
    fig.suptitle(f"SALT vs. TDE fit reduced chisquare", fontsize=14)

    ax.scatter(
        salt_red_chisq,
        tde_red_chisq,
        marker=".",
        s=2,
    )

    ax.set_xlabel("SALT fit red. chisq.")
    ax.set_ylabel("TDE fit red. chisq.")

    ax.set_xlim([0, 25])
    ax.set_ylim([0, 25])

    outfile_zoom = os.path.join(io.LOCALSOURCE_plots, "salt_vs_tde_chisq_zoom.pdf")
    outfile = os.path.join(io.LOCALSOURCE_plots, "salt_vs_tde_chisq.pdf")
    plt.tight_layout()
    plt.savefig(outfile_zoom)

    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
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
    wise_bayesian: dict = None,
    snt_threshold=3,
):
    """Plot a lightcurve"""

    wise_df.to_csv("test.csv")

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

        bandname = utils.ztf_filterid_to_band(filterid, short=True)

        if bl_correction:
            ampl_column = "ampl_corr"
            ampl_err_column = "ampl_err_corr"
            fig.suptitle(f"{ztfid} (baseline corrected)", fontsize=14)
        else:
            ampl_column = "ampl"
            ampl_err_column = "ampl.err"
            fig.suptitle(f"{ztfid} (no baseline correction)", fontsize=14)

        obsmjd = _df.obsmjd.values

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            F0 = 10 ** (_df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * _df.magzpunc
            Fratio = _df[ampl_column] / F0
            Fratio_err = np.sqrt(
                (_df[ampl_err_column] / F0) ** 2
                + (_df[ampl_column] * F0_err / F0**2) ** 2
            )
            abmag = -2.5 * np.log10(Fratio)
            abmag_err = 2.5 / np.log(10) * Fratio_err / Fratio

            _df["flux_Jy"] = utils.abmag_to_flux_density(abmag)

        if snt_threshold:
            snt_limit = Fratio_err * snt_threshold
            abmag = np.where(Fratio > snt_limit, abmag, 99)
            abmag_err = np.where(Fratio > snt_limit, abmag_err, 0.0000000001)
            placeholder_obsmjd = obsmjd[np.argmin(abmag)]
            obsmjd = np.where(abmag < 99, obsmjd, placeholder_obsmjd)

        if magplot:

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
                    ax.errorbar(
                        wise_df.mean_mjd,
                        wise_df.W2_mean_mag_ab,
                        fmt="o",
                        mec="brown",
                        ecolor="brown",
                        alpha=1,
                        ms=3,
                        elinewidth=1,
                    )

        else:
            ax.errorbar(
                _df.obsmjd,
                utils.band_frequency(bandname) * _df["flux_Jy"],
                _df[ampl_err_column] / 100000000000000000000,
                fmt="o",
                mec=color_dict[filterid],
                ecolor=color_dict[filterid],
                mfc="None",
                alpha=0.7,
                ms=2,
                elinewidth=0.5,
            )
            if wise_df is not None:
                if len(wise_df) > 0:
                    flux_W1 = utils.abmag_to_flux_density(wise_df.W1_mean_mag_ab)
                    flux_W2 = utils.abmag_to_flux_density(wise_df.W2_mean_mag_ab)

                    ax.errorbar(
                        wise_df.mean_mjd,
                        # wise_df.W1_mean_mag_ab,
                        utils.band_frequency("W1") * flux_W1,
                        fmt="o",
                        mec="black",
                        ecolor="black",
                        mfc="black",
                        alpha=1,
                        ms=5,
                        elinewidth=1,
                    )
                    ax.errorbar(
                        wise_df.mean_mjd,
                        # wise_df.W2_mean_mag_ab,
                        utils.band_frequency("W2") * flux_W2,
                        fmt="o",
                        mec="gray",
                        mfc="gray",
                        ecolor="gray",
                        alpha=1,
                        ms=5,
                        elinewidth=1,
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
