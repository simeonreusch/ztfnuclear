#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, warnings, typing

from typing import Optional, Union

import astropy  # type: ignore
from astropy import units as u  # type: ignore
from astropy.coordinates import Angle  # type: ignore
from astropy import constants as const  # type: ignore

import matplotlib
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns

from ztfnuclear import io, utils
from ztfnuclear.database import MetadataDB, SampleInfo

GOLDEN_RATIO = 1.62

logger = logging.getLogger(__name__)

config = io.load_config()
config["fritz_sn_ia"] = config["fritz_sn_ia"] + [
    f"SN {i}" for i in config["fritz_sn_ia"]
]
config["fritz_sn_other"] = config["fritz_sn_other"] + [
    f"SN {i}" for i in config["fritz_sn_other"]
]


def get_tde_selection(
    flaring_only: bool = False,
    cut: Optional[str] = None,
    sampletype: str = "nuclear",
) -> pd.DataFrame:
    """
    Apply selection cuts to a pandas dataframe
    """
    meta = MetadataDB(sampletype=sampletype)

    if sampletype == "nuclear":
        salt_key = "salt_loose_bl"
        class_key = "fritz_class"
    else:
        salt_key = "salt"
        class_key = "type"

    res = meta.read_parameters(
        params=[
            "tde_fit_exp",
            "_id",
            class_key,
            "crossmatch",
            "WISE_bayesian",
            salt_key,
            "ZTF_bayesian",
        ]
    )

    def snia_diag_cut(x):
        return 3.34 - 2.2 * x

    def aggressive_snia_diag_cut(x):
        return 3.55 - 2.29 * x

    tde_res = res["tde_fit_exp"]
    all_ztfids = res["_id"]
    fritz_class_all = res[class_key]
    crossmatch_all = res["crossmatch"]
    wise_bayesian = res["WISE_bayesian"]
    salt = res[salt_key]
    ztf_bayesian = res["ZTF_bayesian"]

    risetimes = []
    decaytimes = []
    temperatures = []
    d_temps = []
    red_chisqs = []
    chisqs = []
    wise_w1w2 = []
    wise_w2w3 = []
    salt_red_chisqs = []
    plateaustarts = []
    ztfids = []
    fritz_class = []
    tns_class = []
    wise_strength1 = []
    wise_strength2 = []
    overlapping_regions = []
    milliquas = []

    info_db = SampleInfo()

    if flaring_only:
        selected_subset = set(info_db.read()["flaring"]["ztfids"])
    else:
        selected_subset = set(info_db.read()["all"]["ztfids"])

    has_wise = []
    for i, entry in enumerate(wise_bayesian):
        if entry is not None:
            if entry["bayesian"] is not None:
                if (
                    "Wise_W1" in entry["bayesian"].keys()
                    and "Wise_W2" in entry["bayesian"].keys()
                ):
                    has_wise.append(all_ztfids[i])

    # selected_subset = set(has_wise)
    selected_subset = set(all_ztfids)

    for i, entry in enumerate(tde_res):
        ztfid = all_ztfids[i]
        if entry:
            if "success" in entry.keys():
                if entry["success"] == True:
                    chisq = entry["chisq"]
                    ndof = entry["ndof"]
                    chisqs.append(chisq)
                    red_chisq = chisq / ndof
                    if ztfid in selected_subset:
                        paramdict = entry["paramdict"]
                        salt_res = salt[i]
                        if salt_res != "failure" and salt_res != None:
                            salt_red_chisq = salt_res["chisq"] / salt_res["ndof"]
                            salt_red_chisqs.append(salt_red_chisq)
                        else:
                            salt_red_chisqs.append(99999)
                        # wise_dict = wise_bayesian[i]["bayesian"]
                        # wise_w1 = wise_dict["Wise_W1"]
                        # wise_w2 = wise_dict["Wise_W2"]
                        # wise_strength1.append(wise_w1["strength_sjoert"][0])
                        # wise_strength2.append(wise_w2["strength_sjoert"][0])
                        risetimes.append(paramdict["risetime"])
                        decaytimes.append(paramdict["decaytime"])
                        temperatures.append(paramdict["temperature"])
                        d_temps.append(paramdict["d_temp"])
                        plateaustarts.append(paramdict["plateaustart"])
                        red_chisqs.append(red_chisq)
                        ztfids.append(ztfid)
                        if ztf_bayesian[i] == None:
                            overlapping_regions.append(-99)
                        else:
                            overlapping_regions.append(
                                ztf_bayesian[i]["bayesian"]["overlapping_regions_count"]
                            )
                        fritz_class.append(fritz_class_all[i])

                        _tns_class = None
                        if crossmatch_all[i] is not None:
                            if "TNS" in crossmatch_all[i].keys():
                                if "type" in crossmatch_all[i]["TNS"].keys():
                                    _tns_class = crossmatch_all[i]["TNS"]["type"]
                        tns_class.append(_tns_class)

                        _milliquas_class = "noclass"
                        if crossmatch_all[i] is not None:
                            if "Milliquas" in crossmatch_all[i].keys():
                                if "type" in crossmatch_all[i]["Milliquas"].keys():
                                    _milliquas_class = crossmatch_all[i]["Milliquas"][
                                        "type"
                                    ]
                        milliquas.append(_milliquas_class)

                        _w1w2 = 999
                        _w2w3 = 999
                        if crossmatch_all[i] is not None:
                            if "WISE_cat" in crossmatch_all[i].keys():
                                if wise := crossmatch_all[i]["WISE_cat"]:
                                    _w1w2 = wise["Mag_W1"] - wise["Mag_W2"]
                                    _w2w3 = wise["Mag_W2"] - wise["Mag_W3"]

                        wise_w1w2.append(_w1w2)
                        wise_w2w3.append(_w2w3)

    sample = pd.DataFrame()
    sample["ztfid"] = ztfids
    sample["rise"] = risetimes
    sample["decay"] = decaytimes
    sample["temp"] = temperatures
    sample["d_temp"] = d_temps
    sample["fritz_class"] = fritz_class
    sample["tns_class"] = tns_class
    sample["red_chisq"] = red_chisqs
    sample["wise_w1w2"] = wise_w1w2
    sample["wise_w2w3"] = wise_w2w3
    sample["chisq"] = chisqs
    sample["salt_red_chisq"] = salt_red_chisqs
    sample["d_temp_length"] = plateaustarts
    sample["milliquas"] = milliquas
    sample["overlapping_regions"] = overlapping_regions
    sample["total_d_temp"] = np.asarray(plateaustarts) * np.asarray(d_temps)
    sample["snia_cut"] = aggressive_snia_diag_cut(np.asarray(risetimes))

    sample.query(
        "fritz_class not in @config['fritz_bogus'] and fritz_class not in @config['fritz_agn_star']",
        inplace=True,
    )

    wise_cut = "(wise_w1w2<0.3 or wise_w1w2>1.8) or (wise_w2w3 <1.5 or wise_w2w3>3.5)"

    boundary_cut = "rise>0.1 and rise < 3 and decay > 0.1 and decay<4 and wise_w2w3 < 900 and wise_w1w2 < 900"

    # that's the one for dtemp = 15k, evolution from peak, bolcorr
    tde_selection_finetuned = "temp > 3.93 and temp < 4.38 and d_temp>-90 and d_temp < 140 and rise>0.85 and rise<2.05 and decay>1.1 and decay<3 and red_chisq<6 and red_chisq < salt_red_chisq"

    tde_selection = "temp > 3.9 and temp < 4.4 and d_temp>-100 and d_temp < 150 and rise>0.8 and rise<2.05 and decay>1.1 and decay<3 and red_chisq<6 and red_chisq < salt_red_chisq"

    # RERUN OVERLAPPING REGION FOR EVERYTHING
    if cut:
        if cut == "full":
            sample.query(tde_selection, inplace=True)
            sample.query("snia_cut < decay", inplace=True)
            sample.query("overlapping_regions == 1", inplace=True)
            sample.query("milliquas == 'noclass'", inplace=True)
            sample.query(wise_cut, inplace=True)
        if cut == "boundary":
            sample.query(boundary_cut, inplace=True)
        if cut == "wise":
            sample.query(boundary_cut, inplace=True)
            sample.query(wise_cut, inplace=True)

    def simple_class(row):
        """Add simple classification labels"""
        if row["fritz_class"] == "Tidal Disruption Event":
            return "tde"
        if row["fritz_class"] in config["fritz_sn_ia"]:
            return "snia"
        if row["fritz_class"] in config["fritz_sn_other"]:
            return "sn_other"
        if row["fritz_class"] in config["fritz_agn_star"]:
            return "agn_star"
        return "other"

    sample["classif"] = sample.apply(lambda row: simple_class(row), axis=1)

    return sample


def plot_tde_scatter(
    flaring_only: bool = False,
    ingest: bool = False,
    x_values: str = "rise",
    y_values: str = "decay",
    cut: str = "full",
    sampletype: str = "nuclear",
):
    """
    Plot the rise vs. fadetime of the TDE fit results
    """
    info_db = SampleInfo(sampletype=sampletype)

    sample = get_tde_selection(cut=cut, sampletype=sampletype)

    fig, ax = plt.subplots(figsize=(7, 7 / GOLDEN_RATIO), dpi=300)
    fig.suptitle(
        f"TDE fit {x_values} vs. {y_values} ({len(sample)} transients)",
        fontsize=14,
    )
    for key in config["fritz_queries"].keys():
        _df = sample.query(config["fritz_queries"][key])
        ax.scatter(
            _df[x_values],
            _df[y_values],
            marker=config["pl_props"][key]["m"],
            s=config["pl_props"][key]["s"],
            c=config["pl_props"][key]["c"],
            alpha=config["pl_props"][key]["a"],
            label=config["pl_props"][key]["l"] + f" ({len(_df[x_values])})",
            zorder=config["pl_props"][key]["order"],
        )

    plt.legend(title="Fritz classification")

    ax.set_xlabel(config["axislabels"][x_values])
    ax.set_ylabel(config["axislabels"][y_values])

    # x = np.arange(0.75, 1.08, 0.01)
    # y = aggressive_snia_diag_cut(x)
    # if cut:
    # ax.plot(x, y)

    if sampletype == "ztfnuclear":
        local = io.LOCALSOURCE_plots
    else:
        local = io.LOCALSOURCE_bts_plots

    if flaring_only:
        outfile = os.path.join(local, f"tde_{x_values}_{y_values}_flaring_{cut}.pdf")

    else:
        outfile = os.path.join(local, f"tde_{x_values}_{y_values}_{cut}.pdf")

    plt.tight_layout()

    plt.savefig(outfile)

    plt.close()

    if ingest:

        info_db.ingest_ztfid_collection(
            ztfids=sample.ztfid.values, collection_name="tde_selection"
        )


def plot_tde_scatter_seaborn(
    ingest: bool = False,
    x_values: str = "rise",
    y_values: str = "decay",
    cut: str = "boundary",
    sampletype: str = "nuclear",
):
    """
    Plot the rise vs. fadetime of the TDE fit results
    """
    info_db = SampleInfo(sampletype=sampletype)

    sample = get_tde_selection(cut=cut, sampletype=sampletype)

    sample_reduced = sample.query("classif != 'tde' or ztfid == 'ZTF22aagyuao'")
    # we need one TDE to survive, so we have a handle for the legend (don't ask,
    # I DID try to make this not hacky, but failed)

    g = sns.jointplot(
        data=sample_reduced,
        x=x_values,
        y=y_values,
        hue="classif",
        hue_order=["other", "sn_other", "snia", "tde"],
        alpha=0.8,
        legend=True,
        kind="kde",
        fill=True,
        common_norm=True,
    )
    g.ax_joint.scatter(
        x=sample.query("classif == 'tde'")[x_values],
        y=sample.query("classif == 'tde'")[y_values],
        c=config["pl_props"]["tde"]["c"],
        s=config["pl_props"]["tde"]["s"],
        marker=config["pl_props"]["tde"]["m"],
    )

    leg = g.ax_joint.get_legend()

    leg.set_title("Fritz classification")

    new_labels = [
        config["pl_props"]["other"]["l"]
        + " ({length})".format(length=len(sample_reduced.query("classif == 'other'"))),
        config["pl_props"]["sn_other"]["l"]
        + " ({length})".format(
            length=len(sample_reduced.query("classif == 'sn_other'"))
        ),
        config["pl_props"]["snia"]["l"]
        + " ({length})".format(length=len(sample_reduced.query("classif == 'snia'"))),
        config["pl_props"]["tde"]["l"]
        + " ({length})".format(length=len(sample.query("classif == 'tde'"))),
    ]
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)

    if sampletype == "nuclear":
        local = io.LOCALSOURCE_plots
    else:
        local = io.LOCALSOURCE_bts_plots

    outfile = os.path.join(local, f"tde_{x_values}_{y_values}_seaborn_{cut}.pdf")

    g.fig.savefig(outfile)
    plt.close()

    if ingest:

        info_db.ingest_ztfid_collection(
            ztfids=sample.ztfid.values, collection_name="tde_selection"
        )


def plot_mag_cdf(cuts: str = Optional["agn"]):
    """Plot the magnitude distribution of the transients"""

    meta = MetadataDB()
    meta_info = meta.read_parameters(params=["_id", "ZTF_bayesian", "fritz_class"])
    bayesian = meta_info["ZTF_bayesian"]
    ztfids = meta_info["_id"]
    fritz_class = meta_info["fritz_class"]

    mags = []
    tde_mags = []

    for i, entry in enumerate(ztfids):
        bayes = bayesian[i]
        if bayes:
            fluxes = []
            for fil in ["ZTF_g", "ZTF_r", "ZTF_i"]:
                b = bayes.get("bayesian", {}).get(fil, {})
                if isinstance(b, dict):
                    maximum = b.get("max_mag_excess_region", {})
                    if maximum:
                        flux = max(maximum)
                        if isinstance(flux, list):
                            fluxes.append(flux[0])

            if fluxes:
                max_flux = max(fluxes)
                mag = -2.5 * np.log10(max_flux)
                if cuts == "agn":
                    classification = fritz_class[i]
                    if (
                        classification not in fritz_sn_ia
                        and classification not in fritz_sn_other
                        and classification != "Tidal Disruption Event"
                    ):
                        mags.append(mag)
                    elif classification == "Tidal Disruption Event":
                        tde_mags.append(mag)
                else:
                    mags.append(mag)

    fig, ax = plt.subplots(figsize=(6, 6 / GOLDEN_RATIO), dpi=300)
    if cuts == "agn":
        title = f"ZTF Nuclear Sample (n={len(mags)}), classified as AGN or no class"
    else:
        title = f"ZTF Nuclear Sample (n={len(mags)})"

    fig.suptitle(
        title,
        fontsize=14,
    )

    ax.set_xlabel("Peak magnitude (AB)")
    ax.set_ylabel("CDF")
    ax.set_yscale("log")

    ax.hist(
        mags,
        bins=100,
        range=[16, 20.2],
        cumulative=True,
        density=True,
        label="AGN/None",
    )
    ax.hist(
        tde_mags,
        bins=100,
        range=[16, 20.2],
        cumulative=True,
        density=True,
        label="TDE",
        alpha=0.5,
    )
    plt.legend()

    if cuts == "agn":
        outfile = os.path.join(io.LOCALSOURCE_plots, "magnitude_cdf_agn.pdf")
    else:
        outfile = os.path.join(io.LOCALSOURCE_plots, "magnitude_cdf.pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


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
    saltres = meta.read_parameters(params=["salt_loose_bl"])["salt_loose_bl"]

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


def plot_tde(sampletype: str = "nuclear"):
    """Plot the salt fit results from the Mongo DB"""

    meta = MetadataDB(sampletype=sampletype)
    tde_res = meta.read_parameters(params=["tde_fit_loose_bl"])["tde_fit_loose_bl"]

    red_chisq = []
    for entry in tde_res:
        if entry:
            if entry != "failure":
                chisq = float(entry["chisq"])
                ndof = float(entry["ndof"])
                red_chisq.append(chisq / ndof)

    fig, ax = plt.subplots(figsize=(8, 8 / GOLDEN_RATIO), dpi=300)
    fig.suptitle(
        f"TDE fit reduced chisquare distribution (n={len(red_chisq)})", fontsize=14
    )

    ax.hist(red_chisq, bins=100, range=[0, 10])

    outfile = os.path.join(io.LOCALSOURCE_plots, "tde_chisq_dist.pdf")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_salt_tde_chisq():
    """Plot the salt fit vs. TDE fit chisq"""

    meta = MetadataDB()
    metadata = meta.read_parameters(params=["_id", "tde_fit_loose_bl", "salt_loose_bl"])

    ztfids = metadata["_id"]
    tde_res = metadata["tde_fit_loose_bl"]
    salt_res = metadata["salt_loose_bl"]

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
    z: float = None,
    tns_name: str = None,
    magplot: bool = True,
    wise_df: pd.DataFrame = None,
    wise_bayesian: dict = None,
    snt_threshold=3,
    plot_png: bool = False,
    wide: bool = False,
    thumbnail: bool = False,
    sampletype: str = "nuclear",
) -> list:
    """Plot a lightcurve"""
    if magplot:
        logger.debug("Plotting lightcurve (in magnitude space)")
    else:
        logger.debug("Plotting lightcurve (in flux space)")

    color_dict = {1: "green", 2: "red", 3: "orange"}
    filtername_dict = {1: "ZTF g", 2: "ZTF r", 3: "ZTF i"}

    if sampletype == "nuclear":
        local = io.LOCALSOURCE_plots
    else:
        local = io.LOCALSOURCE_bts_plots

    if magplot:
        plot_dir = os.path.join(local, "lightcurves", "mag")
    else:
        if thumbnail:
            plot_dir = os.path.join(local, "lightcurves", "thumbnails")
        else:
            plot_dir = os.path.join(local, "lightcurves", "flux")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if wide:
        figwidth = 8 / (GOLDEN_RATIO + 0.52)
    else:
        figwidth = 8 / GOLDEN_RATIO

    if thumbnail:
        fig, ax = plt.subplots(figsize=(8 / 4, figwidth / 4), dpi=100)
    else:
        fig, ax = plt.subplots(figsize=(8, figwidth), dpi=300)

    bl_correction = True if "ampl_corr" in df.keys() else False

    for filterid in sorted(df["filterid"].unique()):
        _df = df.query("filterid == @filterid")

        bandname = utils.ztf_filterid_to_band(filterid, short=True)

        if bl_correction:
            ampl_column = "ampl_corr"
            ampl_err_column = "ampl_err_corr"
            if not thumbnail:
                if tns_name:
                    fig.suptitle(
                        f"{ztfid} ({tns_name}) - baseline corrected", fontsize=14
                    )
                else:
                    fig.suptitle(f"{ztfid} - baseline corrected", fontsize=14)
        else:
            ampl_column = "ampl"
            ampl_err_column = "ampl.err"
            if not thumbnail:
                if tns_name:
                    fig.suptitle(
                        f"{ztfid} ({tns_name}) - no baseline correction", fontsize=14
                    )
                else:
                    fig.suptitle(f"{ztfid} - no baseline correction", fontsize=14)

        obsmjd = _df.obsmjd.values

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            F0 = 10 ** (_df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * _df.magzpunc
            flux = _df[ampl_column] / F0
            flux_err = np.sqrt(
                (_df[ampl_err_column] / F0) ** 2
                + (_df[ampl_column] * F0_err / F0**2) ** 2
            )
            abmag = -2.5 * np.log10(flux)
            abmag_err = 2.5 / np.log(10) * flux_err / flux
            obsmjd = _df.obsmjd.values

        if snt_threshold:
            snt_limit = flux_err.values * snt_threshold
            mask = np.less(flux.values, snt_limit)
            flux = ma.masked_array(flux.values, mask=mask).compressed()
            flux_err = ma.masked_array(flux_err.values, mask=mask).compressed()
            abmag = ma.masked_array(abmag.values, mask=mask).compressed()
            abmag_err = ma.masked_array(abmag_err.values, mask=mask).compressed()
            obsmjd = ma.masked_array(obsmjd, mask=mask).compressed()

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
                label=filtername_dict[filterid],
            )
            ax.set_ylim([23, 15])
            ax.set_ylabel("Mag (AB)")

        else:

            nu_fnu = utils.band_frequency(bandname) * flux * 1e-23
            nu_fnu_err = utils.band_frequency(bandname) * flux_err * 1e-23

            ax.set_yscale("log")

            if thumbnail:
                ms = 1
            else:
                ms = 2

            ax.errorbar(
                obsmjd,
                nu_fnu,
                nu_fnu_err,
                fmt="o",
                mec=color_dict[filterid],
                ecolor=color_dict[filterid],
                mfc="None",
                alpha=0.7,
                ms=ms,
                elinewidth=0.5,
                label=filtername_dict[filterid],
            )

            ax.set_ylabel(r"$\nu$ F$_\nu$ (erg s$^{-1}$ cm$^{-2}$)", fontsize=12)

            if z is not None and thumbnail is False:

                from astropy.cosmology import FlatLambdaCDM

                cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

                lumidist = cosmo.luminosity_distance(z)
                lumidist = lumidist.to(u.cm).value

                lumi = lambda flux: flux * 4 * np.pi * lumidist**2
                flux = lambda lumi: lumi / (4 * np.pi * lumidist**2)

                ax2 = ax.secondary_yaxis("right", functions=(lumi, flux))

                ax2.set_ylabel(r"$\nu$ L$_\nu$ (erg s$^{-1}$)", fontsize=12)

    if magplot:
        if wise_df is not None:
            if len(wise_df) > 0:
                ax.errorbar(
                    wise_df.mean_mjd,
                    wise_df.W1_mean_mag_ab,
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
                    wise_df.W2_mean_mag_ab,
                    fmt="o",
                    mec="gray",
                    mfc="gray",
                    ecolor="gray",
                    alpha=1,
                    ms=5,
                    elinewidth=1,
                )

    else:
        if wise_df is not None:
            if len(wise_df) > 0:

                flux_W1 = wise_df["W1_mean_flux_density_bl_corr"] / 1000
                flux_W1_err = wise_df["W1_mean_flux_density_bl_corr_err"] / 1000
                flux_W2 = wise_df["W2_mean_flux_density_bl_corr"] / 1000
                flux_W2_err = wise_df["W2_mean_flux_density_bl_corr_err"] / 1000

                if thumbnail:
                    ms = 3
                    elinewidth = 0.08
                else:
                    ms = 5
                    elinewidth = 0.4

                ax.errorbar(
                    wise_df.mean_mjd,
                    utils.band_frequency("W1") * flux_W1 * 1e-23,
                    utils.band_frequency("W1") * flux_W1_err * 1e-23,
                    fmt="o",
                    mec="black",
                    ecolor="black",
                    mfc="black",
                    alpha=1,
                    ms=ms,
                    elinewidth=elinewidth,
                    label="WISE W1",
                )
                ax.errorbar(
                    wise_df.mean_mjd,
                    utils.band_frequency("W2") * flux_W2 * 1e-23,
                    utils.band_frequency("W2") * flux_W2_err * 1e-23,
                    fmt="o",
                    mec="gray",
                    mfc="gray",
                    ecolor="gray",
                    alpha=1,
                    ms=ms,
                    elinewidth=elinewidth,
                    label="WISE W2",
                )

    if not thumbnail:
        ax.set_xlabel("Date (MJD)", fontsize=12)
        ax.grid(which="both", b=True, axis="both", alpha=0.3)
        plt.legend()
    else:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.tick_params(bottom=False)
        plt.tick_params(left=False)

    if plot_png:
        if thumbnail:
            outfile = os.path.join(plot_dir, ztfid + "_thumbnail.png")
        else:
            outfile = os.path.join(plot_dir, ztfid + ".png")
    else:
        outfile = os.path.join(plot_dir, ztfid + ".pdf")

    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()

    xlim1, xlim2 = ax.get_xlim()
    # ylim1, ylim2 = ax.get_ylim()

    axlims = {"xlim": [xlim1, xlim2]}

    del fig, ax

    return axlims


def plot_lightcurve_irsa(
    df: pd.DataFrame,
    ztfid: str,
    ra: float,
    dec: float,
    wide: bool = False,
    magplot: bool = False,
    plot_png: bool = False,
    axlims: dict = None,
):
    """
    Get the non-difference alert photometry for a transient and plot it
    """
    if wide:
        figwidth = 8 / (GOLDEN_RATIO + 0.52)
    else:
        figwidth = 8 / GOLDEN_RATIO

    if magplot:
        plot_dir = os.path.join(io.LOCALSOURCE_plots_irsa, "mag")
    else:
        plot_dir = os.path.join(io.LOCALSOURCE_plots_irsa, "flux")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, ax = plt.subplots(figsize=(8, figwidth), dpi=300)

    cmap = {"zg": "g", "zr": "r", "zi": "orange"}
    wl = {
        "zg": 472.27,
        "zr": 633.96,
        "zi": 788.61,
    }
    fig.suptitle(f"{ztfid} - non-subtracted alert photometry", fontsize=14)

    for fc in ["zg", "zr", "zi"]:
        mask = df["filtercode"] == fc

        mags = list(df["mag"][mask]) * u.ABmag

        magerrs = list((df["magerr"][mask] + df["mag"][mask])) * u.ABmag

        if magplot:
            ax.invert_yaxis()
            ax.errorbar(
                df["mjd"][mask],
                mags.value,
                yerr=df["magerr"][mask],
                marker="o",
                linestyle=" ",
                markersize=2,
                c=cmap[fc],
                label=f"ZTF {fc[-1]}",
            )

        else:

            flux_j = mags.to(u.Jansky)

            f = (const.c / (wl[fc] * u.nm)).to("Hz")

            flux = (flux_j * f).to("erg cm-2 s-1")

            jerrs = magerrs.to(u.Jansky)
            ferrs = np.abs((jerrs * f).to("erg cm-2 s-1").value - flux.value)
            ax.set_yscale("log")

            ax.errorbar(
                df["mjd"][mask],
                flux.to("erg cm-2 s-1").value,
                yerr=ferrs,
                fmt="o",
                mfc="None",
                alpha=0.7,
                ms=2,
                elinewidth=0.5,
                c=cmap[fc],
                label=f"ZTF {fc[-1]}",
            )

            ax.set_ylabel(r"$\nu$ F$_\nu$ (erg s$^{-1}$ cm$^{-2}$)", fontsize=12)

    ax.set_xlabel("Date (MJD)", fontsize=12)

    if axlims:
        if "xlim" in axlims.keys():
            ax.set_xlim(axlims["xlim"])
        if "ylim" in axlims.keys():
            ax.set_ylim(axlims["ylim"])

    ax.grid(which="both", b=True, axis="both", alpha=0.3)

    plt.tight_layout()
    plt.legend()

    if plot_png:
        outfile = os.path.join(plot_dir, ztfid + ".png")
    else:
        outfile = os.path.join(plot_dir, ztfid + ".pdf")

    plt.savefig(outfile)


def plot_tde_fit(
    df: pd.DataFrame,
    ztfid: str,
    tde_params: dict,
    z: float = None,
    tns_name: str = None,
    snt_threshold=3.0,
    savepath: str = None,
    sampletype: str = "nuclear",
):
    """
    Plot the TDE fit result if present
    """
    from ztfnuclear.tde_fit import TDESource_exp_flextemp, TDESource_pl_flextemp
    import sncosmo
    from sfdmap import SFDMap  # type: ignore[import]

    logger.debug("Plotting TDE fit lightcurve (in flux space)")

    color_dict = {1: "green", 2: "red", 3: "orange"}
    filtername_dict = {1: "ZTF g", 2: "ZTF r", 3: "ZTF i"}

    if sampletype == "nuclear":
        plot_dir = os.path.join(io.LOCALSOURCE_plots, "lightcurves", "tde_fit")
    else:
        plot_dir = os.path.join(io.LOCALSOURCE_bts_plots, "lightcurves", "tde_fit")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    figwidth = 8 / (GOLDEN_RATIO + 0.52)

    fig, ax = plt.subplots(figsize=(8, figwidth), dpi=300)

    # initialize the TDE source
    phase = np.linspace(-50, 100, 10)
    wave = np.linspace(1000, 10000, 5)

    if "alpha" in tde_params.keys():
        tde_source = TDESource_pl_flextemp(phase, wave, name="tde")
    else:
        tde_source = TDESource_exp_flextemp(phase, wave, name="tde")

    dust = sncosmo.models.CCM89Dust()
    dustmap = SFDMap()

    fitted_model = sncosmo.Model(
        source=tde_source,
        effects=[dust],
        effect_names=["mw"],
        effect_frames=["obs"],
    )

    fitted_model.update(tde_params)

    ylim_upper = 0
    ylim_lower = 1

    for filterid in sorted(df["filterid"].unique()):
        _df = df.query("filterid == @filterid")

        bandname = utils.ztf_filterid_to_band(filterid, short=True)
        bandname_sncosmo = utils.ztf_filterid_to_band(filterid, sncosmo=True)

        ampl_column = "ampl_corr"
        ampl_err_column = "ampl_err_corr"
        if tns_name:
            fig.suptitle(f"{ztfid} ({tns_name}) - TDE-fit", fontsize=14)
        else:
            fig.suptitle(f"{ztfid} - TDE-fit", fontsize=14)

        obsmjd = _df.obsmjd.values

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            F0 = 10 ** (_df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * _df.magzpunc
            flux = _df[ampl_column] / F0 * 3630.78
            flux_err = (
                np.sqrt(
                    (_df[ampl_err_column] / F0) ** 2
                    + (_df[ampl_column] * F0_err / F0**2) ** 2
                )
                * 3630.78
            )
            abmag = -2.5 * np.log10(flux)
            abmag_err = 2.5 / np.log(10) * flux_err / flux
            obsmjd = _df.obsmjd.values

        if snt_threshold:
            snt_limit = flux_err.values * snt_threshold
            mask = np.less(flux.values, snt_limit)
            flux = ma.masked_array(flux.values, mask=mask).compressed()
            flux_err = ma.masked_array(flux_err.values, mask=mask).compressed()
            abmag = ma.masked_array(abmag.values, mask=mask).compressed()
            abmag_err = ma.masked_array(abmag_err.values, mask=mask).compressed()
            obsmjd = ma.masked_array(obsmjd, mask=mask).compressed()

        nu_fnu = utils.band_frequency(bandname) * flux * 1e-23
        nu_fnu_err = utils.band_frequency(bandname) * flux_err * 1e-23

        ax.set_yscale("log")

        ms = 2

        if len(nu_fnu) == 0:
            continue

        max_nufnu = np.max(nu_fnu)
        min_nufnu = np.min(nu_fnu)

        if max_nufnu > ylim_upper:
            ylim_upper = max_nufnu

        if min_nufnu < ylim_lower:
            ylim_lower = min_nufnu

        ax.errorbar(
            obsmjd,
            nu_fnu,
            nu_fnu_err,
            fmt="o",
            mec=color_dict[filterid],
            ecolor=color_dict[filterid],
            mfc="None",
            alpha=1,
            ms=ms,
            elinewidth=0.5,
            label=filtername_dict[filterid],
        )

        t0 = tde_params["t0"]
        x_range = np.linspace(t0 - 100, t0 + 600, 200)

        modelflux = (
            fitted_model.bandflux(bandname_sncosmo, x_range, zp=25, zpsys="ab")
            / 1e10
            * 3631
            * utils.band_frequency(bandname)
            * 1e-23
        )

        ax.plot(x_range, modelflux, c=color_dict[filterid], alpha=0.8)

    if z is not None:

        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        lumidist = cosmo.luminosity_distance(z)
        lumidist = lumidist.to(u.cm).value

        lumi = lambda flux: flux * 4 * np.pi * lumidist**2
        flux = lambda lumi: lumi / (4 * np.pi * lumidist**2)

        ax2 = ax.secondary_yaxis("right", functions=(lumi, flux))

        ax2.set_ylabel(r"$\nu$ L$_\nu$ (erg s$^{-1}$)", fontsize=12)

    ax.set_xlabel("Date (MJD)", fontsize=12)
    ax.set_ylabel(r"$\nu$ F$_\nu$ (erg s$^{-1}$ cm$^{-2}$)", fontsize=12)
    ax.grid(which="both", b=True, axis="both", alpha=0.3)
    plt.legend()

    ylim_upper = ylim_upper + ylim_upper * 0.2
    ylim_lower = ylim_lower - ylim_lower * 0.2

    ax.set_ylim([ylim_lower, ylim_upper])

    if "alpha" in tde_params.keys():
        suffix = "_pl"
    else:
        suffix = "_exp"

    if not savepath:
        outfile = os.path.join(plot_dir, ztfid + suffix + ".png")
    else:
        outfile = os.path.join(savepath, ztfid + suffix + ".png")

    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()

    del fig, ax
