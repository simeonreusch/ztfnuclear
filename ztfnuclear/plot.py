#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import copy
import logging
import os
import typing
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import astropy  # type: ignore
import matplotlib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
from astropy import constants as const  # type: ignore
from astropy import units as u  # type: ignore
from astropy.coordinates import Angle  # type: ignore
from ztfnuclear import io, utils
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.wise import is_in_wise_agn_box

nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Helvetica",
}
matplotlib.rcParams.update(nice_fonts)
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

GOLDEN_RATIO = 1.62

logger = logging.getLogger(__name__)

config = io.load_config()
config["sn_ia"] = config["sn_ia"] + [f"SN {i}" for i in config["sn_ia"]]
config["sn_other"] = config["sn_other"] + [f"SN {i}" for i in config["sn_other"]]


def get_tde_selection(
    flaring_only: bool = False,
    cuts: None | list = ["nocut"],
    sampletype: str = "nuclear",
    purity_sel: str | None = "tde",
    rerun: bool = False,
    xgclass: bool = False,
    require_fitsuccess: bool = True,
    reject_bogus: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply selection cuts to a pandas dataframe
    """
    from ztfnuclear.sample import NuclearSample

    def snia_diag_cut(x):
        return 3.55 - 2.29 * x

    def expand_cuts(cuts):
        cutdict = {}
        for cut in cuts:
            subcut_values = [config["selections"][k] for k in config["cuts"][cut]]
            subcut_names = config["cuts"][cut]

            for i, name in enumerate(subcut_names):
                cutdict.update({name: subcut_values[i]})

        return cutdict

    if sampletype == "nuclear":
        salt_key = "salt_loose_bl"
        class_key = "fritz_class"
    elif sampletype == "bts":
        salt_key = "salt"
        class_key = "type"

    if cuts is None:
        cuts = ["nocut"]

    info_db = SampleInfo(sampletype=sampletype)

    cutdict = expand_cuts(cuts)
    ztfids_surviving = {}

    if rerun:
        cuts_to_do = cutdict
    else:
        cuts_to_do = {}
        for name, c in cutdict.items():
            cutres = info_db.read_collection(f"cut_{sampletype}_{name}")
            if cutres is None:
                cuts_to_do.update({name: c})
            else:
                ztfids_surviving.update({name: cutres})

    if len(cuts_to_do) > 0:
        s = NuclearSample(sampletype=sampletype)
        params = params = [
            "tde_fit_exp",
            "_id",
            "distnr",
            class_key,
            "crossmatch",
            "WISE_bayesian",
            salt_key,
            "ZTF_bayesian",
            "peak_mags",
        ]
        if xgclass:
            params.append("xgclass")
        sample = s.meta.get_dataframe(params=params)

        # cut everything flagged as bogus
        if reject_bogus:
            if sampletype == "nuclear":
                sample.query(config["selections"]["bogus"], inplace=True)
            elif sampletype == "bts":
                sample.query(config["selections"]["bogusbts"], inplace=True)

        def simple_class(row, sampletype, purity_sel):
            """Add simple classification labels"""
            if sampletype == "nuclear":
                class_key = "fritz_class"
            elif sampletype == "bts":
                class_key = "type"

            if "crossmatch_bts_class" in row.keys():
                bts_crossmatch = row["crossmatch_bts_class"]
            else:
                bts_crossmatch = None

            if "crossmatch_Marshal_class" in row.keys():
                marshal_crossmatch = row["crossmatch_Marshal_class"]
            else:
                marshal_crossmatch = None

            if purity_sel == "gold":
                if row["ztfid"] in gold_sample:
                    return "gold"
            if (
                row[class_key] in config["tde"]
                or row["tns_class"] in config["tde"]
                or bts_crossmatch in config["tde"]
                or marshal_crossmatch in config["tde"]
            ):
                return "tde"
            if (
                row[class_key] in config["sn_ia"]
                or row["tns_class"] in config["sn_ia"]
                or bts_crossmatch in config["sn_ia"]
                or marshal_crossmatch in config["sn_ia"]
            ):
                return "snia"
            if (
                row[class_key] in config["sn_other"]
                or row["tns_class"] in config["sn_other"]
                or bts_crossmatch in config["sn_other"]
                or marshal_crossmatch in config["sn_other"]
            ):
                return "sn_other"
            if (
                row[class_key] in config["agn"]
                or bts_crossmatch in config["agn"]
                or marshal_crossmatch in config["agn"]
            ):
                return "agn"
            if (
                row[class_key] in config["star"]
                or row["tns_class"] in config["star"]
                or bts_crossmatch in config["star"]
                or marshal_crossmatch in config["star"]
            ):
                return "star"
            if (
                row[class_key] in config["other"]
                or bts_crossmatch in config["other"]
                or marshal_crossmatch in config["other"]
            ):
                return "other"
            return "unclass"

        sample["classif"] = sample.apply(
            lambda row: simple_class(row, sampletype, purity_sel), axis=1
        )
        sample.query("classif != 'other'", inplace=True)
        s.populate_db_from_df(sample[["classif"]])

        n_tot = len(sample)
        n_nofit = len(sample.query("success.isnull()"))
        n_fit = len(sample.query("not success.isnull()"))
        n_fitfail = len(sample.query("success == False"))

        # Only use transients with fit success
        if require_fitsuccess:
            sample.query("success == True", inplace=True)

        sample["snia_cut"] = snia_diag_cut(sample["rise"])
        sample["in_wise_agn_box"] = sample.apply(
            lambda row: is_in_wise_agn_box(
                row["wise_w1w2"],
                row["wise_w2w3"],
            ),
            axis=1,
        )
        s.populate_db_from_df(sample[["snia_cut"]])

        # Now we cut!
        for name, c in cuts_to_do.items():
            sample_cut = sample.query(c)

            info_db = SampleInfo(sampletype=sampletype)
            info_db.ingest_ztfid_collection(
                ztfids=sample_cut.ztfid.values,
                collection_name=f"cut_{sampletype}_{name}",
            )
            logger.info(f"Ingested {name} cut ztfids")
            ztfids_surviving.update({name: sample_cut.ztfid.values})

    ztfid_list = []
    for name, ztfids in ztfids_surviving.items():
        ztfid_list.append(ztfids)

    surviving = list(set(ztfid_list[0]).intersection(*[set(i) for i in ztfid_list[1:]]))

    params = [
        "distnr",
        class_key,
        "peak_mags",
        "ampel_z",
        "classif",
        "tde_fit_exp",
        "crossmatch",
    ]

    if xgclass:
        params.append("xgclass")
    sample = NuclearSample(sampletype=sampletype).meta.get_dataframe(
        params=params,
        ztfids=surviving,
    )

    return sample


def plot_tde_scatter(
    flaring_only: bool = False,
    ingest: bool = False,
    x_values: str = "rise",
    y_values: str = "decay",
    cuts: list | None = None,
    sampletype: str = "nuclear",
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    purity_sel: str | None = None,
    plot_ext: str = "pdf",
    rerun: bool = False,
):
    """
    Plot the rise vs. fadetime of the TDE fit results
    """
    info_db = SampleInfo(sampletype=sampletype)

    sample = get_tde_selection(
        cuts=cuts, sampletype=sampletype, purity_sel=purity_sel, rerun=rerun
    )

    fig, ax = plt.subplots(figsize=(7, 7 / GOLDEN_RATIO), dpi=300)

    title = f"Nuclear sample: {len(sample)}\n"
    title += f"cut stage: {config['cutlabels'][cuts[-1]]} "
    fig.suptitle(title, fontsize=14)

    if purity_sel is not None and stats is not None:
        title += f"\nPurity: {stats['frac_pur']:.1f}% / Efficiency: {stats['frac_eff']:.1f} %"

    fig.suptitle(
        title,
        fontsize=14,
    )

    for key in sample.classif.unique():
        _df = sample.query("classif == @key")
        ax.scatter(
            _df[x_values],
            _df[y_values],
            marker=config["pl_props"][key]["m"],
            s=config["pl_props"][key]["s"],
            c=config["colordict_highlight"][key],
            # alpha=config["pl_props"][key]["a"],
            label=config["pl_props"][key]["l"] + f" ({len(_df[x_values])})",
            zorder=config["pl_props"][key]["order"],
        )

    plt.legend(title="Fritz classification", loc="lower right")

    ax.set_xlabel(config["axislabels"][x_values])
    ax.set_ylabel(config["axislabels"][y_values])

    if sampletype == "nuclear":
        local = io.LOCALSOURCE_plots
    elif sampletype == "bts":
        local = io.LOCALSOURCE_bts_plots

    if flaring_only:
        outfile = os.path.join(
            local, f"tde_{x_values}_{y_values}_flaring_{cuts}.{plot_ext}"
        )

    else:
        outfile = os.path.join(local, f"tde_{x_values}_{y_values}_{cuts}.{plot_ext}")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    plt.savefig(outfile)

    logger.debug(f"Saved to {outfile}")

    plt.close()


def plot_tde_scatter_seaborn(
    ingest: bool = False,
    x_values: str = "rise",
    y_values: str = "decay",
    cuts: list | None = ["nocut"],
    sampletype: str = "nuclear",
    purity_sel: str | None = "tde",
    rerun=False,
):
    """
    Plot the rise vs. fadetime of the TDE fit results
    """
    info_db = SampleInfo(sampletype=sampletype)

    sample, stats = get_tde_selection(cuts=cuts, sampletype=sampletype, rerun=rerun)

    sample_reduced = sample.query("classif != 'tde' or ztfid == 'ZTF22aagyuao'")
    # we need one TDE to survive, so we have a handle for the legend (don't ask,
    # I DID try to make this not hacky, but failed)

    g = sns.jointplot(
        data=sample_reduced,
        x=x_values,
        y=y_values,
        hue="classif",
        hue_order=["unclass", "sn_other", "snia", "tde"],
        alpha=0.8,
        legend=True,
        kind="kde",
        fill=True,
        common_norm=True,
    )
    g.ax_joint.scatter(
        x=sample.query("classif == 'tde'")[x_values],
        y=sample.query("classif == 'tde'")[y_values],
        c=config["colordict"]["tde"],
        s=config["pl_props"]["tde"]["s"],
        marker=config["pl_props"]["tde"]["m"],
    )

    leg = g.ax_joint.get_legend()

    title = f"Nuclear sample: {len(sample)}\n"
    title += f"cut stage: {config['cutlabels'][cuts[-1]]} "
    fig.suptitle(title, fontsize=14)

    if purity_sel is not None and stats is not None:
        title += f"\nPurity: {stats['frac_pur']:.1f}% / Efficiency: {stats['frac_eff']:.1f} %"

    g.fig.suptitle(
        title,
        fontsize=14,
    )

    leg.set_title("Fritz classification")

    new_labels = [
        config["pl_props"]["unclass"]["l"]
        + " ({length})".format(
            length=len(sample_reduced.query("classif == 'unclass'"))
        ),
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
    elif sampletype == "bts":
        local = io.LOCALSOURCE_bts_plots

    outfile = os.path.join(local, f"tde_{x_values}_{y_values}_seaborn_{cuts}.pdf")

    g.fig.savefig(outfile)
    plt.close()

    logger.info(f"Saved to {outfile}")


def plot_mag_hist(cuts: list | None = None, logplot=True, plot_ext="pdf", rerun=False):
    """
    Plot the mag histogram
    """

    classifs = ["tde", "other", "agn_star", "sn_other", "snia", "unclass"]
    colordict = {
        "unclass": "#5799c7",
        "other": "#ff9f4a",
        "snia": "#61b861",
        "tde": "#e15d5e",
        "sn_other": "#af8dce",
        "agn_star": "#a98078",
    }
    legendpos = {"nuclear": "upper left", "bts": "upper right"}

    sample_nuc = get_tde_selection(cuts=cuts, sampletype="nuclear", rerun=rerun)
    sample_bts = get_tde_selection(cuts=cuts, sampletype="bts", rerun=rerun)

    sample_nuc["sample"] = "nuclear"
    sample_bts["sample"] = "bts"

    combined = pd.concat([sample_nuc, sample_bts])

    combined["peak_mag"] = [
        k if not np.isnan(k) else combined.peak_mags_r.values[i]
        for i, k in enumerate(combined.peak_mags_g.values)
    ]
    combined.query("not peak_mag.isnull()", inplace=True)

    if logplot:
        figsize = (9, 4.5)
    else:
        figsize = (9, 9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    sample_ax = {ax1: "nuclear", ax2: "bts"}

    for ax in [ax1, ax2]:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        peak_mags = []

        colors_used = []
        classifs_used = []
        sample_title = sample_ax[ax]

        for c in classifs:
            df = combined.query(f"classif == @c and sample == @sample_title")
            if len(df) > 0:
                peak_mags.append(df["peak_mag"].values)
                colors_used.append(config["colordict"][c])
                classname_long = config["classlabels"][c]
                classifs_used.append(classname_long + f" ({len(df)})")

        ax.hist(
            peak_mags,
            bins=12,
            edgecolor="black",
            density=False,
            histtype="bar",
            stacked=True,
            range=(16, 21),
            color=colors_used,
            label=classifs_used,
        )
        if logplot:
            ax.set_yscale("log")

        if sample_title == "nuclear":
            ax.set_ylabel("Count", fontsize=11)
        ax.set_xlabel("Peak mag (AB)", fontsize=11)
        if logplot:
            ax.set_ylim((0.9, 1500))

        ax.legend(fontsize=11, loc=legendpos[sample_title])

    len_nuc = len(combined.query("sample == 'nuclear'"))
    len_bts = len(combined.query("sample == 'bts'"))
    title = f"Nuclear: {len_nuc} / BTS: {len_bts}\n"

    title += f"cut stage: {config['cutlabels'][cuts[-1]]} "
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if logplot:
        outfile = os.path.join(
            io.LOCALSOURCE_plots, "maghist", f"maghist_{cuts}_log.{plot_ext}"
        )
    else:
        outfile = os.path.join(
            io.LOCALSOURCE_plots, "maghist", f"maghist_{cuts}_lin.{plot_ext}"
        )
    plt.savefig(outfile)
    plt.close()


def plot_mag_hist_2x2(
    cuts: list | None = None,
    logplot=True,
    plot_ext="pdf",
    compare: str = "nuc_bts",
    rerun=False,
):
    """
    Plot the mag histogram
    """

    classifs = ["tde", "other", "agn", "star", "sn_other", "snia", "unclass"]

    colordict = {
        "unclass": "#5799c7",
        "other": "#ff9f4a",
        "snia": "#61b861",
        "tde": "#e15d5e",
        "sn_other": "#af8dce",
        "agn_star": "#a98078",
    }
    legendpos = {
        "nuclear_noagn": "upper left",
        "nuclear_keepagn": "upper left",
        "bts_noagn": "upper right",
        "bts_keepagn": "upper right",
        "nuclear_noagn_xg": "upper left",
        "nuclear_keepagn_xg": "upper left",
    }
    titles = {
        "nuclear_noagn": "Nuclear - veto AGN",
        "nuclear_keepagn": "Nuclear - only AGN",
        "bts_noagn": "BTS - veto AGN",
        "bts_keepagn": "BTS - only AGN",
        "nuclear_noagn_xg": "Nuclear XGBoost - veto AGN",
        "nuclear_keepagn_xg": "Nuclear XGBoost - only AGN",
    }

    cuts_noagn = copy.deepcopy(cuts)
    cuts_keepagn = copy.deepcopy(cuts)

    cuts_noagn.append("milliquas_noagn")
    cuts_keepagn.append("milliquas_keepagn")

    sample_nuc_noagn = get_tde_selection(
        cuts=cuts_noagn, sampletype="nuclear", rerun=rerun, xgclass=False
    )
    sample_nuc_keepagn = get_tde_selection(
        cuts=cuts_keepagn, sampletype="nuclear", rerun=rerun, xgclass=False
    )
    sample_bts_noagn = get_tde_selection(cuts=cuts_noagn, sampletype="bts", rerun=rerun)

    sample_bts_keepagn = get_tde_selection(
        cuts=cuts_keepagn, sampletype="bts", rerun=rerun
    )
    sample_nuc_noagn_xg = get_tde_selection(
        cuts=cuts_noagn, sampletype="nuclear", rerun=rerun, xgclass=True
    )
    sample_nuc_keepagn_xg = get_tde_selection(
        cuts=cuts_keepagn, sampletype="nuclear", rerun=rerun, xgclass=True
    )

    sample_nuc_noagn["sample"] = "nuclear_noagn"
    sample_nuc_keepagn["sample"] = "nuclear_keepagn"
    sample_bts_noagn["sample"] = "bts_noagn"
    sample_bts_keepagn["sample"] = "bts_keepagn"
    sample_nuc_noagn_xg["sample"] = "nuclear_noagn_xg"
    sample_nuc_keepagn_xg["sample"] = "nuclear_keepagn_xg"

    if compare == "nuc_bts":
        combined = pd.concat(
            [sample_nuc_noagn, sample_bts_noagn, sample_nuc_keepagn, sample_bts_keepagn]
        )
    elif compare == "nuc_xg":
        combined = pd.concat(
            [
                sample_nuc_noagn,
                sample_nuc_noagn_xg,
                sample_nuc_keepagn,
                sample_nuc_keepagn_xg,
            ]
        )

    combined["peak_mag"] = [
        k if not np.isnan(k) else combined.peak_mags_r.values[i]
        for i, k in enumerate(combined.peak_mags_g.values)
    ]
    combined.query("not peak_mag.isnull()", inplace=True)

    figsize = (9, 7)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)

    if compare == "nuc_bts":
        sample_ax = {
            axes[0, 0]: "nuclear_noagn",
            axes[0, 1]: "bts_noagn",
            axes[1, 0]: "nuclear_keepagn",
            axes[1, 1]: "bts_keepagn",
        }
    elif compare == "nuc_xg":
        sample_ax = {
            axes[0, 0]: "nuclear_noagn",
            axes[0, 1]: "nuclear_noagn_xg",
            axes[1, 0]: "nuclear_keepagn",
            axes[1, 1]: "nuclear_keepagn_xg",
        }

    for ax in axes.flat:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        peak_mags = []

        colors_used = []
        classifs_used = []
        sample_title = sample_ax[ax]

        if "xg" in sample_title:
            classif_key = "xgclass"
        else:
            classif_key = "classif"

        for c in classifs:
            df = combined.query(f"{classif_key} == @c and sample == @sample_title")
            if len(df) > 0:
                peak_mags.append(df["peak_mag"].values)
                colors_used.append(config["colordict"][c])
                classname_long = config["classlabels"][c]
                classifs_used.append(classname_long + f" ({len(df)})")

        if len(colors_used) == 0:
            colors_used = None

        ax.hist(
            peak_mags,
            bins=12,
            edgecolor="black",
            density=False,
            histtype="bar",
            stacked=True,
            range=(16, 21),
            color=colors_used,
            label=classifs_used,
        )
        ax.set_yscale("log")

        if sample_title in ["nuclear_noagn", "nuclear_keepagn"]:
            ax.set_ylabel("Count", fontsize=11)
        ax.set_xlabel("Peak mag (AB)", fontsize=11)
        ax.set_ylim((0.9, 1500))

        ax.legend(fontsize=11, loc=legendpos[sample_title])
        ax.set_title(
            f"{titles[sample_title]} ({len(combined.query('sample == @sample_title'))})",
            fontsize=13,
        )

    # len_nuc = len(combined.query("sample == 'nuclear'"))
    # len_bts = len(combined.query("sample == 'bts'"))

    title = f"cut stage: {config['cutlabels'][cuts[-1]]} "
    fig.suptitle(title, fontsize=15)

    plt.tight_layout()

    outfile = os.path.join(
        io.LOCALSOURCE_plots, "maghist", f"maghist_{cuts}_comb.{plot_ext}"
    )

    if plot_ext == "png":
        dpi = 400
    else:
        dpi = None
    plt.savefig(outfile, dpi=dpi)
    plt.close()


def plot_dist_hist(
    classif="all", plotrange: List[float] | None = [0, 5], plot_ext: str = "pdf"
):
    """
    Plot the core-distance distribution for BTS and nuclear sample
    """
    from ztfnuclear.sample import NuclearSample

    sample_nuc = get_tde_selection(cuts=["nocut"], sampletype="nuclear")
    sample_bts = get_tde_selection(cuts=["nocut"], sampletype="bts")

    if classif != "all":
        sample_nuc.query(config["classes"][classif], inplace=True)
        sample_bts.query(config["classes"][classif], inplace=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.5 / 1.62))
    fig.suptitle(f"Classification: {config['classlabels'][classif]}")
    ax.hist(
        sample_nuc["distnr_distnr"] ** 2,
        range=plotrange,
        bins=100,
        label=f"nuclear ({len(sample_nuc)})",
        cumulative=False,
        histtype="step",
    )
    ax.hist(
        sample_bts["distnr_distnr"] ** 2,
        range=plotrange,
        bins=100,
        label=f"BTS ({len(sample_bts)})",
        cumulative=False,
        histtype="step",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Core distance (arcsec$^2$)")
    plt.tight_layout()
    plt.legend()
    outdir = os.path.join(io.LOCALSOURCE_plots, "dist")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, f"dist_hist_{classif}.{plot_ext}")
    plt.savefig(outfile, dpi=300)
    logger.info(f"Saved to {outfile}")


def plot_sgscore_hist(
    classif="all", plotrange: List[float] | None = [0, 1], plot_ext="pdf"
):
    """
    Plot the core-distance distribution for BTS and nuclear sample
    """
    from ztfnuclear.sample import NuclearSample

    sample_nuc = get_tde_selection(cuts=["nocut"], sampletype="nuclear")
    sample_bts = get_tde_selection(cuts=["nocut"], sampletype="bts")

    if classif != "all":
        sample_nuc.query(config["classes"][classif], inplace=True)
        sample_bts.query(config["classes"][classif], inplace=True)

    fig, ax = plt.subplots(figsize=(5.5, 5.5 / 1.62))
    fig.suptitle(f"Classification: {classif}")
    ax.hist(
        sample_nuc["crossmatch_sgscore_sgscore"],
        range=plotrange,
        bins=100,
        label=f"nuclear ({len(sample_nuc)})",
        cumulative=False,
        histtype="step",
    )
    ax.hist(
        sample_bts["crossmatch_sgscore_sgscore"] ** 2,
        range=plotrange,
        bins=100,
        label=f"BTS ({len(sample_bts)})",
        cumulative=False,
        histtype="step",
    )

    ax.set_yscale("log")
    ax.set_xlabel("sgscore (0 = galaxy, 1 = star)")
    plt.tight_layout()
    plt.legend()
    outdir = os.path.join(io.LOCALSOURCE_plots, "sgscore")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, f"sgscore_hist_{classif}.{plot_ext}")
    plt.savefig(outfile, dpi=300)
    logger.info(f"Saved to {outfile}")


def plot_mag_cdf(cuts: str | None = "agn"):
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
                        and classification not in fritz_tde
                    ):
                        mags.append(mag)
                    elif classification in fritz_tde:
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


def plot_bts_classes(plot_ext: str = "pdf", classified_only: bool = False):
    """
    Plot the BTS classifications as pie chart
    """
    sample = get_tde_selection(
        cuts=["nocut"],
        sampletype="bts",
        rerun=True,
        reject_bogus=False,
        require_fitsuccess=False,
    )

    # sample.sort_values(by="classif", inplace=True)

    sample.query("not classif.isnull()", inplace=True)

    if classified_only:
        sample.query("classif != 'unclass'", inplace=True)

    total = len(sample)

    fig, ax = plt.subplots(figsize=(5.5, 5.5 / 1.62))
    fig.suptitle(f"BTS Classifications")

    entries = []
    labels = []
    colors = []
    for cl in sample.classif.unique():
        len_class = len(sample.query("classif == @cl"))
        entries.append(len_class)
        colors.append(config["colordict_highlight"][cl])
        clabel = config["classlabels"][cl]
        clabel += f"\n{len_class/total*100:.1f}\%"
        labels.append(clabel)
    ax.pie(entries, labels=labels, colors=colors)  # , autopct="%1.2f%%")
    plt.tight_layout()
    outfile = os.path.join(io.LOCALSOURCE_bts_plots, f"BTS_full.{plot_ext}")
    plt.savefig(outfile)
    sample = get_tde_selection(
        cuts=["nocut"],
        sampletype="bts",
        rerun=True,
        reject_bogus=True,
        require_fitsuccess=True,
    )


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


def plot_tde(sampletype: str = "nuclear", outfile: str | None = None):
    """Plot the tde fit results from the Mongo DB"""

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

    if outfile is None:
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
    primary_grid: bool = False,
    wise_df: pd.DataFrame = None,
    wise_bayesian: dict = None,
    snt_threshold=3,
    plot_png: bool = False,
    wide: bool = False,
    thumbnail: bool = False,
    sampletype: str = "nuclear",
    no_magrange: bool = False,
    classif: str | None = None,
    xlim: list[float] | None = None,
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
    elif sampletype == "bts":
        local = io.LOCALSOURCE_bts_plots
    elif sampletype == "train":
        local = io.LOCALSOURCE_train_plots

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

    if "fid" in list(df.keys()):
        fid_key = "fid"
    else:
        fid_key = "filterid"

    for filterid in sorted(df[fid_key].unique()):
        _df = df.query(f"{fid_key} == @filterid")

        bandname = utils.ztf_filterid_to_band(filterid, short=True)

        if bl_correction:
            ampl_column = "ampl_corr"
            ampl_err_column = "ampl_err_corr"
            if not thumbnail:
                if tns_name:
                    fig.suptitle(
                        f"{ztfid} ({tns_name}) - baseline corrected\nclassif = {classif}",
                        fontsize=14,
                    )
                else:
                    fig.suptitle(
                        f"{ztfid} - baseline corrected\nclassif = {classif}",
                        fontsize=14,
                    )
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
            flux = _df[ampl_column] / F0 * 3630.78
            flux_err = (
                np.sqrt(
                    (_df[ampl_err_column] / F0) ** 2
                    + (_df[ampl_column] * F0_err / F0**2) ** 2
                )
                * 3630.78
            )

            abmag = -2.5 * np.log10(flux / 3630.78)
            abmag_err = 2.5 / np.log(10) * flux_err / flux

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
            ax.set_ylabel("Magnitude (AB)")

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
        if not no_magrange:
            ax.set_ylim([23, 15])
        else:
            ax.invert_yaxis()

    if wise_df is not None:
        if len(wise_df) > 0:
            flux_W1 = wise_df["W1_mean_flux_density_bl_corr"] / 1000
            flux_W1_err = wise_df["W1_mean_flux_density_bl_corr_err"] / 1000
            flux_W2 = wise_df["W2_mean_flux_density_bl_corr"] / 1000
            flux_W2_err = wise_df["W2_mean_flux_density_bl_corr_err"] / 1000

            nufnu_W1 = utils.band_frequency("W1") * flux_W1 * 1e-23
            nufnu_W1_err = utils.band_frequency("W1") * flux_W1_err * 1e-23
            nufnu_W2 = utils.band_frequency("W2") * flux_W2 * 1e-23
            nufnu_W2_err = utils.band_frequency("W2") * flux_W2_err * 1e-23

            abmag_W1 = -2.5 * np.log10(flux_W1 / 3630.78)
            abmag_W1_err = np.abs(2.5 / np.log(10) * flux_W1_err / flux_W1)
            abmag_W2 = -2.5 * np.log10(flux_W2 / 3630.78)
            abmag_W2_err = np.abs(2.5 / np.log(10) * flux_W2_err / flux_W2)

            if magplot:
                ax.errorbar(
                    wise_df.mean_mjd,
                    abmag_W1,
                    abmag_W1_err,
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
                    abmag_W2,
                    abmag_W2_err,
                    fmt="o",
                    mec="gray",
                    mfc="gray",
                    ecolor="gray",
                    alpha=1,
                    ms=5,
                    elinewidth=1,
                )

            else:
                if thumbnail:
                    ms = 3
                    elinewidth = 0.08
                else:
                    ms = 5
                    elinewidth = 0.4

                ax.errorbar(
                    wise_df.mean_mjd,
                    nufnu_W1,
                    nufnu_W1_err,
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
                    nufnu_W2,
                    nufnu_W2_err,
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

    if not xlim:
        xlim1, xlim2 = ax.get_xlim()
        axlims = {"xlim": [xlim1, xlim2]}
    else:
        ax.set_xlim(xlim)
        axlims = {"xlim": xlim}

    plt.tight_layout()

    plt.savefig(outfile)
    plt.close()

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
    import sncosmo
    from sfdmap import SFDMap  # type: ignore[import]

    from ztfnuclear.tde_fit import TDESource_exp_flextemp, TDESource_pl_flextemp

    logger.debug("Plotting TDE fit lightcurve (in flux space)")

    color_dict = {1: "green", 2: "red", 3: "orange"}
    filtername_dict = {1: "ZTF g", 2: "ZTF r", 3: "ZTF i"}

    if sampletype == "nuclear":
        plot_dir = Path(io.LOCALSOURCE_plots) / "lightcurves" / "tde_fit"
    elif sampletype == "bts":
        plot_dir = Path(io.LOCALSOURCE_bts_plots) / "lightcurves" / "tde_fit"
    elif sampletype == "train":
        plot_dir = Path(io.LOCALSOURCE_train_plots) / "lightcurves" / "tde_fit"

    if not plot_dir.is_dir():
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

    if "fid" in list(df.keys()):
        fid_key = "fid"
    else:
        fid_key = "filterid"

    for filterid in sorted(df[fid_key].unique()):
        _df = df.query(f"{fid_key} == @filterid")

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
