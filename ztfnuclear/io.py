#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re, subprocess, json
from typing import Optional, List, Dict

from ztfnuclear import utils

from ztfquery.io import LOCALSOURCE  # type: ignore
from ztfquery.lightcurve import LCQuery  # type: ignore

import pandas as pd  # type: ignore
import numpy as np

if os.getenv("ZTFDATA"):

    _SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
    LOCALSOURCE = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    LOCALSOURCE_pickle = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "FINAL_SAMPLE", "overview.pkl"
    )
    LOCALSOURCE_pickle_flaring = os.path.join(
        str(os.getenv("ZTFDATA")),
        "nuclear_sample",
        "FINAL_SAMPLE",
        "overview_flaring.pkl",
    )
    LOCALSOURCE_dfs = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "data")
    LOCALSOURCE_bts_dfs = os.path.join(LOCALSOURCE, "BTS", "data")
    LOCALSOURCE_irsa = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "irsa")
    LOCALSOURCE_bts_irsa = os.path.join(LOCALSOURCE, "BTS", "irsa")
    LOCALSOURCE_fitres = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "fitres")
    LOCALSOURCE_ampelz = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "ampel_z.json")
    LOCALSOURCE_location = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "location.csv")
    LOCALSOURCE_bts_info = os.path.join(LOCALSOURCE, "BTS", "bts.csv")
    LOCALSOURCE_peak_dates = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "peak_dates.csv")
    LOCALSOURCE_ZTF_tdes = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "ztf_tdes.csv")
    LOCALSOURCE_WISE = os.path.join(LOCALSOURCE, "WISE")
    LOCALSOURCE_WISE_lc_by_pos = os.path.join(
        LOCALSOURCE, "FINAL_SAMPLE", "wise_lightcurves_by_pos.json"
    )
    LOCALSOURCE_WISE_lc_by_id = os.path.join(
        LOCALSOURCE, "FINAL_SAMPLE", "wise_lightcurves_by_id.json"
    )
    LOCALSOURCE_WISE_bayesian = os.path.join(
        LOCALSOURCE, "FINAL_SAMPLE", "wise_bayesian_blocks.json"
    )
    LOCALSOURCE_plots = os.path.join(LOCALSOURCE, "plots")
    LOCALSOURCE_bts_plots = os.path.join(LOCALSOURCE, "plots_bts")
    LOCALSOURCE_plots_irsa = os.path.join(LOCALSOURCE, "plots", "lightcurves_irsa")
    LOCALSOURCE_baseline = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "baseline")
    LOCALSOURCE_bts_baseline = os.path.join(LOCALSOURCE, "BTS", "baseline")

    DOWNLOAD_URL_SAMPLE = (
        "https://syncandshare.desy.de/index.php/s/GHeGQYxgk5FeToY/download"
    )
    DOWNLOAD_URL_WISE = (
        "https://syncandshare.desy.de/index.php/s/iweHsggyCaecSKE/download"
    )

else:
    raise ValueError(
        "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See github.com/mickaelrigault/ztfquery"
    )

logger = logging.getLogger(__name__)


for p in [
    LOCALSOURCE,
    LOCALSOURCE_plots,
    LOCALSOURCE_bts_plots,
    LOCALSOURCE_baseline,
    LOCALSOURCE_bts_baseline,
    LOCALSOURCE_WISE,
    LOCALSOURCE_plots_irsa,
]:
    if not os.path.exists(p):
        os.makedirs(p)


def download_if_neccessary(sampletype="nuclear"):
    """
    Check if the dataframes have been downloaded and do so if neccessary
    """
    if sampletype == "nuclear":
        local = LOCALSOURCE_dfs
    elif sampletype == "bts":
        local = LOCALSOURCE_bts_dfs

    if os.path.exists(local):
        csvs = []
        for name in os.listdir(local):
            if name[-4:] == ".csv":
                csvs.append(name)
        number_of_files = len(csvs)
        logger.info(f"{number_of_files} dataframes found.")

    else:
        logger.info("Dataframe directory is not present, proceed to download files.")
        download_sample()

    if sampletype == "nuclear":
        if not os.path.isfile(os.path.join(LOCALSOURCE_WISE, "WISE.parquet")):
            logger.info("WISE dataframe does not exist, proceed to download.")
            download_wise()


def download_sample():
    """
    Downloads the sample from DESY Syncandshare
    """
    cmd = f"curl --create-dirs -J -O --output-dir {LOCALSOURCE} {DOWNLOAD_URL_SAMPLE}; unzip {LOCALSOURCE}/FINAL_SAMPLE.zip -d {LOCALSOURCE}; rm {LOCALSOURCE}/FINAL_SAMPLE.zip"

    subprocess.run(cmd, shell=True)
    logger.info("Sample download complete")


def download_wise():
    """
    Downloads the WISE location parquet file from DESY Syncandshare
    """
    cmd = (
        f"curl --create-dirs -J -O --output-dir {LOCALSOURCE_WISE} {DOWNLOAD_URL_WISE}"
    )

    subprocess.run(cmd, shell=True)
    logger.info("WISE location file download complete")


def get_all_ztfids(sampletype="nuclear") -> List[str]:
    """
    Checks the download folder and gets all ztfids
    """
    if sampletype == "nuclear":
        local = LOCALSOURCE_dfs
    elif sampletype == "bts":
        local = LOCALSOURCE_bts_dfs

    ztfids = []
    for name in os.listdir(local):
        if name[-4:] == ".csv":
            ztfids.append(name[:-4])
    return ztfids


def is_valid_ztfid(ztfid: str) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    is_match = re.match("^ZTF[1-2]\d[a-z]{7}$", ztfid)
    if is_match:
        return True
    else:
        return False


def is_valid_wiseid(wiseid: str) -> bool:
    """
    Checks if a string adheres to the (internal) WISE naming scheme
    """
    is_match = re.match(r"^WISE\d[0-9]{0,}$", wiseid)

    if is_match:
        return True
    else:
        return False


def get_locations(sampletype="nuclear") -> pd.DataFrame:
    """
    Gets the metadata dataframe for the full sample
    """
    if sampletype == "nuclear":
        local = LOCALSOURCE_location
    elif sampletype == "bts":
        local = LOCALSOURCE_bts_info

    df = pd.read_csv(local, index_col=0)

    if sampletype == "bts":
        df = df[["RA", "Dec"]]
    return df


def get_thumbnail_count() -> int:
    """
    Checks how many thumbnails there are on the disk
    """
    thumbnail_dir = os.path.join(LOCALSOURCE_plots, "lightcurves", "thumbnails")
    _, _, files = next(os.walk(thumbnail_dir))
    file_count = len(files)

    return file_count


def get_ztfid_dataframe(
    ztfid: str, sampletype: str = "nuclear"
) -> Optional[pd.DataFrame]:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        if sampletype == "nuclear":
            filepath = os.path.join(LOCALSOURCE_dfs, f"{ztfid}.csv")
        else:
            filepath = os.path.join(LOCALSOURCE_bts_dfs, f"{ztfid}.csv")
        try:
            df = pd.read_csv(filepath, comment="#")
            return df
        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_ztfid_header(ztfid: str, sampletype="nuclear") -> Optional[dict]:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid):
        if sampletype == "nuclear":
            filepath = os.path.join(LOCALSOURCE_dfs, f"{ztfid}.csv")
        else:
            filepath = os.path.join(LOCALSOURCE_bts_dfs, f"{ztfid}.csv")

        try:
            with open(filepath, "r") as input_file:
                headerkeys = []
                headervals = []

                for i, line in enumerate(input_file):
                    if len(line) >= 300:
                        break
                    key = line.split(",", 2)[0].split("=")[0].lstrip("#")
                    headerkeys.append(key)
                    val = line.split(",", 2)[0].split("=")[1][:-1]
                    headervals.append(val)

                returndict = {}
                for i, key in enumerate(headerkeys):
                    returndict.update({key: headervals[i]})

                returndict["ztfid"] = returndict.pop("name")

                return returndict

        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def parse_json(filepath: str) -> dict:
    """
    Read a json file and extract the dictionary
    """
    if not os.path.isfile(filepath):
        raise ValueError("No file at given filepath")

    with open(filepath) as json_file:
        data = json.load(json_file)

    return data


def airflares_stock_to_ztfid():
    """ """
    infile = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "airflares_raw.json")
    with open(infile, "r") as f:
        wise_lcs = json.load(f)
    stock_to_ztfid = {}

    for key in wise_lcs.keys():
        stock_to_ztfid[key] = wise_lcs[key]["id"]

    return stock_to_ztfid


def ztfid_to_airflares_stock():
    """ """
    stock_to_ztfid = airflares_stock_to_ztfid()

    ztfid_to_stock = {v: k for k, v in stock_to_ztfid.items()}

    return ztfid_to_stock


def parse_ampel_json(filepath: str, parameter_name: str) -> dict:
    """Reads the mongodb export from Ampel"""
    if not os.path.isfile(filepath):
        raise ValueError("No file at given filepath")

    resultdict = {}
    if parameter_name == "wise_bayesian":
        airflares_to_ztf = airflares_stock_to_ztfid()

    with open(filepath) as json_file:
        data = json.load(json_file)
        for entry in data:
            stockid = entry["stock"]

            if parameter_name != "wise_bayesian":
                ztfid = utils.stockid_to_ztfid(stockid)
            else:
                ztfid = airflares_to_ztf[str(stockid)]

            if "body" in entry.keys():
                body = entry["body"][0]

                if parameter_name in [
                    "salt",
                    "salt_loose_bl",
                    "tde_fit",
                    "tde_fit_loose_bl",
                ]:
                    if "sncosmo_result" in body.keys():
                        sncosmo_result = body["sncosmo_result"]
                        chisq = sncosmo_result["chisq"]
                        ndof = sncosmo_result["ndof"]
                        paramdict = sncosmo_result["paramdict"]
                        resultdict.update(
                            {
                                ztfid: {
                                    parameter_name: {
                                        "chisq": chisq,
                                        "ndof": ndof,
                                        "paramdict": paramdict,
                                    }
                                }
                            }
                        )
                    else:
                        resultdict.update({ztfid: {parameter_name: "failure"}})

                elif parameter_name == "ampel_z":
                    if "ampel_z" in body.keys():
                        z = body["ampel_z"]
                        z_dist = body["ampel_dist"]
                        z_group = body["group_z_nbr"]
                        z_precision = body["group_z_precision"]
                        resultdict.update(
                            {
                                ztfid: {
                                    "ampel_z": {
                                        "z": z,
                                        "z_dist": z_dist,
                                        "z_group": z_group,
                                        "z_precision": z_precision,
                                    }
                                }
                            }
                        )
                    else:
                        resultdict.update({ztfid: {"ampel_z": {}}})

                elif parameter_name == "wise_bayesian":
                    unit = entry["unit"]

                    if unit == "T2BayesianBlocks":
                        resultdict.update(
                            {ztfid: {"WISE_bayesian": {"bayesian": body}}}
                        )

                    elif unit == "T2DustEchoEval":
                        if "result" in body.keys():
                            resultdict[ztfid]["WISE_bayesian"].update(
                                {"dustecho": body["result"]}
                            )
                        else:
                            resultdict[ztfid]["WISE_bayesian"].update(
                                {"dustecho": None}
                            )
                    else:
                        raise ValueError(
                            "There is something wrong with your mongo export file."
                        )

                elif parameter_name == "ztf_bayesian":
                    unit = entry["unit"]

                    if unit == "T2BayesianBlocks":
                        resultdict.update({ztfid: {"ZTF_bayesian": {"bayesian": body}}})
                    elif unit == "T2DustEchoEval":
                        resultdict[ztfid]["ZTF_bayesian"].update({"dustecho": body})

                else:
                    raise ValueError("Parameter_name is not know")

    logger.info(f"Imported {len(resultdict)} entries from {filepath}")

    return resultdict


def load_irsa(ra: float, dec: float, radius_arcsec: float = 0.5) -> pd.DataFrame:
    """
    Get lightcuve from IPAC
    """

    logger.debug("Querying IPAC")
    df = LCQuery.from_position(ra, dec, radius_arcsec).data

    logger.debug(f"Found {len(df)} datapoints")

    if len(df) == 0:
        logger.info("No data found.")
        return df

    else:
        mask = df.catflags > 0

        flags = list(set(df.catflags))

        logger.info(
            f"Found {len(df)} datapoints, masking {np.sum(mask)} datapoints with bad flags."
        )

        for flag in sorted(flags):
            logger.debug(f"{np.sum(df.catflags == flag)} datapoints with flag {flag}")

        df = df.drop(df[mask].index)
        return df
