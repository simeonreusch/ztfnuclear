#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re, subprocess, json
from typing import Optional, List, Dict

from ztfnuclear import utils

import pandas as pd  # type: ignore

if os.getenv("ZTFDATA"):

    _SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
    LOCALSOURCE = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    LOCALSOURCE_dfs = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "data")
    LOCALSOURCE_fitres = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "fitres")
    LOCALSOURCE_ampelz = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "ampel_z.json")
    LOCALSOURCE_location = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "location.csv")
    LOCALSOURCE_peak_dates = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "peak_dates.csv")
    LOCALSOURCE_ZTF_tdes = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "ztf_tdes.csv")
    LOCALSOURCE_WISE = os.path.join(LOCALSOURCE, "WISE")
    LOCALSOURCE_WISE_lc_by_pos = os.path.join(
        LOCALSOURCE, "FINAL_SAMPLE", "wise_lightcurves_by_pos.json"
    )
    LOCALSOURCE_WISE_lc_by_id = os.path.join(
        LOCALSOURCE, "FINAL_SAMPLE", "wise_lightcurves_by_id.json"
    )
    LOCALSOURCE_plots = os.path.join(LOCALSOURCE, "plots")
    LOCALSOURCE_baseline = os.path.join(LOCALSOURCE, "baseline")

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


for p in [LOCALSOURCE, LOCALSOURCE_plots, LOCALSOURCE_baseline, LOCALSOURCE_WISE]:
    if not os.path.exists(p):
        os.makedirs(p)


def download_if_neccessary():
    """
    Check if the dataframes have been downloaded and do so if neccessary
    """
    if os.path.exists(LOCALSOURCE_dfs):
        csvs = []
        for name in os.listdir(LOCALSOURCE_dfs):
            if name[-4:] == ".csv":
                csvs.append(name)
        number_of_files = len(csvs)
        logger.info(f"{number_of_files} dataframes found.")

    else:
        logger.info("Dataframe directory is not present, proceed to download files.")
        download_sample()

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


def get_all_ztfids() -> List[str]:
    """
    Checks the download folder and gets all ztfids
    """
    ztfids = []
    for name in os.listdir(LOCALSOURCE_dfs):
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


def get_locations() -> pd.DataFrame:
    """
    Gets the metadata dataframe for the full sample
    """
    df = pd.read_csv(LOCALSOURCE_location, index_col=0)
    return df


def get_ztfid_dataframe(ztfid: str) -> Optional[pd.DataFrame]:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        filepath = os.path.join(LOCALSOURCE_dfs, f"{ztfid}.csv")
        try:
            df = pd.read_csv(filepath, comment="#")
            return df
        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_ztfid_header(ztfid: str) -> Optional[dict]:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid):
        filepath = os.path.join(LOCALSOURCE_dfs, f"{ztfid}.csv")

        try:
            with open(filepath, "r") as input_file:
                headerlines = []
                for i, line in enumerate(input_file):
                    if i == 6:
                        break
                    line = line.split(",", 2)[0].split("=")[1][:-1]
                    headerlines.append(line)
                ztfid, ra, dec, lastobs, lastdownload, lastfit = headerlines

                returndict = {
                    "ztfid": ztfid,
                    "ra": ra,
                    "dec": dec,
                    "lastobs": lastobs,
                    "lastdownload": lastdownload,
                    "lastfit": lastfit,
                }

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


def parse_ampel_json(filepath: str, parameter_name: str) -> dict:
    """Reads the mongodb export from Ampel"""
    if not os.path.isfile(filepath):
        raise ValueError("No file at given filepath")

    resultdict = {}

    with open(filepath) as json_file:
        data = json.load(json_file)
        for entry in data:
            stockid = entry["stock"]
            ztfid = utils.stockid_to_ztfid(stockid)

            if "body" in entry.keys():
                body = entry["body"][0]

                if parameter_name in ["salt", "tde_fit", "tde_fit_loose_bl"]:
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

                else:
                    raise ValueError("Parameter_name is not know")

    logger.info(f"Imported {len(resultdict)} entries from {filepath}")

    return resultdict
