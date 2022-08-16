#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re, subprocess
from typing import Optional, List, Dict

import pandas as pd  # type: ignore

if os.getenv("ZTFDATA"):

    _SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
    LOCALSOURCE = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    LOCALSOURCE_dfs = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "data")
    LOCALSOURCE_metadata = os.path.join(LOCALSOURCE, "FINAL_SAMPLE", "metadata.csv")
    LOCALSOURCE_plots = os.path.join(LOCALSOURCE, "plots")
    LOCALSOURCE_baseline = os.path.join(LOCALSOURCE, "baseline")

    DOWNLOAD_URL = "https://syncandshare.desy.de/index.php/s/GHeGQYxgk5FeToY/download"

else:
    raise ValueError(
        "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See github.com/mickaelrigault/ztfquery"
    )

logger = logging.getLogger(__name__)


for p in [LOCALSOURCE, LOCALSOURCE_plots, LOCALSOURCE_baseline]:
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
        download()


def download():
    """
    Downloads the dataframe from DESY Syncandshare
    """
    cmd = f"wget {DOWNLOAD_URL} -P {LOCALSOURCE}; mv {LOCALSOURCE}/download {LOCALSOURCE}/download.tar; tar -xvf {LOCALSOURCE}/download.tar -C {LOCALSOURCE}; rm {LOCALSOURCE}/download.tar"

    subprocess.run(cmd, shell=True)
    logger.info("Download complete")


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


def get_metadata() -> pd.DataFrame:
    """
    Gets the metadata dataframe for the full sample
    """
    df = pd.read_csv(LOCALSOURCE_metadata, index_col=0)
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
