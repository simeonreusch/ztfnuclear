#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, re, subprocess, json, yaml
from pathlib import Path
from typing import Optional, List, Dict

from ztfnuclear import utils

from ztfquery.io import LOCALSOURCE  # type: ignore
from ztfquery.lightcurve import LCQuery  # type: ignore

import pandas as pd  # type: ignore
import numpy as np

if os.getenv("ZTFDATA"):
    _SOURCEDIR = os.path.dirname(os.path.realpath(__file__))
    BASE = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    SRC_nuclear = os.path.join(BASE, "NUCLEAR")
    SRC_bts = os.path.join(BASE, "BTS")
    SRC_train = os.path.join(BASE, "TRAIN")

    MODEL_dir = Path(BASE) / "models"

    SRC_nuclear_pickle = os.path.join(SRC_nuclear, "overview.pkl")
    SRC_nuclear_pickle_flaring = os.path.join(
        SRC_nuclear,
        "overview_flaring.pkl",
    )

    LOCALSOURCE_dfs = os.path.join(SRC_nuclear, "data")
    LOCALSOURCE_bts_dfs = os.path.join(SRC_bts, "data")
    LOCALSOURCE_train_dfs = os.path.join(SRC_train, "data")

    LOCALSOURCE_irsa = os.path.join(SRC_nuclear, "irsa")
    LOCALSOURCE_bts_irsa = os.path.join(SRC_bts, "irsa")

    LOCALSOURCE_fitres = os.path.join(SRC_nuclear, "fitres")
    LOCALSOURCE_train_fitres = os.path.join(SRC_train, "fitres")
    LOCALSOURCE_ampelz = os.path.join(SRC_nuclear, "ampel_z.json")
    LOCALSOURCE_location = os.path.join(SRC_nuclear, "location.csv")
    LOCALSOURCE_bts_info = os.path.join(SRC_bts, "bts.csv")
    LOCALSOURCE_peak_dates = os.path.join(SRC_nuclear, "peak_dates.csv")
    LOCALSOURCE_bts_peak_dates = os.path.join(SRC_bts, "peak_dates.csv")
    LOCALSOURCE_distnr = os.path.join(SRC_nuclear, "distnr.csv")
    LOCALSOURCE_bts_distnr = os.path.join(SRC_bts, "distnr.csv")
    LOCALSOURCE_peak_mags = os.path.join(SRC_nuclear, "peak_mag.csv")
    LOCALSOURCE_bts_peak_mags = os.path.join(SRC_bts, "peak_mag.csv")
    LOCALSOURCE_ZTF_tdes = os.path.join(SRC_nuclear, "ztf_tdes.csv")
    LOCALSOURCE_WISE = os.path.join(BASE, "WISE")
    LOCALSOURCE_sarah_agn = os.path.join(BASE, "ML_reconstructed_AGN_catalogue.fits")
    LOCALSOURCE_timewise = os.path.join(
        SRC_nuclear, "timewise", "timewise_lightcurves_nuclear.json"
    )
    LOCALSOURCE_timewise_raw = os.path.join(
        SRC_nuclear, "timewise", "timewise_data_product_tap__chunk0.json"
    )
    LOCALSOURCE_timewise_bts = os.path.join(
        SRC_bts, "timewise", "timewise_lightcurves_bts.json"
    )
    LOCALSOURCE_timewise_bts_raw = os.path.join(
        SRC_bts, "timewise", "timewise_data_product_tap__chunk0.json"
    )
    LOCALSOURCE_WISE_bayesian = os.path.join(SRC_nuclear, "wise_bayesian_nuclear.json")
    LOCALSOURCE_WISE_bayesian_bts = os.path.join(SRC_bts, "wise_bayesian_bts.json")
    LOCALSOURCE_WISE_dust = os.path.join(SRC_nuclear, "wise_dust_nuclear.json")
    LOCALSOURCE_WISE_dust_bts = os.path.join(SRC_bts, "wise_dust_bts.json")
    LOCALSOURCE_plots = os.path.join(BASE, "plots")
    LOCALSOURCE_bts_plots = os.path.join(BASE, "plots_bts")
    LOCALSOURCE_train_plots = os.path.join(BASE, "plots_train")
    LOCALSOURCE_plots_irsa = os.path.join(BASE, "plots", "lightcurves_irsa")
    LOCALSOURCE_baseline = os.path.join(SRC_nuclear, "baseline")
    LOCALSOURCE_bts_baseline = os.path.join(SRC_bts, "baseline")

    DOWNLOAD_URL_SAMPLE = (
        "https://syncandshare.desy.de/index.php/s/GHeGQYxgk5FeToY/download"
    )
    DOWNLOAD_URL_BTS = (
        "https://syncandshare.desy.de/index.php/s/xnr7GPSRieNCWkA/download"
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
    elif sampletype == "train":
        local = LOCALSOURCE_train_dfs
    else:
        raise ValueError("sampletype needs to be 'nuclear', 'bts' or 'train'")

    if os.path.exists(local):
        csvs = []
        for name in os.listdir(local):
            if name[-4:] == ".csv":
                csvs.append(name)
        number_of_files = len(csvs)
        logger.debug(f"{number_of_files} dataframes found.")

    else:
        logger.info("Dataframe directory is not present, proceed to download files.")
        download_sample(sampletype=sampletype)

    if sampletype == "nuclear":
        if not os.path.isfile(os.path.join(LOCALSOURCE_WISE, "WISE.parquet")):
            logger.info("WISE dataframe does not exist, proceed to download.")
            download_wise()


def download_sample(sampletype="nuclear"):
    """
    Downloads the sample from DESY Syncandshare
    """
    if sampletype == "nuclear":
        cmd = f"curl --create-dirs -J -O --output-dir {LOCALSOURCE} {DOWNLOAD_URL_SAMPLE}; unzip {LOCALSOURCE}/NUCLEAR.zip -d {LOCALSOURCE}; rm {LOCALSOURCE}/NUCLEAR.zip"
    elif sampletype == "bts":
        cmd = f"curl --create-dirs -J -O --output-dir {LOCALSOURCE} {DOWNLOAD_URL_SAMPLE}; unzip {LOCALSOURCE}/BTS.zip -d {LOCALSOURCE}; rm {LOCALSOURCE}/BTS.zip"
    elif sampletype == "train":
        raise ValueError(
            f"No training data available online. They are expected to be in {LOCALSOURCE_train_dfs}"
        )

    subprocess.run(cmd, shell=True)
    logger.info(f"{sampletype} sample download complete")


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
    elif sampletype == "train":
        local = LOCALSOURCE_train_dfs

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


def is_valid_train_ztfid(ztfid: str) -> bool:
    """
    Checks if a string adheres to the ztfparsnip naming scheme
    """
    is_match = re.match("^ZTF[1-2]\d[a-z]{7}_[0-9]\d*$", ztfid)
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


def get_locations(sampletype="nuclear") -> pd.DataFrame | None:
    """
    Gets the metadata dataframe for the full sample
    """
    if sampletype == "nuclear":
        local = LOCALSOURCE_location
    elif sampletype == "bts":
        local = LOCALSOURCE_bts_info
    elif sampletype == "train":
        return None

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


def to_csv(df: pd.DataFrame, header: dict, outpath: Path):
    """
    Wrapper around pandas df.to_csv() to properly save header
    """
    headerstr = ""
    for i, val in header.items():
        headerstr += f"#{i}={val}\n"

    if outpath.is_file():
        os.remove(outpath)

    with open(outpath, "w") as f:
        f.write(headerstr)
        df.to_csv(f)
        f.close()


def get_ztfid_dataframe(
    ztfid: str, sampletype: str = "nuclear"
) -> Optional[pd.DataFrame]:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid) or is_valid_train_ztfid(ztfid):
        if sampletype == "nuclear":
            filepath = Path(LOCALSOURCE_dfs) / f"{ztfid}.csv"
        elif sampletype == "bts":
            filepath = Path(LOCALSOURCE_bts_dfs) / f"{ztfid}.csv"
        elif sampletype == "train":
            filepath = PathLOCALSOURCE_train_dfs / f"{ztfid}.csv"

        if filepath.is_file():
            df = pd.read_csv(filepath, comment="#")
        else:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_ztfid_header(
    ztfid: str, sampletype="nuclear", baseline: bool = False
) -> Optional[dict]:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid) or is_valid_train_ztfid(ztfid):
        if sampletype == "nuclear":
            if not baseline:
                filepath = Path(LOCALSOURCE_dfs) / f"{ztfid}.csv"
            else:
                filepath = Path(LOCALSOURCE_baseline) / f"{ztfid}_bl.csv"
        elif sampletype == "bts":
            if not baseline:
                filepath = Path(LOCALSOURCE_bts_dfs) / f"{ztfid}.csv"
            else:
                filepath = Path(LOCALSOURCE_bts_baseline) / f"{ztfid}_bl.csv"
        elif sampletype == "train":
            filepath = Path(LOCALSOURCE_train_dfs) / f"{ztfid}.csv"

        try:
            with open(filepath, "r") as input_file:
                headerkeys = []
                headervals = []

                for i, line in enumerate(input_file):
                    if len(line) >= 300:
                        break
                    if line == "\n":
                        break
                    if ",ampl_corr" in line:
                        break
                    key = line.split(",", 2)[0].split("=")[0].lstrip("#")
                    headerkeys.append(key)
                    val = line.split(",", 2)[0].split("=")[1][:-1]
                    headervals.append(val)

                returndict = {}
                for i, key in enumerate(headerkeys):
                    if headervals[i] == "-":
                        returnval = None
                    else:
                        returnval = headervals[i]
                    returndict.update({key: returnval})

                returndict["ztfid"] = returndict.get("name")

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


def airflares_stock_to_ztfid(sampletype="nuclear"):
    """ """
    if sampletype == "nuclear":
        infile = LOCALSOURCE_timewise_raw
    elif sampletype == "bts":
        infile = LOCALSOURCE_timewise_bts_raw

    with open(infile, "r") as f:
        wise_lcs = json.load(f)
    stock_to_ztfid = {}

    for key in wise_lcs.keys():
        stock_to_ztfid[key] = wise_lcs[key].get("id")

    return stock_to_ztfid


def ztfid_to_airflares_stock(sampletype="nuclear"):
    """ """
    stock_to_ztfid = airflares_stock_to_ztfid(sampletype=sampletype)

    ztfid_to_stock = {v: k for k, v in stock_to_ztfid.items()}

    return ztfid_to_stock


def parse_ampel_json(
    filepath: str, parameter_name: str, sampletype: str = "nuclear"
) -> dict:
    """Reads the mongodb export from Ampel"""
    if not os.path.isfile(filepath):
        raise ValueError("No file at given filepath")

    resultdict = {}

    if parameter_name in ["wise_bayesian", "wise_dust"]:
        airflares_to_ztf = airflares_stock_to_ztfid(sampletype=sampletype)

    with open(filepath) as json_file:
        data = json.load(json_file)
        for entry in data:
            stockid = entry["stock"]

            if parameter_name not in ["wise_bayesian", "wise_dust"]:
                if sampletype in ["nuclear", "bts"]:
                    ztfid = utils.stockid_to_ztfid(stockid)
                elif sampletype == "train":
                    ztfid = utils.stockid_ztfid_noisified(stockid)
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

                elif parameter_name in ["wise_bayesian", "wise_dust"]:
                    unit = entry["unit"]

                    if unit == "T2BayesianBlocks":
                        resultdict.update(
                            {ztfid: {"WISE_bayesian": {"bayesian": body}}}
                        )

                    elif unit == "T2DustEchoEval":
                        resultdict.update({ztfid: {"WISE_dust": {"dust": body}}})

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


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Loads the user-specific config
    """
    if not config_path:
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "..", "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config
