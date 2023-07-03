#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import getpass
import logging
import os

import keyring
import pandas as pd
import requests

import backoff
import ztfquery
from astropy import units as u
from astroquery.ipac.irsa import Irsa
from ztfnuclear import io
from ztfnuclear.ampel_api import ampel_api_catalog, ampel_api_distnr, ampel_api_sgscore
from ztfnuclear.database import WISE, SarahAGN

logger = logging.getLogger(__name__)


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=600,
)
def query_catwise(ra_deg: float, dec_deg: float, searchradius_arcsec=5):
    """
    Query the CatWISE 2020 catalog
    """

    logger.debug("Querying CatWISE2020")

    table = Irsa.query_region(
        catalog="catwise_2020",
        coordinates=f"{ra_deg},{dec_deg}",
        spatial="Cone",
        radius=searchradius_arcsec * u.arcsec,
        # selcols="ra,dec,dist",
    )
    df = table.to_pandas()
    if len(df) == 0:
        return {"catWISE2020": {}}

    first_row = df.iloc[0]
    ra = first_row["ra"]
    dec = first_row["dec"]
    w1 = first_row["w1mpro"]
    w2 = first_row["w2mpro"]
    dist = first_row["dist"]

    resdict = {"RA": ra, "Dec": dec, "Mag_W1": w1, "Mag_W2": w2, "dist_arcsec": dist}

    return {"catWISE2020": resdict}


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=600,
)
def query_marshal(ztfid):
    """
    I was thinking I would not need the Marshal. Sweet summerchild...
    Anyway, with this we parse the html for a transient page
    """
    from bs4 import BeautifulSoup as bs

    marshal_baseurl = (
        "http://skipper.caltech.edu:8080/cgi-bin/growth/view_source.cgi?name="
    )
    logger.debug(f"Querying the Growth Marshal for {ztfid}")
    username = keyring.get_password("marshal", f"marshal_user")
    password = keyring.get_password("marshal", f"marshal_password")

    if username is None:
        username = input(f"Enter your Growth Marshal login: ")
        password = getpass.getpass(
            prompt=f"Enter your Growth Marshal password: ", stream=None
        )
        keyring.set_password("marshal", f"marshal_user", username)
        keyring.set_password("marshal", f"marshal_password", password)
    auth = (username, password)

    url = marshal_baseurl + ztfid

    response = requests.get(url, auth=auth)
    df = pd.read_html(response.text)[0]
    rowid = df.apply(lambda row: row.astype(str).str.contains(ztfid).any(), axis=1)
    useful_rows = df.loc[rowid]
    if len(useful_rows) > 0:
        maybe_class = useful_rows[1].values[0].split("  ")[1]
        if len(maybe_class.split(":")) > 1:
            return {"Marshal": {}}
        else:
            logger.debug(f"{ztfid}: Classification found ({maybe_class})")
            return {"Marshal": {"class": maybe_class}}
    else:
        return {"Marshal": {}}


def query_ned_for_z(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 10
) -> dict | None:
    """Function to obtain redshifts from NED (via the AMPEL API)"""
    logger.info(f"Querying NEDz for redshift")

    res = ampel_api_catalog(
        catalog="NEDz_extcats",
        catalog_type="extcats",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
        search_type="nearest",
    )

    if res:
        return {"NEDz": float(res["body"]["z"]), "NEDz_dist": float(res["dist_arcsec"])}

    else:
        return {"NEDz": {}}


def query_crts(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 5
) -> dict | None:
    logger.debug(f"Querying CRTS DR1 if variable star")

    res = ampel_api_catalog(
        catalog="CRTS_DR1",
        catalog_type="extcats",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
    )

    if res:
        logger.debug("CRTS: Match")
        if "name" in res[0]["body"].keys():
            return {
                "CRTS": {
                    "name": str(res[0]["body"]["name"]),
                    "dist": float(res[0]["dist_arcsec"]),
                }
            }
        else:
            logger.debug("CRTS: No match")
            return {"CRTS": {}}

    else:
        logger.debug("CRTS: No match")
        return {"CRTS": {}}


def query_milliquas(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 1.5
) -> dict | None:
    """Query Milliquas"""

    logger.debug("Querying Milliquas for AGN/Quasar")

    res = ampel_api_catalog(
        catalog="milliquas",
        catalog_type="extcats",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
    )

    if res:
        if len(res) == 1:
            if "body" in res[0].keys():
                if res[0]["body"]["broad_type"]:
                    if "q" in res[0]["body"]["broad_type"]:
                        logger.debug("Milliquas: QSO match found")
                        return {
                            "Milliquas": {
                                "name": str(res[0]["body"]["name"]),
                                "type": "QSO",
                                "qso_prob": float(res[0]["body"]["qso_prob"]),
                                "dist": float(res[0]["dist_arcsec"]),
                            }
                        }

                    else:
                        logger.debug("Milliquas: Non-QSO match found")
                        return {
                            "Milliquas": {
                                "name": str(res[0]["body"]["name"]),
                                "type": str(res[0]["body"]["broad_type"]),
                                "dist": float(res[0]["dist_arcsec"]),
                            }
                        }
                else:
                    logger.debug("Milliquas: No match")
                    return {"Milliquas": {}}
            else:
                logger.debug("Milliquas: No match")
                return {"Milliquas": {}}
        else:
            logger.debug("Milliquas: Multiple matches found")
            return {"Milliquas": "multiple_matches"}
    else:
        logger.debug("Milliquas: No match")
        return {"Milliquas": {}}


def query_gaia(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 1.5
) -> dict | None:
    """Query Gaia"""
    logger.debug("Querying Gaia for parallax")
    res = ampel_api_catalog(
        catalog="GAIADR2",
        catalog_type="catsHTM",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=5.0,
    )
    if res:
        if res[0]["body"]["Plx"] is not None:
            plx_sig = float(res[0]["body"]["Plx"]) / float(res[0]["body"]["ErrPlx"])
            if plx_sig > 3.0:
                logger.debug("Gaia: Match.")
                return {
                    "Gaia": {"parallax_sigma": plx_sig, "dist": res[0]["dist_arcsec"]}
                }
            else:
                logger.debug("Gaia: Match, but no significant parallax found")
                return {"Gaia": {}}
        else:
            logger.debug("Gaia: Match, but no parallax")
            return {"Gaia": {}}
    else:
        logger.debug("Gaia: No match found")
        return {"Gaia": {}}


def query_sdss(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 1.5
) -> dict | None:
    """Query SDSS"""
    logger.debug("Querying SDSS for probable star")

    res = ampel_api_catalog(
        catalog="SDSSDR10",
        catalog_type="catsHTM",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
    )
    if res:
        if len(res) == 1:
            if float(res[0]["body"]["type"]) == 6.0:
                logger.debug("SDSS: Match found, it's a star")
                return {"SDSS": {"type": "star", "dist": res[0]["dist_arcsec"]}}
            else:
                logger.debug("SDSS: Match found, but no star")
                return {"SDSS": {}}
        else:
            logger.debug("SDSS: Multiple matches found")
            return {"SDSS": "multiple_matches"}

    else:
        return {"SDSS": {}}


def query_tns(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 1.5
) -> dict | None:
    """
    Query the AMPEL-hosted copy of TNS
    """
    logger.debug("Querying TNS")

    res = ampel_api_catalog(
        catalog="TNS",
        catalog_type="extcats",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
        search_type="nearest",
    )

    if res:
        logger.debug("TNS: Match found")
        res_body = res["body"]
        name = res_body["objname"]
        prefix = res_body["name_prefix"]
        full_name = prefix + name
        classification = res_body.get("object_type", {}).get("name")
        redshift = res_body.get("redshift")
        dist_arcsec = res["dist_arcsec"]

        return {
            "TNS": {
                "name": full_name,
                "type": classification,
                "dist": dist_arcsec,
                "z": redshift,
            },
        }

    return {"TNS": {}}


def query_wise_cat(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 3
) -> dict | None:
    """
    Query the AMPEL-hosted WISE catalogue
    """
    logger.debug("Querying WISE catalogue")

    res = ampel_api_catalog(
        catalog="WISE",
        catalog_type="catsHTM",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        search_radius_arcsec=searchradius_arcsec,
        search_type="nearest",
    )

    if res:
        final_res = (
            {"WISE_cat": res["body"]} if "body" in res.keys() else {"WISE_cat": {}}
        )
        import numpy as np

        if final_res["WISE_cat"]:
            final_res["WISE_cat"].update(
                {
                    "RA": np.degrees(final_res["WISE_cat"]["RA"]),
                    "Dec": np.degrees(final_res["WISE_cat"]["Dec"]),
                }
            )
            logger.debug("WISE cat (AMPEL): Match found")
            return final_res

    return {"WISE_cat": {}}


def query_wise(ra_deg: float, dec_deg: float, searchradius_arcsec: float = 20) -> dict:
    """
    Obtains WISE object RA and Dec from parquet file
    """
    wise = WISE()

    res = wise.query(
        ra_deg=ra_deg, dec_deg=dec_deg, searchradius_arcsec=searchradius_arcsec
    )

    if res:
        logger.debug("WISE: Match found")
        return {"WISE": res}

    return {"WISE": {}}


def query_sarah_agn(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 5
) -> dict:
    """
    Query the local AGN catalog that Sarah curated
    """
    agn = SarahAGN()
    res = agn.query(
        ra_deg=ra_deg, dec_deg=-dec_deg, searchradius_arcsec=searchradius_arcsec
    )
    if res:
        logger.debug("Sarah AGN: Match found")
        return {"Sarah_agn": res}

    return {"Sarah_agn": {}}


def query_bts(ztfid) -> dict:
    """
    Query the local BTS csv file
    """
    bts_df = pd.read_csv(io.LOCALSOURCE_bts_info)
    bts_df.query("ZTFID == @ztfid", inplace=True)

    res = {}

    if len(bts_df["type"]) > 0:
        res.update({"class": bts_df["type"].values[0]})

    return res


def query_ampel_sgscore(ztfid: str) -> dict:
    """
    Gets the median core distance from Ampel alerts
    """
    res = ampel_api_sgscore(ztfid=ztfid)

    if res is not None:
        logger.debug("Ampel sgscore: Match found")
    else:
        res = {}

    return {"sgscore": res}


def query_ampel_dist(ztfid: str) -> dict:
    """
    Gets the median core distance from Ampel alerts
    """
    res = ampel_api_distnr(ztfid=ztfid)

    if res is not None:
        logger.debug("Ampel dist: Match found")
    else:
        res = {}

    return {"distnr": res}
