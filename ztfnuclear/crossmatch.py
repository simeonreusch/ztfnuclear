#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

import pandas as pd

from typing import Optional, Tuple
from ztfnuclear.ampel_api import ampel_api_catalog
from ztfnuclear import io
from ztfnuclear.database import WISE

logger = logging.getLogger(__name__)


def query_ned_for_z(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 10
) -> Optional[dict]:
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
) -> Optional[dict]:

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
) -> Optional[dict]:
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
) -> Optional[dict]:
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
) -> Optional[dict]:
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


def query_wise(
    ra_deg: float, dec_deg: float, searchradius_arcsec: float = 20
) -> Optional[dict]:
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
    else:
        return {"WISE": {}}
