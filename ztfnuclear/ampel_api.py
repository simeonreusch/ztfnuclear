#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging
from json import JSONDecodeError

import numpy as np

import backoff
import requests


API_BASEURL = "https://ampel.zeuthen.desy.de/api"
API_CATALOGMATCH_URL = API_BASEURL + "/catalogmatch"
API_ZTF_ARCHIVE_URL = API_BASEURL + "/ztf/archive/v3"

logging.getLogger("backoff").setLevel(logging.ERROR)


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=600,
)
def ampel_api_catalog(
    catalog: str,
    catalog_type: str,
    ra_deg: float,
    dec_deg: float,
    search_radius_arcsec: float = 10,
    search_type: str = "all",
) -> dict | None:
    """
    Method for querying catalogs via the Ampel API
    'catalog' must be the name of a supported catalog, e.g.
    SDSS_spec, PS1, NEDz_extcats...
    For a full list of catalogs, confer
    https://ampel.zeuthen.desy.de/api/catalogmatch/catalogs

    """
    assert catalog_type in ["extcats", "catsHTM"]
    assert search_type in ["all", "nearest"]

    logger = logging.getLogger(__name__)

    queryurl_catalogmatch = API_CATALOGMATCH_URL + "/cone_search/" + search_type

    # First, we create a json body to post
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    query = {
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "catalogs": [
            {"name": catalog, "rs_arcsec": search_radius_arcsec, "use": catalog_type}
        ],
    }

    logger.debug(queryurl_catalogmatch)
    logger.debug(query)

    response = requests.post(url=queryurl_catalogmatch, json=query, headers=headers)

    if response.status_code == 503:
        if response.headers:
            logger.debug(response.headers)
        raise requests.exceptions.RequestException

    try:
        res = response.json()[0]
    except JSONDecodeError:
        if response.headers:
            logger.debug(response.headers)
        raise requests.exceptions.RequestException

    return res


@backoff.on_exception(
    backoff.expo,
    requests.exceptions.RequestException,
    max_time=30,
)
def ampel_api_ztfid(ztfid: str, limit: int = 100) -> dict | None:
    """
    Query alerts via the Ampel API
    """
    import keyring, getpass

    ampel_api_token = keyring.get_password("ampel_api_token", "password")
    if ampel_api_token is None:
        ampel_api_token = getpass.getpass(
            prompt=f"Enter your AMPEL API token: ", stream=None
        )
        keyring.set_password("ampel_api_token", "password", ampel_api_token)

    logger = logging.getLogger(__name__)

    queryurl_ztf_name = (
        API_ZTF_ARCHIVE_URL
        + f"/object/{ztfid}/alerts?with_history=true&with_cutouts=false&limit={limit}"
    )

    logger.debug(queryurl_ztf_name)

    headers = {"Authorization": f"Bearer {ampel_api_token}"}

    response = requests.get(
        queryurl_ztf_name,
        headers=headers,
    )

    if response.status_code == 503:
        if response.headers:
            logger.debug(response.headers)
        raise requests.exceptions.RequestException

    try:
        query_res = [i for i in response.json()]
        query_res = merge_alerts(query_res)

    except JSONDecodeError:
        if response.headers:
            logger.debug(response.headers)
        raise requests.exceptions.RequestException

    return query_res


def ampel_api_distnr(ztfid: str) -> float | None:
    """
    Calculate median distnr from alerts
    """
    query_res = ampel_api_ztfid(ztfid)

    distnrs = []

    if len(query_res) > 0:

        hist = query_res[0].get("prv_candidates")
        for cand in hist:
            distnr = cand.get("distnr")
            if distnr is not None:
                distnrs.append(distnr)

        if len(distnrs) > 0:
            distnr = np.median(distnrs)
            return distnr

    else:
        return None


def merge_alerts(alert_list: list) -> list:
    """ """
    merged_list = []
    keys = list(set([x["objectId"] for x in alert_list]))

    for objectid in keys:
        alerts = [x for x in alert_list if x["objectId"] == objectid]
        if len(alerts) == 1:
            merged_list.append(alerts[0])
        else:
            jds = [x["candidate"]["jd"] for x in alerts]
            order = [jds.index(x) for x in sorted(jds)[::-1]]
            latest = alerts[jds.index(max(jds))]
            latest["candidate"]["jdstarthist"] = min(
                [x["candidate"]["jdstarthist"] for x in alerts]
            )

            for index in order[1:]:

                x = alerts[index]

                # Merge previous detections

                for prv in x["prv_candidates"] + [x["candidate"]]:
                    if prv not in latest["prv_candidates"]:
                        latest["prv_candidates"] = [prv] + latest["prv_candidates"]

            merged_list.append(latest)

    return merged_list
