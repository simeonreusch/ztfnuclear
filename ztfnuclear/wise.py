#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, json

import pandas as pd
import numpy as np

from timewise.parent_sample_base import ParentSampleBase
from timewise.wise_data_by_visit import WiseDataByVisit

test_sample = pd.read_csv("test_sample.csv", index_col=0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("timewise.wise_data_by_visit").setLevel(logging.DEBUG)

DEFAULT_KEYMAP = {"ra": "RA", "dec": "Dec", "id": "ztfid"}
SERVICE = "tap"


def get_wise_photometry(
    sample_df: pd.DataFrame, searchradius_arcsec: float = 5, positional: bool = True
) -> dict:
    """
    Execute two runs of Timewise. One to obtain the list of AllWISE IDs, one to obtain photometry given the list of IDs
    """

    class NuclearSampleInit(ParentSampleBase):
        """The nuclear sample gets initialized here"""

        def __init__(self):
            super(NuclearSampleInit, self).__init__(base_name="ztfnuclear")
            self.df = sample_df
            self.default_keymap = DEFAULT_KEYMAP

    class NuclearSampleWithID(ParentSampleBase):
        """The nuclear sample gets initialized here"""

        def __init__(self):
            super(NuclearSampleWithID, self).__init__(base_name="ztfnuclear")
            self.df = pd.read_csv(self.local_sample_copy, index_col=0)
            self.default_keymap = DEFAULT_KEYMAP

    tw_sample_init = WiseDataByVisit(
        base_name="ztfnuclear",
        parent_sample_class=NuclearSampleInit,
        min_sep_arcsec=searchradius_arcsec,
        n_chunks=1,
    )

    if positional:
        wise_lcs = tw_sample_init.load_binned_lcs(service=SERVICE)
        if not wise_lcs:
            logger.info("No data present, issuing WISE query for the sample.")
            tw_sample_init.get_photometric_data(service=SERVICE, nthreads=1)
            wise_lcs = tw_sample_init.load_binned_lcs(service=SERVICE)

    else:
        tw_sample_init.match_all_chunks()
        tw_sample_with_id = WiseDataByVisit(
            base_name="ztfnuclear",
            parent_sample_class=NuclearSampleWithID,
            min_sep_arcsec=searchradius_arcsec,
            n_chunks=1,
        )
        wise_lcs = tw_sample_with_id.load_binned_lcs(service=SERVICE)
        if not wise_lcs:
            tw_sample_with_id.get_photometric_data(
                service=SERVICE, query_type="by_allwise_id", nthreads=1
            )
            wise_lcs = tw_sample_with_id.load_binned_lcs(service=SERVICE)

    # Now we reindex with the proper ZTF IDs
    _wise_lcs = copy.deepcopy(wise_lcs)
    for key in _wise_lcs.keys():
        ztfid = sample_df.iloc[int(key)]["ztfid"]
        wise_lcs[ztfid] = wise_lcs.pop(key)

    for vegamag_key in ["W1_mean_mag", "W2_mean_mag"]:
        for ztfid in wise_lcs.keys():
            vegamags = wise_lcs[ztfid][vegamag_key]
            abmags = {}
            for epoch in vegamags.keys():
                vegamag = vegamags[epoch]
                abmag = vega_to_abmag(vegamag, vegamag_key[:2])
                abmags.update({epoch: abmag})

            wise_lcs[ztfid][vegamag_key + "_ab"] = abmags

    returndict = {}
    for ztfid in sample_df.ztfid:
        if ztfid in wise_lcs.keys():
            returndict.update({ztfid: {"WISE_lc": wise_lcs[ztfid]}})
        else:
            returndict.update({ztfid: {"WISE_lc": {}}})
    return returndict


def vega_to_abmag(vegamag: float, band: str) -> float:
    """
    Convert WISE Vega magnitudes to AB magnitudes
    """
    conversion = {"W1": 2.699, "W2": 3.339, "W3": 5.174, "W4": 6.620}

    abmag = vegamag + conversion[band]
    return abmag


if __name__ == "__main__":
    full_sample = pd.read_csv("full_sample.csv", index_col=0)

    wise_lcs = get_wise_photometry(
        sample_df=full_sample, searchradius_arcsec=5, positional=True
    )

    with open("wise_full_by_id.json", "w") as fp:
        json.dump(wise_lcs, fp)
