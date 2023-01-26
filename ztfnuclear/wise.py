#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, json, copy

import pandas as pd
import numpy as np

from ztfnuclear import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("timewise.wise_data_by_visit").setLevel(logging.DEBUG)

DEFAULT_KEYMAP = {"ra": "RA", "dec": "Dec", "id": "ztfid"}
SERVICE = "tap"

SAMPLE = "bts"


def is_in_wise_agn_box(w1w2: float, w2w3: float) -> bool:
    """
    Code for AGN box by Jannis Necker
    https://gitlab.desy.de/jannis.necker/air_flares
    Cuts are taken from Hvding et al. (2022)
    https://iopscience.iop.org/article/10.3847/1538-3881/ac5e33
    """

    # deal with mock values
    if w1w2 > 998:
        return False

    agn_box = {
        "W2-W3": [1.734, 3.916],
        "W1-W2/W2-W3": [
            [0.0771, 0.319],
            [0.261, -0.260],
        ],
    }
    if (
        (w2w3 > agn_box["W2-W3"][0])
        and (w2w3 < agn_box["W2-W3"][1])
        and (
            w1w2 > (agn_box["W1-W2/W2-W3"][0][0] * w2w3 + agn_box["W1-W2/W2-W3"][0][1])
        )
        and (
            w1w2 > (agn_box["W1-W2/W2-W3"][1][0] * w2w3 + agn_box["W1-W2/W2-W3"][1][1])
        )
    ):
        return True
    else:
        return False
    # agn_mask =
    #     (W2W3 > AGN_BOX["W2-W3"][0])
    #     & (W2W3 < AGN_BOX["W2-W3"][1])
    #     & (
    #         W1W2
    #         > AGN_BOX["W1-W2 / W2-W3 parameters"][0][0] * W2W3
    #         + AGN_BOX["W1-W2 / W2-W3 parameters"][0][1]
    #     )
    #     & (
    #         W1W2
    #         > AGN_BOX["W1-W2 / W2-W3 parameters"][1][0] * W2W3
    #         + AGN_BOX["W1-W2 / W2-W3 parameters"][1][1]
    #     )
    # )


def get_wise_photometry(
    sample_df: pd.DataFrame, searchradius_arcsec: float = 5, positional: bool = True
) -> dict:
    """
    Execute two runs of Timewise. One to obtain the list of AllWISE IDs, one to obtain photometry given the list of IDs
    """
    from timewise.parent_sample_base import ParentSampleBase
    from timewise.wise_data_by_visit import WiseDataByVisit

    class NuclearSampleInit(ParentSampleBase):
        """The nuclear sample gets initialized here"""

        def __init__(self):
            super(NuclearSampleInit, self).__init__(base_name=SAMPLE)
            self.df = sample_df
            self.default_keymap = DEFAULT_KEYMAP

    tw_sample_init = WiseDataByVisit(
        base_name=SAMPLE,
        parent_sample_class=NuclearSampleInit,
        min_sep_arcsec=searchradius_arcsec,
        n_chunks=1,
    )

    wise_lcs = tw_sample_init.load_data_product(service=SERVICE)
    if not wise_lcs:
        logger.info("No data present, issuing WISE query for the sample.")
        tw_sample_init.get_photometric_data(service=SERVICE, nthreads=1)
        wise_lcs = tw_sample_init.load_data_product(service=SERVICE)

    # Now we reindex with the proper ZTF IDs
    _wise_lcs = copy.deepcopy(wise_lcs)
    for key in _wise_lcs.keys():
        ztfid = sample_df.iloc[int(key)]["ztfid"]
        wise_lcs[ztfid] = wise_lcs.pop(key)

    for vegamag_key in ["W1_mean_mag", "W2_mean_mag"]:
        for ztfid in wise_lcs.keys():
            # print(wise_lcs[ztfid]["timewise_lightcurve"].keys())
            vegamags = wise_lcs[ztfid]["timewise_lightcurve"][vegamag_key]
            abmags = {}
            for epoch in vegamags.keys():
                vegamag = vegamags[epoch]
                abmag = vega_to_abmag(vegamag, vegamag_key[:2])
                abmags.update({epoch: abmag})

            wise_lcs[ztfid]["timewise_lightcurve"][vegamag_key + "_ab"] = abmags

    returndict = {}

    wise_key = "WISE_lc_by_pos"

    for ztfid in sample_df.ztfid:
        if ztfid in wise_lcs.keys():
            returndict.update({ztfid: {wise_key: wise_lcs[ztfid]}})
        else:
            returndict.update({ztfid: {wise_key: {}}})

    return returndict


def vega_to_abmag(vegamag: float, band: str) -> float:
    """
    Convert WISE Vega magnitudes to AB magnitudes
    """
    conversion = {"W1": 2.699, "W2": 3.339, "W3": 5.174, "W4": 6.620}

    abmag = vegamag + conversion[band]
    return abmag


if __name__ == "__main__":
    """
    Run the WISE query
    """
    positional_search = True

    # test_sample_ztfnuclear = pd.read_csv(
    #     "/Users/simeon/ztfnuclear/timewise_nuclear_test.csv", index_col=0
    # )
    full_sample_nuclear = pd.read_csv(
        "/Users/simeon/ztfnuclear/timewise_nuclear.csv", index_col=0
    )
    full_sample_bts = pd.read_csv(
        "/Users/simeon/ztfnuclear/timewise_bts.csv", index_col=0
    )

    if SAMPLE == "nuclear":
        use_sample = full_sample_nuclear
    else:
        use_sample = full_sample_bts

    wise_lcs = get_wise_photometry(
        sample_df=use_sample, searchradius_arcsec=5, positional=positional_search
    )

    if SAMPLE == "nuclear":
        outpath = os.path.join(io.LOCALSOURCE, "NUCLEAR", "timewise")
    else:
        outpath = os.path.join(io.LOCALSOURCE, "BTS", "timewise")

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(os.path.join(outpath, f"timewise_lightcurves_{SAMPLE}.json"), "w") as fp:
        json.dump(wise_lcs, fp)
