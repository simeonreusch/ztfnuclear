#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import warnings

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy.ma as ma
from ztfnuclear.sample import (
    NuclearSample,
)
from ztfnuclear import utils
from ztfnuclear import io

SAMPLE = "bts"
snt_threshold = 6

resdict = {}
s = NuclearSample(sampletype=SAMPLE)
for t in tqdm(s.get_transients(), total=len(s.ztfids)):
    # for t in tqdm([s.transient("ZTF21acjqalv")]):
    _dict = {}
    lc = t.raw_lc
    ampl_column = "ampl"
    ampl_err_column = "ampl.err"

    for filterid in sorted(lc["filterid"].unique()):
        _df = lc.query("filterid == @filterid")

        bandname = utils.ztf_filterid_to_band(filterid, short=True)
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

        snt_limit = flux_err.values * snt_threshold
        mask = np.less(flux.values, snt_limit)
        flux = ma.masked_array(flux.values, mask=mask).compressed()
        flux_err = ma.masked_array(flux_err.values, mask=mask).compressed()
        abmag = ma.masked_array(abmag.values, mask=mask).compressed()

        if len(abmag) > 0:
            _dict.update({bandname: np.min(abmag)})

        resdict.update({t.ztfid: _dict})

if SAMPLE == "nuclear":
    outfile = io.LOCALSOURCE_peak_mags
else:
    outfile = io.LOCALSOURCE_bts_peak_mags
df = pd.DataFrame.from_dict(resdict, orient="index")
df.to_csv(outfile)

# Re-ingest to database
s.meta.delete_keys(keys=["peak_mags"])
s = NuclearSample(sampletype=SAMPLE)
