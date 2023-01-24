#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause
import warnings

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from ztfnuclear.sample import (
    NuclearSample,
)
from ztfnuclear import utils

resdict = {}
s = NuclearSample(sampletype="bts")
# for t in tqdm(s.get_transients(), total=len(s.ztfids)):
for t in tqdm([s.transient("ZTF18aakvxqj")]):
    _dict = {}
    if len(t.baseline) == 0:
        lc = t.raw_lc
    else:
        lc = t.baseline
    bl_correction = False  # if "ampl_corr" in lc.keys() else False
    if bl_correction:
        ampl_column = "ampl_corr"
        ampl_err_column = "ampl_err_corr"
    else:
        ampl_column = "ampl"
        ampl_err_column = "ampl.err"

    for filterid in sorted(lc["filterid"].unique()):
        _df = lc.query("filterid == @filterid")
        bandname = utils.ztf_filterid_to_band(filterid, short=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            F0 = 10 ** (_df.magzp / 2.5)
            F0_err = F0 / 2.5 * np.log(10) * _df.magzpunc
            flux = _df[ampl_column] / F0
            flux_err = np.sqrt(
                (_df[ampl_err_column] / F0) ** 2
                + (_df[ampl_column] * F0_err / F0**2) ** 2
            )
            abmag = -2.5 * np.log10(flux)
            abmag_err = 2.5 / np.log(10) * flux_err / flux
            obsmjd = _df.obsmjd.values
            _dict.update({bandname: np.min(abmag)})
        resdict.update({t.ztfid: _dict})

df = pd.DataFrame.from_dict(resdict, orient="index")
df.to_csv("peak_mag.csv")
