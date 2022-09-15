#!/usr/bin/env python3
# The coded used here was developed by Valéry Brinnel
# License: BSD-3-Clause

import os, logging, warnings

from typing import Optional

import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz"
rg = (6, 5, 4, 3, 2, 1, 0)
dec_ztf_years = {i: str(i + 17) for i in range(16)}
wise_fnu_jy = {"W1": 309.54, "W2": 171.787}
wise_appcor = {"W1": 0.222, "W2": 0.280}


def stockid_to_ztfid(stockid: int) -> str:
    """Converts AMPEL internal stock ID to ZTF ID"""
    year = dec_ztf_years[stockid & 15]

    # Shift base10 encoded value 4 bits to the right
    stockid = stockid >> 4

    # Convert back to base26
    l = ["a", "a", "a", "a", "a", "a", "a"]
    for i in rg:
        l[i] = alphabet[stockid % 26]
        stockid //= 26
        if not stockid:
            break

    return f"ZTF{year}{''.join(l)}"


def flux_density_to_abmag(
    flux_density: float, correct_apcor_bug: bool = True, band: Optional[str] = None
) -> float:
    """
    Convert flux density in Jy to AB magnitude
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        abmag = -2.5 * np.log10(flux_density) + 8.9

    if correct_apcor_bug:
        abmag = abmag + wise_appcor[band]

    return abmag


def abmag_to_flux_density(abmag: float) -> float:
    """
    Convert abmag to flux density in Jy
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        flux_density = 10 ** ((8.9 - abmag) / 2.5)
    return flux_density
