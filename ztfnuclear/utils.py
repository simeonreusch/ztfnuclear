#!/usr/bin/env python3
# The coded used here was developed by Valéry Brinnel
# License: BSD-3-Clause

import os, logging, warnings, re

from typing import Optional

from astropy import units as u

import numpy as np

alphabet = "abcdefghijklmnopqrstuvwxyz"
rg = (6, 5, 4, 3, 2, 1, 0)
dec_ztf_years = {i: str(i + 17) for i in range(16)}
wise_fnu_jy = {"W1": 309.54, "W2": 171.787}
wise_appcor = {"W1": 0.222, "W2": 0.280}
wl_angstrom = {
    "g": 4722.7,
    "r": 6339.6,
    "i": 7886.1,
    "W1": 33526,
    "W2": 46028,
}


def is_ztf_name(name) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    if re.match(r"^ZTF[1-2]\d[a-z]{7}$", name):
        match = True
    else:
        match = False
    return match


def mjd_to_jd(mjd: float) -> float:
    """
    Convert MJD to JD
    """
    return mjd + 2400000.5


def jd_to_mjd(jd: float) -> float:
    """
    Convert MJD to JD
    """
    return jd - 2400000.5


def is_tns_name(name) -> bool:
    """
    Checks if a string adheres to the TNS naming scheme
    """
    if re.match(r"^(AT|SN)(19|20)\d{2}[a-z]{3,4}$", name):
        match = True
    else:
        match = False
    return match


def ztf_filterid_to_band(filterid: int, short: bool = False, sncosmo: bool = False):
    """
    Get the band name associated with a ZTF filter id
    """
    bands = {1: "ZTF_g", 2: "ZTF_r", 3: "ZTF_i"}
    sncosmo_bands = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

    band = bands[filterid]

    if short:
        return band[4:]
    elif sncosmo:
        return sncosmo_bands[filterid]
    else:
        return band


def band_frequency(band: str) -> float:
    """
    Get the frequency associated with a ZTF or WISE band
    """
    wl_a = wl_angstrom[band]
    wl = wl_a / 1e10
    c = 2.998e8
    nu = c / wl

    return nu


def band_wavelength(band: str) -> float:
    """
    Get the wavelength associated with a ZTF or WISE band
    """
    wl_a = wl_angstrom[band]
    wl = wl_a / 1e10

    return wl


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
    flux_density: float, correct_apcor_bug: bool = False, band: Optional[str] = None
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


def flux_density_to_luminosity(flux_density: float, z: float) -> float:
    """
    Convert flux density to luminosity (given redshift z)
    """
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    conversion_factor = (
        4 * np.pi * cosmo.luminosity_distance(z=z).to(u.cm) ** 2.0 / (1 + z)
    )

    return flux_density * conversion_factor


def flux_density_err_to_abmag_err(
    flux_density: float,
    flux_density_err: float,
) -> float:
    """
    Convert flux density error to AB magnitude error
    """
    abmag_err = 1.08574 / flux_density * flux_density_err

    return abmag_err


def abmag_to_flux_density(abmag: float) -> float:
    """
    Convert AB magnitude to flux density in Jy
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        flux_density = 10 ** ((8.9 - abmag) / 2.5)
    return flux_density


def abmag_err_to_flux_density_err(abmag: float, abmag_err: float) -> float:
    """
    Convert AB magnitude error to flux density error (in Jy) -> Check if this is correct
    """
    # print(abmag_err)
    flux_density_err = 3344.07 * np.exp(-0.921034 * np.abs(abmag)) * abmag_err

    return flux_density_err
