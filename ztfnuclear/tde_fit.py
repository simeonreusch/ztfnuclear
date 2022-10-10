#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, warnings, time, logging
import numpy as np
import sncosmo  # type: ignore[import]
import pandas as pd
from sfdmap import SFDMap  # type: ignore[import]
from typing import Union, Optional
import errno, os, backoff, copy

import astropy.cosmology as cosmo
from astropy.table import Table
from typing import Literal, Sequence

logger = logging.getLogger(__name__)


class TDESource(sncosmo.Source):

    _param_names: list = ["risetime", "decaytime", "temperature", "amplitude"]
    param_names_latex: list = [
        "Rise Time",
        "Decay Time",
        "Temperature~[log10~K]",
        "Amplitude",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        # add redshift as parameter
        name: str = "TDE",
        version: str = "1.0",
    ) -> None:

        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        self.nu_kc: float = 3e8 / 4770e-10  # Reference wavelength used for K-correction
        self.z: float = 0.1

        # Some constants
        self.sigma_SB: float = 5.6704e-5  # Stefan-Boltzmann constant (erg/s/cm^2/K^4)
        self.c: float = 2.9979e10  # Speed of light (cm/s)
        self.h: float = 6.6261e-27  # Planck's constant (erg s)
        self.k: float = 1.3806e-16  # Boltzman's constant (erg/K)
        self.h_over_k: float = self.h / self.k
        self.lumdis: float = float(
            cosmo.FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(self.z).value
            * 1e6
            * 3.08568e18
        )

        # Fit parameters
        # defaults - peaktime = 0, rise=0.85 / decay=1.5 / T=4.5 / peakflux = 32
        self._parameters: np.ndarray = np.array([0.85, 1.5, 4.5, 32])

    @staticmethod
    def _wl_to_nu(self, wl: np.ndarray) -> np.ndarray:
        """
        Convert wavelength in Angstrom to frequency in Hz

        """
        nu = 2.9979e18 / wl

        return nu

    @staticmethod
    def _Planck(self, nu: np.ndarray, T: float) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :nu: np.ndarray, Array containing frequencies in Hz
        :T: np.float, temperature in K
        """
        bb = (
            2
            * self.h
            / self.c**2
            * nu**3
            / (np.exp(self.h * nu / (self.k * T)) - 1)
        )
        return bb

    @staticmethod
    def _cc_bol(self, nu: Union[float, np.ndarray], T: float):
        return self._Planck(self, nu, T) * nu / (self.sigma_SB * T**4 / np.pi)

    @staticmethod
    def _get_kc(self, nu: np.ndarray, T: Union[float, np.float64] = None) -> np.ndarray:
        """
        Calculate a k-correction for each wavelength
        """
        if T is None:
            T = self._parameters[2]
        T = 10**T
        kc = (
            (np.exp(self.h_over_k * self.nu_kc / T) - 1)
            / (np.exp(self.h_over_k * nu / T) - 1)
            * (nu / self.nu_kc) ** 4
        )
        return kc

    @staticmethod
    def _gauss(x: float, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_exp(self, phase: np.float64) -> np.float64:
        """
        Calculate a gaussian rise and decay for the lightcurve
        """

        risetime = self._parameters[0]
        decaytime = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]

        a1 = (
            10**peakflux
        )  # luminosity/flux at peak (this can be bolometric or L(nu_kc) depending on the setting)
        b1 = 10**risetime  # rise rate
        b2 = 10**decaytime  # decay rate
        a2 = a1 * self._gauss(0, b1)

        # are we in the rise or decay phase?
        # risephase
        if phase <= 0:
            val = a1 * self._gauss(phase, b1)

        # decayphase
        else:
            val = a2 * np.exp(-(phase) / b2)

        # kc = self._get_kc(self, nu)  # conversion from model curve to nuLnu of the data

        # return val * kc
        return val

    @staticmethod
    def _gauss_exp_test(self, phases: np.ndarray) -> np.ndarray:
        risetime = self._parameters[0]
        decaytime = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]

        a1 = (
            10**peakflux
        )  # luminosity/flux at peak (this can be bolometric or L(nu_kc) depending on the setting)
        b1 = 10**risetime  # rise rate
        b2 = 10**decaytime  # decay rate
        a2 = a1 * self._gauss(0, b1)

        # print(phases)
        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)
        vals_decay = a2 * np.exp(-(phases_decay) / b2)

        returnvals = np.concatenate((vals_rise, vals_decay))

        return returnvals

    @staticmethod
    def _lum2flux(
        L: np.ndarray,
        cm: float,
        nu: np.ndarray,
    ):
        """
        erg/s to Jansky
        >> flux = lum2flux(L, z, nu=1.4e9) # in Jy
        input:
         - L: luminosity in erg/s
         - cm: luminosity distance in cm

        note, no K-correction
        """
        return L / (nu * 4 * np.pi * cm**2) * 1e23

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        temp = self._parameters[2]

        if np.ndim(phase) == 0:
            # phase_iter = [phase]
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        model_flux = np.empty((len(phase_iter), len(wave)))

        nu = self._wl_to_nu(self, wave)
        nu_z = nu * (1 + self.z)

        kc = self._get_kc(self, nu)
        cc_bol = self._cc_bol(self, T=10**temp, nu=self.nu_kc)
        cc = self._cc_bol(self, T=10**temp, nu=nu * (1 + self.z))
        luminosity = 1 / (nu_z * 4 * np.pi * self.lumdis**2) * 1e23 * (1 + self.z)

        models = self._gauss_exp_test(self, phases=phase_iter)

        corrected_models = np.outer(models, kc) / cc_bol * cc
        model_flux = corrected_models * luminosity

        return model_flux


def fit(df: pd.DataFrame, ra: float, dec: float, baseline_info: dict):
    """
    Create TDE model.
    """
    if "t_peak" in baseline_info.keys():
        t_peak = baseline_info["t_peak"]

    else:
        logger.warn("No peak time in baseline correction, skipping plot")

    obsmjd = df.obsmjd.values

    ampl_column = "ampl_corr"
    ampl_err_column = "ampl_err_corr"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        F0 = 10 ** (df.magzp / 2.5)
        F0_err = F0 / 2.5 * np.log(10) * df.magzpunc
        Fratio = df[ampl_column] / F0
        Fratio_err = np.sqrt(
            (df[ampl_err_column] / F0) ** 2 + (df[ampl_column] * F0_err / F0**2) ** 2
        )

    df.replace(
        {
            "ZTF g": "ztfg",
            "ZTF_g": "ztfg",
            "ZTF r": "ztfr",
            "ZTF_r": "ztfr",
            "ZTF i": "ztfi",
            "ZTF_i": "ztfi",
        },
        inplace=True,
    )

    phot_tab = Table()
    phot_tab["flux"] = Fratio * 1e10
    phot_tab["fluxerr"] = Fratio_err * 1e10
    phot_tab["mjd"] = obsmjd
    phot_tab["zp"] = 25
    phot_tab["zpsys"] = "ab"
    phot_tab["band"] = df["filter"].values
    phot_tab.sort("mjd")

    phase = np.linspace(-50, 100, 10)
    wave = np.linspace(1000, 10000, 5)

    # initialize the TDE source
    tde_source = TDESource(phase, wave, name="tde")

    dust = sncosmo.models.CCM89Dust()
    dustmap = SFDMap()

    sncosmo_model = sncosmo.Model(
        source=tde_source,
        effects=[dust],
        effect_names=["mw"],
        effect_frames=["obs"],
    )

    transient_mwebv = dustmap.ebv((ra, dec))
    sncosmo_model.set(mwebv=transient_mwebv)

    fit_params = copy.deepcopy(sncosmo_model.param_names)
    fit_params.remove("mwebv")
    fit_params.remove("z")
    fit_params.remove("mwr_v")

    default_param_vals = sncosmo_model.parameters

    result, fitted_model = sncosmo.fit_lc(
        phot_tab,
        sncosmo_model,
        fit_params,
        bounds={
            "t0": [t_peak - 15, t_peak + 15],
            "temperature": [3.5, 5],
            "risetime": [0, 10],
            "decaytime": [0, 10],
            "amplitude": [10, 50],
        },
    )

    result["parameters"] = result["parameters"].tolist()

    NoneType = type(None)

    if not isinstance(result["covariance"], NoneType):
        result["covariance"] = result["covariance"].tolist()
    else:
        result["covariance"] = [None]

    result.pop("data_mask")

    # For filtering purposes we want a proper dict
    result["paramdict"] = {}
    for ix, pname in enumerate(result["param_names"]):
        result["paramdict"][pname] = result["parameters"][ix]

    result.pop("param_names")
    result.pop("vparam_names")
    result.pop("parameters")

    plot = sncosmo.plot_lc(phot_tab, model=fitted_model, errors=result.errors)
    plot.savefig("test.png")

    return result

    # testsource = TDESource(phase=phase, wave=wave)
    # testmodel = sncosmo.Model(source=testsource)

    # print(result.paramdict)

    # testmodel = sncosmo.Model(
    #     source=tde_source,
    #     effects=[dust],
    #     effect_names=["mw"],
    #     effect_frames=["obs"],
    # )

    # testmodel.set(
    #     z=0.0,
    #     t0=58667.457413334996,
    #     risetime=-9.353907955219253,
    #     decaytime=1.4189022370025717,
    #     temperature=3.9855297648445154,
    #     amplitude=31.472917556951845,
    #     mwebv=transient_mwebv,
    # )

    # lol1 = fitted_model.bandflux(
    #     "ztfg",
    #     [58668.0, t_peak - 10, t_peak - 5, t_peak, t_peak + 5],
    #     zp=25,
    #     zpsys="ab",
    # )
    # lol2 = testmodel.bandflux(
    #     "ztfg",
    #     [58668.0, t_peak - 10, t_peak - 5, t_peak, t_peak + 5],
    #     zp=25,
    #     zpsys="ab",
    # )
    # print(lol1)
    # print(lol2)
    # print(fitted_model)
    # print(testmodel)

    # lol = sncosmo.plot_lc(phot_tab, model=fitted_model, errors=result.errors)

    # lol.savefig("test.png")
