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

from astropy import constants as c
from astropy import units as u

logger = logging.getLogger(__name__)


class TDESource_exp_simple(sncosmo.Source):
    _param_names: list = [
        "risetime",
        "decaytime",
        "temperature",
        "amplitude",
    ]
    param_names_latex: list = [
        "Rise Time~[log10 day]",
        "Decay Time~[log10 day]",
        "Temperature~[log10~K]",
        "Amplitude",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        name: str = "TDE_exp_simple",
        version: str = "1.0",
    ) -> None:
        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        self._parameters: np.ndarray = np.array([1.584, 2.278, 3.998, 1e-25])

    @staticmethod
    def _planck_lam(
        self, wave: np.ndarray, T: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :wave: np.ndarray, array containing wavelength in AA
        :T: np.ndarray, array containing temperatures in K
        """
        wave = wave * u.AA
        wave = wave.to(u.m)

        prefactor = 2 * c.h * c.c**2 / wave**5

        prefactor = np.tile(prefactor, (len(T), 1)).transpose()

        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)
        bb = bb * u.sr

        return bb

    @staticmethod
    def _gauss(x: float, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_exp(self, phases: np.ndarray) -> np.ndarray:
        risetime = self._parameters[0]
        decaytime = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]

        # Gaussian rise
        a1 = peakflux
        b1 = 10**risetime
        b2 = 10**decaytime
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)

        phases_decay = phases[(phases > 0)]

        # exponential decay
        vals_decay = a2 * np.exp(-(phases_decay) / b2)

        returnvals = np.concatenate((vals_rise, vals_decay))

        return returnvals

    def _temp_evolution(self, phase: np.ndarray) -> np.ndarray:
        """
        Create an array with a linear temperature evolution
        """
        temp = self._parameters[2]

        t_evo = (10**temp) + phase * 0

        return t_evo

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        wave_m = wave * 1e-10

        t_evo = self._temp_evolution(phase=phase)

        if np.ndim(phase) == 0:
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        rise_decay = self._gauss_exp(self, phases=phase_iter)

        model_flux = (rise_decay * bb_lam).transpose()

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)

        return model_flux_cgi


class TDESource_exp_flextemp(sncosmo.Source):
    _param_names: list = [
        "risetime",
        "decaytime",
        "temperature",
        "amplitude",
        "d_temp",
        "plateaustart",
    ]
    param_names_latex: list = [
        "Rise Time~[log10 day]",
        "Decay Time~[log10 day]",
        "Temperature~[log10~K]",
        "Amplitude",
        "delta Temperature (K/day)",
        "plateaustart (day)",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        name: str = "TDE_exp_flextemp",
        version: str = "1.0",
        priors: np.ndarray = None,
        debug: bool = None,
    ) -> None:
        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        if priors is None:
            self._parameters: np.ndarray = np.array(
                [1.584, 2.278, 3.998, 6.43e-25, 0, 300]
            )
        else:
            self._parameters = priors

        if debug:
            print("flextemp fit priors:")
            print(self._parameters)

    @staticmethod
    def _planck_nu(
        self, nu: np.ndarray, T: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :nu: np.ndarray, array containing frequencies in Hz
        :T: np.ndarray, array containing temperatures in K
        """
        nu = nu * u.Hz
        prefactor = 2 * c.h / c.c**2 * nu**3

        prefactor = np.tile(prefactor, (len(T), 1)).transpose()

        exponential_term = np.outer((c.h.value * nu.value), 1 / (c.k_B.value * T))

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        bb = bb.value * u.J / u.s / u.m**2 / u.Hz / u.sr

        # returns spectral radiance: J s-1 sr-1 m-2 Hz-1

        return bb

    @staticmethod
    def _planck_lam(
        self, wave: np.ndarray, T: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :wave: np.ndarray, array containing wavelength in AA
        :T: np.ndarray, array containing temperatures in K
        """
        wave = wave * u.AA
        wave = wave.to(u.m)

        prefactor = 2 * c.h * c.c**2 / wave**5

        prefactor = np.tile(prefactor, (len(T), 1)).transpose()

        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)

        peak_temp = 10 ** self._parameters[2]
        bol_corr = peak_temp**4 / T**4

        bb = bb * u.sr * bol_corr

        return bb

    @staticmethod
    def _gauss(x: float, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_exp(self, phases: np.ndarray) -> np.ndarray:
        risetime = self._parameters[0]
        decaytime = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]

        # Gaussian rise
        a1 = peakflux
        b1 = 10**risetime
        b2 = 10**decaytime
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)

        phases_decay = phases[(phases > 0)]

        # exponential decay
        vals_decay = a2 * np.exp(-(phases_decay) / b2)

        returnvals = np.concatenate((vals_rise, vals_decay))

        return returnvals

    def _temp_evolution(self, phase: np.ndarray) -> np.ndarray:
        """
        Create an array with a linear temperature evolution
        """
        temp = self._parameters[2]
        d_temp = self.parameters[4]
        plateau_start = self.parameters[5]

        phase_clip = np.clip(
            phase, -0.01, plateau_start
        )  # stop temperature evolution at -30 and +365 days

        t_evo = (10**temp) + (phase_clip * d_temp)

        # okay we could try the following:
        temp_max_delta = 15000
        t_evo_clip = np.clip(
            t_evo, 10**temp - temp_max_delta, 10**temp + temp_max_delta
        )

        # clip temperatures outside 1000 and 100,000 K
        # t_evo_clip = np.clip(t_evo, 1e3, 1e6)

        return t_evo_clip

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        t_evo = self._temp_evolution(phase=phase)

        if np.ndim(phase) == 0:
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        # now t is an array, that's why we do all this matrix gymnastics
        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        rise_decay = self._gauss_exp(self, phases=phase_iter)

        model_flux = (rise_decay * bb_lam).transpose()

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)

        return model_flux_cgi


class TDESource_pl_simple(sncosmo.Source):
    _param_names: list = [
        "risetime",
        "alpha",
        "temperature",
        "amplitude",
        "normalization",
    ]
    param_names_latex: list = [
        "Rise Time",
        "Alpha",
        "Temperature~[log10~K]",
        "Amplitude",
        "normalization",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        name: str = "TDE_pl_simple",
        version: str = "1.0",
    ) -> None:
        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        # defaults - peaktime = 0, rise=0.85 / alpha=-1.5 / T=4.0 / peakflux = 6.5e-25 / powerlaw normalization=1.5

        self._parameters: np.ndarray = np.array([1.5, -1.5, 4.0, 1e-24, 1.5])

    @staticmethod
    def _planck_lam(
        self, wave: np.ndarray, T: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :wave: np.ndarray, array containing wavelength in AA
        :T: np.ndarray, array containing temperatures in K
        """
        wave = wave * u.AA
        wave = wave.to(u.m)

        prefactor = 2 * c.h * c.c**2 / wave**5

        prefactor = np.tile(prefactor, (len(T), 1)).transpose()

        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)
        bb = bb * u.sr

        return bb

    @staticmethod
    def _gauss(x: float, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_pl(self, phases: np.ndarray) -> np.ndarray:
        risetime = self._parameters[0]
        alpha = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]
        normalization = 10 ** self._parameters[4]

        # Gaussian rise
        a1 = peakflux
        b1 = 10**risetime  # rise rate
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)  # exp rise
        vals_decay = (
            a2 * ((phases_decay + normalization) / normalization) ** alpha
        )  # powerlaw decay

        returnvals = np.concatenate((vals_rise, vals_decay))

        return returnvals

    def _temp_evolution(self, phase: np.ndarray) -> np.ndarray:
        """
        Create an array with a linear temperature evolution
        """
        temp = self._parameters[2]

        t_evo = (10**temp) + phase * 0

        return t_evo

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        wave_m = wave * 1e-10

        t_evo = self._temp_evolution(phase=phase)

        if np.ndim(phase) == 0:
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        rise_decay = self._gauss_pl(self, phases=phase_iter)

        model_flux = (rise_decay * bb_lam).transpose()

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)

        return model_flux_cgi


class TDESource_pl_flextemp(sncosmo.Source):
    _param_names: list = [
        "risetime",
        "alpha",
        "temperature",
        "amplitude",
        "normalization",
        "d_temp",
        "plateaustart",
    ]
    param_names_latex: list = [
        "Rise Time",
        "Alpha",
        "Temperature~[log10~K]",
        "Amplitude",
        "normalization",
        "delta Temperature (K/day)",
        "plateaustart (day)",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        name: str = "TDE_pl_flextemp",
        version: str = "1.0",
        priors: np.ndarray = None,
        debug: bool = None,
    ) -> None:
        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        if priors is None:
            self._parameters: np.ndarray = np.array(
                [1.584, -1.5, 4.0, 6.5e-25, 1.5, 0, 300]
            )
        else:
            self._parameters = priors

        if debug:
            print("flextemp fit priors:")
            print(self._parameters)

    @staticmethod
    def _planck_lam(
        self, wave: np.ndarray, T: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        """
        Calculate the spectral radiance of a blackbody
        :wave: np.ndarray, array containing wavelength in AA
        :T: np.ndarray, array containing temperatures in K
        """
        wave = wave * u.AA
        wave = wave.to(u.m)

        prefactor = 2 * c.h * c.c**2 / wave**5

        prefactor = np.tile(prefactor, (len(T), 1)).transpose()

        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)

        peak_temp = 10 ** self._parameters[2]
        bol_corr = peak_temp**4 / T**4

        bb = bb * u.sr * bol_corr

        return bb

    @staticmethod
    def _gauss(x: float, sigma: Union[float, np.float64]) -> Union[float, np.float64]:
        """
        Calculate a Gaussian
        """
        gauss = np.exp(-0.5 * x**2 / (sigma**2))
        return gauss

    @staticmethod
    def _gauss_pl(self, phases: np.ndarray) -> np.ndarray:
        risetime = self._parameters[0]
        alpha = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]
        normalization = 10 ** self._parameters[4]

        # Gaussian rise
        a1 = peakflux
        b1 = 10**risetime  # rise rate
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)  # exp rise
        vals_decay = (
            a2 * ((phases_decay + normalization) / normalization) ** alpha
        )  # powerlaw decay

        returnvals = np.concatenate((vals_rise, vals_decay))

        return returnvals

    def _temp_evolution(self, phase: np.ndarray) -> np.ndarray:
        """
        Create an array with a linear temperature evolution
        """
        temp = self._parameters[2]
        d_temp = self.parameters[5]
        plateau_start = self.parameters[6]

        phase_clip = np.clip(
            phase, -30, plateau_start
        )  # stop temperature evolution at -30 and +365 days

        t_evo = (10**temp) + (phase_clip * d_temp)

        t_evo_clip = np.clip(
            t_evo, 1e3, 1e6
        )  # clip temperatures outside 1000 and 100,000 K

        return t_evo_clip

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        t_evo = self._temp_evolution(phase=phase)

        if np.ndim(phase) == 0:
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        # now t is an array, that's why we do all this matrix gymnastics
        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        rise_decay = self._gauss_pl(self, phases=phase_iter)

        model_flux = (rise_decay * bb_lam).transpose()

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)

        return model_flux_cgi


def fit(
    df: pd.DataFrame,
    ra: float,
    dec: float,
    baseline_info: dict,
    powerlaw: bool = True,
    plateau: bool = True,
    ztfid: str = None,
    simplefit_only: bool = False,
    debug: bool = False,
):
    """
    Fit TDE model
    """

    if "t_peak" in baseline_info.keys():
        t_peak = baseline_info["t_peak"]

    else:
        logger.warn("No peak time in baseline correction, skipping plot")

    ampl_column = "ampl_corr"
    ampl_err_column = "ampl_err_corr"

    df["phase"] = df.obsmjd - t_peak
    # df.query("phase < 365 and phase > -30", inplace=True) -> leads to erroneous risetime for some TDEs

    if len(df.query("phase < 365 and phase > -30")) < 10:
        logger.warn(
            "Too little datapoints within the year around peak (-30, +365), skipping fit"
        )
        return {"success": False}

    obsmjd = df.obsmjd.values

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        F0 = 10 ** (df.magzp / 2.5)
        F0_err = F0 / 2.5 * np.log(10) * df.magzpunc
        Fratio = df[ampl_column] / F0
        Fratio_err = np.abs(
            np.sqrt(
                (df[ampl_err_column] / F0) ** 2
                + (df[ampl_column] * F0_err / F0**2) ** 2
            )
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

    if not powerlaw:
        tde_source_simple = TDESource_exp_simple(
            phase,
            wave,
            name="tde",
        )
    else:
        tde_source_simple = TDESource_pl_simple(
            phase,
            wave,
            name="tde",
        )

    dust = sncosmo.models.CCM89Dust()
    dustmap = SFDMap()
    transient_mwebv = dustmap.ebv((ra, dec))

    sncosmo_model_simple = sncosmo.Model(
        source=tde_source_simple,
        effects=[dust],
        effect_names=["mw"],
        effect_frames=["obs"],
    )

    sncosmo_model_simple.set(mwebv=transient_mwebv)

    fit_params = copy.deepcopy(sncosmo_model_simple.param_names)
    fit_params.remove("mwebv")
    fit_params.remove("mwr_v")
    fit_params.remove("z")  # let's not fit z here

    default_param_vals = sncosmo_model_simple.parameters

    try:
        if powerlaw:
            bounds_simple = {
                "t0": [t_peak - 30, t_peak + 30],
                "temperature": [3.5, 5.0],
                "risetime": [0, 5],
                "alpha": [-15, 0],
                "normalization": [0, 5],
            }

        else:
            bounds_simple = {
                "t0": [t_peak - 30, t_peak + 30],
                "temperature": [3.5, 5.0],
                "risetime": [0, 5],
                "decaytime": [0, 5],
            }

        result, fitted_model_simple = sncosmo.fit_lc(
            phot_tab,
            sncosmo_model_simple,
            fit_params,
            bounds=bounds_simple,
        )

        if debug:
            fig = sncosmo.plot_lc(
                data=phot_tab, model=fitted_model_simple, zpsys="ab", zp=25
            )
            outpath = "/Users/simeon/Desktop/flextemp_test/diagnostic"
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if powerlaw:
                fig.savefig(os.path.join(outpath, f"{ztfid}_pl.png"))
            else:
                fig.savefig(os.path.join(outpath, f"{ztfid}_exp.png"))

        result["parameters"] = result["parameters"].tolist()

        NoneType = type(None)

        if not isinstance(result["covariance"], NoneType):
            result["covariance"] = result["covariance"].tolist()
        else:
            result["covariance"] = [None]

        result.pop("data_mask")

        result["paramdict"] = {}
        for ix, pname in enumerate(result["param_names"]):
            result["paramdict"][pname] = result["parameters"][ix]

        result.pop("param_names")
        result.pop("vparam_names")

        if debug:
            print(result["paramdict"])

        if simplefit_only:
            result.pop("parameters")

        if simplefit_only:
            print(result)
            return result

        else:
            # doing a second round of fitting, flexible temperature evolution

            params = result["parameters"]
            pdict = result["paramdict"]

            if not powerlaw:
                priors = [
                    pdict["risetime"],
                    pdict["decaytime"],
                    pdict["temperature"],
                    pdict["amplitude"],
                    0.0,
                    365.0,
                ]
            else:
                priors = [
                    pdict["risetime"],
                    pdict["alpha"],
                    pdict["temperature"],
                    pdict["amplitude"],
                    pdict["normalization"],
                    0,
                    365.0,
                ]

            t0 = pdict["t0"]

            if not powerlaw:
                tde_source_flextemp = TDESource_exp_flextemp(
                    phase, wave, name="tde", priors=priors, debug=debug
                )

            else:
                tde_source_flextemp = TDESource_pl_flextemp(
                    phase, wave, name="tde", priors=priors, debug=debug
                )

            sncosmo_model_flextemp = sncosmo.Model(
                source=tde_source_flextemp,
                effects=[dust],
                effect_names=["mw"],
                effect_frames=["obs"],
            )

            sncosmo_model_flextemp.set(mwebv=transient_mwebv)

            fit_params = copy.deepcopy(sncosmo_model_flextemp.param_names)
            fit_params.remove("mwebv")
            fit_params.remove("mwr_v")
            fit_params.remove("z")  # let's not fit z here

            # if pdict["risetime"] < 2:
            # lol = 1
            # fit_params.remove("risetime")
            # sncosmo_model_flextemp.set(risetime=pdict["risetime"])

            default_param_vals = sncosmo_model_flextemp.parameters

            bounds_flextemp = bounds_simple
            bounds_flextemp["d_temp"] = [-1500, 1500]
            bounds_flextemp["plateaustart"] = [100, 1200]

            result, fitted_model = sncosmo.fit_lc(
                phot_tab,
                sncosmo_model_flextemp,
                fit_params,
                bounds=bounds_flextemp,
            )

            result["parameters"] = result["parameters"].tolist()

            NoneType = type(None)

            if not isinstance(result["covariance"], NoneType):
                result["covariance"] = result["covariance"].tolist()
            else:
                result["covariance"] = [None]

            result.pop("data_mask")

            result["paramdict"] = {}
            for ix, pname in enumerate(result["param_names"]):
                result["paramdict"][pname] = result["parameters"][ix]

            result.pop("param_names")
            result.pop("vparam_names")
            result.pop("parameters")

            if "Hesse" in result["message"]:
                result["success"] = True

            if debug:
                fig = sncosmo.plot_lc(
                    data=phot_tab, model=fitted_model, zpsys="ab", zp=25
                )
                outpath = "/Users/simeon/Desktop/flextemp_test/diagnostic"
                if powerlaw:
                    fig.savefig(os.path.join(outpath, f"{ztfid}_pl_flextemp.png"))
                else:
                    fig.savefig(os.path.join(outpath, f"{ztfid}_exp_flextemp.png"))

                print(result)

            return result

    except:
        return {"success": False}
