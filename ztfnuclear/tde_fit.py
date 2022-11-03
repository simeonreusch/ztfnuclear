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
        name: str = "TDE_exp_flextemp",
        version: str = "1.0",
    ) -> None:

        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        self._parameters: np.ndarray = np.array([1.584, 2.278, 3.998, 6.43e-25])

    @staticmethod
    def _wl_to_nu(self, wl: np.ndarray) -> np.ndarray:
        """
        Convert wavelength in Angstrom to frequency in Hz

        """
        nu = 2.9979e18 / wl

        return nu

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

        # exponential_term = np.outer((self.h * self.c), 1 / (wave * self.k * T))
        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)

        bb = bb * 4 * np.pi * u.sr

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
        # plateau_start = self.parameters[5]

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

        nu = self._wl_to_nu(self, wave)

        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        rise_decay = self._gauss_exp(self, phases=phase_iter)

        model_flux = (rise_decay * bb_lam).transpose()

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)
        print(np.max(model_flux_cgi))

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
    ) -> None:

        self.name: str = name
        self.version: str = version
        self._phase: np.ndarray = phase
        self._wave: np.ndarray = wave

        # Some constants
        self.sigma_SB: float = 5.6704e-5  # Stefan-Boltzmann constant (erg/s/cm^2/K^4)
        self.c: float = 2.9979e10  # Speed of light (cm/s)
        self.h: float = 6.6261e-27  # Planck's constant (erg s)
        self.k: float = 1.3806e-16  # Boltzman's constant (erg/K)
        self.h_over_k: float = self.h / self.k

        # Fit parameters
        # defaults - peaktime = 0 (days), rise=0.85 (log days) / decay=1.5 (log days) / T=4.5 (log K) / peakflux = 1e-16 / d_temp = 10 K/day
        # self._parameters: np.ndarray = np.array([1, 1.2, 4.2, 1.3e-24, 1, 300])
        if priors is None:
            self._parameters: np.ndarray = np.array(
                [1.584, 2.278, 3.998, 6.43e-25, 0, 300]
            )
        else:
            self._parameters = priors

        print(self.parameters)

    @staticmethod
    def _wl_to_nu(self, wl: np.ndarray) -> np.ndarray:
        """
        Convert wavelength in Angstrom to frequency in Hz

        """
        nu = 2.9979e18 / wl

        return nu

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

        # exponential_term = np.outer((self.h * self.c), 1 / (wave * self.k * T))
        exponential_term = (
            c.h.value * c.c.value * 1 / np.outer(wave.to(u.m).value, T) / c.k_B.value
        )

        bb = prefactor * 1 / (np.exp(exponential_term) - 1) / u.sr

        # returns spectral radiance: J s-1 sr-1 m-3

        return bb

    @staticmethod
    def _cc_bol_nu(self, nu: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_nu(self, nu, T)
        bb = bb * u.sr

        nu = nu * u.Hz
        T = T * u.K

        sigma_SB = 5.6704e-8 * u.W / u.m**2 / u.K**4

        bol_corr = np.tile(nu, (len(T), 1)).transpose() / (sigma_SB * T**4 / np.pi)
        bol_corr_dimless = np.tile(nu.value, (len(T), 1)).transpose() / (
            sigma_SB.value * T.value**4 / np.pi
        )

        return bb * bol_corr_dimless

    @staticmethod
    def _cc_bol_lam(self, wave: Union[float, np.ndarray], T: np.ndarray):
        bb = self._planck_lam(self, wave, T)
        # wave = wave * u.AA
        # wave = wave.to(u.m)
        # bol_corr = (
        #     (np.pi * c.c.value)
        #     / np.tile(wave.value, (len(T), 1)).transpose()
        #     / (5.6704e-8 * T**4)
        # )
        bb = bb * 4 * np.pi * u.sr

        return bb  # * testcorr  # * bol_corr

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
        # plateau_start = self.parameters[5]

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
            phase, -30, plateau_start
        )  # stop temperature evolution at -30 and +365 days

        t_evo = (10**temp) + (phase_clip * d_temp)

        t_evo_clip = np.clip(
            t_evo, 1e-10, 1e6
        )  # clip temperatures outside 1000 and 100,000 K

        return t_evo_clip

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        # from matplotlib import pyplot as plt

        # fig, ax = plt.subplots(dpi=200)

        # minwave = np.min(wave)
        # maxwave = np.max(wave)

        # wave = np.arange(1000, 30000, 10)
        wave_m = wave * 1e-10

        # for divider in [5.4, 6.0, 6.1, 6.2, 6.3, 6.8]:
        t_evo = self._temp_evolution(phase=phase)

        if np.ndim(phase) == 0:
            phase_iter = np.asarray(phase)
        else:
            phase_iter = phase

        nu = self._wl_to_nu(self, wave)

        # now t is an array, that's why we do all this matrix gymnastics
        # bb_nu = self._cc_bol_nu(self, T=t_evo, nu=nu)
        bb_lam = self._cc_bol_lam(self, T=t_evo, wave=wave)

        # conversion_factor_nu_lam = (nu * u.Hz) ** 2 / c.c

        # print(bb_nu[0, 0])
        # print(bb_lam[0, 0] / conversion_factor_nu_lam[0])
        # quit()

        rise_decay = self._gauss_exp(self, phases=phase_iter)

        # for i, p in enumerate(phase):
        #     if i % 30 == 0:
        #         # print(t_evo[i])

        #         ax.plot(nu, bb_nu[:, i], label=f"{t_evo[i]:.2f}")
        #         # ax.plot(wave_m, bb_lam[:, i], label=f"{t_evo[i]:.2f}")

        # plt.legend()
        # plt.savefig(f"test_{t_evo[0]:.0f}_{np.mean(wave):.2f}.png")
        # plt.close()

        # rise_decay = np.tile(rise_decay, (len(nu), 1))

        model_flux = (rise_decay * bb_lam).transpose()
        # model_flux = (rise_decay * bb_nu).transpose() * conversion_factor_nu_lam

        model_flux_cgi = model_flux.to(u.erg / u.s / u.cm**2 / u.AA)
        print(np.max(model_flux_cgi))

        return model_flux_cgi


class TDESource_pl(sncosmo.Source):

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
        name: str = "TDE_pl",
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
        # defaults - peaktime = 0, rise=0.85 / alpha=-1.5 / T=4.5 / peakflux = 32 / powerlaw normalization=1.5
        self._parameters: np.ndarray = np.array([0.85, -1.5, 4.5, 32, 1.5])

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
    def _gauss_exp_pl(self, phases: np.ndarray) -> np.ndarray:

        risetime = self._parameters[0]
        alpha = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]
        normalization = 10 ** self._parameters[4]

        a1 = 10**peakflux
        b1 = 10**risetime  # rise rate
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay = phases[(phases > 0)]

        vals_rise = a1 * self._gauss(phases_rise, b1)
        vals_decay = a2 * ((phases_decay + normalization) / normalization) ** alpha

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
        input:
         - L: luminosity in erg/s
         - cm: luminosity distance in cm
        """
        return L / (nu * 4 * np.pi * cm**2) * 1e23

    def _temp_evolution(self, phase: np.ndarray) -> np.ndarray:
        """
        Create an array with a linear temperature evolution
        """
        temp = self._parameters[2]
        d_temp = self.parameters[4]

    def _flux(self, phase: np.ndarray, wave: np.ndarray) -> np.ndarray:
        """
        Calculate the model flux for given wavelengths and phases
        """
        temp = self._parameters[2]

        if np.ndim(phase) == 0:
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

        models = self._gauss_exp_pl(self, phases=phase_iter)

        corrected_models = np.outer(models, kc) / cc_bol * cc
        model_flux = corrected_models * luminosity

        return model_flux


class TDESource_pl_plateau(sncosmo.Source):

    _param_names: list = [
        "risetime",
        "alpha",
        "temperature",
        "amplitude",
        "normalization",
        "plateaustart",
        "plateauend",
    ]
    param_names_latex: list = [
        "Rise Time",
        "Alpha",
        "Temperature~[log10~K]",
        "Amplitude",
        "normalization",
        "plateau start",
        "plateau end",
    ]

    def __init__(
        self,
        phase: np.ndarray,
        wave: np.ndarray,
        name: str = "TDE_pl",
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
        # defaults - peaktime = 0, rise=0.85 / alpha=-1.5 / T=4.5 / peakflux = 32 / powerlaw normalization=1.5 / plateau start (150 days after peak) / plateau end (151) days after peak
        self._parameters: np.ndarray = np.array([0.85, -1.5, 4.5, 32, 1.5, 30, 40])

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
    def _gauss_exp_pl(self, phases: np.ndarray) -> np.ndarray:

        risetime = self._parameters[0]
        alpha = self._parameters[1]
        temp = self._parameters[2]
        peakflux = self._parameters[3]
        normalization = 10 ** self._parameters[4]
        plateau_start = self._parameters[5]
        plateau_end = self._parameters[6]

        if plateau_end < plateau_start:
            temp = plateau_end
            plateau_end = plateau_start
            plateau_start = temp

        a1 = 10**peakflux
        b1 = 10**risetime  # rise rate
        a2 = a1 * self._gauss(0, b1)

        phases_rise = phases[(phases <= 0)]
        phases_decay_full = phases[(phases > 0)]

        phases_decay1 = phases_decay_full[phases_decay_full < plateau_start]
        phases_plateau_untrimmed = phases_decay_full[phases_decay_full >= plateau_start]
        phases_plateau = phases_plateau_untrimmed[
            phases_plateau_untrimmed < plateau_end
        ]
        phases_decay2 = phases_decay_full[phases_decay_full >= plateau_end]

        plateau_length = plateau_end - plateau_start

        vals_rise = a1 * self._gauss(phases_rise, b1)
        vals_decay1 = a2 * ((phases_decay1 + normalization) / normalization) ** alpha

        if len(phases_plateau) > 0:
            vals_plateau = [float(vals_decay1[-1:])] * len(phases_plateau)
        else:
            vals_plateau = []

        vals_decay2 = (
            a2
            * (((phases_decay2 - plateau_length) + normalization) / normalization)
            ** alpha
        )

        returnvals = np.concatenate((vals_rise, vals_decay1, vals_plateau, vals_decay2))

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
        # self._parameters: np.ndarray = np.array([0.85, -1.5, 4.5, 32, 1.5, 150, 151])

        temp = self._parameters[2]

        if np.ndim(phase) == 0:
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

        models = self._gauss_exp_pl(self, phases=phase_iter)

        corrected_models = np.outer(models, kc) / cc_bol * cc
        model_flux = corrected_models * luminosity

        return model_flux


def fit(
    df: pd.DataFrame,
    ra: float,
    dec: float,
    baseline_info: dict,
    powerlaw: bool = True,
    plateau: bool = True,
    ztfid: str = None,
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

    obsmjd = df.obsmjd.values

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
    # phase = np.linspace(-50, 1000, 10)
    wave = np.linspace(1000, 10000, 5)

    tde_source = TDESource_exp_simple(
        phase,
        wave,
        name="tde",
    )

    dust = sncosmo.models.CCM89Dust()
    dustmap = SFDMap()
    transient_mwebv = dustmap.ebv((ra, dec))

    sncosmo_model = sncosmo.Model(
        source=tde_source, effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )

    sncosmo_model.set(mwebv=transient_mwebv)
    # sncosmo_model.set(z=0.0222)

    fit_params = copy.deepcopy(sncosmo_model.param_names)
    fit_params.remove("mwebv")
    fit_params.remove("mwr_v")
    fit_params.remove("z")  # let's not fit z here

    default_param_vals = sncosmo_model.parameters

    result, fitted_model = sncosmo.fit_lc(
        phot_tab,
        sncosmo_model,
        fit_params,
        bounds={
            "t0": [t_peak - 30, t_peak + 30],
            "temperature": [3.5, 5],
            "risetime": [0, 5],
            "decaytime": [0, 5],
        },
    )
    params = result["parameters"]
    t0 = params[1]
    priors = params[2:-2]
    priors = np.append(priors, [0, 365])

    tde_source = TDESource_exp_flextemp(
        phase,
        wave,
        name="tde",
        priors=None,
    )

    sncosmo_model = sncosmo.Model(
        source=tde_source, effects=[dust], effect_names=["mw"], effect_frames=["obs"]
    )

    sncosmo_model.set(mwebv=transient_mwebv)
    # sncosmo_model.set(z=0.0222)

    fit_params = copy.deepcopy(sncosmo_model.param_names)
    fit_params.remove("mwebv")
    fit_params.remove("mwr_v")
    fit_params.remove("z")  # let's not fit z here

    default_param_vals = sncosmo_model.parameters

    result, fitted_model = sncosmo.fit_lc(
        phot_tab,
        sncosmo_model,
        fit_params,
        bounds={
            "t0": [t0 - 30, t0 + 30],
            "temperature": [3.5, 5],
            "risetime": [0.1, 5],
            "decaytime": [0, 5],
            "d_temp": [-1000, 1000],
            "plateaustart": [100, 1200],
        },
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

    # fig = sncosmo.plot_lc(data=phot_tab, model=fitted_model, zpsys="ab", zp=25)

    # if powerlaw:
    #     if plateau:
    #         fig.savefig("test_pl_plateau.png")
    #     else:
    #         fig.savefig("test_pl.png")
    # else:
    #     fig.savefig(f"/Users/simeon/Desktop/flextemp_test/diagnostic/{ztfid}.png")

    print(result)
    return result

    # except:
    #     return {"success": False}
