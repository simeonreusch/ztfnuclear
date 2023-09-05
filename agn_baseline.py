import gc
import glob
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd  # type: ignore
from astropy.time import Time  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import median_abs_deviation  # type: ignore
from ztfnuclear import io, utils

pd.options.mode.chained_assignment = None  # default='warn'


logger = logging.getLogger(__name__)

dcast = {
    "sigma": float,
    "sigma.err": float,
    "ampl": float,
    "ampl.err": float,
    "fval": float,
    "chi2": float,
    "chi2dof": float,
    "humidity": float,
    "obsmjd": float,
    "ccdid": int,
    "amp_id": int,
    "gain": float,
    "readnoi": float,
    "darkcur": float,
    "magzp": float,
    "magzpunc": float,
    "magzprms": float,
    "clrcoeff": float,
    "clrcounc": float,
    "zpclrcov": float,
    "zpmed": float,
    "zpavg": float,
    "zprmsall": float,
    "clrmed": float,
    "clravg": float,
    "clrrms": float,
    "qid": int,
    "rcid": int,
    "seeing": float,
    "airmass": float,
    "nmatches": int,
    "maglim": float,
    "status": int,
    "infobits": int,
    "filterid": int,
    "fieldid": int,
    "moonalt": float,
    "moonillf": float,
    "target_x": float,
    "target_y": float,
    "data_hasnan": bool,
    "pass": int,
    "flag": int,
    "cloudy": int,
    "fcqfid": int,
    "baseline": float,
    "baseline_err_mult": float,
    "n_baseline": int,
    "pre_or_post": int,
    "not_baseline": int,
    "ampl_corr": float,
    "ampel_err_corr": float,
}

ZTF_FILTER_MAP = {"ZTF_g": 1, "ZTF_r": 2, "ZTF_i": 3}


def baseline(
    transient,
    window: str = "10D",
    min_peak_snr: float = 3,
    risetime: float = 80,
    falltime: float = 100,
    excl_poor_conditions: bool = True,
    primary_grid_only: bool = False,
    min_det_per_field_band: int = 10,
    zp_max_deviation_from_median: float = 0.5,
    reference_days_before_peak: Optional[float] = 50,
    pivot_zeropoint: float = 25,
    plot: bool = True,
    plot_suffix: str = "pdf",
) -> pd.DataFrame:
    """
    For each unique baseline combination, estimate and store baseline.
    Partially taken from
    https://github.com/BrightTransientSurvey/ztf_forced_phot/blob/main/bts_phot/calibrate_fps.py.

    risetime (float): days prior to peak to discard from baseline

    falltime ('co'|float): if 'co' this will be estimated from peak mag assuming Co decay, if int this will be taken as direct days.

    primary_grid_only (bool): if this is set to True,
    all observations in the secondary ZTF grid will be
    discarded

    min_det_per_field_band (int): minimum detections required
    per unique combination of field and band

    zp_max_deviation_from_median (float): maximum deviation of the zeropoint from the median zeropoint that is allowed for each obs to survive

    reference_days_before_peak (Opt[float]): number of days the reference images used for a filter/ccd/band-combination have to been made before the estimated peak date (to avoid contamination of the reference images with transient light)

    """
    df = pd.read_csv(transient, comment="#")
    with open(transient, "r") as input_file:
        headerkeys = []
        headervals = []

        for i, line in enumerate(input_file):
            if len(line) >= 300:
                break
            if line == "\n":
                break
            if ",ampl_corr" in line:
                break
            key = line.split(",", 2)[0].split("=")[0].lstrip("#")
            headerkeys.append(key)
            val = line.split(",", 2)[0].split("=")[1][:-1]
            headervals.append(val)

        header = {}
        for i, key in enumerate(headerkeys):
            if headervals[i] == "-":
                returnval = None
            else:
                returnval = headervals[i]
            header.update({key: returnval})

        header["ztfid"] = header.get("name")
        ztfid = header["ztfid"]

    if excl_poor_conditions and "pass" in df.keys():
        df = df[(df["pass"] == 1)]

    if "ampl_corr" in df.keys():
        df.rename(
            columns={"ampl_corr": "ampl", "ampl_err_corr": "ampl.err"}, inplace=True
        )

    if "fieldid" not in df.keys():
        df["fieldid"] = df["fid"]
        df["ccdid"] = df["fid"]
        df["qid"] = df["fid"]
        df["filterid"] = df["fid"]
        reference_days_before_peak = False
        df["filter"] = df["fid"].apply(lambda x: utils.ztf_filterid_to_band(x))

    df["ampl_zp_scale"] = 10 ** ((pivot_zeropoint - df["magzp"]) / 2.5)
    df["ampl"] *= df["ampl_zp_scale"]
    df["ampl.err"] *= df["ampl_zp_scale"]

    logger.info(f"{ztfid}: Transient df has {len(df)} entries.")
    df["fcqfid"] = np.array(
        df.fieldid.values * 10000
        + df.ccdid.values * 100
        + df.qid.values * 10
        + df.filterid.values
    )

    if primary_grid_only:
        for fid in df.fieldid.unique():
            if len(str(fid)) > 3:
                df = df.query("fieldid != @fid").reset_index(drop=True)

    """
    Remove all combinations of fieldid and filterid where
    minimum detection number per field and band are not met
    (min_det_per_field_band, default 10)
    """
    counts = df.groupby(by=["fieldid", "filterid"]).size().reset_index(name="counts")

    for i, row in counts.iterrows():
        if row["counts"] < min_det_per_field_band:
            fieldid = row["fieldid"]
            filterid = row["filterid"]
            added = fieldid + filterid
            df.query("fieldid + filterid != @added", inplace=True)
    df = df.reset_index(drop=True)

    # Cut datapoints for which magzp deviates too much from median magzp
    median_zp = np.median(df.magzp)
    df["zp_median_deviation"] = np.abs(np.log10(median_zp / df.magzp))
    df.query("zp_median_deviation < @zp_max_deviation_from_median", inplace=True)

    unique_fid = np.unique(df.fcqfid.values).astype(int)

    if df.shape[0] == 0:
        logger.warn(f"{ztfid}: No datapoints survived baseline correction.")
        return (df, {})

    # Time index for use for rolling window
    df = df.sort_values("obsmjd")
    obs_jd = Time(df["obsmjd"].values, format="mjd")
    df = df.set_index(pd.to_datetime(obs_jd.datetime))

    # Find time of peak in each field/filter/... combo
    fcqfid_dict: dict[str, dict[str, Any]] = {}
    t_peak_list = []
    for ufid in unique_fid:
        fcqfid_dict[str(ufid)] = {}
        this_fcqfid = np.where(df.fcqfid.values == ufid)

        if (ufid % 10 == 3) or (len(this_fcqfid[0]) < 2):
            continue

        fcqf_df = df.iloc[this_fcqfid].copy()
        # Use the pulls from mean to find largest deviation
        pull_series = fcqf_df.ampl / fcqf_df["ampl.err"]

        roll_med = pull_series.rolling(window, center=True).median().values

        # Only use medians with a min nbr of values (otherwise we get edge results)
        t_max = fcqf_df.obsmjd.values[np.argmax(roll_med)]
        #        flux_max = np.max(roll_avg)
        flux_max = fcqf_df.ampl.values[np.argmax(roll_med)]
        flux_scatt = median_abs_deviation(fcqf_df.ampl.values, scale="normal")
        peak_snr = flux_max / flux_scatt
        if (peak_snr > min_peak_snr) and (ufid < 10000000):
            fcqfid_dict[str(ufid)]["det_sn"] = True
            fcqfid_dict[str(ufid)]["t_max"] = t_max
            fcqfid_dict[str(ufid)]["flux_max"] = flux_max
            t_peak_list.append(t_max)
        else:
            fcqfid_dict[str(ufid)]["det_sn"] = False
            fcqfid_dict[str(ufid)]["t_max"] = t_max
            fcqfid_dict[str(ufid)]["flux_max"] = flux_max
            t_peak_list.append(t_max)

    """
    For all field/ccd/fid combination where we have determined a peak, we can now check if the reference image end date is comfortably prior to the peak and remove this combo if not
    """
    if reference_days_before_peak:
        ufids_to_check = []

        for ufid, res in fcqfid_dict.items():
            if "t_max" in res.keys():
                ufids_to_check.append(ufid)

        if ufids_to_check:
            ref_mjd_dict = get_reference_mjds(fcqfid_list=ufids_to_check)

            for ufid, ref_end_mjd in ref_mjd_dict.items():
                t_max = fcqfid_dict[ufid]["t_max"]

                if (t_max - ref_end_mjd) < reference_days_before_peak:
                    ufid = int(ufid)
                    df.query("fcqfid != @ufid", inplace=True)

    # should we not first convert to a common zeropoint or flux scale (jansky?)
    df["baseline"] = np.zeros_like(df.ampl.values)
    df["baseline_err_mult"] = np.zeros_like(df.ampl.values)
    df["n_baseline"] = np.zeros_like(df.ampl.values).astype(int)
    df["pre_or_post"] = np.zeros_like(df.ampl.values).astype(int)
    df["not_baseline"] = np.zeros_like(df.ampl.values).astype(int)

    if len(t_peak_list) > 0:
        t_peak = np.mean(t_peak_list)
        if len(t_peak_list) > 1 and np.std(t_peak_list, ddof=1) > 20:
            print("Warning! Large scatter in time of maximum")
        fcqfid_dict["t_peak"] = t_peak
        if falltime == "co":
            around_max = np.where(
                (df.obsmjd.values - t_peak > -10) & (df.obsmjd.values - t_peak < 10)
            )
            if len(around_max[0]) > 0:
                diff_flux_around_max = df.ampl.values[around_max]
                mag_min = np.nanmin(
                    df.magzp.values[around_max] - 2.5 * np.log10(diff_flux_around_max)
                )
                # calculate time when SN signal is "gone" via Co56 decay at z ~ 0.09
                t_faded = t_peak + (22.5 - mag_min) / 0.009
            else:
                t_faded = t_peak + 611  # catch strange cases where t_gmax != t_rmax
        elif isinstance(falltime, (float, int)):
            t_faded = t_peak + falltime
        t_risetime = t_peak - risetime
        outside_baseline = np.where(
            (df.obsmjd.values >= t_risetime) & (df.obsmjd.values <= t_faded)
        )
        df.iloc[outside_baseline[0], df.columns.get_loc("not_baseline")] = np.ones(
            len(outside_baseline[0])
        )

        for ufid in unique_fid:
            if ufid % 10 == 4:  # JN: Not sure what this check does
                continue
            else:
                this_fcqfid = np.where(df.fcqfid.values == ufid)
                fcqf_df = df.iloc[this_fcqfid].copy()

                # measure the baseline pre-peak
                pre_bl = np.where((t_peak - fcqf_df.obsmjd.values > 100))
                fcqfid_dict[str(ufid)]["N_pre_peak"] = 0
                if len(pre_bl[0]) > 1:
                    # base_mjd = fcqf_df.obsmjd.values[pre_bl]
                    base_flux = fcqf_df.ampl.values[pre_bl]
                    base_flux_unc = fcqf_df["ampl.err"].values[
                        pre_bl
                    ]  # Would ampl.err work?
                    mask = np.where(
                        np.abs((base_flux - np.median(base_flux)) / base_flux_unc) <= 5
                    )
                    if len(mask[0]) > 1:
                        Cmean = np.average(
                            base_flux[mask], weights=1 / base_flux_unc[mask] ** 2
                        )
                        sum_diff_sq = np.sum(
                            ((base_flux[mask] - Cmean) / (base_flux_unc[mask])) ** 2
                        )
                        chi = 1 / (len(mask[0]) - 1) * sum_diff_sq
                        fcqfid_dict[str(ufid)]["C_pre"] = Cmean
                        fcqfid_dict[str(ufid)]["chi_pre"] = chi
                        fcqfid_dict[str(ufid)]["N_pre_peak"] = len(mask[0])

                # measure the baseline post-peak
                post_bl = np.where((fcqf_df.obsmjd.values > t_faded))
                fcqfid_dict[str(ufid)]["N_post_peak"] = 0
                if len(post_bl[0]) > 1:
                    # local variable 'base_jd' is assigned to but never used
                    # base_jd = fcqf_df.jd.values[post_bl]
                    base_flux = fcqf_df.ampl.values[post_bl]
                    base_flux_unc = fcqf_df["ampl.err"].values[post_bl]
                    mask = np.where(
                        np.abs((base_flux - np.median(base_flux)) / base_flux_unc) <= 5
                    )
                    if len(mask[0]) > 1:
                        Cmean = np.average(
                            base_flux[mask], weights=1 / base_flux_unc[mask] ** 2
                        )
                        sum_diff_sq = np.sum(
                            ((base_flux[mask] - Cmean) / (base_flux_unc[mask])) ** 2
                        )
                        chi = 1 / (len(mask[0]) - 1) * sum_diff_sq
                        fcqfid_dict[str(ufid)]["C_post"] = Cmean
                        fcqfid_dict[str(ufid)]["chi_post"] = chi
                        fcqfid_dict[str(ufid)]["N_post_peak"] = len(mask[0])

                # Decide which baseline to use
                if (fcqfid_dict[str(ufid)]["N_pre_peak"] >= 25) or (
                    (fcqfid_dict[str(ufid)]["N_pre_peak"] > 10)
                    and (fcqfid_dict[str(ufid)]["N_post_peak"] < 25)
                ):
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline")
                    ] = fcqfid_dict[str(ufid)]["C_pre"]
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline_err_mult")
                    ] = np.ones(len(this_fcqfid[0])) * max(
                        np.sqrt(fcqfid_dict[str(ufid)]["chi_pre"]), 1
                    )
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("n_baseline")
                    ] = fcqfid_dict[str(ufid)]["N_pre_peak"]
                    df.iloc[this_fcqfid[0], df.columns.get_loc("pre_or_post")] = -1
                    fcqfid_dict[str(ufid)]["which_baseline"] = "pre"
                elif (fcqfid_dict[str(ufid)]["N_post_peak"] >= 25) or (
                    (fcqfid_dict[str(ufid)]["N_pre_peak"] < 10)
                    and (fcqfid_dict[str(ufid)]["N_post_peak"] >= 25)
                ):
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline")
                    ] = fcqfid_dict[str(ufid)]["C_post"]
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("baseline_err_mult")
                    ] = np.ones(len(this_fcqfid[0])) * max(
                        np.sqrt(fcqfid_dict[str(ufid)]["chi_post"]), 1
                    )
                    df.iloc[
                        this_fcqfid[0], df.columns.get_loc("n_baseline")
                    ] = fcqfid_dict[str(ufid)]["N_post_peak"]
                    df.iloc[this_fcqfid[0], df.columns.get_loc("pre_or_post")] = 1
                    fcqfid_dict[str(ufid)]["which_baseline"] = "post"
                else:
                    fcqfid_dict[str(ufid)]["which_baseline"] = None

    filepath_new = transient_path.parent.parent / "AGN_commented" / transient_path.name

    t_peak_mjd = fcqfid_dict.get("t_peak")
    t_peak_jd = t_peak_mjd + 2400000.5
    header["bts_peak_jd"] = t_peak_jd

    xmatch = pd.read_csv("/Users/simeon/ztfnuclear/sarah_agn_crossmatch_with_z.csv")
    xmatch.query("ztf_id == @ztfid", inplace=True)
    header["sgscore"] = xmatch["sgscore"].values[0]
    header["distnr"] = xmatch["distnr"].values[0]
    header["bts_z"] = xmatch["z"].values[0]
    header["bts_class"] = "AGN"

    print(ztfid)

    with open(filepath_new, "w") as f:
        for i, (key, val) in enumerate(header.items()):
            writestr = f"#{key}={val}\n"
            f.write(writestr)

    with open(filepath_new, "a") as f:
        df.to_csv(f, index=False)


def get_reference_mjds(fcqfid_list: list) -> dict:
    """
    Get list of references from IPAC and return dates for all unique combinations of fieldid, CCD and filter
    """
    from planobs.utils import get_references

    fieldids = list(
        set([int(str(fcqfid)[: len(str(fcqfid)) - 4]) for fcqfid in fcqfid_list])
    )

    references = get_references(fieldids)

    ref_mjd_dict: dict[int, float] = {}

    for fcqfid in fcqfid_list:
        if len(str(fcqfid)) == 7:
            i = 0
        else:
            i = 1

        fieldid = int(str(fcqfid)[: 3 + i])
        ccdid = int(str(fcqfid)[3 + i : 5 + i])
        qid = int(str(fcqfid)[5 + i : 6 + i])
        fid = int(str(fcqfid)[6 + i : 7 + i])
        _ref = references.query(
            "field == @fieldid and ccdid == @ccdid and qid == @qid and fid == @fid"
        )
        endobsdate = _ref.endobsdate.values[0].split("+")[0]
        endobsdate_mjd = float(Time(endobsdate, format="iso").mjd)
        ref_mjd_dict.update({fcqfid: endobsdate_mjd})

    return ref_mjd_dict


basepath = Path("/Users/simeon/ZTFDATA/ztfparsnip/AGN/")

for transient_path in basepath.iterdir():
    # try:
    baseline(transient=transient_path)
    # except:
    # continue
