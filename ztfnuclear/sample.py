#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import base64
import datetime
import logging
import os
import pickle
import time
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from ztfnuclear import baseline, io, utils
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.fritz import FritzAPI

logger = logging.getLogger(__name__)

meta = MetadataDB()
meta_bts = MetadataDB(sampletype="bts")
meta_train = MetadataDB(sampletype="train")
info_db = SampleInfo()
info_db_bts = SampleInfo(sampletype="bts")


class NuclearSample(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(self, sampletype="nuclear"):
        super(NuclearSample, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing sample")
        self.sampletype = sampletype
        io.download_if_neccessary(sampletype=self.sampletype)

        self.get_all_ztfids()
        self.location()
        self.meta = MetadataDB(sampletype=self.sampletype)
        self.info_db = SampleInfo(sampletype=self.sampletype)

        if self.sampletype in ["nuclear", "bts"]:
            if self.info_db.read().get("flaring") is None:
                logger.info("Flaring info not found, ingesting")
                self.get_flaring()

        if self.sampletype == "train":
            db_check = self.meta.get_statistics()
            if db_check.get("count", 0) == 0:
                self.populate_db_from_headers()

        if self.sampletype == "nuclear":
            db_check = self.meta.get_statistics()

            if not db_check["has_ra"]:
                self.populate_db_from_csv(filepath=io.LOCALSOURCE_location)

            if not db_check["has_peak_dates"]:
                self.populate_db_from_csv(
                    filepath=io.LOCALSOURCE_peak_dates, name="peak_dates"
                )
            if not db_check["has_peak_mags"]:
                self.populate_db_from_csv(
                    filepath=io.LOCALSOURCE_peak_mags, name="peak_mags"
                )
            if not db_check["has_distnr"]:
                self.populate_db_from_csv(filepath=io.LOCALSOURCE_distnr, name="distnr")
            if not db_check["has_salt"]:
                saltfit_res = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_fitres, "saltfit.json"),
                    parameter_name="salt",
                )
                self.populate_db_from_dict(data=saltfit_res)

            if not db_check["has_salt_loose_bl"]:
                saltfit_res = io.parse_ampel_json(
                    filepath=os.path.join(
                        io.LOCALSOURCE_fitres, "saltfit_loose_bl.json"
                    ),
                    parameter_name="salt_loose_bl",
                )
                self.populate_db_from_dict(data=saltfit_res)

            if not db_check["has_ampel_z"]:
                ampelz = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_ampelz),
                    parameter_name="ampel_z",
                )
                self.populate_db_from_dict(data=ampelz)

            if not db_check["has_wise_bayesian"]:
                wise_bayesian = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_WISE_bayesian),
                    parameter_name="wise_bayesian",
                    sampletype="nuclear",
                )
                self.populate_db_from_dict(data=wise_bayesian)

            if not db_check["has_wise_dust"]:
                wise_dust = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_WISE_dust),
                    parameter_name="wise_dust",
                    sampletype="nuclear",
                )
                self.populate_db_from_dict(data=wise_dust)

            if not db_check["has_wise_lc"]:
                wise_lcs = io.parse_json(filepath=io.LOCALSOURCE_timewise)
                self.populate_db_from_dict(data=wise_lcs)

        elif self.sampletype == "bts":
            db_check = self.meta.get_statistics()
            if not db_check["has_wise_bayesian"]:
                wise_bayesian = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_WISE_bayesian_bts),
                    parameter_name="wise_bayesian",
                    sampletype="bts",
                )
                self.populate_db_from_dict(data=wise_bayesian)
            if not db_check["has_wise_dust"]:
                wise_dust = io.parse_ampel_json(
                    filepath=os.path.join(io.LOCALSOURCE_WISE_dust_bts),
                    parameter_name="wise_dust",
                    sampletype="bts",
                )
                self.populate_db_from_dict(data=wise_dust)
            if not db_check["has_peak_mags"]:
                self.populate_db_from_csv(
                    filepath=io.LOCALSOURCE_bts_peak_mags, name="peak_mags"
                )
            if not db_check["has_peak_dates"]:
                self.populate_db_from_csv(
                    filepath=io.LOCALSOURCE_bts_peak_dates, name="peak_dates"
                )
            if not db_check["has_wise_lc"]:
                wise_lcs = io.parse_json(filepath=io.LOCALSOURCE_timewise_bts)
                self.populate_db_from_dict(data=wise_lcs)

            if not db_check["has_distnr"]:
                self.populate_db_from_csv(
                    filepath=io.LOCALSOURCE_bts_distnr, name="distnr"
                )
            if db_check["count"] == 0:
                self.populate_db_from_csv(filepath=io.LOCALSOURCE_bts_info)
            db_check = self.meta.get_statistics()

        elif self.sampletype == "train":
            # if not db_check["has_salt"]:
            #     saltfit_res = io.parse_ampel_json(
            #         filepath=os.path.join(io.LOCALSOURCE_train_fitres, "saltfit.json"),
            #         parameter_name="salt",
            #         sampletype=self.sampletype,
            #     )
            #     self.populate_db_from_dict(data=saltfit_res)

            # if not db_check["has_tdefit_exp"]:
            #     tdefit_path = os.path.join(
            #         io.LOCALSOURCE_train_fitres, "tde_fit_exp.json"
            #     )
            #     self.logger.info(f"Importing TDE fit results from {tdefit_path}")
            #     self.meta.key_update_from_json(
            #         json_path=tdefit_path, mongo_key="tde_fit_exp"
            #     )

            # if not db_check["has_crossmatch"]:
            #     self.get_parent_crossmatch(sampletype=self.sampletype)

            # if not db_check["has_ztf_bayesian"]:
            #     bayesian_res = io.parse_ampel_json(
            #         filepath=os.path.join(io.SRC_train, "ztf_bayesian.json"),
            #         parameter_name="ztf_bayesian",
            #         sampletype=self.sampletype,
            #     )
            #     self.populate_db_from_dict(data=bayesian_res)

            if not db_check["has_distnr_scaled"]:
                self.get_scaled_distnr(sampletype=self.sampletype)

            if not db_check["has_peakmag_scaled"]:
                self.get_scaled_peakmag(sampletype=self.sampletype)

        if self.sampletype == "nuclear":
            assert db_check["count"] == 11687
        elif self.sampletype == "bts":
            assert db_check["count"] == 7131 or db_check["count"] == 7130

    def get_scaled_distnr(self, sampletype):
        """
        Convert the core distance at a certain redshift
        to core distance at another redshift
        """
        from astropy import units as u
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        self.logger.info("Updating train metadata DB with z-corrected distnr values")

        assert self.sampletype == "train"

        for t in self.get_transients():
            median_distnr = t.meta.get("median_distnr")
            if median_distnr is not None:
                distnr = float(median_distnr)
            else:
                distnr = float(t.meta["distnr"])
            if len(t.ztfid.split("_")) > 1:
                distnr_deg = distnr / 3600 * u.deg
                z = float(t.meta["z"])
                parent_z = float(t.meta["bts_z"])
                parent_lumidist = cosmo.luminosity_distance(parent_z)
                new_lumidist = cosmo.luminosity_distance(z)
                theta = distnr_deg.to(u.radian).value
                D_A = parent_lumidist / (1 + parent_z) ** 2
                dist = theta * D_A

                theta_new = dist * (1 + z) ** 2 / new_lumidist
                distnr_new = ((theta_new * u.rad).to(u.deg) * 3600).value
            else:
                distnr_new = distnr

            data = {"distnr": distnr_new}

            self.meta.update_transient(ztfid=t.ztfid, data=data)

    def get_scaled_peakmag(self, sampletype):
        """
        Convert the peak apparent mag at a certain redshift to one at another redshift
        """
        from astropy import units as u
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.logger.info("Updating train metadata DB with z-corrected peakmag values")

        assert self.sampletype == "train"

        for t in self.get_transients():
            peakmag_try = t.meta.get("bts_peak_mag")
            if peakmag_try is not None:
                peakmag_parent = float(peakmag_try)
            else:
                peakjd = float(t.meta.get("bts_peak_jd"))
                peak_mjd = peakjd - 2400000.5
                parent_ztfid = t.meta["parent_ztfid"]

                parent = Transient(ztfid=parent_ztfid, sampletype="train")
                parentdf = parent.df.query("band == 'ztfg'")
                if len(parentdf) == 0:
                    parentdf = parent.df.query("band == 'ztfr'")
                if len(parentdf) == 0:
                    parentdf = parent.df.query("band == 'ztfi'")

                dist_to_peak = np.abs(peak_mjd - parentdf["obsmjd"])

                parentdf["peakdist"] = dist_to_peak
                peakmag_parent = parentdf["magpsf"].values[np.argmin(dist_to_peak)]
                newdata = {"_id": t.ztfid, "bts_peak_mag": peakmag_parent}
                t.update(data=newdata)

            if len(t.ztfid.split("_")) > 1:
                z = float(t.meta["z"])
                parent_z = float(t.meta["bts_z"])
                lumidist_pc = cosmo.luminosity_distance(parent_z).to(u.pc).value
                peak_absmag_parent = peakmag_parent - 5 * (np.log10(lumidist_pc) - 1)
                lumidist_new_pc = cosmo.luminosity_distance(z).to(u.pc).value
                peakmag_new = peak_absmag_parent + 5 * np.log10(lumidist_new_pc) - 5
            else:
                peakmag_new = peakmag_parent

            data = {"peakmag": peakmag_new}
            self.meta.update_transient(ztfid=t.ztfid, data=data)

    def get_parent_crossmatch(self, sampletype):
        """
        Add the parent lightcurve crossmatch info
        """
        assert self.sampletype == "train"

        self.logger.info("Updating train metadata DB with parent crossmatch info")

        for t in self.get_transients():
            parent_ztfid = t.meta["parent_ztfid"]
            try:
                test = Transient(ztfid=parent_ztfid, sampletype="bts")
            except:
                test = Transient(ztfid=parent_ztfid, sampletype="nuclear")
            crossmatch = test.meta["crossmatch"]
            data = {"crossmatch": crossmatch}

            self.meta.update_transient(ztfid=t.ztfid, data=data)

    def get_flaring(self, after_optical_peak: bool = True):
        """
        Get all the IR flares after optical peak
        """
        from ztfnuclear.utils import mjd_to_jd

        flaring_ztfids = []
        logger.info("Ingesting WISE flare data")
        for t in tqdm(self.get_transients(), total=len(self.ztfids)):
            flaring_status = None
            flaring_status = t.meta.get("WISE_dust", {}).get("dust", {}).get("status")
            if flaring_status:
                if flaring_status != "No further investigation":
                    bayes = t.meta.get("WISE_bayesian", {}).get("bayesian", {})
                    peaks = []
                    for wise_filter in ["Wise_W1", "Wise_W2"]:
                        jd_excess_regions = bayes.get(wise_filter, {}).get(
                            "jd_excess_regions"
                        )
                        max_mag_excess_region = bayes.get(wise_filter, {}).get(
                            "max_mag_excess_region"
                        )
                        peakindex = np.argmax(max_mag_excess_region)
                        start_peak_region = min(jd_excess_regions[peakindex])
                        peaks.append(start_peak_region)
                    wise_peak = np.max(peaks)
                    if after_optical_peak:
                        if t.meta.get("peak_dates") is not None:
                            ztf_peak_mjd = min(list(t.meta.get("peak_dates").values()))
                            ztf_peak_jd = mjd_to_jd(ztf_peak_mjd)
                            # we allow for 50 days before optical peak to have some wiggle room
                            if wise_peak > (ztf_peak_jd - 100):
                                flaring_ztfids.append(t.ztfid)
                    else:
                        flaring_ztfids.append(t.ztfid)

        self.logger.info(f"Found {len(flaring_ztfids)} flaring transients")
        self.info_db.ingest_ztfid_collection(flaring_ztfids, "flaring")

    def get_transient(self, ztfid: str):
        df = io.get_ztfid_dataframe(ztfid=ztfid)
        header = io.get_ztfid_header(ztfid=ztfid)
        return header, df

    def transient(self, ztfid: str):
        return Transient(ztfid=ztfid, sampletype=self.sampletype)

    def get_all_ztfids(self):
        all_ztfids = io.get_all_ztfids(sampletype=self.sampletype)
        self.ztfids = all_ztfids

    def location(self):
        df = io.get_locations(sampletype=self.sampletype)
        self.location = df

    def create_baseline(self):
        self.logger.info("Creating baseline corrections for the full sample")
        success = []
        failed = []
        for ztfid in tqdm(self.ztfids):
            t = Transient(ztfid, sampletype=self.sampletype)
            try:
                t.recreate_baseline()
                success.append(ztfid)
            except:
                self.logger.warn(f"{ztfid} failed")
                failed.append(ztfid)

        self.logger.info(f"{len(failed)} objects failed the baseline creation:")
        self.logger.info(failed)

    def populate_db_from_csv(self, filepath, name=None):
        self.logger.info(f"Populating the database from a csv file ({filepath})")
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath, comment="#", index_col=0)
            ztfids = []
            data = []
            for s in df.iterrows():
                ztfid = s[0]
                if ztfid in self.ztfids:
                    ztfids.append(ztfid)
                    if name:
                        write_dict = {name: s[1].to_dict()}
                    else:
                        write_dict = s[1].to_dict()
                    data.append(write_dict)

            self.meta.update_many(ztfids=ztfids, data=data)
            self.logger.info(f"Wrote {len(ztfids)} entries to db")
        else:
            raise ValueError(f"File {filepath} does not exist")

    def populate_db_from_headers(self):
        """
        Read the lightcurve headers and populate the DB from these
        """
        for ztfid in tqdm(self.ztfids):
            header = io.get_ztfid_header(ztfid=ztfid, sampletype=self.sampletype)
            header["RA"] = header.pop("ra")
            header["Dec"] = header.pop("dec")
            data = {ztfid: header}
            self.populate_db_from_dict(data=data)

    def populate_db_from_dict(self, data: dict):
        """
        Use a dict of the form {ztfid: data} to update the Mongo DB
        """
        self.logger.info(
            f"Populating the {self.meta.coll.name} database from a dictionary"
        )

        ztfids = data.keys()
        data_list = []
        for ztfid in ztfids:
            data_list.append(data[ztfid])

        if len(data_list) > 0:
            if list(data_list[0].keys())[0] == "WISE_lc_by_pos":
                data_reformatted = []
                for entry in data_list:
                    entry["WISE_lc"] = entry.pop("WISE_lc_by_pos")
                    data_reformatted.append(entry)
                data_list = data_reformatted

        self.meta.update_many(ztfids=ztfids, data=data_list)

    def populate_db_from_df(self, df: pd.DataFrame):
        """
        Use a pandas dataframe to update the MongoDB
        """
        self.logger.info("Populating the database from a pandas dataframe")
        self.logger.debug(df)

        ztfids = df.index.values
        data_dict = df.to_dict(orient="index")
        data_list = []
        for ztfid in ztfids:
            data_list.append(data_dict[ztfid])
        self.meta.update_many(ztfids=ztfids, data=data_list)

    def populate_db_from_ampel_json(self, json_path: str):
        """
        Use an AMPEL json export to populate the sample db
        json filename must be equal to parameter name!
        """
        param_name = os.path.basename(json_path).split(".")[0]

        self.logger.info(
            f"Populating the database from an AMPEL json; parameter name: {param_name}"
        )

        json_content = io.parse_ampel_json(
            filepath=json_path,
            parameter_name=param_name,
        )

        self.populate_db_from_dict(data=json_content)

    def crossmatch(self, startindex: int = 0):
        """Crossmatch the full sample"""
        self.logger.info("Crossmatching the full sample")

        for i, ztfid in tqdm(
            enumerate(self.ztfids[startindex:]), total=len(self.ztfids[startindex:])
        ):
            self.logger.debug(f"{ztfid}: Crossmatching")
            self.logger.debug(f"Transient {i+startindex} of {len(self.ztfids)}")
            t = self.transient(ztfid)
            t.crossmatch()
        info = SampleInfo()
        date_now = datetime.datetime.now().replace(microsecond=0)
        info.update(data={"crossmatch_info": {"crossmatch": True, "date": date_now}})

    def fritz(self, startindex: int = 0):
        """
        Query Fritz for the full sample
        """
        self.logger.info("Obtaining metadata on full sample from Fritz")
        for i, ztfid in tqdm(
            enumerate(self.ztfids[startindex:]), total=len(self.ztfids[startindex:])
        ):
            self.logger.debug(f"{ztfid}: Querying Fritz")
            self.logger.debug(f"Transient {i+startindex} of {len(self.ztfids)}")
            t = Transient(ztfid)
            t.fritz()
        info = SampleInfo()
        date_now = datetime.datetime.now().replace(microsecond=0)
        info.update(data={"fritz_info": {"fritz": True, "date": date_now}})

    def irsa(self, startindex: int = 0):
        """
        Obtain non-difference photometry lightcurves for full sample
        """
        self.logger.info("Obtaining IRSA lightcurves for full sample")

        for i, ztfid in tqdm(
            enumerate(self.ztfids[startindex:]), total=len(self.ztfids[startindex:])
        ):
            self.logger.debug(f"{ztfid}: Obtaining IRSA lc")
            t = Transient(ztfid)
            t.irsa()

    def generate_thumbnails(self, startindex: int = 0):
        """
        Generate thumbnail lightcurve plots for all transients
        """
        self.logger.info("Generating thumbnails for full sample")

        for i, ztfid in tqdm(
            enumerate(self.ztfids[startindex:]), total=len(self.ztfids[startindex:])
        ):
            t = Transient(ztfid)
            t.plot(plot_png=True, thumbnail=True)

    def get_ratings(self, username: str = None, select="all") -> dict:
        """
        Get all ratings for the sample (for all or a given user)
        """
        ratings = self.meta.get_rating_overview()

        rating_to_value = {
            "interesting": 3,
            "maybe": 2,
            "boring": 1,
        }

        value_to_rating = {v: k for k, v in rating_to_value.items()}

        returndict = {}

        if select == "all" and username is None:
            returndict = ratings

        elif select == "all" and username is not None:
            for k, v in ratings.items():
                if v is not None:
                    for user in v.keys():
                        if user == username:
                            returndict.update({k: v})

        elif username is not None:
            for k, v in ratings.items():
                if v is not None:
                    for user in v.keys():
                        if user == username:
                            if v[username] == rating_to_value[select]:
                                returndict.update({k: v})

        else:
            for k, v in ratings.items():
                if v is not None:
                    for user in v.keys():
                        if v[user] == rating_to_value[select]:
                            returndict.update({k: v})

        return returndict

    def next_transient(self, ztfid: str, flaring: bool = False):
        """
        Get the transient following the current one in the sample
        """
        if flaring:
            flaring_ztfids = self.info_db.read()["flaring"]
            idx = flaring_ztfids.index(ztfid)

            if idx == len(flaring_ztfids) - 1:
                return flaring_ztfids[idx]
            return flaring_ztfids[idx + 1]

        else:
            idx = self.ztfids.index(ztfid)
            if idx == len(self.ztfids) - 1:
                return self.ztfids[idx]
            return self.ztfids[idx + 1]

    def previous_transient(self, ztfid: str, flaring: bool = False):
        """
        Get the transient following the current one in the sample
        """
        if flaring:
            flaring_ztfids = self.info_db.read()["flaring"]
            idx = flaring_ztfids.index(ztfid)
            return flaring_ztfids[idx - 1]

        else:
            idx = self.ztfids.index(ztfid)
            return self.ztfids[idx - 1]

    def get_transients(
        self,
        start: int | None = 0,
        end: int | None = None,
        ztfids: List[str] | None = None,
    ):
        """
        Loop over all transients in sample (or over all ztfids if given) and return a Transient Object
        """
        if not end:
            if ztfids is None:
                end = len(self.ztfids)
            else:
                end = len(ztfids)

        if ztfids is None:
            for ztfid in tqdm(
                self.ztfids[start:end], total=len(self.ztfids[start:end])
            ):
                t = Transient(ztfid, sampletype=self.sampletype)
                yield t

        else:
            for ztfid in ztfids[start:end]:
                t = Transient(ztfid, sampletype=self.sampletype)
                yield t

    def get_transients_pickled(self, flaring_only: bool = False):
        """
        Read the pickled transient overview and return from that (for webpage performance reasons)
        """
        if flaring_only:
            if not os.path.isfile(io.LOCALSOURCE_pickle_flaring):
                raise FileExistsError(
                    "You have to run self.generate_overview_pickled first"
                )
        else:
            if not os.path.isfile(io.LOCALSOURCE_pickle):
                raise FileExistsError(
                    "You have to run self.generate_overview_pickled first"
                )

        if flaring_only:
            with open(io.LOCALSOURCE_pickle_flaring, "rb") as f:
                transients = pickle.load(f)
        else:
            with open(io.LOCALSOURCE_pickle, "rb") as f:
                transients = pickle.load(f)

        for t in transients:
            yield t

    def get_flaring_transients(self, n: Optional[int] = None):
        """
        Loop over all infrared flaring transients in sample and return a Transient object
        """
        flaring_ztfids = self.info_db.read()["flaring"]["ztfids"]

        if not n:
            n = len(flaring_ztfids)

        for ztfid in flaring_ztfids[:n]:
            t = Transient(ztfid)
            yield t

    def get_transients_subset_chisq(
        self,
        fitname: str,
        max_red_chisq: float,
        n: Optional[int] = None,
    ):
        """
        Loop over all transients in sample and return those that match a maximum reduced chisquare from one of the fit distributions
        """

        selected_ztfids = []

        meta = MetadataDB(sampletype=self.sampletype)
        db_res = meta.read_parameters(params=["_id", fitname])
        fitres_all = db_res[fitname]
        ztfids_all = db_res["_id"]

        for i, ztfid in enumerate(ztfids_all):
            fitres = fitres_all[i]
            if fitres != "failure" and fitres != None:
                chisq = float(fitres["chisq"])
                ndof = float(fitres["ndof"])
                red_chisq = chisq / ndof
                if red_chisq <= max_red_chisq:
                    selected_ztfids.append(ztfid)

        if n is None:
            n = len(selected_ztfids)

        for sel_ztfid in selected_ztfids[:n]:
            t = Transient(sel_ztfid)
            yield

    def get_transients_subset_chisq_list(
        self,
        fitname: str,
        max_red_chisq: float,
        n: Optional[int] = None,
    ):
        """
        Loop over all transients in sample and return a ztfid-list of those that match a maximum reduced chisquare from one of the fit distributions
        """

        selected_ztfids = []

        meta = MetadataDB(sampletype=self.sampletype)
        db_res = meta.read_parameters(params=["_id", fitname])
        fitres_all = db_res[fitname]
        ztfids_all = db_res["_id"]

        for i, ztfid in enumerate(ztfids_all):
            fitres = fitres_all[i]
            if fitres != "failure" and fitres != None:
                chisq = float(fitres["chisq"])
                ndof = float(fitres["ndof"])
                red_chisq = chisq / ndof
                if red_chisq <= max_red_chisq:
                    selected_ztfids.append(ztfid)

        if n is None:
            n = len(selected_ztfids)

        return selected_ztfids[:n]

    def generate_ir_flare_sample(self):
        """
        Create a list of transients that match the WISE IR flare selection and write them to the info db
        """
        meta = MetadataDB(sampletype=self.sampletype)
        db_res = meta.read_parameters(
            params=[
                "_id",
                "WISE_bayesian",
            ]
        )

        final_ztfids = []

        for i, entry in enumerate(db_res["WISE_bayesian"]):
            ztfid = db_res["_id"][i]

            if entry is not None:
                if "bayesian" in entry.keys():
                    if entry["bayesian"] is not None:
                        if "start_excess" in entry["bayesian"].keys():
                            start_excess = entry["bayesian"]["start_excess"]
                    else:
                        start_excess = None
                else:
                    start_excess = None

                if "dustecho" in entry.keys():
                    if entry["dustecho"] is not None:
                        if "status" in entry["dustecho"]:
                            status = entry["dustecho"]["status"]
                        else:
                            status = None
                    else:
                        status = None
                else:
                    status = None

                if start_excess != None and status != "No further investigation":
                    if start_excess >= 2458239.50000:
                        final_ztfids.append(ztfid)

        self.info_db.update(data={"flaring": {"ztfids": final_ztfids}})

    def generate_overview_pickled(self, flaring_only: bool = False):
        """
        Pickle all transients with certain properties (to make webpage fast)
        """
        if flaring_only:
            transients = self.get_flaring_transients()
        else:
            transients = self.get_transients()

        if flaring_only:
            self.logger.info("Pickling the flaring transient overview")
        else:
            self.logger.info("Pickling the transient overview")

        transients_pickled = []

        if flaring_only:
            flaring_ztfids = self.info_db.read()["flaring"]["ztfids"]
            length = len(flaring_ztfids)
        else:
            length = len(self.ztfids)

        for t in tqdm(transients, total=length):
            t.thumbnail
            t.meta
            t.fritz_class
            t.z
            t.z_dist
            t.tns_class
            t.crossmatch_info
            t.tde_res_exp
            t.salt_res
            transients_pickled.append(t)

        if flaring_only:
            with open(io.LOCALSOURCE_pickle_flaring, "wb") as f:
                pickle.dump(transients_pickled, f)
        else:
            with open(io.LOCALSOURCE_pickle, "wb") as f:
                pickle.dump(transients_pickled, f)

    def get_gold_transients(self):
        """
        get gold sample
        """

        golden_sample_ztfids = []

        # get all transients rated as interesting by at least one person
        interesting_ztfids = self.get_ratings(select="interesting")

        # at least two persons must have rated the transient as interesting
        for k, v in interesting_ztfids.items():
            if len(v) > 1:
                rated_interesting = 0
                for entry in v.keys():
                    if v[entry] == 3:
                        rated_interesting += 1

                if rated_interesting >= 2:
                    golden_sample_ztfids.append(k)

        self.logger.info(
            f"Golden sample consists of {len(golden_sample_ztfids)} transients at the moment"
        )

        for ztfid in golden_sample_ztfids:
            t = Transient(ztfid, sampletype=self.sampletype)
            yield t


class Transient(object):
    """
    This class contains all info for a given transient
    """

    def __init__(self, ztfid: str, sampletype: str = "nuclear"):
        super(Transient, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.ztfid = ztfid

        self.sampletype = sampletype

        if self.sampletype == "nuclear":
            transient_info = meta.read_transient(self.ztfid)
        elif self.sampletype == "bts":
            transient_info = meta_bts.read_transient(self.ztfid)
        elif self.sampletype == "train":
            transient_info = meta_train.read_transient(self.ztfid)

        if transient_info is None:
            raise ValueError(f"{ztfid} is not in {sampletype} sample")

        self.ra = float(transient_info["RA"])
        self.dec = float(transient_info["Dec"])

        self.location = {"RA": self.ra, "Dec": self.dec}

    @cached_property
    def header(self) -> dict:
        header = io.get_ztfid_header(ztfid=self.ztfid, sampletype=self.sampletype)
        return header

    @cached_property
    def df(self) -> pd.DataFrame:
        df = io.get_ztfid_dataframe(ztfid=self.ztfid, sampletype=self.sampletype)
        return df

    @cached_property
    def sample_ids(self) -> List[str]:
        ids = NuclearSample(sampletype=self.sampletype).ztfids
        return ids

    @cached_property
    def peak_mag(self) -> Tuple[float | None, str | None]:
        returnvals = None, None
        peak_mags_dict = self.meta.get("peak_mags")
        if peak_mags_dict is not None:
            peak_mag_band = min(peak_mags_dict, key=peak_mags_dict.get)
            if not np.isnan(peak_mags_dict[peak_mag_band]):
                returnvals = peak_mags_dict[peak_mag_band], peak_mag_band

        return returnvals

    @cached_property
    def peak_date(self) -> str | None:
        peak_date = None
        peak_dates_dict = self.meta.get("peak_dates")

        peak_mag, peak_band = self.peak_mag
        if peak_band is not None:
            peak_date = peak_dates_dict[peak_band]

        return peak_date

    @cached_property
    def baseline(self) -> Optional[pd.DataFrame]:
        """
        Obtain the baseline correction, create if not present
        """
        if self.sampletype == "nuclear":
            bl_file = Path(io.LOCALSOURCE_baseline) / f"{self.ztfid}_bl.csv"
        elif self.sampletype == "bts":
            bl_file = Path(io.LOCALSOURCE_bts_baseline) / f"{self.ztfid}_bl.csv"
        elif self.sampletype == "train":
            bl_file = Path(io.LOCALSOURCE_train_dfs) / f"{self.ztfid}.csv"

        if bl_file.is_file():
            bl = pd.read_csv(bl_file, comment="#")
            if "filter" not in list(bl.keys()):
                bl["filter"] = bl["fid"].apply(lambda x: utils.ztf_filterid_to_band(x))
        else:
            # create empty df
            bl = pd.DataFrame()

        return bl

    @cached_property
    def baseline_info(self) -> dict:
        """
        Obtain the baseline correction metadata, create if not present
        """
        if self.sampletype in ["nuclear", "bts"]:
            return self.meta.get("bl_info")

        else:
            parent_ztfid = self.meta["parent_ztfid"]
            try:
                t = Transient(ztfid=parent_ztfid, sampletype="bts")
                bl_info = t.meta.get("bl_info")
            except:
                t_peak_bts = float(self.meta.get("bts_peak_jd")) - 2400000.5
                bl_info = {"t_peak": t_peak_bts}

        return bl_info

    def recreate_baseline(self):
        """
        Recreate the baseline
        """
        bl, bl_info = baseline.baseline(transient=self, plot=False)

        self.baseline = bl
        self.baseline_info = bl_info

        if self.sampletype == "nuclear":
            meta.update_transient(self.ztfid, data={"bl_info": bl_info})
        elif self.sampletype == "bts":
            meta_bts.update_transient(self.ztfid, data={"bl_info": bl_info})

    def update(self, data: dict):
        """
        Update the database with new metadata
        """
        if self.sampletype == "nuclear":
            transient_info = meta.update_transient(self.ztfid, data=data)
        elif self.sampletype == "bts":
            transient_info = meta_bts.update_transient(self.ztfid, data=data)
        elif self.sampletype == "train":
            transient_info = meta_train.update_transient(self.ztfid, data=data)

    @cached_property
    def raw_lc(self) -> pd.DataFrame:
        """
        Read the lightcurve_dataframe
        """
        if self.sampletype == "nuclear":
            lc_path = os.path.join(io.LOCALSOURCE_dfs, self.ztfid + ".csv")
        elif self.sampletype == "bts":
            lc_path = os.path.join(io.LOCALSOURCE_bts_dfs, self.ztfid + ".csv")
        lc = pd.read_csv(lc_path, comment="#")

        return lc

    @cached_property
    def tns_name(self) -> Optional[str]:
        """
        Get the TNS name if one is present in metadata
        """
        if "TNS_name" in self.meta.keys():
            return self.meta["TNS_name"]
        else:
            return None

    @cached_property
    def tns_class(self) -> Optional[str]:
        """
        Get the TNS classification if one is present in metadata
        """
        # USE DEFAULTDICT TO FIX THIS MESS
        if "crossmatch" in self.meta.keys():
            if "TNS" in self.meta["crossmatch"].keys():
                if "type" in self.meta["crossmatch"]["TNS"].keys():
                    return self.meta["crossmatch"]["TNS"]["type"]
        else:
            return None

    @cached_property
    def fritz_class(self) -> Optional[str]:
        """
        Get the Fritz classification if one is present in metadata
        """
        if "fritz_class" in self.meta.keys():
            fritz_class = self.meta["fritz_class"]
            return fritz_class
        else:
            return None

    @cached_property
    def tde_res_exp(self) -> Optional[float]:
        """
        Get the TDE fit reduced chisq
        """
        if "tde_fit_exp" in self.meta.keys():
            if self.meta["tde_fit_exp"] is not None:
                if "success" in self.meta["tde_fit_exp"].keys():
                    if self.meta["tde_fit_exp"]["success"] != False:
                        chisq = self.meta["tde_fit_exp"]["chisq"]
                        ndof = self.meta["tde_fit_exp"]["ndof"]
                        red_chisq = chisq / ndof
                        return red_chisq
        else:
            return None

    @cached_property
    def salt_res(self) -> Optional[float]:
        """
        Get the SALT fit reduced chisq
        """
        if "salt_loose_bl" in self.meta.keys():
            if self.meta["salt_loose_bl"] != "failure":
                chisq = self.meta["salt_loose_bl"]["chisq"]
                ndof = self.meta["salt_loose_bl"]["ndof"]
                red_chisq = chisq / ndof
                return red_chisq
        else:
            return None

    @cached_property
    def z(self) -> Optional[float]:
        """
        Get the AMPEL redshift from the database
        """
        if fritz_z := self.meta.get("fritz_z"):
            return float(fritz_z)

        if "ampel_z" in self.meta.keys():
            if "z" in self.meta["ampel_z"].keys():
                ampel_z = self.meta["ampel_z"]["z"]
                return ampel_z
        else:
            return None

    @cached_property
    def z_dist(self) -> Optional[float]:
        """
        Get the redshift distance from the database. In case we have a Fritz redshift, the distance is set to 0 (we trust Fritz)
        """
        if self.meta.get("fritz_z"):
            return 0.0

        if "ampel_z" in self.meta.keys():
            if "z_dist" in self.meta["ampel_z"].keys():
                ampel_z_dist = self.meta["ampel_z"]["z_dist"]
                return ampel_z_dist

        else:
            return None

    @cached_property
    def z_precision(self) -> Optional[float]:
        """
        Get the estimated redshift precision based on Ampel T2DigestRedshift group
        """
        if "ampel_z" in self.meta.keys():
            if "z_precision" in self.meta["ampel_z"].keys():
                z_precision = self.meta["ampel_z"]["z_precision"]
                return z_precision
        else:
            return None

    @property
    def meta(self) -> dict | None:
        """
        Read all metadata for the transient from the database
        """
        if self.sampletype == "nuclear":
            transient_metadata = meta.read_transient(ztfid=self.ztfid)
        elif self.sampletype == "bts":
            transient_metadata = meta_bts.read_transient(ztfid=self.ztfid)
        elif self.sampletype == "train":
            transient_metadata = meta_train.read_transient(ztfid=self.ztfid)
        if transient_metadata:
            return transient_metadata
        else:
            return None

    @cached_property
    def wise_lc(self) -> pd.DataFrame | None:
        """
        Get the corresponding WISE lightcurve (if available) as pandas df
        """
        df = None
        wise_dict = self.meta.get("WISE_lc", {}).get("timewise_lightcurve")
        if wise_dict is not None:
            df = pd.DataFrame.from_dict(wise_dict)
            if len(df) == 0:
                return None

        return df

    @cached_property
    def thumbnail(self) -> str | None:
        """
        Read the thumbnail image and return as base64 string
        """
        if self.sampletype == "nuclear":
            plot_dir = os.path.join(io.LOCALSOURCE_plots, "lightcurves", "thumbnails")
        elif self.sampletype == "bts":
            plot_dir = os.path.join(
                io.LOCALSOURCE_bts_bplots, "lightcurves", "thumbnails"
            )
        thumb_file = os.path.join(plot_dir, self.ztfid + "_thumbnail.png")

        if os.path.isfile(thumb_file):
            thumb_data = open(thumb_file, "rb")
            thumb_b64 = base64.b64encode(thumb_data.read()).decode("ascii")
            return thumb_b64

        else:
            return None

    def irsa(self):
        """
        Load the IRSA lightcurve if not locally present
        """
        if self.sampletype == "nuclear":
            path_to_lc = os.path.join(io.LOCALSOURCE_irsa, f"{self.ztfid}.csv")
        elif self.sampletype == "bts":
            path_to_lc = os.path.join(io.LOCALSOURCE_bts_irsa, f"{self.ztfid}.csv")
        if not os.path.isfile(path_to_lc):
            df = io.load_irsa(ra=self.ra, dec=self.dec, radius_arcsec=0.5)
            df.to_csv(path_to_lc)
        else:
            df = pd.read_csv(path_to_lc)
        return df

    def get_rating(self, username: str) -> Optional[int]:
        """
        Read the rating from the DB (3: interesting, 2: maybe, 1: boring. If none is found, return 0)
        """
        if "rating" in self.meta.keys():
            if self.meta["rating"] is not None:
                if username in self.meta["rating"].keys():
                    rating = int(self.meta["rating"][username])
                else:
                    rating = 0
            else:
                rating = 0
        else:
            rating = 0

        return rating

    def get_rating_overview(self) -> dict:
        """
        Read all the ratings (from different users) for a transient and return the individual ratings
        """
        if "rating" in self.meta.keys():
            return self.meta["rating"]
        else:
            return {}

    def get_rating_average(self) -> float:
        """
        Read all the ratings (from different users) for a transient and return the average rating
        """
        ratings = self.get_rating_overview()
        if ratings:
            avg_list = []
            for key in ratings.keys():
                avg_list.append(ratings[key])

            return np.average(avg_list)
        else:
            return 0

    def set_rating(self, rating: int, username: str):
        """
        Update the transient rating
        """
        if "rating" in self.meta.keys():
            rating_dict = self.meta["rating"]
        else:
            rating_dict = {}

        rating_dict.update({username: rating})

        meta.update_transient(self.ztfid, data={"rating": rating_dict})

    def get_comments(self) -> Optional[dict]:
        """
        Read the transient comments from the database
        """
        if "comments" in self.meta.keys():
            if self.meta["comments"] is not None:
                return self.meta["comments"]
            else:
                return None
        else:
            return None

    def get_comment_count(self) -> int:
        """
        Return number of comments for transient
        """
        if "comments" in self.meta.keys():
            if self.meta["comments"] is not None:
                return len(self.meta["comments"])
            else:
                return None
        else:
            return 0

    def delete_comment(self, timestamp):
        """
        Delete a comment matching to a timecode
        """
        all_comments = self.get_comments()

        if timestamp in all_comments.keys():
            del all_comments[timestamp]

        if self.sampletype == "nuclear":
            meta.update_transient(self.ztfid, data={"comments": all_comments})
        elif self.sampletype == "bts":
            meta_bts.update_transient(self.ztfid, data={"comments": all_comments})

    def delete_all_comments(self):
        """
        Delete all comments for a transient
        """
        if self.sampletype == "nuclear":
            meta.update_transient(self.ztfid, data={"comments": {}})
        elif self.sampletype == "bts":
            meta_bts.update_transient(self.ztfid, data={"comments": {}})

    def get_comments_generator(self):
        """
        Iterate over transient comments
        """
        comments = self.get_comments()

        if comments is None:
            yield None

        else:
            for timestamp, content in dict(sorted(comments.items())).items():
                yield {"timestamp": timestamp, "content": content}

    def add_comment(self, username: str, comment: str):
        """
        Add a comment to the database
        """
        if "comments" in self.meta.keys():
            if self.meta["comments"] is not None:
                comments_dict = self.meta["comments"]
            else:
                comments_dict = {}
        else:
            comments_dict = {}

        timestamp = str(time.time())

        comments_dict.update(
            {timestamp: {"username": username, "comment": str(comment)}}
        )

        if self.sampletype == "nuclear":
            meta.update_transient(self.ztfid, data={"comments": comments_dict})
        elif self.sampletype == "bts":
            meta_bts.update_transient(self.ztfid, data={"comments": comments_dict})

    @cached_property
    def crossmatch_info(self) -> Optional[str]:
        """
        Read the crossmatch results from the DB and return a dict with found values
        """
        exclude_list = ["WISE", "TNS", "WISE_cat"]

        xmatch = self.meta["crossmatch"]

        message = ""
        for key in xmatch.keys():
            subdict = xmatch[key]
            if len(subdict) > 0:
                if (
                    subdict is not None
                    and key not in exclude_list
                    and isinstance(subdict, dict)
                ):
                    message += key
                    if "type" in subdict.keys():
                        if key == "TNS" and subdict["type"] is not None:
                            message += f" {subdict['type']}"
                    if "dist" in subdict.keys():
                        message += f" {subdict['dist']:.5f}  "

        if len(message) == 0:
            return None
        else:
            return message

    @cached_property
    def crossmatch_dict(self) -> dict:
        return self.meta.get("crossmatch", {})

    def get_crossmatch_for_viewer(self, include: list = ["WISE"]) -> Optional[str]:
        """
        Read the crossmatch results from the DB and return a string with found values
        """
        xmatch = self.meta["crossmatch"]
        message = ""
        for key in ["Milliquas", "TNS", "Marshal"]:
            entry = xmatch.get(key, {})
            if len(entry) > 0:
                if key == "TNS" or key == "Milliquas":
                    classif = entry.get("type")
                    if classif is not None:
                        message += f"{key}: {classif}  "

                if key == "Marshal":
                    classif = entry.get("class")
                    if classif is not None:
                        message += f"{key}: {classif}"

        if len(message) == 0:
            return None
        else:
            return message

    def crossmatch(self, crossmatch_types: list | None = None):
        """
        Do all kinds of crossmatches for the transient
        """
        from ztfnuclear.crossmatch import (
            query_ampel_dist,
            query_ampel_sgscore,
            query_bts,
            query_crts,
            query_gaia,
            query_marshal,
            query_milliquas,
            query_sarah_agn,
            query_sdss,
            query_tns,
            query_wise,
            query_wise_cat,
            query_catwise,
        )

        results = self.meta.get("crossmatch", {})
        res_list = []

        if crossmatch_types is None:
            res_list.append(query_crts(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_wise_cat(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_milliquas(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_sdss(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_gaia(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_wise(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_tns(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_sarah_agn(ra_deg=self.ra, dec_deg=self.dec))
            res_list.append(query_catwise(ra_deg=self.ra, dec_deg=self.dec))

        else:
            if "crts" in crossmatch_types:
                res_list.append(query_crts(ra_deg=self.ra, dec_deg=self.dec))
            if "wise_cat" in crossmatch_types:
                res_list.append(query_wise_cat(ra_deg=self.ra, dec_deg=self.dec))
            if "milliquas" in crossmatch_types:
                res_list.append(query_milliquas(ra_deg=self.ra, dec_deg=self.dec))
            if "sdss" in crossmatch_types:
                res_list.append(query_sdss(ra_deg=self.ra, dec_deg=self.dec))
            if "gaia" in crossmatch_types:
                res_list.append(query_gaia(ra_deg=self.ra, dec_deg=self.dec))
            if "wise" in crossmatch_types:
                res_list.append(query_wise(ra_deg=self.ra, dec_deg=self.dec))
            if "tns" in crossmatch_types:
                res_list.append(query_tns(ra_deg=self.ra, dec_deg=self.dec))
            if "dist" in crossmatch_types:
                res_list.append(
                    {"distnr": {"dist": query_ampel_dist(ztfid=self.ztfid)}}
                )
            if "sgscore" in crossmatch_types:
                res_list.append({"sgscore": query_ampel_sgscore(ztfid=self.ztfid)})
            if "sarah_agn" in crossmatch_types:
                res_list.append(query_sarah_agn(ra_deg=self.ra, dec_deg=self.dec))

            if "bts" in crossmatch_types:
                res_list.append({"bts": query_bts(ztfid=self.ztfid)})
            if "marshal" in crossmatch_types:
                res_list.append(query_marshal(ztfid=self.ztfid))

            if "catwise2020" in crossmatch_types:
                res_list.append(query_catwise(ra_deg=self.ra, dec_deg=self.dec))

            for res in res_list:
                results.update(res)

        self.crossmatch = {"crossmatch": results}

        if self.sampletype == "nuclear":
            m_db = meta
        elif self.sampletype == "bts":
            m_db = meta_bts

        if "name" in self.crossmatch["crossmatch"].get("TNS", {}).keys():
            tns_name = self.crossmatch["crossmatch"]["TNS"]["name"]
            m_db.update_transient(ztfid=self.ztfid, data={"TNS_name": tns_name})

        m_db.update_transient(ztfid=self.ztfid, data=self.crossmatch)

    def fritz(self):
        """
        Query Fritz for transient info and update the database
        """
        fritz = FritzAPI()
        fritzinfo = fritz.get_transient(self.ztfid)
        data = fritzinfo[self.ztfid]
        if self.sampletype == "nuclear":
            meta.update_transient(ztfid=self.ztfid, data=data)
        elif self.sampletype == "bts":
            meta_bts.update_transient(ztfid=self.ztfid, data=data)

    def plot(
        self,
        force_baseline_correction: bool = False,
        plot_raw: bool = False,
        magplot: bool = False,
        include_wise: bool = True,
        wise_baseline_correction: bool = True,
        snt_threshold: float = 3.0,
        save: bool = True,
        plot_png: bool = False,
        wide: bool = False,
        thumbnail: bool = False,
        no_magrange: bool = False,
        xlim: list[float] | None = None,
        outdir: Path | str | None = None,
    ):
        """
        Plot the transient lightcurve
        """
        if include_wise is False:
            wise_df_to_plot = None
        else:
            wise_df = self.wise_lc
            wise_df_to_plot = None

            if wise_baseline_correction:
                if "WISE_bayesian" in self.meta.keys():
                    if (
                        self.meta.get("WISE_bayesian", {}).get("bayesian", {})
                        is not None
                    ):
                        if (
                            "bayesian" in self.meta["WISE_bayesian"].keys()
                            and self.meta["WISE_bayesian"]["bayesian"] is not None
                        ):
                            bl_W1 = self.meta["WISE_bayesian"]["bayesian"]["Wise_W1"][
                                "baseline"
                            ]
                            bl_W2 = self.meta["WISE_bayesian"]["bayesian"]["Wise_W2"][
                                "baseline"
                            ]

                            bl_W1_err = self.meta["WISE_bayesian"]["bayesian"][
                                "Wise_W1"
                            ]["baseline_rms"]
                            bl_W2_err = self.meta["WISE_bayesian"]["bayesian"][
                                "Wise_W2"
                            ]["baseline_rms"]
                            wise_df["W1_mean_flux_density_bl_corr"] = (
                                wise_df["W1_mean_flux_density"] - bl_W1
                            )
                            wise_df["W2_mean_flux_density_bl_corr"] = (
                                wise_df["W2_mean_flux_density"] - bl_W2
                            )
                            wise_df["W1_mean_flux_density_bl_corr_err"] = np.sqrt(
                                wise_df["W1_flux_density_rms"].values ** 2
                                + bl_W1_err**2
                            )
                            wise_df["W2_mean_flux_density_bl_corr_err"] = np.sqrt(
                                wise_df["W2_flux_density_rms"].values ** 2
                                + bl_W2_err**2
                            )

                            wise_df["W1_mean_mag_ab"] = utils.flux_density_to_abmag(
                                flux_density=wise_df["W1_mean_flux_density_bl_corr"]
                                / 1000,  # convert from mJy to Jy
                                band="W1",
                            )
                            wise_df["W2_mean_mag_ab"] = utils.flux_density_to_abmag(
                                flux_density=wise_df["W2_mean_flux_density_bl_corr"]
                                / 1000,  # convert from mJy to Jy
                                band="W2",
                            )
                            wise_df_to_plot = wise_df

        if force_baseline_correction:
            df_to_plot = self.baseline
        else:
            if len(self.baseline) == 0 or plot_raw:
                df_to_plot = self.raw_lc
            else:
                df_to_plot = self.baseline

        if self.z is not None:
            if self.z_dist < 1:
                z = self.z
            else:
                z = None
        else:
            z = None

        from ztfnuclear.plot import plot_lightcurve

        cl = self.meta.get("classif")
        if cl is not None:
            cl += f" / xgclass: {self.meta.get('xgclass')}"

        axlims = plot_lightcurve(
            df=df_to_plot,
            ztfid=self.ztfid,
            z=z,
            tns_name=self.tns_name,
            magplot=magplot,
            wise_df=wise_df_to_plot,
            snt_threshold=snt_threshold,
            plot_png=plot_png,
            wide=wide,
            thumbnail=thumbnail,
            sampletype=self.sampletype,
            no_magrange=no_magrange,
            classif=cl,
            xlim=xlim,
            outdir=outdir,
        )

        return axlims

    def fit_tde(
        self,
        powerlaw: bool = False,
        plateau: bool = False,
        simplefit_only: bool = False,
        debug: bool = False,
    ):
        """
        Re-fit the transient TDE lightcurve
        """
        from ztfnuclear import tde_fit

        if self.sampletype == "nuclear":
            m_db = meta
        elif self.sampletype == "bts":
            m_db = meta_bts
        elif self.sampletype == "train":
            m_db = meta_train

        # self.sampletype == "trai"

        if len(self.baseline) > 0:
            fitresult = tde_fit.fit(
                df=self.baseline,
                ra=self.ra,
                dec=self.dec,
                baseline_info=self.baseline_info,
                powerlaw=powerlaw,
                plateau=plateau,
                ztfid=self.ztfid,
                simplefit_only=simplefit_only,
                debug=debug,
            )
            if powerlaw:
                m_db.update_transient(self.ztfid, data={"tde_fit_pl": fitresult})
            else:
                m_db.update_transient(self.ztfid, data={"tde_fit_exp": fitresult})

        else:
            self.logger.info(
                f"{self.ztfid}: No datapoints survived baseline correction, skipping plot"
            )

    def plot_tde(
        self,
        powerlaw: bool = False,
        debug: bool = False,
        params: dict = None,
        savepath: str | None = None,
    ) -> bool:
        """
        Plot the TDE fit result if present
        """
        if powerlaw:
            keyword = "tde_fit_pl"
        else:
            keyword = "tde_fit_exp"

        if debug:
            savepath = "/Users/simeon/Desktop/flextemp_test/"

        success = False

        from ztfnuclear.plot import plot_tde_fit

        if keyword in self.meta.keys():
            if self.meta[keyword] is not None and params is None:
                if "success" in self.meta[keyword].keys():
                    if self.meta[keyword]["success"] == True:
                        if self.z is not None:
                            if self.z_dist < 1:
                                z = self.z
                            else:
                                z = None
                        else:
                            z = None

                        plot_tde_fit(
                            df=self.baseline,
                            ztfid=self.ztfid,
                            z=z,
                            tns_name=self.tns_name,
                            tde_params=self.meta[keyword]["paramdict"],
                            savepath=savepath,
                            sampletype=self.sampletype,
                        )

                        success = True

            elif params:
                plot_tde_fit(
                    df=self.baseline,
                    ztfid=self.ztfid,
                    z=None,
                    tns_name=self.tns_name,
                    tde_params=params,
                    savepath=savepath,
                    sampletype=self.sampletype,
                )

                success = True

        return success

    def plot_irsa(
        self,
        wide: bool = False,
        magplot: bool = False,
        plot_png: bool = False,
        axlims: dict = None,
    ):
        """
        Get the non-difference alert photometry for this transient from IRSA and plot it
        """
        from ztfnuclear.plot import plot_lightcurve_irsa

        df = self.irsa()

        plot_lightcurve_irsa(
            df=df,
            ztfid=self.ztfid,
            ra=self.ra,
            dec=self.dec,
            wide=wide,
            magplot=magplot,
            plot_png=plot_png,
            axlims=axlims,
        )
