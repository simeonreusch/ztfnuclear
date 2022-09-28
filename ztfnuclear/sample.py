#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, datetime, base64, time

from functools import cached_property
from typing import Optional, List

from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from ztfnuclear import io, baseline, utils
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.plot import plot_lightcurve, plot_lightcurve_irsa
from ztfnuclear.fritz import FritzAPI

logger = logging.getLogger(__name__)

meta = MetadataDB()
info_db = SampleInfo()


class NuclearSample(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(self):
        super(NuclearSample, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing sample")

        io.download_if_neccessary()

        self.get_all_ztfids()
        self.location()
        self.meta = MetadataDB()
        self.info_db = info_db
        db_check = self.meta.get_statistics()

        if not db_check["has_ra"]:
            self.populate_db_from_csv(filepath=io.LOCALSOURCE_location)

        if not db_check["has_peak_dates"]:
            self.populate_db_from_csv(
                filepath=io.LOCALSOURCE_peak_dates, name="peak_dates"
            )
        if not db_check["has_salt"]:
            saltfit_res = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_fitres, "saltfit.json"),
                parameter_name="salt",
            )
            self.populate_db_from_dict(data=saltfit_res)

        if not db_check["has_salt_loose_bl"]:
            saltfit_res = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_fitres, "saltfit_loose_bl.json"),
                parameter_name="salt_loose_bl",
            )
            self.populate_db_from_dict(data=saltfit_res)

        if not db_check["has_tdefit"]:
            tdefit_res = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_fitres, "tdefit.json"),
                parameter_name="tde_fit",
            )
            self.populate_db_from_dict(data=tdefit_res)

        if not db_check["has_tdefit_loose_bl"]:
            tdefit_res = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_fitres, "tdefit_loose_bl.json"),
                parameter_name="tde_fit_loose_bl",
            )
            self.populate_db_from_dict(data=tdefit_res)

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
            )
            self.populate_db_from_dict(data=wise_bayesian)

        if not db_check["has_wise_lc_by_pos"]:
            wise_lcs_by_pos = io.parse_json(filepath=io.LOCALSOURCE_WISE_lc_by_pos)
            self.populate_db_from_dict(data=wise_lcs_by_pos)

        if not db_check["has_wise_lc_by_id"]:
            wise_lcs_by_id = io.parse_json(filepath=io.LOCALSOURCE_WISE_lc_by_id)
            self.populate_db_from_dict(data=wise_lcs_by_id)

        db_check = self.meta.get_statistics()
        assert db_check["count"] == 11687

    def get_transient(self, ztfid: str):
        df = io.get_ztfid_dataframe(ztfid=ztfid)
        header = io.get_ztfid_header(ztfid=ztfid)
        return header, df

    def get_all_ztfids(self):
        all_ztfids = io.get_all_ztfids()
        self.ztfids = all_ztfids

    def location(self):
        df = io.get_locations()
        self.location = df

    def create_baseline(self):
        self.logger.info("Creating baseline corrections for the full sample")
        for ztfid in tqdm(self.ztfids):
            t = Transient(ztfid)
            t.create_baseline()

    def populate_db_from_csv(self, filepath, name=None):
        self.logger.info("Populating the database from a csv file")
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath, comment="#", index_col=0)
            ztfids = []
            data = []
            for s in df.iterrows():
                ztfid = s[0]
                ztfids.append(ztfid)
                if name:
                    write_dict = {name: s[1].to_dict()}
                else:
                    write_dict = s[1].to_dict()
                data.append(write_dict)

            self.meta.update_many(ztfids=ztfids, data=data)
        else:
            raise ValueError("File does not exist")

    def populate_db_from_dict(self, data: dict):
        """Use a dict of the form {ztfid: data} to update the Mongo DB"""
        self.logger.info("Populating the database from a dictionary")

        ztfids = data.keys()
        data_list = []
        for ztfid in ztfids:
            data_list.append(data[ztfid])

        self.meta.update_many(ztfids=ztfids, data=data_list)

    def crossmatch(self, startindex: int = 0):
        """Crossmatch the full sample"""
        self.logger.info("Crossmatching the full sample")
        for i, ztfid in tqdm(enumerate(self.ztfids[startindex:])):
            self.logger.debug(f"{ztfid}: Crossmatching")
            self.logger.debug(f"Transient {i+startindex} of {len(self.ztfids)}")
            t = Transient(ztfid)
            t.crossmatch()
        info = SampleInfo()
        date_now = datetime.datetime.now().replace(microsecond=0)
        info.update(data={"crossmatch_info": {"crossmatch": True, "date": date_now}})

    def fritz(self, startindex: int = 0):
        """
        Query Fritz for the full sample
        """
        self.logger.info("Obtaining metadata on full sample from Fritz")
        for i, ztfid in tqdm(enumerate(self.ztfids[startindex:])):
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
        ratings = self.meta.get_rating_overview(username=username)

        rating_to_value = {
            "interesting": 3,
            "maybe": 2,
            "boring": 1,
        }

        value_to_rating = {v: k for k, v in rating_to_value.items()}

        returndict = {}

        if select == "all":
            returndict = ratings
        else:
            for k, v in ratings.items():
                if v[username] == rating_to_value[select]:
                    returndict.update({k: v})

        return returndict

    def next_transient(self, ztfid: str, flaring: bool = False):
        """
        Get the transient following the current one in the sample
        """
        if flaring:
            flaring_ztfids = self.info_db.read()["flaring"]["ztfids"]
            idx = flaring_ztfids.index(ztfid)
            return flaring_ztfids[idx + 1]

        else:
            idx = self.ztfids.index(ztfid)
            return self.ztfids[idx + 1]

    def previous_transient(self, ztfid: str, flaring: bool = False):
        """
        Get the transient following the current one in the sample
        """
        if flaring:
            flaring_ztfids = self.info_db.read()["flaring"]["ztfids"]
            idx = flaring_ztfids.index(ztfid)
            return flaring_ztfids[idx - 1]

        else:
            idx = self.ztfids.index(ztfid)
            return self.ztfids[idx - 1]

    def get_transients(
        self, n: Optional[int] = None, ztfids: Optional[List[str]] = None
    ):
        """
        Loop over all transients in sample (or over all ztfids if given) and return a Transient Object
        """
        if not n:
            if ztfids is None:
                n = len(self.ztfids)
            else:
                n = len(ztfids)

        if ztfids is None:
            for ztfid in self.ztfids[:n]:
                t = Transient(ztfid)
                yield t

        else:
            for ztfid in ztfids[:n]:
                t = Transient(ztfid)
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

        meta = MetadataDB()
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

        meta = MetadataDB()
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
        meta = MetadataDB()
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


class Transient(object):
    """
    This class contains all info for a given transient
    """

    def __init__(self, ztfid: str):
        super(Transient, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ztfid = ztfid

        transient_info = meta.read_transient(self.ztfid)
        self.ra = transient_info["RA"]
        self.dec = transient_info["Dec"]

        self.location = {"RA": self.ra, "Dec": self.dec}

    @cached_property
    def header(self):
        header = io.get_ztfid_header(ztfid=self.ztfid)
        return header

    @cached_property
    def df(self):
        df = io.get_ztfid_dataframe(ztfid=self.ztfid)
        return df

    @cached_property
    def sample_ids(self):
        ids = NuclearSample().ztfids
        return ids

    @cached_property
    def baseline(self) -> pd.DataFrame:
        """
        Obtain the baseline correction, create if not present
        """
        bl_file = os.path.join(io.LOCALSOURCE_baseline, self.ztfid + "_bl.csv")
        if os.path.isfile(bl_file):
            bl = pd.read_csv(bl_file)
            return bl
        else:
            self.logger.info(
                f"{self.ztfid}: No baseline correction file, trying to apply baseline correction"
            )
            bl, bl_info = baseline.baseline(transient=self)
            return bl

    def recreate_baseline(self):
        """
        Recreate the baseline
        """
        bl, bl_info = baseline.baseline(transient=self, primary_grid_only=False)
        self.baseline = bl

    @cached_property
    def raw_lc(self) -> pd.DataFrame:
        """
        Read the lightcurve_dataframe
        """
        lc_path = os.path.join(io.LOCALSOURCE_dfs, self.ztfid + ".csv")
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
        if "crossmatch" in self.meta.keys():
            if "TNS" in self.meta["crossmatch"].keys():
                if "type" in self.meta["crossmatch"]["TNS"].keys():
                    return self.meta["crossmatch"]["TNS"]["type"]
        else:
            return None

    @cached_property
    def tde_res(self) -> Optional[float]:
        """
        Get the TDE fit reduced chisq
        """
        if "tde_fit_loose_bl" in self.meta.keys():
            if self.meta["tde_fit_loose_bl"] != "failure":
                chisq = self.meta["tde_fit_loose_bl"]["chisq"]
                ndof = self.meta["tde_fit_loose_bl"]["ndof"]
                red_chisq = chisq / ndof
                return red_chisq
        else:
            return None

    @cached_property
    def salt_res(self) -> Optional[float]:
        """
        Get the SALT fit reduced chisq
        """
        if "tde_fit_loose_bl" in self.meta.keys():
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
        if "ampel_z" in self.meta.keys():
            if "z" in self.meta["ampel_z"].keys():
                ampel_z = self.meta["ampel_z"]["z"]
                return ampel_z
        else:
            return None

    @cached_property
    def z_dist(self) -> Optional[float]:
        """
        Get the AMPEL redshift distance from the database
        """
        if "ampel_z" in self.meta.keys():
            if "z_dist" in self.meta["ampel_z"].keys():
                ampel_z_dist = self.meta["ampel_z"]["z_dist"]
                return ampel_z_dist
        else:
            return None

    @cached_property
    def meta(self) -> Optional[dict]:
        """
        Read all metadata for the transient from the database
        """
        transient_metadata = meta.read_transient(ztfid=self.ztfid)
        if transient_metadata:
            return transient_metadata
        else:
            return None

    @cached_property
    def wise_lc(self) -> Optional[pd.DataFrame]:
        """
        Get the corresponding WISE lightcurve (if available) as pandas df
        """
        wise_dict = self.meta["WISE_lc"]
        df = pd.DataFrame.from_dict(wise_dict)
        if len(df) == 0:
            return None
        else:
            return df

    @cached_property
    def thumbnail(self) -> Optional[str]:
        """
        Read the thumbnail image and return as base64 string
        """
        plot_dir = os.path.join(io.LOCALSOURCE_plots, "lightcurves", "thumbnails")
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
        path_to_lc = os.path.join(io.LOCALSOURCE_irsa, f"{self.ztfid}.csv")
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
            if username in self.meta["rating"].keys():
                rating = int(self.meta["rating"][username])
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
            return self.meta["comments"]
        else:
            return None

    def get_comment_count(self) -> int:
        """
        Return number of comments for transient
        """
        if "comments" in self.meta.keys():
            return len(self.meta["comments"])
        else:
            return 0

    def delete_comments(self):
        """
        Delete all comments for a transient
        """
        meta.update_transient(self.ztfid, data={"comments": {}})

    def get_comments_generator(self):
        """
        Iterate over transient comments
        """
        comments = self.get_comments()

        if comments is None:
            yield None

        else:
            for timestamp, content in dict(sorted(comments.items())).items():
                yield content

    def add_comment(self, username: str, comment: str):
        """
        Add a comment to the database
        """
        if "comments" in self.meta.keys():
            comments_dict = self.meta["comments"]
        else:
            comments_dict = {}

        timestamp = str(time.time())

        comments_dict.update(
            {timestamp: {"username": username, "comment": str(comment)}}
        )

        meta.update_transient(self.ztfid, data={"comments": comments_dict})

    def get_crossmatch_info(self, exclude: list = ["WISE"]) -> Optional[str]:
        """
        Read the crossmatch results from the DB and return a dict with found values
        """
        xmatch = self.meta["crossmatch"]
        message = ""
        for key in xmatch.keys():
            subdict = xmatch[key]
            if len(subdict) > 0:
                if (
                    subdict is not None
                    and key not in exclude
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

    def crossmatch(self):
        """
        Do all kinds of crossmatches for the transient
        """
        from ztfnuclear.crossmatch import (
            query_crts,
            query_milliquas,
            query_sdss,
            query_gaia,
            query_wise,
            query_tns,
        )

        results = {}

        crts_res = query_crts(ra_deg=self.ra, dec_deg=self.dec)
        millliquas_res = query_milliquas(ra_deg=self.ra, dec_deg=self.dec)
        sdss_res = query_sdss(ra_deg=self.ra, dec_deg=self.dec)
        gaia_res = query_gaia(ra_deg=self.ra, dec_deg=self.dec)
        wise_res = query_wise(ra_deg=self.ra, dec_deg=self.dec)
        tns_res = query_tns(ra_deg=self.ra, dec_deg=self.dec)

        for res in [crts_res, millliquas_res, sdss_res, gaia_res, wise_res, tns_res]:
            results.update(res)

        self.crossmatch = {"crossmatch": results}
        if "name" in self.crossmatch["crossmatch"]["TNS"].keys():
            tns_name = self.crossmatch["crossmatch"]["TNS"]["name"]
            meta.update_transient(ztfid=self.ztfid, data={"TNS_name": tns_name})

        meta.update_transient(ztfid=self.ztfid, data=self.crossmatch)

    def fritz(self):
        """
        Query Fritz for transient info and update the database
        """
        fritz = FritzAPI()
        fritzinfo = fritz.get_transient(self.ztfid)
        data = fritzinfo[self.ztfid]
        meta.update_transient(ztfid=self.ztfid, data=data)

    def plot(
        self,
        force_baseline_correction: bool = False,
        magplot: bool = False,
        include_wise: bool = True,
        wise_baseline_correction: bool = True,
        snt_threshold: float = 3.0,
        save: bool = True,
        plot_png: bool = False,
        wide: bool = False,
        thumbnail: bool = False,
    ):
        """
        Plot the transient lightcurve
        """
        if include_wise:
            wise_df = self.wise_lc
            wise_df_to_plot = None

            if wise_baseline_correction:
                if "WISE_bayesian" in self.meta.keys():
                    if (
                        "bayesian" in self.meta["WISE_bayesian"].keys()
                        and self.meta["WISE_bayesian"]["bayesian"] is not None
                    ):
                        bl_W1_uncorr = self.meta["WISE_bayesian"]["bayesian"][
                            "Wise_W1"
                        ]["baseline"][0]
                        bl_W2_uncorr = self.meta["WISE_bayesian"]["bayesian"][
                            "Wise_W2"
                        ]["baseline"][0]

                        bl_W1_err_uncorr = self.meta["WISE_bayesian"]["bayesian"][
                            "Wise_W1"
                        ]["baseline_rms"][0]
                        bl_W2_err_uncorr = self.meta["WISE_bayesian"]["bayesian"][
                            "Wise_W2"
                        ]["baseline_rms"][0]

                        bl_W1 = utils.flux_density_bug_correction(
                            flux_density=bl_W1_uncorr, band="W1"
                        )
                        bl_W2 = utils.flux_density_bug_correction(
                            flux_density=bl_W2_uncorr, band="W2"
                        )
                        bl_W1_err = utils.flux_density_bug_correction(
                            flux_density=bl_W1_err_uncorr, band="W1"
                        )
                        bl_W2_err = utils.flux_density_bug_correction(
                            flux_density=bl_W2_err_uncorr, band="W2"
                        )

                        wise_df[
                            "W1_mean_flux_density"
                        ] = utils.flux_density_bug_correction(
                            flux_density=wise_df["W1_mean_flux_density"], band="W1"
                        )
                        wise_df[
                            "W2_mean_flux_density"
                        ] = utils.flux_density_bug_correction(
                            flux_density=wise_df["W2_mean_flux_density"], band="W2"
                        )
                        wise_df[
                            "W1_flux_density_rms"
                        ] = utils.flux_density_bug_correction(
                            flux_density=wise_df["W1_flux_density_rms"], band="W1"
                        )
                        wise_df[
                            "W2_flux_density_rms"
                        ] = utils.flux_density_bug_correction(
                            flux_density=wise_df["W2_flux_density_rms"], band="W2"
                        )

                        wise_df["W1_mean_flux_density_bl_corr"] = (
                            wise_df["W1_mean_flux_density"] - bl_W1
                        )
                        wise_df["W2_mean_flux_density_bl_corr"] = (
                            wise_df["W2_mean_flux_density"] - bl_W2
                        )
                        wise_df["W1_mean_flux_density_bl_corr_err"] = np.sqrt(
                            wise_df["W1_flux_density_rms"].values ** 2 + bl_W1_err**2
                        )
                        wise_df["W2_mean_flux_density_bl_corr_err"] = np.sqrt(
                            wise_df["W2_flux_density_rms"].values ** 2 + bl_W2_err**2
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
            if len(self.baseline) == 0:
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
        )

        return axlims

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
