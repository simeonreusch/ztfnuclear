#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, datetime

from functools import cached_property
from typing import Optional

from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from ztfnuclear import io, baseline
from ztfnuclear.database import MetadataDB, SampleInfo
from ztfnuclear.plot import plot_lightcurve
from ztfnuclear.fritz import FritzAPI

logger = logging.getLogger(__name__)


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

        if not db_check["has_tdefit"]:
            tdefit_res = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_fitres, "tdefit.json"),
                parameter_name="tde_fit",
            )
            self.populate_db_from_dict(data=tdefit_res)

        if not db_check["has_ampel_z"]:
            ampelz = io.parse_ampel_json(
                filepath=os.path.join(io.LOCALSOURCE_ampelz),
                parameter_name="ampel_z",
            )
            self.populate_db_from_dict(data=ampelz)

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
        """Query Fritz for the full sample"""
        self.logger.info("Obtaining metadata on full sample from Fritz")
        for i, ztfid in tqdm(enumerate(self.ztfids[startindex:])):
            self.logger.debug(f"{ztfid}: Querying Fritz")
            self.logger.debug(f"Transient {i+startindex} of {len(self.ztfids)}")
            t = Transient(ztfid)
            t.fritz()
        info = SampleInfo()
        date_now = datetime.datetime.now().replace(microsecond=0)
        info.update(data={"fritz_info": {"fritz": True, "date": date_now}})


class Transient(object):
    """
    This class contains all info for a given transient
    """

    def __init__(self, ztfid: str):
        super(Transient, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ztfid = ztfid

        self.df = io.get_ztfid_dataframe(ztfid=self.ztfid)
        self.header = io.get_ztfid_header(ztfid=self.ztfid)

        self.ra = float(self.header["ra"])
        self.dec = float(self.header["dec"])

        location_all = io.get_locations()
        self.location = location_all.loc[self.ztfid].to_dict()

    @cached_property
    def baseline(self) -> pd.DataFrame:
        """Obtain the baseline correction, recalculate if not present"""

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
        """Recalculate the baseline"""
        bl, bl_info = baseline.baseline(transient=self, primary_grid_only=True)
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
    def z(self) -> Optional[float]:
        """
        Get the AMPEL redshift from the database
        """
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
        if "z_dist" in self.meta["ampel_z"].keys():
            ampel_z_dist = self.meta["ampel_z"]["z_dist"]
            return ampel_z_dist
        else:
            return None

    @cached_property
    def meta(self) -> Optional[dict]:
        """
        Read all metadata  for transient from the database
        """
        meta = MetadataDB()
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
        )

        results = {}

        crts_res = query_crts(ra_deg=self.ra, dec_deg=self.dec)
        millliquas_res = query_milliquas(ra_deg=self.ra, dec_deg=self.dec)
        sdss_res = query_sdss(ra_deg=self.ra, dec_deg=self.dec)
        gaia_res = query_gaia(ra_deg=self.ra, dec_deg=self.dec)
        wise_res = query_wise(ra_deg=self.ra, dec_deg=self.dec)

        for res in [crts_res, millliquas_res, sdss_res, gaia_res, wise_res]:
            results.update(res)

        self.crossmatch = {"crossmatch": results}
        meta = MetadataDB()
        meta.update_transient(ztfid=self.ztfid, data=self.crossmatch)

    def fritz(self):
        """
        Query Fritz for transient info and update the database
        """
        fritz = FritzAPI()
        fritzinfo = fritz.get_transient(self.ztfid)
        data = fritzinfo[self.ztfid]
        meta = MetadataDB()
        meta.update_transient(ztfid=self.ztfid, data=data)

    def plot(
        self,
        baseline_correction: bool = True,
        magplot: bool = True,
        include_wise: bool = False,
    ):
        """
        Plot the transient lightcurve
        """
        if include_wise:
            wise_df = self.wise_lc
        else:
            wise_df = None

        if baseline_correction:
            plot_lightcurve(
                df=self.baseline, ztfid=self.ztfid, magplot=magplot, wise_df=wise_df
            )
        else:
            plot_lightcurve(
                df=self.raw_lc, ztfid=self.ztfid, magplot=magplot, wise_df=wise_df
            )
