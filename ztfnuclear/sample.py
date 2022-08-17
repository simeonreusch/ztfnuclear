#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging, datetime

from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from ztfnuclear import io, baseline
from ztfnuclear.database import MetadataDB, SampleInfo


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
        self.metadb = MetadataDB()
        db_check = self.metadb.get_statistics()

        if not db_check["has_ra"]:
            self.populate_db_from_csv(filepath=io.LOCALSOURCE_location)

        if not db_check["has_peak_dates"]:
            self.populate_db_from_csv(
                filepath=io.LOCALSOURCE_peak_dates, name="peak_dates"
            )

        db_check = self.metadb.get_statistics()
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
            t = Transient(ztfid, recreate_baseline=True)

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

            self.metadb.update_many(ztfids=ztfids, data=data)
        else:
            raise ValueError("File does not exist")

    def crossmatch(self):
        """Crossmatch the full sample"""
        self.logger.info("Crossmatching the full sample")
        for ztfid in tqdm(self.ztfids):
            t = Transient(ztfid, recreate_baseline=False)
            t.crossmatch()
        info = SampleInfo()
        date_now = datetime.datetime.now().replace(microsecond=0)
        info.update(data={"crossmatch_info": {"crossmatch": True, "date": date_now}})


class Transient(object):
    """
    This class contains all info for a given transient
    """

    def __init__(
        self, ztfid: str, recreate_baseline: bool = False, read_baseline=False
    ):
        super(Transient, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ztfid = ztfid

        self.df = io.get_ztfid_dataframe(ztfid=self.ztfid)
        self.header = io.get_ztfid_header(ztfid=self.ztfid)

        self.ra = float(self.header["ra"])
        self.dec = float(self.header["dec"])

        if recreate_baseline:
            bl, bl_info = baseline.baseline(transient=self)
            self.baseline = bl
        else:
            if read_baseline:
                bl_file = os.path.join(io.LOCALSOURCE_baseline, ztfid + "_bl.csv")
                if os.path.isfile(bl_file):
                    self.baseline = pd.read_csv(bl_file)
                else:
                    self.logger.info(
                        f"{ztfid}: No baseline correction file, trying to apply baseline correction"
                    )
                    bl, bl_info = baseline.baseline(transient=self)
                    self.baseline = bl

        location_all = io.get_locations()
        self.location = location_all.loc[self.ztfid].to_dict()

    def query_z(self):
        """
        Query NEDz via Ampel API
        """
        from ztfnuclear.crossmatch import query_ned_for_z

        ned_res = query_ned_for_z(
            ra_deg=self.ra, dec_deg=self.dec, searchradius_arcsec=10
        )
        if ned_res:
            self.z = ned_res["NEDz"]
            self.z_dist = ned_res["NEDz_dist"]

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
