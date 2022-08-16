#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from ztfnuclear import io, baseline
from ztfnuclear.database import MetadataDB


class NuclearSample(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(self):
        super(NuclearSample, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing sample")

        io.download_if_neccessary()

        self.get_all_ztfids()
        self.meta()

    def get_transient(self, ztfid: str):
        df = io.get_ztfid_dataframe(ztfid=ztfid)
        header = io.get_ztfid_header(ztfid=ztfid)
        return header, df

    def get_all_ztfids(self):
        all_ztfids = io.get_all_ztfids()
        self.ztfids = all_ztfids

    def meta(self):
        df = io.get_metadata()
        self.meta = df

    def create_baseline(self):
        self.logger.info("Creating baseline corrections for the full sample")
        for ztfid in tqdm(self.ztfids):
            t = Transient(ztfid, recreate_baseline=True)

    def populate_db_from_csv(self, filepath, name):
        self.logger.info("Populating the database from a csv file")
        if os.path.isfile(filepath):
            df = pd.read_csv(filepath, comment="#", index_col=0)
            for s in tqdm(df.iterrows()):
                ztfid = s[0]
                if ztfid in self.ztfids:
                    write_dict = {name: s[1].to_dict()}
                    meta = MetadataDB()
                    meta.update(ztfid=ztfid, data=write_dict)
        else:
            raise ValueError("File does not exist")


class Transient(object):
    """
    This class contains all info for a given transient
    """

    def __init__(self, ztfid: str, recreate_baseline: bool = False):
        super(Transient, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.ztfid = ztfid

        self.df = io.get_ztfid_dataframe(ztfid=self.ztfid)
        self.header = io.get_ztfid_header(ztfid=self.ztfid)

        self.ra = self.header["ra"]
        self.dec = self.header["dec"]

        if recreate_baseline:
            bl, bl_info = baseline.baseline(transient=self)
            self.baseline = bl
        else:
            bl_file = os.path.join(io.LOCALSOURCE_baseline, ztfid + "_bl.csv")
            if os.path.isfile(bl_file):
                self.baseline = pd.read_csv(bl_file)
            else:
                self.logger.info(
                    f"{ztfid}: No baseline correction file, trying to apply baseline correction"
                )
                bl, bl_info = baseline.baseline(transient=self)
                self.baseline = bl

        metadata_all = io.get_metadata()
        self.meta = metadata_all.loc[self.ztfid].to_dict()

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
        )

        results = {}

        crts_res = query_crts(ra_deg=self.ra, dec_deg=self.dec)
        millliquas_res = query_milliquas(ra_deg=self.ra, dec_deg=self.dec)
        sdss_res = query_sdss(ra_deg=self.ra, dec_deg=self.dec)
        gaia_res = query_gaia(ra_deg=self.ra, dec_deg=self.dec)

        for res in [crts_res, millliquas_res, sdss_res, gaia_res]:
            results.update(res)

        self.crossmatch = {"crossmatch": results}
