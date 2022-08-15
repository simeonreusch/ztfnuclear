#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, logging

from tqdm import tqdm
import numpy as np
import pandas as pd

from ztfnuclear import io, baseline


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
