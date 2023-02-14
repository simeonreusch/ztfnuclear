#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging

import numpy as np
import pandas as pd

from ztfnuclear.sample import NuclearSample
from ztfnuclear.plot import get_tde_selection


class Train(object):
    """
    Do fancy ML
    """

    def __init__(self):
        super(Train, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.get_training_metadata()

    def get_training_metadata(self) -> pd.DataFrame:
        """
        Read both samples and get feature dataframe for training ML
        """
        nuc = NuclearSample(sampletype="nuclear")
        bts = NuclearSample(sampletype="bts")
        nuc_df = nuc.meta.get_dataframe(for_training=True)
        bts_df = bts.meta.get_dataframe(for_training=True)

        self.meta = pd.concat([nuc_df, bts_df])
        self.meta.query("classif != 'unclass'", inplace=True)

        self.logger.info(f"Read metadata. {len(self.meta)} transients available.")

    def train_test_split(self):
        return 0
