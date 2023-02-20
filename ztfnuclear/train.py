#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
from typing import List

import numpy as np
import pandas as pd
from numpy.random import default_rng

import xgboost as xgb
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)


from ztfnuclear.sample import NuclearSample
from ztfnuclear.plot import get_tde_selection

# ToDo
# -add peakmag
# -add wisecolors


class Model(object):
    """
    Do fancy ML
    """

    def __init__(
        self,
        noisified: bool = True,
        seed: int | None = None,
        validation_fraction: float = 0.1,
        train_test_fraction: float = 0.7,
        n_iter: int = 10,
    ):
        super(Model, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.noisified = noisified
        self.validation_fraction = validation_fraction
        self.train_test_fraction = train_test_fraction
        self.seed = seed
        self.n_iter = n_iter

        self.rng = default_rng(seed=self.seed)
        self.get_training_metadata()
        self.get_validation_sample()

        self.train_test_split()

    def get_training_metadata(self) -> pd.DataFrame:
        """
        Read both samples and get feature dataframe for training ML
        """
        if not self.noisified:
            nuc = NuclearSample(sampletype="nuclear")
            bts = NuclearSample(sampletype="bts")
            nuc_df = nuc.meta.get_dataframe(for_training=True)
            bts_df = bts.meta.get_dataframe(for_training=True)

            self.meta = pd.concat([nuc_df, bts_df])
            self.meta.query("classif != 'unclass'", inplace=True)

        else:
            train = NuclearSample(sampletype="train")
            self.meta = train.meta.get_dataframe(for_training=True)
            self.meta.rename(columns={"simpleclasses": "classif"}, inplace=True)

        self.logger.info(f"Read metadata. {len(self.meta)} transients available.")
        self.meta.to_csv("test.csv")
        self.all_ztfids = self.meta.index.values

        self.parent_ztfids = self.get_parent_ztfids(self.all_ztfids)
        self.logger.info(f"{len(self.parent_ztfids)} parent ZTFIDs available.")

    def get_validation_sample(self) -> List[str]:
        """
        Get a validation sample
        """
        self.validation_parent_ztfids = self.rng.choice(
            self.parent_ztfids,
            size=int(self.validation_fraction * len(self.parent_ztfids)),
            replace=False,
        )
        self.validation_ztfids = self.get_child_ztfids(self.validation_parent_ztfids)
        self.logger.info(
            f"Selected {len(self.validation_parent_ztfids)} validation ZTFIDs from {len(self.parent_ztfids)} parent ZTFIDs. These comprise {len(self.validation_ztfids)} lightcurves."
        )
        self.train_test_parent_ztfids = [
            ztfid
            for ztfid in self.parent_ztfids
            if ztfid not in self.validation_parent_ztfids
        ]
        self.validation_sample = self.meta.query("index in @self.validation_ztfids")

    def train_test_split(self):
        """
        Split the remaining sample (all minus validation) into a test and a training sample
        """
        self.train_parent_ztfids = self.rng.choice(
            self.train_test_parent_ztfids,
            size=int(self.train_test_fraction * len(self.train_test_parent_ztfids)),
            replace=False,
        )
        self.test_parent_ztfids = [
            ztfid
            for ztfid in self.train_test_parent_ztfids
            if ztfid not in self.train_parent_ztfids
        ]

        self.train_ztfids = self.get_child_ztfids(ztfids=self.train_parent_ztfids)
        self.test_ztfids = self.get_child_ztfids(ztfids=self.test_parent_ztfids)

        self.logger.info(
            f"From {len(self.train_test_parent_ztfids)} available parent ZTFIDs selected {len(self.train_parent_ztfids)} for training, {len(self.test_parent_ztfids)} for testing."
        )
        self.logger.info(
            f"Train sample: {len(self.train_ztfids)} lightcurves in total."
        )
        self.logger.info(f"Test sample: {len(self.test_ztfids)} lightcurves in total.")

        df_train = self.meta.query("index in @self.train_ztfids")
        df_test = self.meta.query("index in @self.test_ztfids")

        X_train = df_train.drop(columns="classif").reset_index(drop=True)
        X_test = df_test.drop(columns="classif").reset_index(drop=True)
        y_train = df_train.filter(["classif"]).reset_index(drop=True)["class_short"]
        y_test = df_test.filter(["classif"]).reset_index(drop=True)["class_short"]

        print(y_test)

        # df["class_short"] = y.values

        # df = shuffle(df, random_state=random_state).reset_index(drop=True)

    def get_parent_ztfids(self, ztfids: List[str]):
        parent_ztfids = [i for i in ztfids if len(i.split("_")) == 1]
        return parent_ztfids

    def get_child_ztfids(self, ztfids: List[str], include_parent: bool = True):
        child_ztfids = [
            i
            for i in self.all_ztfids
            if (len(isplit := i.split("_")) == 2 and isplit[0] in ztfids)
        ]

        if include_parent:
            child_ztfids.extend(ztfids)
        return child_ztfids
