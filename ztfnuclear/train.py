#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging, time, os
from typing import List

import joblib

from pathlib import Path

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
from sklearn.preprocessing import LabelEncoder


from ztfnuclear import io
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
        grid_search_sample_size: int = 1000,
        model_dir: Path | str | None = None,
    ):
        super(Model, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.noisified = noisified
        self.validation_fraction = validation_fraction
        self.train_test_fraction = train_test_fraction
        self.seed = seed
        self.n_iter = n_iter
        self.grid_search_sample_size = grid_search_sample_size

        if model_dir is None:
            self.model_dir = io.MODEL_dir
        elif isinstance(model, str):
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = model_dir

        if not self.model_dir.is_dir():
            os.makedirs(self.model_dir)

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

        self.meta.drop(
            columns=["RA", "Dec", "tde_fit_exp_covariance", "sample"], inplace=True
        )
        self.meta = self.meta.astype({"median_distnr": "float64", "classif": "str"})

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

        df_train = shuffle(df_train, random_state=self.seed).reset_index(drop=True)
        df_test = shuffle(df_test, random_state=self.seed).reset_index(drop=True)

        le = LabelEncoder()

        self.X_train = df_train.drop(columns="classif").reset_index(drop=True)
        self.X_test = df_test.drop(columns="classif").reset_index(drop=True)
        self.y_train = df_train.filter(["classif"]).reset_index(drop=True)

        self.y_train["classif"] = le.fit_transform(self.y_train["classif"])
        self.y_train = self.y_train["classif"]

        self.y_test = df_test.filter(["classif"]).reset_index(drop=True)
        self.y_test["classif"] = le.fit_transform(self.y_test["classif"])
        self.y_test = self.y_test["classif"]

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

    def train(self):
        t_start = time.time()

        # print(type(len(self.y_train)))
        # print(np.sum(self.y_train))

        # scale_pos_weight = (len(self.y_train) - np.sum(self.y_train)) / np.sum(
        # self.y_train
        # )

        model = xgb.XGBClassifier(
            # scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            objective="multi:softmax",
            num_class=max(self.y_train) + 1,
            eval_metric="aucpr",
            colsample_bytree=1.0,
        )

        param_grid = {
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "min_child_weight": np.arange(0.0001, 0.5, 0.001),
            "gamma": np.arange(0.0, 40.0, 0.005),
            "learning_rate": np.arange(0.0005, 0.5, 0.0005),
            "subsample": np.arange(0.01, 1.0, 0.01),
            "colsample_bylevel": np.round(np.arange(0.1, 1.0, 0.01)),
            # "colsample_bytree": np.arange(0.1, 1.0, 0.01),
        }

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed + 3)

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=None,
            n_iter=self.n_iter,
            cv=kfold,
            random_state=self.seed + 4,
            verbose=2,
            error_score="raise",
        )

        """
        Now we downsample our training set to do a
        fine-grained grid search. Training will be done
        on the best estimator from that search and uses
        the full sample
        """
        self.logger.info(
            f"Downsampling for grid search to {self.grid_search_sample_size} entries\n"
        )
        X_train_subset = self.X_train.sample(
            n=self.grid_search_sample_size, random_state=self.seed + 5
        )
        y_train_subset = self.y_train.sample(
            n=self.grid_search_sample_size, random_state=self.seed + 5
        )

        grid_result = grid_search.fit(X_train_subset, y_train_subset)

        """
        Run the actual training with the best estimator
        on the full training sample
        """
        self.logger.info("--------------------------------------------")
        self.logger.info(
            "\n\nNow fitting with the best estimator from the grid search. This will take time\n"
        )
        self.logger.info("--------------------------------------------")

        best_estimator = grid_result.best_estimator_.fit(self.X_train, self.y_train)

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        outpath_grid = (
            self.model_dir
            / f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}"
        )

        outpath_model = (
            self.model_dir
            / f"model_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}"
        )

        joblib.dump(grid_result, str(outpath_grid))
        joblib.dump(best_estimator, str(outpath_model))

        t_end = time.time()

        self.logger.info("------------------------------------")
        self.logger.info("           FITTING DONE             ")
        self.logger.info(f"  This took {(t_end-t_start)/60} minutes")
        self.logger.info("------------------------------------")

    def evaluate(self):
        """
        Evaluate the model
        """

        # Load the stuff
        infile_grid = (
            self.model_dir
            / f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
        )

        grid_result = joblib.load(infile_grid)
        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        self.logger.info(f"Loading best estimator. Parameters:\n{self.best_estimator}")

        # Plot feature importance for full set
        self.plot_features()

        """
        Now we cut the test sample so that only one datapoint
        per stock-ID survives
        """
        df_test_subsample = self.get_random_stock_subsample(self.df_test)

        logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        self.df_test_subsample = df_test_subsample

        # We get even sized binning (at least as far as possible)
        evaluation_bins, nbins = self.get_optimal_bins(nbins=14)

        self.evaluation_bins = evaluation_bins

        logger.info(f"\nWe now plot the evaluation using {nbins} time bins")

        precision_list = []
        recall_list = []
        aucpr_list = []
        timebin_mean_list = []

        for timebin in evaluation_bins:
            df_test_bin = df_test_subsample[
                (df_test_subsample["ndet"] >= timebin[0])
                & (df_test_subsample["ndet"] <= timebin[1])
            ]
            X_test = df_test_bin.drop(columns=["class_short", "stock"])

            self.cols_to_use.append("stock")
            y_test = df_test_bin.drop(columns=self.cols_to_use)
            features = X_test
            target = y_test

            pred = best_estimator.predict(features)

            precision_list.append(metrics.precision_score(target, pred))
            recall_list.append(metrics.recall_score(target, pred))
            aucpr_list.append(metrics.average_precision_score(target, pred))

            timebin_mean_list.append(np.mean([timebin[0], timebin[1]]))

        outfiles = [
            os.path.join(
                self.plot_dir,
                f"{i}_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}.png",
            )
            for i in ["precision", "recall", "aucpr"]
        ]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, precision_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("precision")
        ax.set_ylim([0.5, 1])
        fig.savefig(outfiles[0], dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, recall_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("recall")
        ax.set_ylim([0.75, 1])
        fig.savefig(outfiles[1], dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, aucpr_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("aucpr")
        ax.set_ylim([0.5, 1])
        fig.savefig(outfiles[2], dpi=300)
        plt.close()

    def get_optimal_bins(self, nbins=20):
        """
        Determine optimal time bins (requirement: same number
        of alerts per bin). This cannot always be fulfilled, so duplicates=drop is passed.
        """
        out, bins = pd.qcut(
            self.df_test_subsample.ndet.values, nbins, retbins=True, duplicates="drop"
        )
        final_bins = []
        for i in range(len(bins) - 1):
            final_bins.append([int(bins[i]), int(bins[i + 1])])
        nbins = len(final_bins)
        return final_bins, nbins

    def get_random_stock_subsample(self, df):
        """
        Returns a df consisting of one random datapoint for each unique stock ID
        """
        df_sample = df.groupby("stock").sample(n=1, random_state=self.random_state)

        return df_sample

    def plot_features(self):
        """
        Plot the features in their importance for the classification decision
        """

        fig, ax = plt.subplots(figsize=(10, 21))

        cols = self.cols_to_use

        cols.remove("stock")

        ax.barh(cols, self.best_estimator.feature_importances_)
        plt.title("Feature importance", fontsize=25)
        plt.tight_layout()

        outfile = os.path.join(
            self.plot_dir,
            f"feature_importance_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}.png",
        )

        fig.savefig(
            outfile,
            dpi=300,
        )
