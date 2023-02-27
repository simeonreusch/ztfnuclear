#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from numpy.random import default_rng
from sklearn import metrics
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from ztfnuclear import io
from ztfnuclear.plot import get_tde_selection
from ztfnuclear.sample import NuclearSample


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
        noisified_validation: bool = True,
        n_iter: int = 10,
        grid_search_sample_size: int = 1000,
        model_dir: Path | str = Path(io.MODEL_dir),
        plot_dir: Path | str = Path(io.LOCALSOURCE_plots) / "train",
    ):
        super(Model, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.noisified = noisified
        self.noisified_validation = noisified_validation
        self.validation_fraction = validation_fraction
        self.train_test_fraction = train_test_fraction
        self.seed = seed
        self.n_iter = n_iter
        self.grid_search_sample_size = grid_search_sample_size

        self.le = LabelEncoder()

        if isinstance(model_dir, str):
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = model_dir

        if isinstance(plot_dir, str):
            self.plot_dir = Path(plot_dir)
        else:
            self.plot_dir = plot_dir

        for p in [self.model_dir, self.plot_dir]:
            if not p.is_dir():
                os.makedirs(p)

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
        self.meta = self.meta.astype({"distnr": "float64", "classif": "str"})

        self.logger.info(f"Read metadata. {len(self.meta)} transients available.")
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
        if self.noisified_validation:
            self.validation_ztfids = self.get_child_ztfids(
                self.validation_parent_ztfids
            )
        else:
            self.validation_ztfids = self.validation_parent_ztfids

        self.logger.info(
            f"Selected {len(self.validation_parent_ztfids)} validation ZTFIDs from {len(self.parent_ztfids)} parent ZTFIDs. These comprise {len(self.validation_ztfids)} lightcurves."
        )
        self.train_test_parent_ztfids = [
            ztfid
            for ztfid in self.parent_ztfids
            if ztfid not in self.validation_parent_ztfids
        ]
        df_validation = self.meta.query("index in @self.validation_ztfids")
        df_validation = shuffle(df_validation, random_state=self.seed).reset_index(
            drop=True
        )
        self.X_validation = df_validation.drop(columns="classif").reset_index(drop=True)
        self.y_validation = df_validation.filter(["classif"]).reset_index(drop=True)
        self.y_validation["classif"] = self.le.fit_transform(
            self.y_validation["classif"]
        )
        self.y_validation = self.y_validation["classif"]

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

        self.X_train = df_train.drop(columns="classif").reset_index(drop=True)
        self.X_test = df_test.drop(columns="classif").reset_index(drop=True)
        self.y_train = df_train.filter(["classif"]).reset_index(drop=True)

        self.y_train["classif"] = self.le.fit_transform(self.y_train["classif"])
        self.y_train = self.y_train["classif"]

        self.y_test = df_test.filter(["classif"]).reset_index(drop=True)
        self.y_test["classif"] = self.le.fit_transform(self.y_test["classif"])
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
        self.logger.info("\nNow fitting with the best estimator from the grid search.")
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
        self.logger.info(f"  This took {(t_end-t_start)/60:.2f} minutes")
        self.logger.info("------------------------------------")

    def evaluate(self, normalize=True):
        """
        Evaluate the model
        """

        # Load the stuff
        infile_grid = (
            self.model_dir
            / f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}"
        )

        grid_result = joblib.load(infile_grid)
        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        # print(self.le.get_params())
        # quit()

        self.logger.info(f"Loading best estimator. Parameters:\n{self.best_estimator}")

        # Plot feature importance for full set
        self.logger.info("Plotting feature importance")
        self.plot_features()

        self.logger.info("Plotting evaluation")

        features = self.X_validation
        target = self.y_validation

        pred = best_estimator.predict(features)

        # 0: agn
        # 1: sn_ia
        # 2: sn_other
        # 3: star
        # 4: tde

        if normalize:
            norm_str = "true"
        else:
            norm_str = None

        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_true=target,
            y_pred=pred,
            display_labels=["agn", "snia", "sn_other", "star", "tde"],
            normalize=norm_str,
        )
        disp.plot()
        outfile = (
            self.plot_dir
            / f"results_seed_{self.seed}_n_iter_{self.n_iter}_noisified_val_{self.noisified_validation}_normalized_{normalize}.pdf"
        )
        plt.savefig(outfile)
        self.logger.info(f"We saved the evaluation to {outfile}")

    def plot_features(self):
        """
        Plot the features in their importance for the classification decision
        """

        fig, ax = plt.subplots(figsize=(10, 21))

        cols = list(self.X_train.keys())

        ax.barh(cols, self.best_estimator.feature_importances_)
        plt.title("Feature importance", fontsize=25)
        plt.tight_layout()

        outfile = (
            self.plot_dir
            / f"feature_importance_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}.pdf"
        )

        fig.savefig(
            outfile,
            dpi=300,
        )
