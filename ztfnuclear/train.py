#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import itertools
import json
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from tqdm import tqdm
from ztfnuclear import io
from ztfnuclear.plot import get_tde_selection
from ztfnuclear.sample import NuclearSample

GOLDEN_RATIO = 1.62


class Model(object):
    """
    Do fancy ML
    """

    def __init__(
        self,
        noisified: bool = True,
        seed: int | None = None,
        test_fraction: float = 0.1,
        train_validation_fraction: float = 0.7,
        noisified_test: bool = True,
        nuclear_test: bool = True,
        n_iter: int = 10,
        grid_search_sample_size: int = 1000,
        model_dir: Path | str = Path(io.MODEL_dir),
        plot_dir: Path | str = Path(io.LOCALSOURCE_train_plots),
    ):
        super(Model, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.noisified = noisified
        self.noisified_test = noisified_test
        self.nuclear_test = nuclear_test
        self.test_fraction = test_fraction
        self.train_validation_fraction = train_validation_fraction
        self.seed = seed
        self.n_iter = n_iter
        self.grid_search_sample_size = grid_search_sample_size
        self.config = io.load_config()

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
        self.get_test_sample()
        self.train_validation_split()

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
            self.meta = train.meta.get_dataframe(
                for_training=True, include_z_in_training=False
            )
            self.meta.rename(columns={"simpleclasses": "classif"}, inplace=True)
            self.meta.replace({"classif": {"sn_ia": "snia"}}, inplace=True)

        # self.meta.query("classif != 'agn'", inplace=True)

        self.meta.drop(
            columns=[
                "sample",
            ],
            inplace=True,
        )
        self.meta = self.meta.astype({"classif": "str"})
        if "distnr" in self.meta.keys():
            self.meta = self.meta.astype({"distnr": "float64"})
        self.meta_parent = self.meta.copy(deep=True)
        self.meta_parent = self.meta_parent[~self.meta_parent.index.str.contains("_")]

        self.logger.info(f"Read metadata. {len(self.meta)} transients available.")
        self.all_ztfids = self.meta.index.values

        self.parent_ztfids = self.get_parent_ztfids(self.all_ztfids)
        self.logger.info(f"{len(self.parent_ztfids)} parent ZTFIDs available.")

    def get_test_sample(self) -> List[str]:
        """
        Get a test sample
        """
        if self.nuclear_test:
            self.test_parent_ztfids = self.get_test_per_class(
                select_classes=["tde", "agn", "star", "snia", "sn_other"],
                test_fraction=self.test_fraction,
            )
        else:
            self.test_parent_ztfids = self.rng.choice(
                self.parent_ztfids,
                size=int(self.test_fraction * len(self.parent_ztfids)),
                replace=False,
            )

        if self.noisified_test:
            self.test_ztfids = self.get_child_ztfids(self.test_parent_ztfids)
        else:
            self.test_ztfids = self.test_parent_ztfids

        self.logger.info(
            f"Selected {len(self.test_parent_ztfids)} test ZTFIDs from {len(self.parent_ztfids)} parent ZTFIDs. These comprise {len(self.test_ztfids)} lightcurves."
        )
        self.train_validation_parent_ztfids = [
            ztfid
            for ztfid in self.parent_ztfids
            if ztfid not in self.test_parent_ztfids
        ]
        df_test = self.meta.query("index in @self.test_ztfids")
        df_test = shuffle(df_test, random_state=self.seed).reset_index(drop=True)
        self.X_test = df_test.drop(columns="classif").reset_index(drop=True)
        self.y_test = df_test.filter(["classif"]).reset_index(drop=True)

        self.y_test["classif"] = self.le.fit_transform(self.y_test["classif"])
        label_list = self.le.inverse_transform(self.y_test["classif"])

        # create a dictionary to remember which value belongs to which label
        label_mapping = {}
        for i, classif in enumerate(label_list):
            label_mapping.update({self.y_test["classif"].values[i]: classif})

        self.label_mapping = dict(sorted(label_mapping.items()))

        self.y_test = self.y_test["classif"]

    def get_test_per_class(
        self,
        select_classes: list = ["tde", "agn", "star", "snia", "sn_other"],
        test_fraction: float = 0.3,
    ):
        """
        For each class, select val_fraction of all transients belonging to that class for test. Make sure that
            a) all transients that are both in the nuclear and the BTS sample are kept for test
            b) this rule is not applied to TDEs
        """
        nuc = NuclearSample(sampletype="nuclear")

        nuc_ztfids = set(nuc.ztfids)
        bts_ztfids = set(self.parent_ztfids)

        in_both_list = list(nuc_ztfids.intersection(bts_ztfids))
        in_both = self.meta_parent.copy(deep=True)
        in_both.query("index in @in_both_list", inplace=True)
        in_bts_only = self.meta_parent.copy(deep=True)
        in_bts_only.query("index not in @in_both_list", inplace=True)

        test_parent_ztfids = []

        for cl in select_classes:
            if cl == "tde":
                size = int(test_fraction * len(in_both.query("classif == @cl")))
                test_parent_ztfids.extend(
                    self.rng.choice(
                        in_both.query("classif == @cl").index.values,
                        size=size,
                        replace=False,
                    )
                )
            else:
                desired = int(
                    test_fraction * len(self.meta_parent.query("classif == @cl"))
                )
                nuc = in_both.query("classif == @cl")
                available_nuc = len(nuc)
                needed = desired - available_nuc

                self.logger.info(f"Class: {cl}")
                self.logger.info(f"desired val. size: {desired}")
                self.logger.info(f"available in nuclear sample: {available_nuc}")
                self.logger.info(f"needed: {needed}")

                # now we take all the nuclear transients (if we need them all)
                if available_nuc < desired:
                    test_parent_ztfids.extend(nuc.index.values)
                else:
                    test_parent_ztfids.extend(
                        self.rng.choice(
                            nuc.index.values,
                            size=desired,
                            replace=False,
                        )
                    )

                # and fill with bts only transients (if we must)
                if needed > 0:
                    test_parent_ztfids.extend(
                        self.rng.choice(
                            in_bts_only.query("classif == @cl").index.values,
                            size=needed,
                            replace=False,
                        )
                    )

        return test_parent_ztfids

    def train_validation_split(self):
        """
        Split the remaining sample (all minus test) into a validation and a training sample
        """
        self.train_parent_ztfids = self.rng.choice(
            self.train_validation_parent_ztfids,
            size=int(
                self.train_validation_fraction
                * len(self.train_validation_parent_ztfids)
            ),
            replace=False,
        )
        self.validation_parent_ztfids = [
            ztfid
            for ztfid in self.train_validation_parent_ztfids
            if ztfid not in self.train_parent_ztfids
        ]

        self.train_ztfids = self.get_child_ztfids(ztfids=self.train_parent_ztfids)
        self.validation_ztfids = self.get_child_ztfids(
            ztfids=self.validation_parent_ztfids
        )

        self.logger.info(
            f"From {len(self.train_validation_parent_ztfids)} available parent ZTFIDs selected {len(self.train_parent_ztfids)} for training, {len(self.validation_parent_ztfids)} for validationing."
        )
        self.logger.info(
            f"Train sample: {len(self.train_ztfids)} lightcurves in total."
        )
        self.logger.info(
            f"Test sample: {len(self.validation_ztfids)} lightcurves in total."
        )

        export_dict = {
            "test": list(self.test_ztfids),
            "test_parentonly": list(self.test_parent_ztfids),
            "trainvalidation": self.train_ztfids + self.validation_ztfids,
        }

        with open("split.json", "w") as f:
            json.dump(export_dict, f)

        df_train = self.meta.query("index in @self.train_ztfids")
        df_validation = self.meta.query("index in @self.validation_ztfids")

        df_train = shuffle(df_train, random_state=self.seed).reset_index(drop=True)
        df_validation = shuffle(df_validation, random_state=self.seed).reset_index(
            drop=True
        )

        self.X_train = df_train.drop(columns="classif").reset_index(drop=True)
        self.X_validation = df_validation.drop(columns="classif").reset_index(drop=True)
        self.y_train = df_train.filter(["classif"]).reset_index(drop=True)

        self.y_train["classif"] = self.le.fit_transform(self.y_train["classif"])
        self.y_train = self.y_train["classif"]

        self.y_validation = df_validation.filter(["classif"]).reset_index(drop=True)
        self.y_validation["classif"] = self.le.fit_transform(
            self.y_validation["classif"]
        )
        self.y_validation = self.y_validation["classif"]

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

        model = xgb.XGBClassifier(
            random_state=self.seed,
            objective="multi:softmax",
            num_class=max(self.y_train) + 1,
            eval_metric="aucpr",
            colsample_bytree=1.0,
        )

        # param_grid = {
        #     "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        #     "min_child_weight": np.arange(0.0001, 0.5, 0.001),
        #     "gamma": np.arange(0.0, 40.0, 0.005),
        #     "learning_rate": np.arange(0.0005, 0.5, 0.0005),
        #     "subsample": np.arange(0.01, 1.0, 0.01),
        #     "colsample_bylevel": np.round(np.arange(0.1, 1.0, 0.01)),
        # }
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.001],
            "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
            "max_depth": [2, 4, 7, 10],
            "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
            "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 1.5, 2, 3, 4.5],
            "min_child_weight": [1, 3, 5, 7],
            "n_estimators": [100, 250, 500, 1000],
        }

        if self.seed is None:
            random_state = None
        else:
            random_state = self.seed + 3

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        if self.seed is None:
            random_state = None
        else:
            random_state = self.seed + 4

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=None,
            n_iter=self.n_iter,
            cv=kfold,
            random_state=random_state,
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

        if self.seed is None:
            random_state = None
        else:
            random_state = self.seed + 5

        from sklearn.utils import class_weight

        X_train_subset = self.X_train.sample(
            n=self.grid_search_sample_size, random_state=random_state
        )
        y_train_subset = self.y_train.sample(
            n=self.grid_search_sample_size, random_state=random_state
        )

        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=y_train_subset
        )

        grid_result = grid_search.fit(
            X_train_subset, y_train_subset, sample_weight=classes_weights
        )

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

        self.logger.info(f"Loading best estimator. Parameters:\n{self.best_estimator}")

        # Plot feature importance for full set
        self.logger.info("Plotting feature importance")
        self.plot_features()

        self.logger.info("Plotting evaluation")

        features = self.X_test
        target = self.y_test

        pred = best_estimator.predict(features)

        self.plot_confusion_matrix(
            y_true=target.values, y_pred=pred, normalize=normalize
        )

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True
    ):
        """
        Plot the confusion matrix
        """
        # Use human readable labels (instead of integers)
        y_true_pretty = self.le.inverse_transform(y_true)
        y_pred_pretty = self.le.inverse_transform(y_pred)
        labels = list(self.label_mapping.values())
        labels_pretty = [self.config["classlabels"][i] for i in labels]

        plt.figure(figsize=(5, 4))

        if normalize:
            norm = "true"
            cmlabel = "Fraction of objects"
            fmt = ".2f"
        else:
            norm = None
            cmlabel = "Objects"
            fmt = ".0f"

        cm = confusion_matrix(
            y_true_pretty, y_pred_pretty, labels=labels, normalize=norm
        )

        if normalize:
            vmax = 1
        else:
            vmax = cm.max()

        im = plt.imshow(
            cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax
        )

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels_pretty, ha="center")
        plt.yticks(tick_marks, labels_pretty)

        thresh = cm.max() / 2.0

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True Type", fontsize=12)
        plt.xlabel("Predicted Type", fontsize=12)

        # Make a colorbar that is lined up with the plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.25)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label=cmlabel, fontsize=12)

        outfile = (
            self.plot_dir
            / f"results_seed_{self.seed}_n_iter_{self.n_iter}_noisified_val_{self.noisified_test}_normalized_{normalize}.pdf"
        )
        plt.tight_layout()
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

    def classify(self):
        """
        Use the trained model to classify the nuclear sample
        """
        # Load the model
        infile_grid = (
            self.model_dir
            / f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}"
        )

        self.grid_result = joblib.load(infile_grid)
        self.best_estimator = self.grid_result.best_estimator_

        # Load the nuclear sample
        s = NuclearSample(sampletype="nuclear")
        nuc_df = s.meta.get_dataframe(for_classification=True)
        # nuc_df.query("crossmatch_Milliquas_type.isnull()", inplace=True)

        nuc_df_noclass = nuc_df.copy(deep=True)
        nuc_df_noclass.drop(columns=["classif"], inplace=True)

        pred = self.best_estimator.predict(nuc_df_noclass)

        pred_pretty = [self.label_mapping[i] for i in pred]
        nuc_df["xgclass"] = pred_pretty
        counts = nuc_df["xgclass"].value_counts()

        self.logger.info("Statistics:\n")
        self.logger.info(f"\n{counts}")

        self.logger.info("Ingesting XGBoost classifications into database")

        for ztfid, row in tqdm(nuc_df.iterrows(), total=len(nuc_df)):
            t = s.transient(ztfid)
            t.update({"xgclass": row["xgclass"]})
