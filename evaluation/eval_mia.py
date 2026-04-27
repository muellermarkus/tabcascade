import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


class MIAScore:
    """
    Evaluates membership inference attack (MIA) using a LightGBM classifier.

    Broadly based on https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy/metric_MIA_classification.py
    """

    def __init__(self, cat_cols, num_cols, max_obs=100_000):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_obs = max_obs

    def prep_data(self, df_trn, df_tst, df_gen, seed=42):
        X_num_train = df_trn.select(self.num_cols).to_pandas()
        X_num_test = df_tst.select(self.num_cols).to_pandas()
        X_num_gen = df_gen.select(self.num_cols).to_pandas()

        # combine all data for consistent encoding
        df_all_cat = df_trn.vstack(df_tst).vstack(df_gen).select(self.cat_cols)
        cat_enc = OrdinalEncoder()
        cat_enc.fit(df_all_cat)
        X_cat_train = cat_enc.transform(df_trn.select(self.cat_cols))
        X_cat_test = cat_enc.transform(df_tst.select(self.cat_cols))
        X_cat_gen = cat_enc.transform(df_gen.select(self.cat_cols))
        X_cat_train = pd.DataFrame(X_cat_train.astype(int), columns=self.cat_cols, dtype="int")
        X_cat_test = pd.DataFrame(X_cat_test.astype(int), columns=self.cat_cols, dtype="int")
        X_cat_gen = pd.DataFrame(X_cat_gen.astype(int), columns=self.cat_cols, dtype="int")

        # subsample if necessary to limit needed resources
        if df_tst.height > self.max_obs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_cat_test.shape[0], self.max_obs, replace=False)
            X_num_test = X_num_test.iloc[idx]
            X_cat_test = X_cat_test.iloc[idx]

        data = {
            "num": {"train": X_num_train, "test": X_num_test, "gen": X_num_gen},
            "cat": {"train": X_cat_train, "test": X_cat_test, "gen": X_cat_gen},
        }

        return data

    def construct_train_and_eval_sets(self, data, seed):
        rng = np.random.default_rng(seed)

        tst_train_num, tst_test_num, tst_train_cat, tst_test_cat = train_test_split(
            data["num"]["test"],
            data["cat"]["test"],
            test_size=0.25,
        )

        # get subset of generated data for training
        idx = rng.choice(data["num"]["gen"].shape[0], tst_train_num.shape[0], replace=False)
        gen_train_num = data["num"]["gen"].iloc[idx]
        gen_train_cat = data["cat"]["gen"].iloc[idx]
        gen_train = pd.concat([gen_train_num, gen_train_cat], axis=1)

        # create training set for GBM
        tst_train = pd.concat([tst_train_num, tst_train_cat], axis=1)
        X_train = pd.concat([gen_train, tst_train], axis=0, ignore_index=True)
        y_train = np.concatenate((np.ones((gen_train_num.shape[0],)), np.zeros((tst_train_num.shape[0],)))).astype(int)

        # create test set for GBM
        assert tst_test_num.shape[0] <= data["num"]["train"].shape[0]
        idx = rng.choice(data["num"]["train"].shape[0], tst_test_num.shape[0], replace=False)
        trn_test_num = data["num"]["train"].iloc[idx]
        trn_test_cat = data["cat"]["train"].iloc[idx]
        trn_test = pd.concat([trn_test_num, trn_test_cat], axis=1)
        tst_test = pd.concat([tst_test_num, tst_test_cat], axis=1)

        X_test = pd.concat([trn_test, tst_test], axis=0, ignore_index=True)
        y_test = np.concatenate((np.ones((trn_test.shape[0],)), np.zeros((tst_test.shape[0],)))).astype(int)

        # construct lgb dataloader
        cat_features = self.cat_cols

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

        return train_data, X_test, y_test

    def estimate_score(self, df_trn, df_tst, df_gen, seed=42, n_iter=5):
        data = self.prep_data(df_trn, df_tst, df_gen, seed=seed)

        params = {
            "objective": "binary",
            "deterministic": True,
            "verbosity": -1,
            "seed": seed,
            "max_depth": 5,
            "num_leaves": 2**5 - 1,
        }

        scores = []
        for i in range(n_iter):
            train_data, X_test, y_test = self.construct_train_and_eval_sets(data, seed=seed + i)

            # train LightGBM model and get predictions
            gbm = lgb.train(params, train_data, num_boost_round=200)
            y_pred = gbm.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            scores.append(1 - (np.maximum(0.5, auc) * 2 - 1))

        results = {"mia_score": np.array(scores).mean().item()}

        return results
