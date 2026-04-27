import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OrdinalEncoder


class DetectionScore:
    """Estimates detection score based on AUC using a lightgbm classifier.

    Similar to detection score in SDMetrics, see https://github.com/sdv-dev/SDMetrics/blob/main/sdmetrics/single_table/detection/base.py.

    A detection score close to 1 indicates that the model cannot distinguish between real and generated data, while a score close to 0 indicates that the model can easily distinguish between the two.

    """

    def __init__(self, cat_cols, num_cols, max_obs=100_000):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.max_obs = max_obs

    def prep_data(self, df_trn, df_gen, drop="none", seed=42):
        X_num_train = df_trn.select(self.num_cols).to_pandas()
        X_num_gen = df_gen.select(self.num_cols).to_pandas()

        cat_enc = OrdinalEncoder()
        cat_enc.fit(df_trn.vstack(df_gen).select(self.cat_cols))
        X_cat_train = cat_enc.transform(df_trn.select(self.cat_cols))
        X_cat_gen = cat_enc.transform(df_gen.select(self.cat_cols))
        X_cat_train = pd.DataFrame(X_cat_train.astype(int), columns=self.cat_cols, dtype="category")
        X_cat_gen = pd.DataFrame(X_cat_gen.astype(int), columns=self.cat_cols, dtype="category")

        # subsample if necessary to limit needed resources
        if df_trn.height > self.max_obs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_cat_train.shape[0], self.max_obs, replace=False)
            X_num_train = X_num_train.iloc[idx]
            X_cat_train = X_cat_train.iloc[idx]

        assert X_num_train.shape[0] == X_num_gen.shape[0], (
            "Training and generated data must have the same number of rows."
        )

        # construct data with 50% fake 50% real observations
        X_real = []
        X_fake = []
        if drop in ["num", "none"]:
            X_real.append(X_cat_train)
            X_fake.append(X_cat_gen)
            cat_features = self.cat_cols

        if drop in ["cat", "none"]:
            X_real.append(X_num_train)
            X_fake.append(X_num_gen)

        if drop == "cat":
            cat_features = []

        X_fake = pd.concat(X_fake, axis=1)
        X_real = pd.concat(X_real, axis=1)

        # y = 1 if fake, y = 0 if real
        labels_real = np.zeros((X_real.shape[0],))
        labels_fake = np.ones((X_fake.shape[0],))
        X = pd.concat((X_real, X_fake), axis=0, ignore_index=True)
        y = np.concatenate((labels_real, labels_fake)).astype(int)

        data = lgb.Dataset(X, label=y, categorical_feature=cat_features)

        return data

    def estimate_score(self, df_trn, df_gen, seed=42, nfold=5, drop="none"):
        train_data = self.prep_data(df_trn, df_gen, drop, seed=seed)
        train_labels = list(train_data.data.columns)

        # by default uses stratified sampling
        params = {
            "objective": "binary",
            "deterministic": True,
            "metric": "auc",
            "early_stopping_round": 10,
            "verbosity": -1,
            "seed": seed,
            "max_depth": 5,
            "num_leaves": 2**5 - 1,
        }
        results = lgb.cv(params, train_data, nfold=nfold, num_boost_round=200, seed=seed, return_cvbooster=True)

        # we take model that achieved best average AUC over validation sets across all iterations (each iteration adds a tree learner)
        avg_auc = max(results["valid auc-mean"])
        score = 1 - (np.maximum(0.5, avg_auc) * 2 - 1)

        # estimate feature importance
        feat_imp = np.zeros((len(train_labels),))
        for i in range(nfold):
            feat_imp += results["cvbooster"].boosters[i].feature_importance(importance_type="gain")
        feat_imp /= nfold
        feat_imp = pl.DataFrame(feat_imp[None,], schema=train_labels)

        results = {
            "detection_score": score.item(),
            "detection_feat_imp": feat_imp,
        }

        return results
