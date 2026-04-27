import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


class MLEScore:
    """Evaluates machine learning efficiency of generated data compared to real data.

    Trains a LightGBM on real and synthetic data and compares the performane on a real test set.
    Also outputs the absolute differences in feature importances.
    Good synthetic data should yield similar performance and feature importance as the real data.

    """

    def __init__(self, cat_cols, num_cols, target, max_obs=100_000, boost_rounds=200):
        self.max_obs = max_obs
        self.boost_rounds = boost_rounds

        # remove target from cat_cols or num_cols
        self.target = target
        self.num_cols = num_cols.copy()
        self.cat_cols = cat_cols.copy()
        self.is_regression = False
        if self.target in cat_cols:
            self.cat_cols.remove(self.target)
        if self.target in num_cols:
            self.num_cols.remove(self.target)
            self.is_regression = True

    def prep_data(self, df_trn, df_test, df_gen, seed=42):
        X_num_trn = df_trn.select(self.num_cols).to_pandas()
        X_num_gen = df_gen.select(self.num_cols).to_pandas()
        X_num_test = df_test.select(self.num_cols)

        if len(self.cat_cols) > 0:
            X_cat_fit = pl.concat([df_trn.vstack(df_gen).select(self.cat_cols), df_test.select(self.cat_cols)])
            cat_enc = OrdinalEncoder().fit(X_cat_fit)
            X_cat_trn = cat_enc.transform(df_trn.select(self.cat_cols))
            X_cat_gen = cat_enc.transform(df_gen.select(self.cat_cols))
            X_cat_trn = pd.DataFrame(X_cat_trn.astype(int), columns=self.cat_cols, dtype="category")
            X_cat_gen = pd.DataFrame(X_cat_gen.astype(int), columns=self.cat_cols, dtype="category")
            X_cat_test = cat_enc.transform(df_test.select(self.cat_cols))

        # get target
        y_trn = df_trn.select(self.target).to_numpy().ravel()
        y_test = df_test.select(self.target).to_numpy().ravel()
        y_gen = df_gen.select(self.target).to_numpy().ravel()

        if not self.is_regression:
            label_enc = LabelEncoder()
            y_trn = label_enc.fit_transform(y_trn)
            y_test = label_enc.transform(y_test)
            y_gen = label_enc.transform(y_gen)
        else:
            y_scaler = StandardScaler()
            y_trn = y_scaler.fit_transform(y_trn.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
            y_gen = y_scaler.transform(y_gen.reshape(-1, 1)).ravel()

        # subsample if necessary to limit needed resources
        if df_trn.height > self.max_obs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_cat_trn.shape[0], self.max_obs, replace=False)
            X_num_trn = X_num_trn.iloc[idx].reset_index(drop=True)
            X_cat_trn = X_cat_trn.iloc[idx].reset_index(drop=True)
            y_trn = y_trn[idx]

        X_trn = pd.concat((X_cat_trn, X_num_trn), axis=1) if len(self.cat_cols) > 0 else X_num_trn
        X_gen = pd.concat((X_cat_gen, X_num_gen), axis=1) if len(self.cat_cols) > 0 else X_num_gen
        X_test = np.column_stack((X_cat_test, X_num_test)) if len(self.cat_cols) > 0 else X_num_test.to_numpy()

        data_trn = lgb.Dataset(X_trn, label=y_trn, categorical_feature=self.cat_cols)
        data_gen = lgb.Dataset(X_gen, label=y_gen, categorical_feature=self.cat_cols)

        return data_trn, data_gen, X_test, y_test

    def train_gbm(self, train_set, X_test, y_test, seed=42):
        objective = "regression" if self.is_regression else "binary"
        params = {
            "objective": objective,
            "deterministic": True,
            "verbosity": -1,
            "seed": seed,
            "max_depth": 5,
            "num_leaves": 2**5 - 1,
        }
        gbm = lgb.train(params, train_set, num_boost_round=self.boost_rounds)

        # retrieve probabiliies for classification and predictions for regression
        y_pred = gbm.predict(X_test)

        if not self.is_regression:
            auc = roc_auc_score(y_test, y_pred)
        else:
            rmse = root_mean_squared_error(y_test, y_pred)

        # get feature importances
        feat_imp = gbm.feature_importance(importance_type="gain")

        return auc if not self.is_regression else rmse, feat_imp

    def get_score(self, df_trn, df_test, df_gen, seed):
        data_trn, data_gen, X_test, y_test = self.prep_data(df_trn, df_test, df_gen, seed)

        # train on real data
        score_real, feat_imp_real = self.train_gbm(data_trn, X_test, y_test, seed)

        # train on generated data
        score_gen, feat_imp_gen = self.train_gbm(data_gen, X_test, y_test, seed)

        # compute rank distance of feature importances
        stat, _ = kendalltau(feat_imp_real, feat_imp_gen, nan_policy="raise")

        return {
            "mle_real": score_real,
            "mle_gen": score_gen,
            "mle_abs_diff": abs(score_real - score_gen),
            "mle_feat_imp_real": feat_imp_real,
            "mle_feat_imp_gen": feat_imp_gen,
            "mle_feat_imp_abs_diff": np.abs(feat_imp_real - feat_imp_gen).mean().item(),
            "mle_feat_imp_rank_dist": stat.item(),
        }
