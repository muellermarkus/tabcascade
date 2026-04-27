from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import OmegaConf
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler

from data.data_utils import CatEncoder
from data.fast_dataloader import FastTensorDataLoader
from model.utils import set_seeds

DATA_DIR = Path("data")


def pick_coeffs(X, idxs_nas):
    d_na = len(idxs_nas)
    coeffs = torch.randn(X.shape[1], d_na, dtype=X.dtype)
    Wx = X.mm(coeffs)
    coeffs /= torch.std(Wx, 0, keepdim=True)

    return coeffs


def fit_intercepts(X, coeffs, p):
    d_na = coeffs.shape[1]
    intercepts = torch.zeros(d_na)
    for j in range(d_na):

        def f(x):
            return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

        intercepts[j] = optimize.bisect(f, -500, 500)

    return intercepts


def digit_distribution(series: pd.Series) -> pd.Series:
    """Return the distribution of decimal digit counts for a numeric series."""
    return (
        series.dropna()
        .astype(str)
        .apply(lambda x: len(x.rstrip("0").split(".")[-1]) if "." in x else 0)
        .value_counts()
        .sort_index()
        .rename_axis("n_decimals")
        .rename("count")
    )


class DataProcessor:
    def __init__(
        self,
        dataset,
        cat_encoding="ordinal",
        seed=42,
        val_prop=0.1,
        test_prop=0.1,
        train_batch_size=4096,
        val_batch_size=4096,
        cat_min_freq=5,
        missing_mechanism=None,
        seed_missings=42,
        p_miss=0.1,
        p_obs=0.3,
    ):
        self.name = dataset
        self.cfg = OmegaConf.load(f"{DATA_DIR}/configs/{dataset}.yaml")
        self.file_path = DATA_DIR / "raw" / dataset / self.cfg.csv_file
        self.separator = self.cfg.sep if "sep" in self.cfg else ","

        self.target = self.cfg.target
        self.cat_cols = self.cfg.cat_features
        if "dequant_features" in self.cfg:
            self.num_cols = self.cfg.int_features + self.cfg.cont_features + self.cfg.dequant_features
        else:
            self.num_cols = self.cfg.int_features + self.cfg.cont_features
        self.is_regression = self.cfg.task == "regression"
        self.task = self.cfg.task
        self.seed = seed
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.cat_min_freq = cat_min_freq
        self.cat_encoding = cat_encoding  # 'ordinal' or 'onehot'
        assert missing_mechanism in [None, "mcar", "mar", "mnar"]
        self.missing_mechanism = missing_mechanism  # None, 'mcar', 'mar', 'mnar'
        self.seed_missings = seed_missings
        self.p_miss = p_miss
        self.p_obs = p_obs

        self.data_preprocessed = False
        self.preprocess()

    def create_splits(self, data):
        # compute proportion of data required to achieve val_prop
        prop = self.val_prop / (1 - self.test_prop)

        # train, validation, test split
        idx = np.arange(len(data))

        if self.is_regression:
            train_val_idx, test_idx = train_test_split(idx, test_size=self.test_prop, random_state=self.seed)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=prop, random_state=self.seed)
        else:
            train_val_idx, test_idx = train_test_split(
                idx, test_size=self.test_prop, random_state=self.seed, stratify=data.select(self.target)
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=prop,
                random_state=self.seed,
                stratify=data.select(self.target)[train_val_idx],
            )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    def preprocess(self):
        set_seeds(self.seed)
        assert not self.data_preprocessed, "Data has already been preprocessed."

        print(f"= Processing {self.name} with seed {self.seed} =")

        nan_vals = ["NA", "NULL", "null", "nan", "NaN", "N/A", "n/a", "", " ", "None", "none", "?"]
        if self.name == "news":
            data = pl.from_pandas(pd.read_csv(self.file_path))
        else:
            data = pl.read_csv(
                self.file_path, null_values=nan_vals, infer_schema_length=100000, separator=self.separator
            )

        n_full_sample = data.shape[0]
        data = data.select(pl.col([self.target] + self.cat_cols + self.num_cols))
        self.orig_cols = data.columns

        # remove rows with missings in target (only needed for ml efficiency tasks)
        data = data.filter(~pl.col(self.target).is_null())
        print(f"Rows with missings in target removed: {n_full_sample - data.shape[0]}")

        # determine missings in numerical features
        n_num_miss = data.select(self.num_cols).null_count().to_numpy().sum().item()
        assert n_num_miss == 0, "Data should not have missings in numerical features."

        # create binary classification tasks
        if self.name == "diabetes":
            data = data.with_columns(
                pl.when(pl.col(self.target) == "NO").then(pl.lit("no")).otherwise(pl.lit("yes")).alias(self.target)
            )
        elif self.name == "covertype":
            data = data.with_columns(pl.when(pl.col(self.target) == 2).then(1).otherwise(0).alias(self.target))

        # depending on task, add target to categorical or numerical features
        if self.cfg.task == "regression":
            self.num_cols = [self.target] + self.num_cols
        else:
            self.cat_cols = [self.target] + self.cat_cols
        print(f"Cat cols: {len(self.cat_cols)}")
        print(f"Num cols: {len(self.num_cols)}")

        # record number of digits to round generated numerical values
        self.col_to_round_digits = {}
        for col in self.num_cols:
            digit_dist = digit_distribution(data[col].to_pandas())
            if digit_dist[digit_dist.index > 0].sum() < 10:
                # treat as integer
                self.col_to_round_digits[col] = 0
            else:
                self.col_to_round_digits[col] = min(digit_dist.index.max().item(), 6)

        # round original data to decimals representable by float32 for fair evaluation
        X_num_gen = data.select(self.num_cols).to_pandas()
        for col_name, decimals in self.col_to_round_digits.items():
            X_num_gen[col_name] = X_num_gen[col_name].round(decimals)
        X_cat_gen = data.select(self.cat_cols)
        data = pl.concat([X_cat_gen, pl.from_pandas(X_num_gen)], how="horizontal").select(self.orig_cols)

        # for categorical features, replace missings to encode as separate category
        data = data.with_columns(pl.col(self.cat_cols).fill_null("_MISSING_"))

        # simulate missings according to selected mechanism
        if self.missing_mechanism is not None:
            assert data.null_count().to_numpy().sum().item() == 0, "Data should have no missings before simulating."

        if self.missing_mechanism == "mcar":
            print("Simulating MCAR...")
            rng = np.random.default_rng(seed=self.seed_missings)
            miss_mask = rng.random((*data.select(self.num_cols).shape,)) < self.p_miss
            miss_mask = torch.tensor(miss_mask, dtype=torch.bool)

            # avoid introducing missings in numerical target
            if self.is_regression:
                miss_mask[:, self.num_cols.index(self.target)] = False

        elif self.missing_mechanism == "mar":
            print("Simulating MAR...")
            miss_mask = self.get_MAR_mask(data)

        elif self.missing_mechanism == "mnar":
            print("Simulating MNAR...")
            miss_mask, miss_mask_cat = self.get_MAR_mask(data, exclude_inputs=True)

        if self.missing_mechanism is not None:
            # apply mask to numerical features
            d = data.select(self.num_cols).to_pandas()
            d[miss_mask.numpy()] = np.nan
            data = data.with_columns([pl.Series(name=col, values=d[col].values) for col in self.num_cols])

            if self.missing_mechanism == "mnar":
                # apply mask to categorical features
                d = data.select(self.cat_cols).to_pandas()
                d[miss_mask_cat.numpy()] = "_MISSING_"
                data = data.with_columns([pl.Series(name=col, values=d[col].tolist()) for col in self.cat_cols])

        # create train/val/test splits
        self.splits = self.create_splits(data)

        # take snapshot of original data
        self.orig_schema = data.schema
        self.orig_data = data.fill_nan(None)

        # set flag to avoid re-doing preprocessing
        self.data_preprocessed = True

    def get_MAR_mask(self, data, exclude_inputs=False):
        """
        Simulates missings in numerical features based on subset of categorical and numerical features.
        Uses a random logistic model to determine the probability of missing values.
        Avoids introducing missings into the target avoid problems for ML efficiency evaluation.

        Based on https://github.com/BorisMuzellec/MissingDataOT/blob/master/utils.py#L144.
        """
        rng = np.random.default_rng(seed=self.seed_missings)

        # remove target in regression case to avoid introducing missings to it
        if self.is_regression:
            num_cols = [col for col in self.num_cols if col != self.target]
        else:
            num_cols = self.num_cols

        X = data.select(num_cols).to_numpy().astype(float)
        std_scaler = StandardScaler()
        X = std_scaler.fit_transform(X)
        X = torch.tensor(X).float()
        miss_mask = torch.zeros_like(X).bool()
        n, k = X.shape

        # number of numerical features without missing values (at least one variable)
        k_obs = max(int(self.p_obs * k), 1)

        # number of numerical features that will have missing values
        k_na = k - k_obs
        assert k_na > 0, "At least one numerical feature should have missing values."

        # sample observed feature indices and those with missing values
        idxs_obs = rng.choice(k, k_obs, replace=False)
        idxs_nas = np.array([i for i in range(k) if i not in idxs_obs])

        # subsample categorical features that determine the missing values
        # if excluding inputs later on, cannot use binary target to avoid introducing missings to it
        if not self.is_regression and exclude_inputs:
            cat_cols = [col for col in self.cat_cols if col != self.target]
        else:
            cat_cols = self.cat_cols

        # if the only categorical feature is the target, skip adding categorical features
        if len(cat_cols) == 0:
            X_combined = X[:, idxs_obs]
        else:
            cat_enc = OrdinalEncoder()
            X_cat = cat_enc.fit_transform(data.select(cat_cols).to_numpy())
            X_cat = torch.tensor(X_cat).float()

            cat_idx = rng.choice(X_cat.shape[1], max(int(self.p_obs * X_cat.shape[1]), 1), replace=False)

            # combine observed numerical and selected categorical features
            X_combined = torch.column_stack((X[:, idxs_obs], X_cat[:, cat_idx]))

        # use randomly initialized logistic model to determine missing value probability for numerical features
        # pick coefficients so that W^Tx has unit variance (avoids shrinking)
        coeffs = pick_coeffs(X_combined, idxs_nas)

        # pick the intercepts to have a desired amount of missing values
        # so per feature, get self.p_miss proportion of missing values
        intercepts = fit_intercepts(X_combined, coeffs, self.p_miss)

        ps = torch.sigmoid(X_combined.mm(coeffs) + intercepts)
        bern = torch.rand(n, k_na)
        miss_mask[:, idxs_nas] = bern < ps

        if exclude_inputs:
            # mask part of the features that are used to determine the missing values
            miss_mask[:, idxs_obs] = torch.rand(n, k_obs) < self.p_miss

            if len(cat_cols) > 0:
                miss_mask_cat = torch.zeros_like(X_cat).bool()
                miss_mask_cat[:, cat_idx] = torch.rand(n, len(cat_idx)) < self.p_miss

            # add columns to mask with False corresponding to targets
            if self.is_regression:
                miss_mask = np.insert(miss_mask, self.num_cols.index(self.target), False, axis=1).bool()
            else:
                if len(cat_cols) > 0:
                    miss_mask_cat = np.insert(miss_mask_cat, self.cat_cols.index(self.target), False, axis=1).bool()
                else:
                    miss_mask_cat = torch.zeros((n, 1), dtype=torch.bool)

            assert miss_mask.shape[1] == len(self.num_cols)
            assert miss_mask_cat.shape[1] == len(self.cat_cols)

            return miss_mask, miss_mask_cat

        if self.is_regression:
            miss_mask = np.insert(miss_mask, self.num_cols.index(self.target), False, axis=1).bool()

        # enforce no missings in classification targets as well in MAR case
        assert miss_mask.shape[1] == len(self.num_cols)

        return miss_mask

    def get_data_splits(self):
        """Returns the data splits as polars DataFrames."""
        train_data = self.orig_data[self.splits["train"], :].clone()
        val_data = self.orig_data[self.splits["val"], :].clone()
        test_data = self.orig_data[self.splits["test"], :].clone()

        return train_data, val_data, test_data

    def get_data_loaders(self, mean_impute=True, include_test=False):
        # include_test is only used for evaluation of LowRes model, to ensure that test data is preprocessed the same way as training data

        train_data, val_data, test_data = self.get_data_splits()
        data = pl.concat([train_data, val_data])  # to ensure consistent values for later evaluation as well

        if include_test:
            pl.concat([train_data, val_data, test_data])

        ####################################################
        # Handle numerical features

        X_num_train = train_data.select(self.num_cols).to_numpy()
        X_num_val = val_data.select(self.num_cols).to_numpy() if val_data.height > 0 else None
        X_num_test = test_data.select(self.num_cols).to_numpy()

        # create missing indicator features for numerical features
        if self.missing_mechanism is not None:
            M_ind_train = []
            M_ind_val = []
            M_ind_test = []
            self.M_ind_cols = []
            for i, label in enumerate(self.num_cols):
                d_train = X_num_train[:, i]
                d_val = X_num_val[:, i] if X_num_val is not None else None
                d_test = X_num_test[:, i]
                if np.isnan(d_train).any():
                    M_ind_train.append(np.isnan(d_train))
                    M_ind_test.append(np.isnan(d_test))
                    if d_val is not None:
                        M_ind_val.append(np.isnan(d_val))
                    self.M_ind_cols.append(label)

                    # mean-impute numerical missings per feature
                    if mean_impute:
                        X_num_train[:, i] = np.nan_to_num(d_train, nan=np.nanmean(d_train), copy=False)
                        X_num_val[:, i] = (
                            np.nan_to_num(d_val, nan=np.nanmean(d_train), copy=False) if d_val is not None else None
                        )
                        X_num_test[:, i] = np.nan_to_num(d_test, nan=np.nanmean(d_train), copy=False)

            M_ind_train = np.column_stack(M_ind_train)
            M_ind_val = np.column_stack(M_ind_val) if val_data.height > 0 else None
            M_ind_test = np.column_stack(M_ind_test)
            M_ind_train = torch.tensor(M_ind_train).bool()
            M_ind_val = torch.tensor(M_ind_val).bool() if val_data.height > 0 else None
            M_ind_test = torch.tensor(M_ind_test).bool()

        # quantile-transform + standardize
        self.num_enc = Pipeline(
            [
                (
                    "quantile_transformer",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=max(min(train_data.height // 30, 1000), 10),
                        subsample=None,
                        random_state=42,
                    ),
                ),
                ("standard_scaler", StandardScaler()),
            ]
        )
        X_num_train = self.num_enc.fit_transform(X_num_train)
        X_num_val = self.num_enc.transform(X_num_val) if val_data.height > 0 else None
        X_num_test = self.num_enc.transform(X_num_test)
        X_num_train = torch.tensor(X_num_train).float()
        X_num_val = torch.tensor(X_num_val).float() if val_data.height > 0 else None
        X_num_test = torch.tensor(X_num_test).float()

        ####################################################
        # Handle categorical features

        if self.cat_encoding == "onehot":
            self.cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        else:
            self.cat_enc = CatEncoder(data.select(self.cat_cols), min_frequency=self.cat_min_freq)

        self.cat_enc.fit(data.select(self.cat_cols))
        X_cat_train = self.cat_enc.transform(train_data.select(self.cat_cols))
        X_cat_val = self.cat_enc.transform(val_data.select(self.cat_cols))
        X_cat_test = self.cat_enc.transform(test_data.select(self.cat_cols)) if include_test else None
        X_cat_train = torch.tensor(X_cat_train).long()
        X_cat_val = torch.tensor(X_cat_val).long() if val_data.height > 0 else None
        X_cat_test = torch.tensor(X_cat_test).long() if include_test else None

        if self.cat_encoding == "onehot":
            self.X_cat_n_classes = [len(cats) for cats in self.cat_enc.categories_]
        else:
            self.X_cat_n_classes = [self.cat_enc.idx_to_stats[i]["n_classes"] for i in range(len(self.cat_cols))]

        # add missing indicator features for numerical features
        if self.missing_mechanism is not None:
            X_cat_train = torch.column_stack((X_cat_train, M_ind_train)).long()
            X_cat_val = torch.column_stack((X_cat_val, M_ind_val)).long() if val_data.height > 0 else None
            X_cat_test = torch.column_stack((X_cat_test, M_ind_test)).long() if include_test else None

            # update n_classes
            self.X_cat_n_classes = self.X_cat_n_classes + [2] * M_ind_train.shape[1]

        ####################################################3
        # Prepare loaders

        train_loader = FastTensorDataLoader(
            X_cat_train,
            X_num_train,
            batch_size=min(self.train_batch_size, train_data.height),
            shuffle=True,
            drop_last=True,
        )

        if val_data.height > 0:
            val_loader = FastTensorDataLoader(
                X_cat_val,
                X_num_val,
                batch_size=min(self.val_batch_size, val_data.height),
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None

        if include_test:
            test_loader = FastTensorDataLoader(
                X_cat_test,
                X_num_test,
                batch_size=min(self.val_batch_size, test_data.height),
                shuffle=False,
                drop_last=False,
            )

            return train_loader, val_loader, test_loader

        return train_loader, val_loader

    def postprocess(self, X_cat_gen, X_num_gen, includes_miss_ind=True):
        """
        Postprocess synthetic data samples.
        Returns full synthetic data frame with same structure as original data.
        """

        X_cat_gen = X_cat_gen.astype(np.int64)
        X_num_gen = X_num_gen.astype(float)

        ####################################################
        # Handle categorical features

        if includes_miss_ind and self.missing_mechanism is not None:
            M_ind_gen = X_cat_gen[:, len(self.cat_cols) :].astype(np.bool)
            assert M_ind_gen.shape[1] == len(self.M_ind_cols)
            X_cat_gen = X_cat_gen[:, : len(self.cat_cols)]
        X_cat_gen = self.cat_enc.inverse_transform(X_cat_gen)

        ####################################################
        # Handle numerical features

        X_num_gen = self.num_enc.inverse_transform(X_num_gen)
        X_num_gen = pd.DataFrame(X_num_gen, columns=self.num_cols)

        # rounding numerical values
        for col_name, decimals in self.col_to_round_digits.items():
            X_num_gen[col_name] = X_num_gen[col_name].round(decimals)

        # if there are missings, replace them with NaN
        if includes_miss_ind and self.missing_mechanism is not None:
            miss_mask = np.zeros_like(X_num_gen, dtype=bool)
            for i, col in enumerate(self.num_cols):
                if col in self.M_ind_cols:
                    miss_mask[:, i] = M_ind_gen[:, self.M_ind_cols.index(col)]

            X_num_gen[miss_mask] = np.nan

        ####################################################
        # Combine data and construct final dataframe

        # combine and reorder columns
        X_num_gen = pl.from_pandas(X_num_gen)
        X_cat_gen = pl.DataFrame(X_cat_gen)
        X_cat_gen.columns = self.cat_cols
        df_gen = pl.concat([X_cat_gen, X_num_gen], how="horizontal")
        df_gen = df_gen.select(self.orig_cols)

        # round (for fair evaluation has to be redone after loading due to floating point precision)
        # df_gen = self.round_df_gen(df_gen) FIXME: is this needed still?

        return df_gen

    def load_parquet(self, path: Path) -> pl.DataFrame:
        """Load and decode scaled integers back to their original decimal values."""
        df = pl.read_parquet(path)

        # rounding
        X_num_gen = df.select(self.num_cols).to_pandas()
        for col_name, decimals in self.col_to_round_digits.items():
            X_num_gen[col_name] = X_num_gen[col_name].round(decimals)
        X_cat_gen = df.select(self.cat_cols)
        df = pl.concat([X_cat_gen, pl.from_pandas(X_num_gen)], how="horizontal").select(self.orig_cols)
        return df

    def round_df_gen(self, df_gen: pl.DataFrame) -> pl.DataFrame:
        """Round numerical columns in generated data according to original data precision."""
        X_num_gen = df_gen.select(self.num_cols).to_pandas()
        for col_name, decimals in self.col_to_round_digits.items():
            X_num_gen[col_name] = X_num_gen[col_name].round(decimals)
        X_cat_gen = df_gen.select(self.cat_cols)
        df_gen = pl.concat([X_cat_gen, pl.from_pandas(X_num_gen)], how="horizontal").select(self.orig_cols)
        return df_gen
