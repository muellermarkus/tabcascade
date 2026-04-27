import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DCRScore:
    """Computes the popular distance to closest record (DCR) score.

    It min max scales continuous features and onehot encodes categorical features.
    Missing values are handled by mean imputation and adding binary masks for missingness.

    """

    def __init__(self, df_trn, df_test, cat_cols, num_cols, max_obs=100_000):
        self.max_obs = max_obs

        # onehot encode categorical features
        if len(cat_cols) > 0:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(df_trn.select(cat_cols))
            X_cat_trn_ohe = self.ohe.transform(df_trn.select(cat_cols))
            X_cat_tst_ohe = self.ohe.transform(df_test.select(cat_cols))
        self.cat_cols = cat_cols

        # min max scale continuous features
        if len(num_cols) > 0:
            self.minmax_scale = MinMaxScaler()
            X_num_trn_scaled = self.minmax_scale.fit_transform(df_trn.select(num_cols))
            X_num_tst_scaled = self.minmax_scale.transform(df_test.select(num_cols))
        self.num_cols = num_cols

        # deal with missings
        self.df_has_missings = False
        if (df_trn.null_count().to_numpy().sum() > 0) and (len(num_cols) > 0):
            self.df_has_missings = True
            # add binary missingsness masks as features
            miss_ind_trn = []
            self.cols_with_missings = []
            for col in self.num_cols:
                if df_trn[col].is_null().any():
                    self.cols_with_missings.append(col)
                    miss_ind_trn.append((df_trn[col].is_null()).cast(int).to_numpy())
            miss_ind_trn = np.column_stack(miss_ind_trn)

            miss_ind_tst = []
            for col in self.cols_with_missings:
                miss_ind_tst.append((df_test[col].is_null()).cast(int).to_numpy())
            miss_ind_tst = np.column_stack(miss_ind_tst)

            # mean impute missing values
            df_trn = df_trn.fill_null(strategy="mean")
            df_test = df_test.fill_null(strategy="mean")

            # reapply scaling after imputation
            X_num_trn_scaled = self.minmax_scale.transform(df_trn.select(num_cols))
            X_num_tst_scaled = self.minmax_scale.transform(df_test.select(num_cols))

            # concate with missingsness masks
            X_num_trn_scaled = np.column_stack((X_num_trn_scaled, miss_ind_trn))
            X_num_tst_scaled = np.column_stack((X_num_tst_scaled, miss_ind_tst))

        X_list_trn = []
        X_list_tst = []
        if len(cat_cols) > 0:
            X_list_trn.append(X_cat_trn_ohe)
            X_list_tst.append(X_cat_tst_ohe)
        if len(num_cols) > 0:
            X_list_trn.append(X_num_trn_scaled)
            X_list_tst.append(X_num_tst_scaled)
        X_trn = np.column_stack(X_list_trn)
        X_tst = np.column_stack(X_list_tst)

        self.nbrs_trn = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_trn)
        self.nbrs_tst = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_tst)

    def compute_dcr(self, df_gen, seed=42):
        if len(self.cat_cols) > 0:
            X_cat_gen_ohe = self.ohe.transform(df_gen.select(self.cat_cols))
        if len(self.num_cols) > 0:
            X_num_gen_scaled = self.minmax_scale.transform(df_gen.select(self.num_cols))

        if self.df_has_missings:
            # if train set had missings, then gen set must also have missings
            assert df_gen.null_count().to_numpy().sum() > 0
        else:
            # if train set had no missings, then gen set must also have no missings
            assert df_gen.null_count().to_numpy().sum() == 0

        if self.df_has_missings:
            miss_ind = []
            for col in self.cols_with_missings:
                miss_ind.append((df_gen[col].is_null()).cast(int).to_numpy())
            miss_ind = np.column_stack(miss_ind)

            # mean impute missing values
            df_gen = df_gen.fill_null(strategy="mean")

            X_num_gen_scaled = self.minmax_scale.transform(df_gen.select(self.num_cols))
            X_num_gen_scaled = np.column_stack((X_num_gen_scaled, miss_ind))

        X_list = []
        if len(self.cat_cols) > 0:
            X_list.append(X_cat_gen_ohe)
        if len(self.num_cols) > 0:
            X_list.append(X_num_gen_scaled)
        X_gen = np.column_stack(X_list)

        # subsample if necessary
        if df_gen.height > self.max_obs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_gen.shape[0], self.max_obs, replace=False)
            X_gen = X_gen[idx]

        # find nearest neighbor in train set
        # note that this takes very long for large datasets
        dist_trn, _ = self.nbrs_trn.kneighbors(X_gen, return_distance=True)
        dist_tst, _ = self.nbrs_tst.kneighbors(X_gen, return_distance=True)

        # compute dcr share
        dcr_share = dist_trn.copy()
        dcr_share[dist_trn < dist_tst] = 1
        dcr_share[dist_trn > dist_tst] = 0
        dcr_share[dist_trn == dist_tst] = 0.5
        dcr_share = dcr_share.mean().item()

        return {
            "dcr_min": dist_trn.min().item(),
            "dcr_max": dist_trn.max().item(),
            "dcr_mean": dist_trn.mean().item(),
            "dcr_median": np.median(dist_trn).item(),
            "dcr_raw": dist_trn.squeeze(),
            "dcr_share": dcr_share,
        }
