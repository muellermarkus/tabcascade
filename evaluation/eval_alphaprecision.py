import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class AlphaPrecision:
    """
    Based on https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py.

    Computes the more popular ''naive'' variant of Alpha Precision, Beta Recall, and Authenticity.
    The naive variant does not train a separate one-class neural network.
    This function is a more lightweight version of the original implementation.
    We accommodate missing values by mean imputation and adding binary missingness masks as features.

    Higher scores are better.
    """

    def __init__(self, cat_cols, n_alphas=30, max_obs=100000):
        super().__init__()
        self.cat_cols = cat_cols
        self.n_alphas = n_alphas  # number of alpha to derive integrated metrics
        self.max_obs = max_obs

    def prepare_data(self, df_trn, df_gen, seed):
        # ohe categorical features
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_trn = ohe.fit_transform(df_trn.select(self.cat_cols))
        X_cat_gen = ohe.transform(df_gen.select(self.cat_cols))

        # scale using min max scaler
        num_cols = [col for col in df_trn.columns if col not in self.cat_cols]
        scaler = MinMaxScaler()
        X_num_trn = scaler.fit_transform(df_trn.select(num_cols))
        X_num_gen = scaler.transform(df_gen.select(num_cols))

        # deal with missings
        if df_trn.null_count().to_numpy().sum() > 0:
            # add binary missingsness masks as features
            miss_ind_trn = []
            miss_ind_gen = []
            for col in num_cols:
                if df_trn[col].is_null().any():
                    miss_ind_trn.append((df_trn[col].is_null()).cast(int).to_numpy())
                    miss_ind_gen.append((df_gen[col].is_null()).cast(int).to_numpy())
            miss_ind_trn = np.column_stack(miss_ind_trn)
            miss_ind_gen = np.column_stack(miss_ind_gen)

            # mean impute missing values
            df_trn = df_trn.fill_null(strategy="mean")
            df_gen = df_gen.fill_null(strategy="mean")

            # reapply scaling after imputation
            X_num_trn = scaler.transform(df_trn.select(num_cols))
            X_num_gen = scaler.transform(df_gen.select(num_cols))

            # concate with missingsness masks
            X_num_trn = np.column_stack((X_num_trn, miss_ind_trn))
            X_num_gen = np.column_stack((X_num_gen, miss_ind_gen))

        else:
            assert df_gen.null_count().to_numpy().sum() == 0, "df_gen should not have any missing values"

        X_trn = np.column_stack((X_cat_trn, X_num_trn))
        X_gen = np.column_stack((X_cat_gen, X_num_gen))

        # subsample if too large
        if X_trn.shape[0] > self.max_obs:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_trn.shape[0], size=self.max_obs, replace=False)
            X_trn = X_trn[idx]

        X_trn = X_trn.astype(np.float32)
        X_gen = X_gen.astype(np.float32)

        return X_trn, X_gen

    def compute_metrics(self, X_trn, X_gen):
        emb_center = np.mean(X_trn, axis=0)
        synth_center = np.mean(X_gen, axis=0)
        alphas = np.linspace(0, 1, self.n_alphas)
        Radii = np.quantile(np.sqrt(np.sum((X_trn - emb_center) ** 2, axis=1)), alphas)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((X_gen - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X_trn)
        real_to_real, _ = nbrs_real.kneighbors(X_trn)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_gen)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X_trn)

        # find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = X_gen[real_to_synth_args]

        real_synth_closest_d = np.sqrt(np.sum((real_synth_closest - synth_center) ** 2, axis=1))
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                ((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k]))
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        # See which one is bigger
        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        alpha_precision = 1 - np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) / np.sum(alphas)

        if alpha_precision < 0:
            raise RuntimeError("negative value detected for alpha_precision")

        beta_recall = 1 - np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) / np.sum(alphas)

        if beta_recall < 0:
            raise RuntimeError("negative value detected for beta_recall")

        scores = {
            "alpha_precision": alpha_precision.item(),
            "beta_recall": beta_recall.item(),
            "authenticity": authenticity.item(),
        }

        return scores

    def estimate_scores(self, df_trn, df_gen, seed=42):
        X_trn, X_gen = self.prepare_data(df_trn, df_gen, seed)
        return self.compute_metrics(X_trn, X_gen)
