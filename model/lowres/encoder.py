import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import OrdinalEncoder

from disttree import DistTree


class Discretizer:
    """
    Encoder that encodes x_num into z_num (discrete groups) using either DT or GMM.

    Variants:
    - DT based (dt)
    - GMM based (gmm)
    """

    def __init__(self, X_num_trn, variant="dt", k_max=20, perc_obs=0.03, seed=42, adjust_means=False, max_depth=3):
        self.seed = seed
        self.variant = variant
        self.adjust_means = adjust_means
        self.max_depth = max_depth
        self.k_max = k_max
        self.perc_obs = perc_obs
        self.fit_gmm_ord_enc = True

        # check for missings in train data (assuming the same holds for validation / synthetic data)
        self.has_missing = torch.isnan(X_num_trn).any(0).numpy()

        # get means of non-missing values to infer mean for missing group (after mean imputation)
        miss_mean = torch.nanmean(X_num_trn, dim=0)

        if self.variant == "gmm":
            self.gmms = self._train_bgmms(X_num_trn, k_max=k_max)
            groups = self._get_gmm_groups(X_num_trn)
        elif self.variant == "dt":
            self.disttree = DistTree(max_depth, seed=seed)
            self.disttree.fit(X_num_trn)
            groups = self.disttree.get_groups(X_num_trn)

        # get group-specific means and stds
        if self.adjust_means:
            means = []
            stds = []
            for i in range(X_num_trn.shape[1]):
                df = pd.DataFrame({"x": X_num_trn[:, i].clone(), "group": groups[:, i]})
                df_stats = df.groupby("group").agg(["mean", "std"]).droplevel(0, axis=1)
                means.append(torch.tensor(df_stats["mean"].to_numpy(), dtype=torch.float32))
                stds.append(torch.tensor(df_stats["std"].to_numpy(), dtype=torch.float32))
        elif self.variant == "gmm":
            means = []
            stds = []
            for i in range(X_num_trn.shape[1]):
                means.append(torch.tensor(self.gmms[i].means_.squeeze(), dtype=torch.float32))
                stds.append(torch.tensor(np.sqrt(self.gmms[i].covariances_.squeeze()), dtype=torch.float32))
        elif self.variant == "dt":
            means = [torch.tensor(m, dtype=torch.float32) for m in self.disttree.means]
            stds = [torch.tensor(s, dtype=torch.float32) for s in self.disttree.stds]

        # check for inflated values (empirical var = 0)
        self.has_inflated = []
        self.infl_groups = []
        for i in range(X_num_trn.shape[1]):
            df = pd.DataFrame({"x": X_num_trn[:, i].clone(), "group": groups[:, i]})
            df_stats = df.groupby("group").agg(["mean", "std"]).droplevel(0, axis=1)
            infl_idx = df_stats.loc[df_stats["std"] == 0].index.to_list()
            self.has_inflated.append(len(infl_idx) > 0)
            self.infl_groups.append(infl_idx)

            # adjust std to zero
            stds[i][infl_idx] = 0

        # adjust means for missings (assign mean, std of group to which average X belongs)
        if self.has_missing.any():
            for i in range(X_num_trn.shape[1]):
                if self.has_missing[i]:
                    # update means with mean of missing group (= mean of group that we get for average x), similar for std dev.
                    means[i] = torch.cat((miss_mean[i].unsqueeze(0), means[i]))
                    stds[i] = torch.cat((torch.zeros(1), stds[i]))
        self.means = means
        self.stds = stds

    def _get_gmm_groups(self, X: torch.Tensor):
        groups = []
        for i in range(X.shape[1]):
            d = X[:, i].clone()
            miss_mask = d.isnan()
            d[miss_mask] = d.nanmean()
            # assign class with highest probability (argmax)
            group = self.gmms[i].predict(d.reshape(-1, 1)).astype(float)
            group[miss_mask] = np.nan
            groups.append(group)
        groups = np.column_stack(groups)

        if self.fit_gmm_ord_enc:
            self.fit_gmm_ord_enc = False
            self.gmm_ord_enc = OrdinalEncoder()
            self.gmm_ord_enc.fit(groups)
        groups = self.gmm_ord_enc.transform(groups)

        return groups

    def encode(self, X: torch.Tensor):
        if self.variant == "gmm":
            groups = self._get_gmm_groups(X)
        elif self.variant == "dt":
            groups = self.disttree.get_groups(X)

        groups, mask = self.postprocess_groups(groups)

        return groups, mask, self.infl_groups, self.has_missing

    def postprocess_groups(self, groups):
        # get inflated mask
        infl_mask = []
        for i in range(groups.shape[1]):
            mask = np.isin(groups[:, i], self.infl_groups[i])
            infl_mask.append(torch.tensor(mask, dtype=torch.bool))
        infl_mask = torch.column_stack(infl_mask)

        # shift other groups by 1, so that group 0 is reserved for missings
        # construct missingness mask
        miss_mask = []
        for i in range(groups.shape[1]):
            g_i = groups[:, i]
            miss_mask.append(np.isnan(g_i))

            # update group IDs, missing = 0
            if self.has_missing[i]:
                new_g_i = g_i.copy() + 1
                new_g_i = np.nan_to_num(new_g_i, nan=0, copy=True)
                groups[:, i] = new_g_i
        miss_mask = np.column_stack(miss_mask) if len(miss_mask) > 0 else None
        miss_mask = torch.tensor(miss_mask, dtype=torch.bool) if self.has_missing.any() else None

        # combine masks
        mask = miss_mask | infl_mask if miss_mask is not None else infl_mask

        # get into correct formats
        groups = torch.tensor(groups, dtype=torch.long)

        return groups, mask

    def _train_bgmms(self, X: torch.Tensor, k_max=20):
        bgmms = []
        for i in range(X.shape[1]):
            d = X[:, i].clone()
            d = d[~d.isnan()]

            bgmm = BayesianGaussianMixture(
                n_components=k_max,
                random_state=self.seed,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=0.001,
                n_init=1,
            )
            bgmms.append(bgmm.fit(d.reshape(-1, 1)))

        return bgmms
