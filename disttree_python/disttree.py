from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from ._tree import DistTree as DT


class DistTree:
    def __init__(self, max_depth=3, seed=42):
        self.seed = seed
        self.max_depth = max_depth
        self.models = []

    def fit(self, x):
        self.ord_enc = []
        self.ord_enc_joint = []
        self.means = []
        self.stds = []

        for i in range(x.shape[1]):
            d = x[:, i].clone()
            df = pd.DataFrame(d, columns=["x"]).dropna()

            model = DT(max_depth=self.max_depth)
            model.fit(df["x"].to_frame(), df["x"])
            self.models.append(model)

            # predict mu, sigma
            preds = model.predict_params(df["x"].to_frame())
            df_preds = pd.DataFrame(preds, columns=["mu", "sigma"])
            ord_enc = OrdinalEncoder(dtype=int)
            df_enc = ord_enc.fit_transform(df_preds)
            self.ord_enc.append(ord_enc)

            # form joint group
            joint_group = df_enc[:, 0].astype(str) + "_" + df_enc[:, 1].astype(str)
            ord_enc_joint = OrdinalEncoder(dtype=int)
            joint_group = ord_enc_joint.fit_transform(joint_group.reshape(-1, 1))
            self.ord_enc_joint.append(ord_enc_joint)

            # retrieve means and sigmas
            df_aux = np.column_stack((df_preds.to_numpy(), joint_group))
            df_aux = pd.DataFrame(df_aux, columns=["mu", "sigma", "group"])
            mus = df_aux.drop_duplicates(["group"]).sort_values("group")["mu"].to_list()
            sigmas = df_aux.drop_duplicates(["group"]).sort_values("group")["sigma"].to_list()
            self.means.append(mus)
            self.stds.append(sigmas)

    def get_groups(self, x):
        groups = []
        for i in range(x.shape[1]):
            d = x[:, i].clone()
            miss_mask = d.isnan()
            d[miss_mask] = d.nanmean()
            model = self.models[i]
            preds = model.predict_params(d.reshape(-1, 1))
            df_preds = pd.DataFrame(preds, columns=["mu", "sigma"])
            df_enc = self.ord_enc[i].transform(df_preds)

            # form joint group
            joint_group = df_enc[:, 0].astype(str) + "_" + df_enc[:, 1].astype(str)
            joint_group = self.ord_enc_joint[i].transform(joint_group.reshape(-1, 1))

            joint_group = joint_group.astype(float).flatten()
            joint_group[miss_mask] = np.nan
            groups.append(joint_group)

        groups = np.column_stack(groups)

        return groups
