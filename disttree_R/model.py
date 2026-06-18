from pathlib import Path

import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import OrdinalEncoder


class DistTree:
    def __init__(self, max_depth=3, seed=42, suppress_warnings=True):
        self.seed = seed
        self.max_depth = max_depth
        self.models = []
        console_out = []

        def f(x):
            # function that append its argument to the list
            console_out.append(x)

        # consolewrite_print_backup = rpy2.rinterface_lib.callbacks.consolewrite_print
        rpy2.rinterface_lib.callbacks.consolewrite_print = f
        if suppress_warnings:
            rpy2.rinterface_lib.callbacks.consolewrite_warnerror = f

        base = Path(__file__).resolve().parent
        rpackages.importr("gamlss.dist")  # import needed to find correct distfamily object (i.e., "NO" = Normal)
        robjects.r["source"](str(base / "disttree.R"))
        robjects.r["source"](str(base / "distfamily.R"))
        robjects.r["source"](str(base / "distfit.R"))
        self.dt = robjects.globalenv["disttree"]

    def fit(self, x):
        set_seed = robjects.r("set.seed")
        set_seed(self.seed)

        self.ord_enc = []
        self.ord_enc_joint = []
        self.means = []
        self.stds = []

        for i in range(x.shape[1]):
            d = x[:, i].clone()
            df = pd.DataFrame(d, columns=["x"]).dropna()
            # convert pandas DataFrame to R DataFrame
            with (robjects.default_converter + pandas2ri.converter).context():
                df_r = robjects.conversion.get_conversion().py2rpy(df)
            mod = self.dt(robjects.Formula("x ~ x"), data=df_r, family="NO", maxdepth=self.max_depth)
            self.models.append(mod)

            preds = robjects.r.predict(mod, df_r)

            # convert R DataFrame to pandas DataFrame
            with (robjects.default_converter + pandas2ri.converter).context():
                df_preds = robjects.conversion.get_conversion().rpy2py(preds)
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
            df = pd.DataFrame(d, columns=["x"])
            # convert pandas DataFrame to R DataFrame
            with (robjects.default_converter + pandas2ri.converter).context():
                df_r = robjects.conversion.get_conversion().py2rpy(df)
            mod = self.models[i]
            preds = robjects.r.predict(mod, df_r)

            # convert R DataFrame to pandas DataFrame
            with (robjects.default_converter + pandas2ri.converter).context():
                df_preds = robjects.conversion.get_conversion().rpy2py(preds)
            df_enc = self.ord_enc[i].transform(df_preds)

            # form joint group
            joint_group = df_enc[:, 0].astype(str) + "_" + df_enc[:, 1].astype(str)
            joint_group = self.ord_enc_joint[i].transform(joint_group.reshape(-1, 1))

            joint_group = joint_group.astype(float).flatten()
            joint_group[miss_mask] = np.nan
            groups.append(joint_group)

        groups = np.column_stack(groups)

        return groups


def main():
    import torch
    from rpy2.robjects.vectors import StrVector

    # first install partykit and gamlss packages (dependencies of disttree)
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    packnames = ("partykit", "gamlss.dist")

    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    # small test that everything works
    x = torch.randn(1000, 2)
    disttree = DistTree(max_depth=3)
    disttree.fit(x)
    groups = disttree.get_groups(x)

    print(groups)


if __name__ == "__main__":
    main()
