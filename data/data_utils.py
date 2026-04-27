import numpy as np
import polars as pl


class CatEncoder:
    def __init__(self, train_data, min_frequency=8):
        self.min_frequency = min_frequency
        self.num_features = train_data.shape[1]
        self.fit(train_data)

    def fit(self, train_data):

        self.idx_to_stats = {}
        for i in range(self.num_features):
            stats = {}
            d = train_data[:, i]
            stats["orig_dtype"] = d.dtype
            d = d.cast(pl.String)
            if d.has_nulls():
                stats["has_missing"] = True
                d = d.fill_null("MISSING")
            else:
                stats["has_missing"] = False
            vals, cnt = np.unique(d, return_counts=True)

            if (cnt < self.min_frequency).any():
                stats["has_rare"] = True
                stats["rare_cats"] = vals[cnt < self.min_frequency]
                stats["rare_counts"] = cnt[cnt < self.min_frequency]
                d = d.replace(stats["rare_cats"], "_UNKNOWN_")
            else:
                stats["has_rare"] = False

            vals, cnt = np.unique(d, return_counts=True)
            stats["values"] = vals
            stats["count"] = cnt
            stats["n_classes"] = len(vals)
            self.idx_to_stats[i] = stats

    def transform(self, X):

        X_enc = np.zeros((X.shape[0], self.num_features), dtype=np.int32)
        for i in range(self.num_features):
            d = X[:, i].cast(pl.String)

            if d.has_nulls():
                assert self.idx_to_stats[i]["has_missing"], (
                    f"Column {i} has missing values but was not fitted with missing values."
                )
                d = d.fill_null("MISSING")

            if self.idx_to_stats[i]["has_rare"]:
                d = d.replace(self.idx_to_stats[i]["rare_cats"], "_UNKNOWN_")

            vals = np.unique(d)
            new_vals = [v for v in vals if v not in self.idx_to_stats[i]["values"]]

            if len(new_vals) > 0:
                d = d.replace(new_vals, "_UNKNOWN_")

            d = d.cast(pl.Enum(self.idx_to_stats[i]["values"]))
            X_enc[:, i] = d.to_physical().to_numpy()

        return X_enc

    def inverse_transform(self, X_enc):
        cols = []
        for i in range(self.num_features):
            d = pl.Series(X_enc[:, i])
            d = d.cast(pl.Enum(self.idx_to_stats[i]["values"])).to_pandas()

            if self.idx_to_stats[i]["has_rare"]:
                # sample category from categories
                n_unknown = (d == "_UNKNOWN_").sum()
                p_choices = self.idx_to_stats[i]["rare_counts"] / self.idx_to_stats[i]["rare_counts"].sum()
                col_choices = self.idx_to_stats[i]["rare_cats"]
                # renormalize probabilities
                draws = np.random.choice(col_choices, size=n_unknown, replace=True, p=p_choices)
                d = d.cat.add_categories(np.unique(draws))
                d[d == "_UNKNOWN_"] = draws

            if self.idx_to_stats[i]["has_missing"]:
                d[d == "MISSING"] = np.nan

            d = pl.Series(d, dtype=self.idx_to_stats[i]["orig_dtype"])
            cols.append(d)
        return pl.DataFrame(cols).to_numpy()
