"""
DistTree: distributional regression tree with ctree-style splitting.

Each leaf node fits a full parametric distribution (via :class:`DistFit`) to
the observations that fall into it.  Variable selection at each internal node
uses the conditional permutation test (Strasser & Weber, 1999; Hothorn et al.
2006) with Bonferroni correction; the split point is found by argmax of the
conditional quadratic statistic over all valid binary indicators.

This matches the algorithm used by R's disttree / partykit packages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._family import DistFamily, GaussianFamily
from ._distfit import DistFit
from ._split import select_variable, find_split


# ---------------------------------------------------------------------------
# Internal node structure
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    """A node in the distributional regression tree.

    Attributes
    ----------
    node_id:
        Unique integer identifier (assigned depth-first, root = 0).
    depth:
        Depth in the tree (root = 0).
    n_samples:
        Number of training observations that reach this node.
    distfit:
        Fitted distribution for this node (present for every node, leaf or not).
    is_leaf:
        True when this node is terminal.
    split_var:
        Index into the feature matrix ``X`` on which to split (internal only).
    split_val:
        Threshold: left child receives ``X[:, split_var] <= split_val``.
    left:
        Left child node (internal only).
    right:
        Right child node (internal only).
    loglik_gain:
        Log-likelihood gain achieved by this split (internal only).
    """
    node_id: int
    depth: int
    n_samples: int
    distfit: DistFit

    is_leaf: bool = True
    split_var: Optional[int] = None
    split_val: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    loglik_gain: float = 0.0


# ---------------------------------------------------------------------------
# DistTree
# ---------------------------------------------------------------------------

class DistTree:
    """Distributional regression tree.

    Each leaf predicts a full parametric distribution fitted by maximum
    likelihood.  Variable selection uses the ctree conditional permutation
    test (Strasser & Weber / Hothorn et al. 2006); split points are found by
    argmax of the conditional quadratic statistic over valid binary splits.

    Parameters
    ----------
    family:
        A :class:`~disttree._family.DistFamily` instance.  Defaults to
        :class:`~disttree._family.GaussianFamily`.
    min_samples_split:
        Minimum number of observations at a node required to attempt a split.
        Defaults to ``ceil(10 * family.npar)``  (e.g. 20 for Gaussian).
    min_samples_leaf:
        Minimum number of observations required in each child after a split.
        Defaults to ``ceil(10 * family.npar)``  (e.g. 20 for Gaussian).
        This matches R disttree's ``minbucket`` default.
    max_depth:
        Maximum tree depth.  *None* means unlimited.
    alpha:
        Significance threshold for the Bonferroni-corrected ctree test.
        A node is split only when the adjusted p-value is below *alpha*.
    minprob:
        Minimum fraction of node observations required in each child.
        The effective minbucket used in the split search is
        ``max(min_samples_leaf, ceil(minprob * n_node))``.
        Defaults to ``0.01``, matching R's ``ctree_control(minprob=0.01)``.

    Attributes (after ``fit``)
    --------------------------
    tree_:
        Root :class:`_Node` of the fitted tree.
    n_leaves_:
        Number of leaf nodes.
    n_features_in_:
        Number of partitioning features.
    feature_importances_:
        1-D array of length ``n_features_in_``.  Each entry is the total
        log-likelihood gain attributable to splits on that feature, normalised
        so that the array sums to 1.  Features never used for splitting have
        importance 0.

    Examples
    --------
    >>> import numpy as np
    >>> from disttree import DistTree
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.uniform(0, 1, n)
    >>> y = np.where(x < 0.5,
    ...              rng.normal(0, 1, n),
    ...              rng.normal(3, 2, n))
    >>> X = x.reshape(-1, 1)
    >>> tree = DistTree(alpha=0.05).fit(X, y)
    >>> tree.n_leaves_
    2
    >>> pred = tree.predict(X)        # predicted means
    >>> params = tree.predict_params(X)  # {'mu': ..., 'sigma': ...}
    >>> node_ids = tree.apply(X)      # leaf node IDs
    """

    def __init__(
        self,
        family: DistFamily | None = None,
        min_samples_split: int | None = None,
        min_samples_leaf: int | None = None,
        max_depth: int | None = None,
        alpha: float = 0.05,
        minprob: float = 0.01,
    ) -> None:
        self.family = family if family is not None else GaussianFamily()
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.alpha = alpha
        self.minprob = minprob

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DistTree":
        """Grow the distributional regression tree.

        Parameters
        ----------
        X:
            Feature matrix of partitioning variables, shape ``(n, p)``.
        y:
            Response vector, shape ``(n,)``.
        sample_weight:
            Optional non-negative observation weights, shape ``(n,)``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        if len(y) != n:
            raise ValueError("X and y must have the same number of rows.")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (n,):
                raise ValueError("sample_weight must have length n.")

        # Resolve default min-sample thresholds
        npar = self.family.npar
        default_size = math.ceil(10 * npar)
        min_split = self.min_samples_split if self.min_samples_split is not None else default_size
        min_leaf  = self.min_samples_leaf  if self.min_samples_leaf  is not None else default_size

        self.n_features_in_ = p
        self._node_counter = 0
        self._feature_gains = np.zeros(p, dtype=float)

        self.tree_ = self._grow(
            indices=np.arange(n),
            X=X,
            y=y,
            sample_weight=sample_weight,
            depth=0,
            min_split=min_split,
            min_leaf=min_leaf,
        )

        self.n_leaves_ = self._count_leaves(self.tree_)

        # Normalise feature importances
        total = self._feature_gains.sum()
        if total > 0:
            self.feature_importances_ = self._feature_gains / total
        else:
            self.feature_importances_ = self._feature_gains.copy()

        return self

    def _next_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _grow(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        depth: int,
        min_split: int,
        min_leaf: int,
    ) -> _Node:
        """Recursively grow a subtree for the given observation indices."""
        n = len(indices)
        y_node = y[indices]
        w_node = None if sample_weight is None else sample_weight[indices]

        # --- Guard: near-constant response → degenerate node, stop early ---
        # Matches R's behaviour of producing sigma=0 (delta distribution) leaves
        # when all observations share essentially the same y value.  The
        # absolute threshold 1e-14 is safely below any real-data variance.
        if float(np.var(y_node)) < 1e-14:
            return _Node(
                node_id=self._next_id(),
                depth=depth,
                n_samples=n,
                distfit=DistFit(family=self.family).fit(y_node, w_node),
            )

        # Fit the distribution to this node
        df = DistFit(family=self.family).fit(y_node, w_node)
        node = _Node(
            node_id=self._next_id(),
            depth=depth,
            n_samples=n,
            distfit=df,
        )

        # --- Guard: non-finite scores → cannot compute test statistic ---
        if not np.all(np.isfinite(df.estfun_)):
            return node

        # --- Stopping checks ---
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if n < min_split:
            return node

        # Effective minbucket = max(absolute min, minprob × node size).
        # Matches R's ctree_control(minprob = 0.01): for n_node > 2000 this
        # is more restrictive than the absolute floor, ruling out very
        # unbalanced splits that R would also reject.
        effective_minleaf = max(min_leaf, math.ceil(self.minprob * n))

        X_node = X[indices]

        # --- Variable selection (ctree conditional permutation test) ---
        j_star = select_variable(
            X=X_node,
            distfit=df,
            alpha=self.alpha,
            bonferroni=True,
        )
        if j_star is None:
            return node  # no significant instability → leaf

        # --- Split-point search (argmax of conditional quadratic statistic) ---
        threshold = find_split(
            x=X_node[:, j_star],
            distfit=df,
            minbucket=effective_minleaf,
        )
        if threshold is None:
            return node  # no valid split found → leaf

        # Partition observations
        x_col = X_node[:, j_star]
        left_mask  = x_col <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < effective_minleaf or right_mask.sum() < effective_minleaf:
            return node

        # Compute log-likelihood gain for feature importance
        gain = 0.0
        try:
            w_l = None if w_node is None else w_node[left_mask]
            w_r = None if w_node is None else w_node[right_mask]
            df_l = DistFit(family=self.family).fit(y_node[left_mask], w_l)
            df_r = DistFit(family=self.family).fit(y_node[right_mask], w_r)
            gain = max(0.0, df_l.loglik_ + df_r.loglik_ - df.loglik_)
        except Exception:
            gain = 0.0

        # Mark as internal and recurse
        self._feature_gains[j_star] += gain
        node.is_leaf = False
        node.split_var = j_star
        node.split_val = threshold
        node.loglik_gain = gain

        node.left = self._grow(
            indices=indices[left_mask],
            X=X, y=y, sample_weight=sample_weight,
            depth=depth + 1,
            min_split=min_split,
            min_leaf=min_leaf,
        )
        node.right = self._grow(
            indices=indices[right_mask],
            X=X, y=y, sample_weight=sample_weight,
            depth=depth + 1,
            min_split=min_split,
            min_leaf=min_leaf,
        )
        return node

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _apply_one(self, x: np.ndarray) -> _Node:
        """Traverse the tree for a single observation and return its leaf."""
        node = self.tree_
        while not node.is_leaf:
            if x[node.split_var] <= node.split_val:
                node = node.left
            else:
                node = node.right
        return node

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the conditional mean E[Y | X = x] for each row of *X*.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n, p)``.

        Returns
        -------
        np.ndarray of shape ``(n,)``
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([
            self._apply_one(row).distfit.family.mean(
                self._apply_one(row).distfit.coef_array_
            )
            for row in X
        ])

    def predict_params(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict distributional parameters for each row of *X*.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n, n_features_in_)``.

        Returns
        -------
        dict mapping each parameter name to a 1-D array of length *n*.

        Example
        -------
        ``{'mu': array([...]), 'sigma': array([...])}``
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        leaves = [self._apply_one(row) for row in X]
        # Use display param names (may differ from family.param_names when the
        # family reparametrises internally, e.g. log_sigma → sigma).
        _, display_names = self.family.to_user_params(
            np.zeros(self.family.npar)
        )
        result: dict[str, list[float]] = {name: [] for name in display_names}
        for leaf in leaves:
            for name, val in leaf.distfit.coef_.items():
                result[name].append(val)
        return {name: np.array(vals) for name, vals in result.items()}

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf node ID for each row of *X*.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n, n_features_in_)``.

        Returns
        -------
        np.ndarray of int, shape ``(n,)``
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.array([self._apply_one(row).node_id for row in X], dtype=int)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _count_leaves(node: _Node) -> int:
        if node.is_leaf:
            return 1
        return DistTree._count_leaves(node.left) + DistTree._count_leaves(node.right)

    def get_leaf_params(self) -> dict[int, dict[str, float]]:
        """Return the fitted parameters for every leaf node.

        Returns
        -------
        dict mapping ``node_id`` to ``{param_name: value}``.
        """
        self._check_fitted()
        result: dict[int, dict[str, float]] = {}
        self._collect_leaves(self.tree_, result)
        return result

    def _collect_leaves(
        self, node: _Node, out: dict[int, dict[str, float]]
    ) -> None:
        if node.is_leaf:
            out[node.node_id] = dict(node.distfit.coef_)
        else:
            self._collect_leaves(node.left, out)
            self._collect_leaves(node.right, out)

    def _check_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self) -> str:
        if not hasattr(self, "tree_"):
            return (
                f"DistTree(family={self.family.__class__.__name__}, "
                f"alpha={self.alpha}, minprob={self.minprob}, unfitted)"
            )
        return (
            f"DistTree(family={self.family.__class__.__name__}, "
            f"n_leaves={self.n_leaves_}, alpha={self.alpha}, "
            f"minprob={self.minprob})"
        )
