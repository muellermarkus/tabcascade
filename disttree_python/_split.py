"""
ctree-style split selection for distributional regression trees.

Variable selection uses the **conditional permutation test** (Strasser &
Weber, 1999; Hothorn, Hornik & Zeileis, 2006):

  * Linear statistic  T = Σᵢ g(Xᵢ) · h(Yᵢ)
    where g(Xⱼᵢ) = Xⱼᵢ  (identity for numeric variables)
          h(Yᵢ)  = wᵢ · ∂ log p(Yᵢ; η̂) / ∂η  (link-scale estfun)

  * Expected value and covariance under the conditional permutation
    distribution (Strasser & Weber formulas):
      μ = (Σᵢ g(Xᵢ)) · h̄
      Σ = [n/(n−1)]·V(h)·(Σᵢ gᵢ²) − [1/(n−1)]·V(h)·(Σᵢ gᵢ)²
    where V(h) = (1/n) Σᵢ (hᵢ − h̄)(hᵢ − h̄)ᵀ  (biased sample covariance)

  * Quadratic-form test statistic c(T) = (T−μ)ᵀ Σ⁺ (T−μ) ~ χ²(rank Σ)

  * Bonferroni correction across all candidate partitioning variables.

Split-point search: argmax over all valid binary split indicators of the
conditional quadratic statistic (Hothorn et al. 2006, §3.2).  This is an
O(n) sweep after one sort.

References
----------
Strasser H, Weber C (1999).
  "On the asymptotic theory of permutation statistics."
  Mathematical Methods of Statistics 8(2), 220–250.

Hothorn T, Hornik K, Zeileis A (2006).
  "Unbiased Recursive Partitioning: A Conditional Inference Framework."
  JCGS 15(3), 651–674.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2

from ._distfit import DistFit


# ---------------------------------------------------------------------------
# Core: linear statistic + permutation moments  (scalar g, vector h)
# ---------------------------------------------------------------------------

def _linstat_scalar_g(
    g: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear statistic T, E[T], Cov(T) for a scalar g-transformation (p=1).

    Parameters
    ----------
    g:
        1-D array of length n — the partitioning-variable transformation
        (e.g. raw values for numeric variables, or binary 0/1 for split test).
    h:
        2-D array of shape (n, q) — the influence function (link-scale
        estfun, already weighted).

    Returns
    -------
    T   : shape (q,) — linear statistic Σᵢ g(xᵢ) hᵢ
    mu  : shape (q,) — expected value under permutation null
    Sigma : shape (q, q) — covariance under permutation null
    """
    n, q = h.shape

    h_bar = h.mean(axis=0)                                  # (q,)
    T = g @ h                                               # (q,)
    mu = g.sum() * h_bar                                    # (q,)

    # Biased sample covariance of h: V(h) = (1/n) Σ (hᵢ−h̄)(hᵢ−h̄)ᵀ
    h_c = h - h_bar                                         # (n, q)
    V_h = (h_c.T @ h_c) / n                                 # (q, q)

    # Strasser-Weber "variance" of g (scalar):
    #   σ_g = [n/(n−1)] Σᵢ gᵢ² − [1/(n−1)] (Σᵢ gᵢ)²
    sum_g = float(g.sum())
    sum_g2 = float(g @ g)
    sigma_g = (n / (n - 1)) * sum_g2 - (1 / (n - 1)) * sum_g ** 2

    Sigma = sigma_g * V_h                                   # (q, q)
    return T, mu, Sigma


# ---------------------------------------------------------------------------
# Quadratic-form test statistic  c(T) = (T−μ)ᵀ Σ⁺ (T−μ)
# ---------------------------------------------------------------------------

def _quadratic_stat(
    T: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
) -> tuple[float, int]:
    """Compute (T−μ)ᵀ Σ⁺ (T−μ) and rank(Σ).

    Uses eigen-decomposition so that near-singular Σ is handled via its
    Moore-Penrose pseudoinverse.

    Returns
    -------
    (stat, rank)
        *stat* is the quadratic form value; *rank* is the effective
        degrees of freedom (= rank of Σ).
    """
    diff = T - mu
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    tol = max(1e-14, 1e-10 * float(np.abs(eigvals).max() or 1.0))
    pos = eigvals > tol
    rank = int(pos.sum())
    if rank == 0:
        return 0.0, 0
    Sigma_pinv = (
        eigvecs[:, pos]
        @ np.diag(1.0 / eigvals[pos])
        @ eigvecs[:, pos].T
    )
    return float(diff @ Sigma_pinv @ diff), rank


# ---------------------------------------------------------------------------
# Public API: variable selection
# ---------------------------------------------------------------------------

def select_variable(
    X: np.ndarray,
    distfit: DistFit,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> int | None:
    """Select the best partitioning variable using the ctree permutation test.

    For each candidate variable j, the quadratic-form test statistic is
    computed using the conditional permutation distribution, and a χ²
    p-value is obtained.  The variable with the smallest p-value is
    selected; a split is attempted only when the Bonferroni-adjusted
    p-value is below *alpha*.

    Parameters
    ----------
    X:
        Partitioning-variable matrix at the current node, shape ``(n, p)``.
    distfit:
        Already-fitted :class:`~disttree.DistFit` for this node.  Its
        ``estfun_`` (link-scale weighted score matrix, shape ``(n, q)``)
        is used as the influence function h.
    alpha:
        Global significance threshold (default 0.05).
    bonferroni:
        Whether to apply Bonferroni correction over the *p* variables
        (default True).

    Returns
    -------
    int or None
        0-based column index of the selected variable, or *None* if no
        variable shows significant instability after adjustment.
    """
    h = distfit.estfun_              # (n, q)
    n, p_vars = X.shape

    p_values = np.ones(p_vars)
    for j in range(p_vars):
        g = X[:, j]
        T, mu, Sigma = _linstat_scalar_g(g, h)
        stat, df = _quadratic_stat(T, mu, Sigma)
        p_values[j] = float(_chi2.sf(stat, df=df)) if df > 0 else 1.0

    j_star = int(np.argmin(p_values))
    p_best = float(p_values[j_star])
    p_corrected = min(p_vars * p_best, 1.0) if bonferroni else p_best

    if p_corrected >= alpha:
        return None
    return j_star


# ---------------------------------------------------------------------------
# Public API: split point search
# ---------------------------------------------------------------------------

def find_split(
    x: np.ndarray,
    distfit: DistFit,
    minbucket: int,
) -> float | None:
    """Find the best binary split via argmax of the conditional quadratic stat.

    For each candidate threshold c (at observed values of *x*), the binary
    split indicator g_c(xᵢ) = I(xᵢ ≤ c) defines a linear statistic whose
    quadratic form under the permutation null is computed.  The threshold
    maximising this statistic is selected.

    This is an O(n) sweep after one sort (the pseudoinverse of V(h) is
    shared across all thresholds).

    The split convention is ``intersplit = FALSE``: the threshold equals the
    last observed value in the left child (not the midpoint between adjacent
    distinct values).

    Parameters
    ----------
    x:
        Values of the selected partitioning variable at this node, shape
        ``(n,)``.
    distfit:
        Already-fitted :class:`~disttree.DistFit` for this node.
    minbucket:
        Minimum number of observations required in each resulting child.

    Returns
    -------
    float or None
        Best split threshold *c* (left child: x ≤ c), or *None* if no
        valid split exists.
    """
    h = distfit.estfun_              # (n, q)
    n, q = h.shape

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    h_sorted = h[order]

    # Precompute V(h) and its pseudoinverse — shared across all split candidates
    h_bar = h.mean(axis=0)                              # (q,)
    h_c = h - h_bar                                     # (n, q)
    V_h = (h_c.T @ h_c) / n                             # (q, q)

    eigvals, eigvecs = np.linalg.eigh(V_h)
    tol = max(1e-14, 1e-10 * float(np.abs(eigvals).max() or 1.0))
    pos = eigvals > tol
    if not pos.any():
        return None
    V_h_pinv = (
        eigvecs[:, pos]
        @ np.diag(1.0 / eigvals[pos])
        @ eigvecs[:, pos].T
    )                                                   # (q, q)

    # Running prefix sums of h (sorted by x): T_k = Σ_{i≤k} h_sorted[i]
    h_cumsum = np.cumsum(h_sorted, axis=0)              # (n, q)

    best_stat = -np.inf
    best_k = None

    for k in range(minbucket, n - minbucket + 1):
        # Only split at gaps between distinct x values
        if x_sorted[k - 1] == x_sorted[k]:
            continue

        T_k = h_cumsum[k - 1]                          # (q,) sum of left child
        diff_k = T_k - k * h_bar

        # Strasser-Weber scalar variance for binary g (0/1, exactly k ones):
        #   σ_g = (n/(n-1))·k − (1/(n-1))·k² = k(n-k)/(n-1)
        sigma_g = k * (n - k) / (n - 1)
        if sigma_g <= 0.0:
            continue

        stat_k = float(diff_k @ V_h_pinv @ diff_k) / sigma_g

        if stat_k > best_stat:
            best_stat = stat_k
            best_k = k

    if best_k is None:
        return None

    # intersplit = FALSE: threshold = value of the last observation in the
    # left child (all obs with x ≤ threshold go left)
    return float(x_sorted[best_k - 1])
