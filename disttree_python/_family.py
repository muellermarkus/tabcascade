"""
Distribution family specifications for distributional regression trees.

A DistFamily encapsulates everything needed to fit a parametric distribution
via MLE at a tree node:
  - number and names of parameters
  - link / inverse-link functions (maps natural scale <-> unconstrained space)
  - log-likelihood, score (gradient), and Hessian
  - starting-parameter heuristic
  - mean prediction
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DistFamily(ABC):
    """Abstract base class for a parametric distribution family."""

    # ------------------------------------------------------------------
    # Subclasses must define these class-level attributes
    # ------------------------------------------------------------------
    npar: int
    param_names: list[str]

    # ------------------------------------------------------------------
    # Link / inverse-link  (unconstrained <-> natural scale)
    # ------------------------------------------------------------------

    @abstractmethod
    def link(self, params: np.ndarray) -> np.ndarray:
        """Map parameter vector from natural scale to unconstrained (link) scale.

        Parameters
        ----------
        params:
            1-D array of length ``npar`` on the natural scale.

        Returns
        -------
        eta:
            1-D array of length ``npar`` on the link scale.
        """

    @abstractmethod
    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Map parameter vector from link scale back to natural scale.

        Parameters
        ----------
        eta:
            1-D array of length ``npar`` on the link scale.

        Returns
        -------
        params:
            1-D array of length ``npar`` on the natural scale.
        """

    # ------------------------------------------------------------------
    # Likelihood / score / Hessian  (all on the *natural* scale)
    # ------------------------------------------------------------------

    @abstractmethod
    def log_likelihood(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Total (weighted) log-likelihood.

        Parameters
        ----------
        y:
            Response vector, shape ``(n,)``.
        params:
            Parameter vector (natural scale), shape ``(npar,)``.
        weights:
            Non-negative observation weights, shape ``(n,)``.  When *None*,
            unit weights are used.

        Returns
        -------
        float
            Scalar log-likelihood value.
        """

    @abstractmethod
    def score(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-observation score matrix (gradient contributions).

        Parameters
        ----------
        y:
            Response vector, shape ``(n,)``.
        params:
            Parameter vector (natural scale), shape ``(npar,)``.
        weights:
            Non-negative observation weights, shape ``(n,)``.  When *None*,
            unit weights are used.

        Returns
        -------
        S:
            Array of shape ``(n, npar)`` where ``S[i, j]`` is the partial
            derivative of ``log f(y_i | params)`` with respect to ``params[j]``.
        """

    @abstractmethod
    def hessian(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Total Hessian of the log-likelihood.

        Parameters
        ----------
        y, params, weights:
            Same as in :meth:`score`.

        Returns
        -------
        H:
            Array of shape ``(npar, npar)``.
        """

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @abstractmethod
    def start_params(
        self,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Heuristic starting parameters for the optimizer (natural scale).

        Parameters
        ----------
        y:
            Response vector, shape ``(n,)``.
        weights:
            Optional weights, shape ``(n,)``.

        Returns
        -------
        params0:
            1-D array of length ``npar``.
        """

    @abstractmethod
    def mean(self, params: np.ndarray) -> float:
        """Return the distribution mean for a given parameter vector.

        Parameters
        ----------
        params:
            1-D array of length ``npar`` on the natural scale.
        """

    # ------------------------------------------------------------------
    # Optional overrides for display / optimiser configuration
    # ------------------------------------------------------------------

    def to_user_params(
        self,
        internal_params: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert internal (link-scale / optimizer) params to user-facing params.

        The default implementation is the identity — internal and display
        representations are the same.  Subclasses may override to apply a
        back-transformation (e.g. ``log_sigma → sigma`` for
        :class:`GaussianFamily`).

        Parameters
        ----------
        internal_params:
            1-D array of length ``npar`` as returned by
            :meth:`inverse_link`.

        Returns
        -------
        display_params:
            1-D array of length ``npar`` on the user-facing natural scale.
        display_names:
            Parameter names corresponding to *display_params*.
        """
        return internal_params.copy(), list(self.param_names)

    def link_bounds(self) -> list[tuple] | None:
        """Bounds on link-scale parameters for the L-BFGS-B optimiser.

        Returns a list of ``(lower, upper)`` tuples of length ``npar``,
        or *None* for fully unconstrained optimisation.  Use *None* as
        either element of a tuple to leave that end unconstrained.

        The default implementation returns *None* (unconstrained).
        Subclasses should override when part of the link space needs to be
        restricted to prevent numerical issues (e.g. ``log_sigma ≥ -100``).
        """
        return None


# ---------------------------------------------------------------------------
# Gaussian family
# ---------------------------------------------------------------------------

class GaussianFamily(DistFamily):
    """Normal / Gaussian distribution N(mu, sigma^2).

    Internal working parametrisation: ``(mu, log_sigma)``.

    The optimiser operates directly on ``(mu, log_sigma)`` — an identity
    link — so that ``sigma = exp(log_sigma) > 0`` is guaranteed without
    any explicit constraint, and the score / Hessian never divide by
    ``sigma``.  The user-facing representation (``coef_``,
    ``predict_params``, etc.) exposes ``sigma`` via :meth:`to_user_params`.

    Parameters
    ----------
    mu:
        Location — identity link.
    log_sigma:
        Log-scale (internal) — identity link; ``sigma = exp(log_sigma)``.
    """

    npar = 2
    param_names = ["mu", "log_sigma"]   # internal / optimizer names

    # ------------------------------------------------------------------
    # Link functions  (identity — optimizer works in natural log-space)
    # ------------------------------------------------------------------

    def link(self, params: np.ndarray) -> np.ndarray:
        """(mu, log_sigma) -> (mu, log_sigma)  [identity]."""
        return np.asarray(params, dtype=float).copy()

    def inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """(mu, log_sigma) -> (mu, log_sigma)  [identity]."""
        return np.asarray(eta, dtype=float).copy()

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------

    def log_likelihood(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        y = np.asarray(y, dtype=float)
        mu, log_sigma = float(params[0]), float(params[1])
        if weights is None:
            weights = np.ones(len(y))
        else:
            weights = np.asarray(weights, dtype=float)

        # log f(y_i) = -0.5*log(2*pi) - log_sigma - 0.5*(y_i-mu)^2 * exp(-2*log_sigma)
        ll = (
            -0.5 * np.log(2.0 * np.pi)
            - log_sigma
            - 0.5 * (y - mu) ** 2 * np.exp(-2.0 * log_sigma)
        )
        return float(np.dot(weights, ll))

    # ------------------------------------------------------------------
    # Score  (analytic, w.r.t. (mu, log_sigma))
    # ------------------------------------------------------------------

    def score(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-observation score matrix, shape (n, 2).

        Using the parametrisation ``(mu, log_sigma)``::

          d log f / d mu        = (y - mu) * exp(-2 * log_sigma)
          d log f / d log_sigma = (y - mu)^2 * exp(-2 * log_sigma) - 1

        No division by ``sigma`` — numerically stable for any
        ``log_sigma`` value.

        Because the link is the identity, this score is simultaneously the
        natural-scale and link-scale gradient, so :class:`DistFit` applies
        the identity Jacobian (no-op chain rule).
        """
        y = np.asarray(y, dtype=float)
        mu, log_sigma = float(params[0]), float(params[1])
        if weights is None:
            weights = np.ones(len(y))
        else:
            weights = np.asarray(weights, dtype=float)

        exp_neg2 = np.exp(-2.0 * log_sigma)   # = 1 / sigma^2
        residual = y - mu
        s_mu       = residual * exp_neg2
        s_log_sigma = residual ** 2 * exp_neg2 - 1.0

        S = np.column_stack([s_mu, s_log_sigma])
        return S * weights[:, np.newaxis]

    # ------------------------------------------------------------------
    # Hessian  (analytic, w.r.t. (mu, log_sigma))
    # ------------------------------------------------------------------

    def hessian(
        self,
        y: np.ndarray,
        params: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Total Hessian of the log-likelihood, shape (2, 2).

        Using the parametrisation ``(mu, log_sigma)``::

          d^2 L / d mu^2            = -n_eff * exp(-2 * log_sigma)
          d^2 L / d log_sigma^2     = -2 * sum_w((y-mu)^2) * exp(-2 * log_sigma)
          d^2 L / d mu d log_sigma  = -2 * sum_w(y-mu) * exp(-2 * log_sigma)
        """
        y = np.asarray(y, dtype=float)
        mu, log_sigma = float(params[0]), float(params[1])
        if weights is None:
            weights = np.ones(len(y))
        else:
            weights = np.asarray(weights, dtype=float)

        exp_neg2 = np.exp(-2.0 * log_sigma)   # = 1 / sigma^2
        n_eff    = float(weights.sum())
        residual = y - mu
        wresid2  = float(np.dot(weights, residual ** 2))

        h_mumu   = -n_eff * exp_neg2
        h_lsls   = -2.0 * wresid2 * exp_neg2
        h_muls   = -2.0 * float(np.dot(weights, residual)) * exp_neg2

        return np.array([[h_mumu,  h_muls],
                         [h_muls,  h_lsls]])

    # ------------------------------------------------------------------
    # Starting parameters
    # ------------------------------------------------------------------

    def start_params(
        self,
        y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if weights is None:
            mu0    = float(np.mean(y))
            sigma0 = float(np.std(y, ddof=1)) or 1.0
        else:
            weights = np.asarray(weights, dtype=float)
            w   = weights / weights.sum()
            mu0    = float(np.dot(w, y))
            sigma0 = float(np.sqrt(np.dot(w, (y - mu0) ** 2))) or 1.0
        return np.array([mu0, np.log(sigma0)])   # internal: (mu, log_sigma)

    # ------------------------------------------------------------------
    # Mean
    # ------------------------------------------------------------------

    def mean(self, params: np.ndarray) -> float:
        return float(params[0])   # params[0] is mu in both internal and display

    # ------------------------------------------------------------------
    # Display conversion and optimiser bounds
    # ------------------------------------------------------------------

    def to_user_params(
        self,
        internal_params: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert ``(mu, log_sigma)`` → ``(mu, sigma)`` for user-facing output."""
        mu        = float(internal_params[0])
        log_sigma = float(internal_params[1])
        return np.array([mu, np.exp(log_sigma)]), ["mu", "sigma"]

    def link_bounds(self) -> list[tuple]:
        """Bound ``log_sigma >= -100`` to guard against degenerate constant-y nodes.

        ``exp(-100) ≈ 3.7e-44``; all powers up to the fourth remain
        representable in float64, so no underflow can occur even if the
        optimiser reaches the boundary.
        """
        return [(None, None), (-100.0, None)]
