"""
DistFit: maximum-likelihood fitting of a parametric distribution to a sample.

This is the leaf-node model used inside :class:`~disttree.DistTree`.  It can
also be used standalone to fit a distribution to any 1-D sample.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy import stats as _stats

from ._family import DistFamily, GaussianFamily


class DistFit:
    """Fit a parametric distribution to a 1-D sample by maximum likelihood.

    Parameters
    ----------
    family:
        A :class:`~disttree._family.DistFamily` instance describing the
        parametric distribution to fit.  Defaults to
        :class:`~disttree._family.GaussianFamily`.

    Examples
    --------
    >>> import numpy as np
    >>> from disttree import DistFit
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(loc=2.0, scale=3.0, size=500)
    >>> fit = DistFit().fit(y)
    >>> fit.coef_          # {'mu': ~2.0, 'sigma': ~3.0}
    >>> fit.loglik_        # total log-likelihood
    """

    def __init__(self, family: DistFamily | None = None) -> None:
        self.family = family if family is not None else GaussianFamily()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DistFit":
        """Fit the distribution to *y* by maximum likelihood.

        Parameters
        ----------
        y:
            1-D response array.
        sample_weight:
            Non-negative observation weights.  When *None*, unit weights
            are assumed.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1-D array.")
        n = len(y)

        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != (n,):
                raise ValueError("sample_weight must have the same length as y.")
            if np.any(w < 0):
                raise ValueError("sample_weight must be non-negative.")

        family = self.family

        # Starting values on the natural scale, then map to link scale
        params0 = family.start_params(y, w)
        eta0 = family.link(params0)

        # Objective: *negative* log-likelihood (L-BFGS-B minimises)
        def neg_loglik(eta: np.ndarray) -> float:
            params = family.inverse_link(eta)
            return -family.log_likelihood(y, params, w)

        def neg_score_eta(eta: np.ndarray) -> np.ndarray:
            """Gradient of the negative log-likelihood w.r.t. eta (link scale).

            Uses the chain rule:
              d(-LL) / d eta = d(-LL) / d params  *  d params / d eta
            """
            params = family.inverse_link(eta)
            # Score in natural-param space, shape (n, npar); sum over obs
            S = family.score(y, params, w)   # (n, npar)
            grad_natural = -S.sum(axis=0)    # (npar,)

            # Jacobian d params / d eta: diagonal for element-wise links
            # Approximate numerically for generality; exact for Gaussian below
            jac = _link_jacobian(family, eta)  # (npar, npar)
            return jac.T @ grad_natural

        result = minimize(
            neg_loglik,
            eta0,
            jac=neg_score_eta,
            method="L-BFGS-B",
            bounds=family.link_bounds(),
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )

        eta_hat = result.x
        params_hat = family.inverse_link(eta_hat)

        # ------------------------------------------------------------------
        # Store fitted quantities
        # ------------------------------------------------------------------
        # Convert internal optimizer params to user-facing natural params
        # (e.g. GaussianFamily: log_sigma → sigma).
        display_params, display_names = family.to_user_params(params_hat)
        self.coef_: dict[str, float] = {
            name: float(v)
            for name, v in zip(display_names, display_params)
        }
        self.coef_array_: np.ndarray = display_params.copy()
        self.eta_: np.ndarray = eta_hat.copy()
        self.loglik_: float = float(-result.fun)
        self.converged_: bool = result.success
        self.n_samples_: int = n

        # Score matrix and Hessian at the MLE (natural scale)
        self.score_matrix_: np.ndarray = family.score(y, params_hat, w)  # (n, npar)
        self.hessian_: np.ndarray = family.hessian(y, params_hat, w)     # (npar, npar)

        # Link-scale estfun: wᵢ · ∂ℓᵢ/∂η (used by ctree split tests)
        # Chain rule: ∂ℓ/∂η_k = Σ_j (∂ℓ/∂params_j) * (∂params_j/∂η_k)
        # For element-wise links the Jacobian is diagonal.
        jac = _link_jacobian(family, eta_hat)               # (npar, npar)
        self.estfun_: np.ndarray = self.score_matrix_ @ jac # (n, npar)

        # Keep references for later prediction queries
        self._y = y
        self._w = w

        return self

    # ------------------------------------------------------------------
    # Prediction / distribution queries
    # ------------------------------------------------------------------

    def predict(self, type: str = "response") -> float | dict[str, float]:
        """Return a prediction from the fitted distribution.

        Parameters
        ----------
        type:
            ``"response"`` (default) — returns the fitted mean ``E[Y]``.
            ``"parameter"`` — returns a dict ``{param_name: value}``.

        Returns
        -------
        float or dict
        """
        self._check_fitted()
        if type == "response":
            return self.family.mean(self.coef_array_)
        if type == "parameter":
            return dict(self.coef_)
        raise ValueError(f"Unknown type {type!r}. Use 'response' or 'parameter'.")

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        """Log-density of the fitted distribution evaluated at *y*.

        Parameters
        ----------
        y:
            Array of values at which to evaluate log f(y).

        Returns
        -------
        np.ndarray of shape ``(len(y),)``.
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float).ravel()
        out = np.empty(len(y))
        for i, yi in enumerate(y):
            out[i] = self.family.log_likelihood(
                np.array([yi]), self.coef_array_, weights=None
            )
        return out

    def cdf(self, y: np.ndarray) -> np.ndarray:
        """CDF of the fitted distribution evaluated at *y*."""
        self._check_fitted()
        y = np.asarray(y, dtype=float).ravel()
        return _gaussian_cdf(y, self.coef_array_, self.family)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Quantile (percent-point) function of the fitted distribution.

        Parameters
        ----------
        q:
            Probabilities in ``(0, 1)``.
        """
        self._check_fitted()
        q = np.asarray(q, dtype=float).ravel()
        return _gaussian_ppf(q, self.coef_array_, self.family)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit() before using this method.")

    def __repr__(self) -> str:
        if not hasattr(self, "coef_"):
            return f"DistFit(family={self.family.__class__.__name__}, unfitted)"
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self.coef_.items())
        return (
            f"DistFit(family={self.family.__class__.__name__}, "
            f"{parts}, loglik={self.loglik_:.4g})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_jacobian(family: DistFamily, eta: np.ndarray) -> np.ndarray:
    """Diagonal Jacobian d params / d eta via numerical differentiation.

    Returns a square ``(npar, npar)`` matrix.  For element-wise links (the
    common case) this is diagonal.
    """
    eps = 1e-7
    params = family.inverse_link(eta)
    npar = len(eta)
    jac = np.zeros((npar, npar))
    for j in range(npar):
        eta_plus = eta.copy()
        eta_plus[j] += eps
        params_plus = family.inverse_link(eta_plus)
        jac[:, j] = (params_plus - params) / eps
    return jac


def _gaussian_cdf(y: np.ndarray, params: np.ndarray, family: DistFamily) -> np.ndarray:
    """CDF — falls back to scipy.stats.norm for GaussianFamily."""
    if isinstance(family, GaussianFamily):
        mu, sigma = params[0], params[1]
        return _stats.norm.cdf(y, loc=mu, scale=sigma)
    raise NotImplementedError("cdf only implemented for GaussianFamily.")


def _gaussian_ppf(q: np.ndarray, params: np.ndarray, family: DistFamily) -> np.ndarray:
    """PPF — falls back to scipy.stats.norm for GaussianFamily."""
    if isinstance(family, GaussianFamily):
        mu, sigma = params[0], params[1]
        return _stats.norm.ppf(q, loc=mu, scale=sigma)
    raise NotImplementedError("ppf only implemented for GaussianFamily.")
