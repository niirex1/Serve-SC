"""Stage-2 service-impact risk model (Eq. 3).

r_c = sigmoid(beta0 + beta1 u + beta2 e + beta3 chi + beta4 q + beta5 p
              + beta6 (u*e) + beta7 (u*chi)).

Eight coefficients, fitted by L2-regularised maximum likelihood (the intercept
is not penalised). The two interaction terms are imposed as a modelling prior,
not selected from data. Coefficient stability is reported via the bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import BOOTSTRAP_CI, BOOTSTRAP_REPLICATES, RISK_L2_LAMBDA, RISK_TERMS


def design_matrix(u, e, chi, q, p) -> np.ndarray:
    """Build the (N, 8) design matrix [1, u, e, chi, q, p, u*e, u*chi]."""
    u, e, chi, q, p = (np.asarray(x, dtype=float) for x in (u, e, chi, q, p))
    ones = np.ones_like(u)
    return np.column_stack([ones, u, e, chi, q, p, u * e, u * chi])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _nll_l2(beta, X, y, lam):
    """Negative log-likelihood + L2 on non-intercept terms."""
    z = X @ beta
    p = _sigmoid(z)
    eps = 1e-12
    nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    reg = lam * np.sum(beta[1:] ** 2)
    return nll + reg


def _grad_l2(beta, X, y, lam):
    p = _sigmoid(X @ beta)
    g = X.T @ (p - y)
    reg = 2 * lam * beta
    reg[0] = 0.0
    return g + reg


@dataclass
class RiskModel:
    lam: float = RISK_L2_LAMBDA
    beta: np.ndarray | None = None

    def fit(self, u, e, chi, q, p, y) -> "RiskModel":
        X = design_matrix(u, e, chi, q, p)
        y = np.asarray(y, dtype=float)
        beta0 = np.zeros(X.shape[1])
        res = minimize(_nll_l2, beta0, args=(X, y, self.lam),
                       jac=_grad_l2, method="L-BFGS-B")
        self.beta = res.x
        return self

    def predict(self, u, e, chi, q, p) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("RiskModel is not fitted")
        return _sigmoid(design_matrix(u, e, chi, q, p) @ self.beta)

    def bootstrap_coefficients(self, u, e, chi, q, p, y,
                               replicates: int = BOOTSTRAP_REPLICATES,
                               ci: float = BOOTSTRAP_CI,
                               seed: int = 0):
        """Resample the development cases, refit, and summarise each coefficient.

        Returns a list of dicts with median and central interval per term. The
        signs/ordering this yields reflect the data it is fitted on -- on real
        case data they are the values that belong in the paper's coefficient
        table; on synthetic data they reflect the planted structure only.
        """
        rng = np.random.default_rng(seed)
        u, e, chi, q, p, y = (np.asarray(x, dtype=float)
                              for x in (u, e, chi, q, p, y))
        n = len(y)
        draws = []
        for _ in range(replicates):
            idx = rng.integers(0, n, size=n)
            try:
                m = RiskModel(self.lam).fit(u[idx], e[idx], chi[idx],
                                            q[idx], p[idx], y[idx])
                draws.append(m.beta)
            except Exception:
                continue
        draws = np.asarray(draws)
        lo_pct = (1 - ci) / 2 * 100
        hi_pct = (1 + ci) / 2 * 100
        out = []
        for j, name in enumerate(RISK_TERMS):
            col = draws[:, j]
            out.append({
                "term": name,
                "median": float(np.median(col)),
                "lo": float(np.percentile(col, lo_pct)),
                "hi": float(np.percentile(col, hi_pct)),
                "excludes_zero": bool(np.percentile(col, lo_pct) > 0
                                      or np.percentile(col, hi_pct) < 0),
            })
        return out
