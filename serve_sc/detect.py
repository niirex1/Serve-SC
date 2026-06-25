"""Stage-1 detector wrapper.

If PyTorch is installed, the heterogeneous graph transformer in ``hgt.py`` is the
model (this is what the paper reports). If not, a lightweight numpy classifier is
used so the pipeline still runs end to end: per-class L2-regularised logistic
regression on the mean function-node features concatenated with the 12 dense
cues. The fallback is a smoke-test convenience, not the method, and is reported
as such in any output it produces.
"""
from __future__ import annotations

import numpy as np

from .config import K
from .hgt import TORCH_AVAILABLE
from .risk import _grad_l2, _nll_l2, _sigmoid  # reuse the L2 logistic solver
from scipy.optimize import minimize


def _graph_vector(graph) -> np.ndarray:
    """Pooled function-node features ++ dense cues."""
    if graph.func_idx.size > 0:
        pooled = graph.node_feats[graph.func_idx].mean(0)
    else:
        pooled = graph.node_feats.mean(0)
    return np.concatenate([pooled, graph.dense_cues]).astype(np.float64)


class NumpyFallbackDetector:
    """Per-class logistic regression. Clearly NOT the HGT."""

    is_fallback = True

    def __init__(self, lam: float = 1e-2):
        self.lam = lam
        self.betas = None
        self.mu = None
        self.sd = None

    def fit(self, graphs, labels):
        X = np.stack([_graph_vector(g) for g in graphs])
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        Xn = np.column_stack([np.ones(len(Xn)), Xn])
        Y = np.asarray(labels, dtype=float)
        self.betas = []
        for c in range(K):
            y = Y[:, c]
            b0 = np.zeros(Xn.shape[1])
            res = minimize(_nll_l2, b0, args=(Xn, y, self.lam),
                           jac=_grad_l2, method="L-BFGS-B")
            self.betas.append(res.x)
        return self

    def predict_proba(self, graphs) -> np.ndarray:
        X = np.stack([_graph_vector(g) for g in graphs])
        Xn = (X - self.mu) / self.sd
        Xn = np.column_stack([np.ones(len(Xn)), Xn])
        return np.column_stack([_sigmoid(Xn @ b) for b in self.betas])


class Detector:
    """Front-end that picks the HGT or the fallback."""

    def __init__(self, force_fallback: bool = False):
        self.use_hgt = TORCH_AVAILABLE and not force_fallback
        self._model = None
        self._impl = None

    @property
    def backend(self) -> str:
        return "hgt" if self.use_hgt else "numpy-fallback"

    def fit(self, graphs, labels, seed: int = 0):
        if self.use_hgt:
            from .hgt import train_hgt
            self._model = train_hgt(graphs, labels, seed=seed)
        else:
            self._impl = NumpyFallbackDetector().fit(graphs, labels)
        return self

    def predict_proba(self, graphs) -> np.ndarray:
        if self.use_hgt:
            from .hgt import predict_hgt
            return predict_hgt(self._model, graphs)
        return self._impl.predict_proba(graphs)
