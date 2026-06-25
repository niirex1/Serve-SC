"""Evaluation metrics.

Detection: macro-F1, one-vs-rest macro-AUC.
Prioritisation: HR@k, NDCG@k, Spearman rho.

Metric definitions are stated explicitly so they can be matched to the paper's
implementation. HR@k here is recall-at-k: the fraction of positive (high-service-
impact) cases that appear within the top-k of the ranking. If your paper uses a
different convention, align this function before quoting numbers from the suite.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0 and (fp == 0 or fn == 0):
        # no positives predicted and/or none present
        return 0.0 if (fp + fn) > 0 else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class F1 over a multi-label {0,1} matrix of shape (N, K)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = y_true.shape[1]
    return float(np.mean([_f1_binary(y_true[:, c], y_pred[:, c]) for c in range(k)]))


def _auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC via the Mann-Whitney statistic. Returns 0.5 for degenerate columns."""
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    r_pos = np.sum(ranks[y_true == 1])
    n_pos, n_neg = len(pos), len(neg)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc_ovr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean one-vs-rest AUC over a (N, K) score matrix."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = y_true.shape[1]
    return float(np.mean([_auc_binary(y_true[:, c], scores[:, c]) for c in range(k)]))


# --------------------------------------------------------------------------- #
# Prioritisation
# --------------------------------------------------------------------------- #
def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    return np.argsort(-np.asarray(scores), kind="mergesort")[:k]


def hit_rate_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Recall-at-k: fraction of positive cases captured in the top-k ranking."""
    relevant = np.asarray(relevant).astype(bool)
    n_pos = int(relevant.sum())
    if n_pos == 0:
        return 0.0
    top = _topk_indices(scores, k)
    return float(relevant[top].sum()) / float(n_pos)


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, k: int) -> float:
    """NDCG@k with (binary or graded) relevance ``gains``."""
    gains = np.asarray(gains, dtype=float)
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    k = min(k, len(scores))
    g = gains[order][:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(g * discount))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal * discount))
    return 0.0 if idcg == 0 else dcg / idcg


def spearman_rho(scores: np.ndarray, true_scores: np.ndarray) -> float:
    """Spearman rank correlation between predicted and reference scores."""
    if len(np.unique(scores)) < 2 or len(np.unique(true_scores)) < 2:
        return 0.0
    rho, _ = spearmanr(scores, true_scores)
    return float(rho)


def bootstrap_ci(values: np.ndarray, ci: float = 0.95):
    """Percentile interval of a bootstrap distribution of a metric."""
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))
