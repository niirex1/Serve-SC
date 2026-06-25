import numpy as np

from serve_sc.metrics import (hit_rate_at_k, macro_auc_ovr, macro_f1,
                              ndcg_at_k, spearman_rho)


def test_macro_f1_perfect():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    assert macro_f1(y, y) == 1.0


def test_macro_f1_known_value():
    y_true = np.array([[1], [1], [0], [0]])
    y_pred = np.array([[1], [0], [0], [0]])  # tp=1 fn=1 fp=0 -> P=1 R=0.5 F1=2/3
    assert abs(macro_f1(y_true, y_pred) - (2 / 3)) < 1e-9


def test_auc_perfect_separation():
    y = np.array([[0], [0], [1], [1]])
    s = np.array([[0.1], [0.2], [0.8], [0.9]])
    assert abs(macro_auc_ovr(y, s) - 1.0) < 1e-9


def test_auc_chance_for_constant_score():
    y = np.array([[0], [1], [0], [1]])
    s = np.array([[0.5], [0.5], [0.5], [0.5]])
    assert abs(macro_auc_ovr(y, s) - 0.5) < 1e-9


def test_hit_rate_recall_at_k():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    rel = np.array([1, 0, 1, 0, 0])           # two positives, both in top 3
    assert hit_rate_at_k(scores, rel, 3) == 1.0
    assert hit_rate_at_k(scores, rel, 1) == 0.5


def test_ndcg_ideal_is_one():
    scores = np.array([3.0, 2.0, 1.0])
    gains = np.array([3.0, 2.0, 1.0])
    assert abs(ndcg_at_k(scores, gains, 3) - 1.0) < 1e-9


def test_spearman_monotone():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([10.0, 20.0, 30.0, 40.0])
    assert abs(spearman_rho(s, t) - 1.0) < 1e-9
