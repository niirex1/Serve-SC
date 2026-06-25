import numpy as np

from serve_sc.risk import RiskModel, design_matrix


def _make_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    e = rng.random(n)
    chi = rng.random(n)
    q = (rng.random(n) < 0.5).astype(float)
    p = rng.beta(2, 5, size=n)
    # known generative coefficients
    logit = (-1.5 + 2.0 * u + 0.5 * e + 0.4 * chi + 0.8 * q + 0.6 * p
             + 1.2 * u * e + 1.0 * u * chi)
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    return u, e, chi, q, p, y


def test_design_matrix_columns():
    X = design_matrix([1], [0.5], [0.4], [1], [0.2])
    # [1, u, e, chi, q, p, u*e, u*chi]
    assert X.shape == (1, 8)
    assert np.allclose(X[0], [1, 1, 0.5, 0.4, 1, 0.2, 0.5, 0.4])


def test_recovers_positive_signs():
    u, e, chi, q, p, y = _make_data()
    m = RiskModel(lam=1e-3).fit(u, e, chi, q, p, y)
    b = m.beta
    # exploitability and the two interactions should be clearly positive
    assert b[1] > 0.5            # u
    assert b[6] > 0.0            # u*e
    assert b[7] > 0.0            # u*chi
    assert b[0] < 0.0            # intercept


def test_predict_in_unit_interval():
    u, e, chi, q, p, y = _make_data(n=500)
    m = RiskModel().fit(u, e, chi, q, p, y)
    r = m.predict(u, e, chi, q, p)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_bootstrap_dominant_terms_exclude_zero():
    u, e, chi, q, p, y = _make_data()
    m = RiskModel(lam=1e-3).fit(u, e, chi, q, p, y)
    coeffs = {c["term"]: c for c in
              m.bootstrap_coefficients(u, e, chi, q, p, y, replicates=200)}
    assert coeffs["u"]["excludes_zero"]
