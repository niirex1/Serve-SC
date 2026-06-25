"""End-to-end SERVE-SC pipeline.

Stage 1 (detection) -> exploitability gate -> Stage 2 (risk scoring) -> ranking,
with detection and prioritisation metrics. Run on synthetic data it produces
DEMO numbers; run on the real datasets (DATA_SCHEMA.md) it produces the paper's
numbers. Nothing here is hardcoded to any target value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CLASSES, DEFAULT_SEED, K
from .detect import Detector
from .exploitability import exploitability
from .graph import build_graph
from .metrics import (bootstrap_ci, hit_rate_at_k, macro_auc_ovr, macro_f1,
                      ndcg_at_k, spearman_rho)
from .risk import RiskModel

BANNER = (
    "SYNTHETIC DEMONSTRATION RESULTS -- computed on generated data. "
    "These are NOT the paper's numbers and do not reproduce them. "
    "Supply the real datasets (see DATA_SCHEMA.md) to reproduce reported tables."
)


def _labels_matrix(irs):
    return np.array([[ir["labels"][c] for c in CLASSES] for ir in irs], dtype=int)


def _service(irs):
    e = np.array([ir["service"]["e"] for ir in irs], dtype=float)
    chi = np.array([ir["service"]["chi"] for ir in irs], dtype=float)
    q = np.array([ir["service"]["q"] for ir in irs], dtype=float)
    p = np.array([ir["service"]["p"] for ir in irs], dtype=float)
    return e, chi, q, p


def _phi(irs):
    return np.array([ir["phi"] for ir in irs], dtype=float)


def run_pipeline(irs, seed: int = DEFAULT_SEED, force_fallback: bool = False,
                 synthetic: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    n = len(irs)
    perm = rng.permutation(n)
    n_tr, n_dev = int(0.6 * n), int(0.2 * n)
    tr, dev, test = perm[:n_tr], perm[n_tr:n_tr + n_dev], perm[n_tr + n_dev:]

    graphs = [build_graph(ir, np.random.default_rng(seed + i))
              for i, ir in enumerate(irs)]
    Y = _labels_matrix(irs)

    # Stage 1: train detector on the training split only
    det = Detector(force_fallback=force_fallback)
    det.fit([graphs[i] for i in tr], Y[tr], seed=seed)
    proba_all = np.zeros((n, K))
    proba_all[test] = det.predict_proba([graphs[i] for i in test])
    proba_all[dev] = det.predict_proba([graphs[i] for i in dev])

    # Detection metrics on the held-out test split
    y_pred_test = (proba_all[test] >= 0.5).astype(int)
    det_f1 = macro_f1(Y[test], y_pred_test)
    det_auc = macro_auc_ovr(Y[test], proba_all[test])

    # Exploitability gate (parameter-free) on dev + test
    phi = _phi(irs)
    u_all = np.zeros(n)
    u_all[dev] = exploitability(proba_all[dev], phi[dev])
    u_all[test] = exploitability(proba_all[test], phi[test])

    # Stage 2: calibrate risk model on dev, evaluate on test
    e, chi, q, p = _service(irs)
    impact = np.array([ir["impact"] for ir in irs], dtype=int)
    risk = RiskModel().fit(u_all[dev], e[dev], chi[dev], q[dev], p[dev], impact[dev])
    r_test = risk.predict(u_all[test], e[test], chi[test], q[test], p[test])

    rel = impact[test]
    pri = {
        "HR@5": hit_rate_at_k(r_test, rel, 5),
        "HR@10": hit_rate_at_k(r_test, rel, 10),
        "NDCG@5": ndcg_at_k(r_test, rel.astype(float), 5),
        "NDCG@10": ndcg_at_k(r_test, rel.astype(float), 10),
        "Spearman": spearman_rho(r_test, rel.astype(float)),
    }

    # Bootstrap confidence intervals for the ranking metrics (Moderate #7)
    b_rng = np.random.default_rng(seed + 1)
    boot = {m: [] for m in pri}
    m_test = len(test)
    for _ in range(300):
        bi = b_rng.integers(0, m_test, size=m_test)
        rb, relb = r_test[bi], rel[bi]
        boot["HR@5"].append(hit_rate_at_k(rb, relb, 5))
        boot["HR@10"].append(hit_rate_at_k(rb, relb, 10))
        boot["NDCG@5"].append(ndcg_at_k(rb, relb.astype(float), 5))
        boot["NDCG@10"].append(ndcg_at_k(rb, relb.astype(float), 10))
        boot["Spearman"].append(spearman_rho(rb, relb.astype(float)))
    pri_ci = {m: bootstrap_ci(np.asarray(v), 0.95) for m, v in boot.items()}

    coeffs = risk.bootstrap_coefficients(
        u_all[dev], e[dev], chi[dev], q[dev], p[dev], impact[dev], seed=seed)

    return {
        "banner": BANNER if synthetic else "Results on user-supplied data.",
        "backend": det.backend,
        "n_contracts": n,
        "splits": {"train": len(tr), "dev": len(dev), "test": len(test)},
        "detection": {"macro_F1": det_f1, "macro_AUC": det_auc},
        "prioritisation": pri,
        "prioritisation_95CI": pri_ci,
        "risk_coefficients": coeffs,
    }


def write_results(result: dict, out_dir: str = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    lines = [f"# SERVE-SC pipeline results", "", f"> {result['banner']}", "",
             f"- backend: `{result['backend']}`",
             f"- contracts: {result['n_contracts']} "
             f"(train/dev/test = {result['splits']['train']}/"
             f"{result['splits']['dev']}/{result['splits']['test']})", "",
             "## Detection",
             f"- macro-F1: {result['detection']['macro_F1']:.3f}",
             f"- macro-AUC: {result['detection']['macro_AUC']:.3f}", "",
             "## Prioritisation"]
    for m, v in result["prioritisation"].items():
        lo, hi = result["prioritisation_95CI"][m]
        lines.append(f"- {m}: {v:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
    lines += ["", "## Risk-model coefficients (bootstrap)",
              "| term | median | 90% range | excludes 0 |",
              "|------|--------|-----------|------------|"]
    for c in result["risk_coefficients"]:
        lines.append(f"| {c['term']} | {c['median']:+.2f} | "
                     f"[{c['lo']:+.2f}, {c['hi']:+.2f}] | {c['excludes_zero']} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    return out / "results.md"
