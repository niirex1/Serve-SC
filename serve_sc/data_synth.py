"""Synthetic dataset generator (DEMONSTRATION ONLY).

Emits contract IRs in the schema of DATA_SCHEMA.md with *planted* structure so
the pipeline is exercised end to end: function-node features and dense cues carry
class signal (the detector should learn above chance), and the high-service-impact
label is drawn from a logistic in which exploitability and its interactions with
exposure and centrality are positive (the risk model should recover those signs).

This is NOT the paper's data and does NOT reproduce the paper's numbers. The
paper's results come from SmartBugs Curated, SolidiFI-benchmark, and the 127-case
financial-service suite. Point the pipeline at those (DATA_SCHEMA.md) to reproduce
the reported tables.
"""
from __future__ import annotations

import numpy as np

from .config import CLASSES, K, NODE_FEAT_DIM, P
from .exploitability import exploitability

# Planted generator coefficients for the impact label (Eq. 3 structure).
# Positive exploitability, exposure, centrality, and the two interactions; these
# are what the fitted risk model should recover on this synthetic data.
_PLANT_BETA = {
    "intercept": -3.8, "u": 2.2, "e": 0.6, "chi": 0.5,
    "q": 0.9, "p": 0.7, "u*e": 1.3, "u*chi": 1.0,
}

# Which precondition axis tends to enable each class (mirrors the A map).
_ENABLER = {
    "reentrancy": 2, "access_control": 0, "unchecked_calls": 2,
    "oracle_misuse": 1, "logic_inconsistency": 3,
}


def _class_directions(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 777)
    return rng.normal(0, 1, size=(K, NODE_FEAT_DIM)).astype(np.float32)


def _cues_for_labels(labels: np.ndarray, rng) -> dict:
    """Dense-cue counts weakly correlated with present classes (overlapping)."""
    def base(present_scale, absent_scale, present):
        scale = present_scale if present else absent_scale
        return float(max(0.0, rng.normal(scale, 1.1)))
    return {
        "reentrant_callsites": base(1.3, 0.5, labels[0]),
        "low_level_calls": base(1.1, 0.5, labels[2]),
        "write_after_call": base(0.9, 0.4, labels[0]),
        "authorization_checkpoints": base(0.5, 1.1, labels[1]),
        "max_cross_contract_depth": base(1.3, 0.7, labels[0] or labels[2]),
        "oracle_reads": base(1.5, 0.5, labels[3]),
        "unchecked_returns": base(1.3, 0.5, labels[2]),
        "integer_arith": base(1.1, 0.7, labels[4]),
        "selfdestruct": float(rng.random() < (0.25 if labels[1] else 0.08)),
        "delegatecall": float(rng.random() < (0.25 if labels[1] else 0.08)),
    }


def generate_contract(idx: int, seed: int, dirs: np.ndarray) -> dict:
    rng = np.random.default_rng(seed * 100003 + idx)
    n_func = int(rng.integers(3, 10))
    funcs = [f"f{idx}_{i}" for i in range(n_func)]
    states = [f"s{idx}_{i}" for i in range(int(rng.integers(1, 4)))]
    extcalls = [f"x{idx}_{i}" for i in range(int(rng.integers(0, 3)))]
    oracles = [f"o{idx}_{i}" for i in range(int(rng.integers(0, 2)))]

    # multi-label vulnerability presence
    base_rates = np.array([0.35, 0.30, 0.30, 0.22, 0.25])
    labels = (rng.random(K) < base_rates).astype(int)
    if labels.sum() == 0:
        labels[rng.integers(0, K)] = 1

    # planted function-node signal from present classes, with contract-level
    # confounding noise so detection is strong but not perfect on the demo
    planted = (dirs * labels[:, None]).sum(0) * 0.30
    planted = planted + rng.normal(0, 0.35, size=NODE_FEAT_DIM).astype(np.float32)

    # static edges
    edges = []
    for a in range(n_func):
        for b in range(n_func):
            if a != b and rng.random() < 0.25:
                edges.append(["calls", funcs[a], funcs[b]])
    for f in funcs:
        for s in states:
            if rng.random() < 0.4:
                edges.append(["reads", f, s])
            if rng.random() < 0.3:
                edges.append(["writes", f, s])
        for o in oracles:
            if rng.random() < 0.5:
                edges.append(["oracle_dep", f, o])

    # transaction sequences (drive tx_interact + behaviour cues)
    n_seq = int(rng.integers(2, 8))
    tx_sequences = []
    for _ in range(n_seq):
        L = int(rng.integers(2, min(5, n_func) + 1))
        tx_sequences.append(list(rng.choice(funcs, size=L, replace=False)))

    cues = _cues_for_labels(labels, rng)

    # exploit preconditions phi (noisy, correlated with enabling classes)
    phi = np.zeros(P, dtype=int)
    for c in range(K):
        if labels[c]:
            j = _ENABLER[CLASSES[c]]
            if rng.random() < 0.7:
                phi[j] = 1
    # occasional spurious preconditions
    for j in range(P):
        if rng.random() < 0.1:
            phi[j] = 1

    # service-context signals
    e = float(rng.random())
    chi = float(rng.random())
    q = int(rng.random() < 0.5)
    p = float(rng.beta(2, 5))

    # planted high-service-impact label from the Eq. (3) structure, using a
    # ground-truth exploitability computed from the true labels and phi
    u_true = float(exploitability(labels[None, :].astype(float), phi[None, :])[0])
    b = _PLANT_BETA
    logit = (b["intercept"] + b["u"] * u_true + b["e"] * e + b["chi"] * chi
             + b["q"] * q + b["p"] * p + b["u*e"] * u_true * e
             + b["u*chi"] * u_true * chi)
    prob = 1.0 / (1.0 + np.exp(-logit))
    impact = int(rng.random() < prob)

    return {
        "id": f"c{idx}",
        "functions": funcs,
        "state_vars": states,
        "ext_calls": extcalls,
        "oracles": oracles,
        "edges": edges,
        "tx_sequences": tx_sequences,
        "static_cues": cues,
        "_planted_func_signal": planted.tolist(),
        "labels": {CLASSES[c]: int(labels[c]) for c in range(K)},
        "phi": phi.tolist(),
        "service": {"e": e, "chi": chi, "q": q, "p": p},
        "impact": impact,
    }


def generate_dataset(n: int = 240, seed: int = 0) -> list:
    dirs = _class_directions(seed)
    return [generate_contract(i, seed, dirs) for i in range(n)]
