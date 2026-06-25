"""Contract-service graph construction (Section IV-A).

Consumes a contract intermediate representation (IR) -- see DATA_SCHEMA.md -- and
produces typed nodes, typed edges (including the transaction-derived
``tx_interact`` relation), per-node features, and the 12-dimensional dense cue
vector m_c. The IR can be produced by a Solidity/bytecode front-end (yours) or by
the synthetic generator in ``data_synth.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .config import DENSE_CUE_DIM, EDGE_TYPES, NODE_FEAT_DIM, NODE_TYPES


@dataclass
class ContractGraph:
    cid: str
    node_types: list                      # type string per node
    node_names: list                      # original identifier per node
    node_feats: np.ndarray                # (num_nodes, NODE_FEAT_DIM)
    edges: dict                           # edge_type -> (2, E) int index array
    dense_cues: np.ndarray                # (DENSE_CUE_DIM,)
    func_idx: np.ndarray                  # indices of func-type nodes
    meta: dict = field(default_factory=dict)


def _node_feature(name: str, rng: np.random.Generator,
                  planted: np.ndarray | None) -> np.ndarray:
    base = rng.normal(0, 1, size=NODE_FEAT_DIM)
    if planted is not None:
        base = base + planted
    return base.astype(np.float32)


def build_graph(ir: dict, rng: np.random.Generator | None = None) -> ContractGraph:
    """Build a :class:`ContractGraph` from one contract IR dict."""
    if rng is None:
        rng = np.random.default_rng(0)

    names, types = [], []
    index = {}

    def add(kind, name):
        key = (kind, name)
        if key not in index:
            index[key] = len(names)
            names.append(name)
            types.append(kind)
        return index[key]

    for f in ir.get("functions", []):
        add("func", f)
    for s in ir.get("state_vars", []):
        add("state", s)
    for x in ir.get("ext_calls", []):
        add("extcall", x)
    for o in ir.get("oracles", []):
        add("oracle", o)

    # planted class signal lives on function-node features (synthetic demo only)
    planted_vec = None
    planted = ir.get("_planted_func_signal")
    if planted is not None:
        planted_vec = np.asarray(planted, dtype=np.float32)

    feats = np.zeros((len(names), NODE_FEAT_DIM), dtype=np.float32)
    for i, (kind) in enumerate(types):
        pv = planted_vec if kind == "func" else None
        feats[i] = _node_feature(names[i], rng, pv)

    # static edges from the IR
    edge_lists = {et: [] for et in EDGE_TYPES}
    type_of_name = {}
    for (kind, name) in index:
        type_of_name.setdefault(name, kind)

    def resolve(name):
        kind = type_of_name.get(name)
        return index.get((kind, name)) if kind is not None else None

    for etype, src, dst in ir.get("edges", []):
        if etype not in edge_lists:
            continue
        si, di = resolve(src), resolve(dst)
        if si is not None and di is not None:
            edge_lists[etype].append((si, di))

    # tx_interact edges: functions co-occurring in the same transaction sequence
    co_pairs = set()
    for seq in ir.get("tx_sequences", []):
        fseq = [resolve(f) for f in seq if resolve(f) is not None]
        for a, b in combinations(sorted(set(fseq)), 2):
            co_pairs.add((a, b))
            co_pairs.add((b, a))
    edge_lists["tx_interact"].extend(sorted(co_pairs))

    edges = {}
    for et, lst in edge_lists.items():
        if lst:
            arr = np.asarray(lst, dtype=np.int64).T
        else:
            arr = np.zeros((2, 0), dtype=np.int64)
        edges[et] = arr

    func_idx = np.asarray([i for i, t in enumerate(types) if t == "func"],
                          dtype=np.int64)
    dense = _dense_cues(ir, type_of_name)

    return ContractGraph(
        cid=ir.get("id", "contract"),
        node_types=types,
        node_names=names,
        node_feats=feats,
        edges=edges,
        dense_cues=dense,
        func_idx=func_idx,
        meta={"num_functions": len(ir.get("functions", []))},
    )


def _dense_cues(ir: dict, type_of_name: dict) -> np.ndarray:
    """12 static cue counts, normalised by function count (Section IV-A)."""
    funcs = ir.get("functions", [])
    nf = max(1, len(funcs))
    cues = ir.get("static_cues", {})

    def c(name, default=0.0):
        return float(cues.get(name, default))

    tx_seqs = ir.get("tx_sequences", [])
    n_tx = len(tx_seqs)
    callers = ir.get("unique_callers", None)
    if callers is None:
        callers = len({t[0] for t in tx_seqs}) if tx_seqs else 0

    vec = np.array([
        c("reentrant_callsites"),
        c("low_level_calls"),
        c("write_after_call"),
        c("authorization_checkpoints"),
        c("max_cross_contract_depth"),
        c("oracle_reads"),
        c("unchecked_returns"),
        c("integer_arith"),
        c("selfdestruct"),
        c("delegatecall"),
        float(n_tx),
        float(callers),
    ], dtype=np.float32)
    vec[:10] = vec[:10] / nf
    assert vec.shape[0] == DENSE_CUE_DIM
    return vec
