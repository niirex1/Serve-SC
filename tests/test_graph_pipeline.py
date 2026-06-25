import numpy as np

from serve_sc.config import DENSE_CUE_DIM, NODE_FEAT_DIM
from serve_sc.data_synth import generate_contract, generate_dataset
from serve_sc.data_synth import _class_directions
from serve_sc.graph import build_graph
from serve_sc.pipeline import run_pipeline


def test_graph_has_types_and_tx_interact():
    dirs = _class_directions(0)
    ir = generate_contract(0, 0, dirs)
    g = build_graph(ir)
    assert "tx_interact" in g.edges
    assert g.node_feats.shape[1] == NODE_FEAT_DIM
    assert g.dense_cues.shape[0] == DENSE_CUE_DIM
    assert g.func_idx.size == len(ir["functions"])
    # every func that co-occurs in a sequence should have at least one tx edge
    if any(len(s) >= 2 for s in ir["tx_sequences"]):
        assert g.edges["tx_interact"].shape[1] > 0


def test_tx_interact_links_cooccurring_functions():
    ir = {
        "id": "t", "functions": ["a", "b", "c"], "state_vars": [],
        "ext_calls": [], "oracles": [], "edges": [],
        "tx_sequences": [["a", "b"]], "static_cues": {},
    }
    g = build_graph(ir)
    ei = g.edges["tx_interact"]
    names = set()
    for s, d in ei.T:
        names.add((g.node_names[s], g.node_names[d]))
    assert ("a", "b") in names and ("b", "a") in names
    # c never co-occurs, so it has no tx_interact edge
    assert all("c" not in pair for pair in names)


def test_pipeline_runs_and_is_bounded():
    irs = generate_dataset(n=120, seed=1)
    res = run_pipeline(irs, seed=1, force_fallback=True, synthetic=True)
    assert 0.0 <= res["detection"]["macro_F1"] <= 1.0
    assert 0.0 <= res["prioritisation"]["HR@10"] <= 1.0
    assert "SYNTHETIC" in res["banner"]
    assert res["backend"] == "numpy-fallback"
    assert len(res["risk_coefficients"]) == 8


def test_detector_learns_above_chance_on_planted_data():
    # with planted class signal the fallback should beat a trivial F1 floor
    irs = generate_dataset(n=200, seed=2)
    res = run_pipeline(irs, seed=2, force_fallback=True, synthetic=True)
    assert res["detection"]["macro_AUC"] > 0.6
