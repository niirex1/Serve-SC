# Input data schema

The pipeline consumes a list of **contract IR** objects. On disk the synthetic
generator writes one JSON object per line (`data/synthetic/contracts.jsonl`). To
reproduce the paper, export your real contracts/cases to the same shape.

Each object:

```jsonc
{
  "id": "c0",                          // unique id (string)

  // --- graph nodes ---
  "functions":  ["f0", "f1"],          // function node ids
  "state_vars": ["s0"],                // state variable node ids
  "ext_calls":  ["x0"],                // external-call node ids
  "oracles":    ["o0"],                // oracle node ids

  // --- static edges (from code structure) ---
  // each edge is [edge_type, src_id, dst_id]; edge_type in:
  //   calls, reads, writes, ctrl_dep, data_dep, oracle_dep
  "edges": [["calls","f0","f1"], ["reads","f0","s0"], ["oracle_dep","f1","o0"]],

  // --- runtime behaviour (drives tx_interact edges and behaviour cues) ---
  // a list of transaction sequences; each is an ordered list of function ids.
  // Functions co-occurring in the same sequence get a tx_interact edge.
  "tx_sequences": [["f0","f1"], ["f1"]],
  "unique_callers": 12,                // optional; else derived from sequences

  // --- 12 static cue raw counts (normalised by function count internally) ---
  "static_cues": {
    "reentrant_callsites": 2, "low_level_calls": 1, "write_after_call": 1,
    "authorization_checkpoints": 0, "max_cross_contract_depth": 3,
    "oracle_reads": 2, "unchecked_returns": 1, "integer_arith": 4,
    "selfdestruct": 0, "delegatecall": 0
    // (transaction count and unique-caller count are taken from the fields above)
  },

  // --- Stage-1 supervision (multi-label, 0/1 per class) ---
  "labels": {
    "reentrancy": 1, "access_control": 0, "unchecked_calls": 0,
    "oracle_misuse": 1, "logic_inconsistency": 0
  },

  // --- exploit preconditions φ_c (5 binary axes, fixed order) ---
  // [public_reachability, oracle_exposure, cross_contract_trigger,
  //  flashloan_feasible, tx_order_sensitive]
  "phi": [1, 1, 0, 0, 0],

  // --- service context z_c ---
  "service": { "e": 0.42, "chi": 0.31, "q": 1, "p": 0.18 },
  // e   asset exposure (TVL share), in [0,1]
  // chi normalised betweenness centrality, in [0,1]
  // q   user-facing criticality, 0/1
  // p   historical-incident prior for the protocol family, in [0,1]

  // --- Stage-2 supervision: high-service-impact label (0/1) ---
  // The binary calibration target. In the paper this is the upper-tier
  // direct-loss + service-disruption case; define it consistently with your
  // annotation and with the HR@k "relevant" criterion.
  "impact": 1
}
```

Notes:

- `_planted_func_signal` appears in the synthetic data only; it injects node
  features so the demo detector can learn. Real exports should omit it — real
  node features come from opcode n-gram / AST embeddings in `graph.py`
  (extend `build_graph` to attach your own features if you compute them
  upstream).
- Fields used for **training** (`labels`) and **calibration/evaluation**
  (`phi`, `service`, `impact`) are read by `pipeline.py`. Keep splits
  contract-disjoint; the pipeline uses a 60/20/20 train/dev/test partition by
  default and never trains the detector on the dev/test contracts.
