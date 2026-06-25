# SERVE-SC — reference implementation and reproducibility artifact

Service-impact prioritisation of smart-contract vulnerabilities. This repository
implements the two-stage method from the paper and ships a small **synthetic**
dataset so the full pipeline runs end to end and the unit tests pass without any
proprietary data.

```
Stage 1  Heterogeneous graph transformer over the contract-service graph
         (typed nodes/edges incl. the runtime tx_interact edge) + 12 static
         cues  ->  five-class vulnerability prediction  (ŷ_c)
   |
Exploitability gate (parameter-free)   u_c = max_k ŷ_{c,k} · π_k(φ_c)     [Eq. 2]
   |
Stage 2  L2-regularised logistic service-impact model                     [Eq. 3]
         r_c = σ(β0 + β1 u + β2 e + β3 χ + β4 q + β5 p + β6 ue + β7 uχ)
   |
Ranked alert list Π (contracts ordered by r_c)
```

## ⚠️ What the synthetic demo does and does not show

The included data is **generated** (`serve_sc/data_synth.py`). Running the demo
produces numbers on that synthetic data; **it does not reproduce the paper's
numbers and is not meant to.** The paper's results come from *SmartBugs Curated*,
*SolidiFI-benchmark*, and the 127-case financial-service suite. Nothing in this
repository is hardcoded to any reported value — the pipeline is data-agnostic:

| Input | What you get |
|-------|--------------|
| Synthetic demo data (default) | demonstration metrics; pipeline/test smoke check |
| Your real datasets (see `DATA_SCHEMA.md`) | the values you report in the paper |

The data generator plants structure on purpose, so the demo shows the
machinery working: the detector learns above chance, and the risk model recovers
the **positive interaction terms** `u·e` and `u·χ` (the paper's central scoring
claim) with bootstrap intervals that exclude zero, while the single-term `u`
coefficient stays noisy on the small development split — the same small-sample
fragility discussed in the paper's Threats to Validity.

## Layout

```
serve_sc/
  config.py          classes, precondition axes, the fixed map A, hyperparameters
  graph.py           contract-service graph from an IR (incl. tx_interact, 12 cues)
  hgt.py             heterogeneous graph transformer (PyTorch) — the Stage-1 model
  detect.py          detector wrapper: HGT if torch present, else numpy fallback
  exploitability.py  parameter-free gate (Eq. 2)
  risk.py            L2 logistic risk model + bootstrap coefficients (Eq. 3)
  metrics.py         macro-F1, macro-AUC, HR@k, NDCG@k, Spearman (definitions documented)
  data_synth.py      synthetic IR generator (DEMONSTRATION ONLY)
  pipeline.py        end-to-end orchestration + results writer
scripts/
  run_demo.py             run the pipeline on synthetic data
  make_synthetic_data.py  dump synthetic IRs to data/synthetic/contracts.jsonl
tests/                    pytest suite (gate, metrics, risk, graph, pipeline)
DATA_SCHEMA.md            input format for plugging in real data
```

## Install

```bash
python -m pip install -r requirements.txt
# PyTorch is optional. With it, Stage 1 uses the HGT (the method in the paper).
# Without it, a numpy logistic fallback runs so the demo/tests still work.
```

## Quickstart

```bash
python scripts/run_demo.py            # runs pipeline, writes results/results.md
python scripts/run_demo.py --fallback # force the numpy path (no torch needed)
python -m pytest -q                   # run the test suite
```

## Reproducing the paper

1. Export your detection corpora and the 127-case suite to the IR format in
   `DATA_SCHEMA.md` (one JSON object per contract/case).
2. Load them instead of `generate_dataset(...)` and call
   `serve_sc.pipeline.run_pipeline(irs, synthetic=False)`.
3. Confirm the metric definitions in `serve_sc/metrics.py` match the paper's
   conventions (HR@k here is recall-at-k); align if needed before quoting.

The detection/scoring code is identical across synthetic and real data; only the
inputs differ. Replace, re-run, report.

## Tests

`python -m pytest -q` — covers the gate (parameter-free correctness), the metrics
(against hand-checked cases), the risk solver (recovers known coefficient signs),
graph construction (tx_interact links co-activating functions), and a full
pipeline smoke test.

## Citation

If you use this code, please cite the paper (BibTeX to be added on acceptance).

## License

See `LICENSE` (MIT placeholder — set the copyright holder before publishing).
