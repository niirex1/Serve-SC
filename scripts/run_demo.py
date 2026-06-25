#!/usr/bin/env python3
"""Run the full SERVE-SC pipeline on synthetic demonstration data.

    python scripts/run_demo.py [--n 240] [--seed 0] [--fallback]

Prints DEMO results and writes results/results.{json,md}. These numbers are on
synthetic data and do NOT reproduce the paper; see README.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve_sc.data_synth import generate_dataset
from serve_sc.pipeline import run_pipeline, write_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fallback", action="store_true",
                    help="force the numpy fallback detector even if torch exists")
    args = ap.parse_args()

    irs = generate_dataset(n=args.n, seed=args.seed)
    result = run_pipeline(irs, seed=args.seed, force_fallback=args.fallback,
                          synthetic=True)
    path = write_results(result)

    print("=" * 72)
    print(result["banner"])
    print("=" * 72)
    print(f"backend: {result['backend']}  |  contracts: {result['n_contracts']}")
    d = result["detection"]
    print(f"detection  macro-F1={d['macro_F1']:.3f}  macro-AUC={d['macro_AUC']:.3f}")
    print("prioritisation:")
    for m, v in result["prioritisation"].items():
        lo, hi = result["prioritisation_95CI"][m]
        print(f"  {m:9s} {v:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("risk coefficients (median [90% range], * = excludes 0):")
    for c in result["risk_coefficients"]:
        star = "*" if c["excludes_zero"] else " "
        print(f"  {c['term']:9s} {c['median']:+.2f} "
              f"[{c['lo']:+.2f}, {c['hi']:+.2f}] {star}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
