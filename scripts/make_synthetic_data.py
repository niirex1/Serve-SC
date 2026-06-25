#!/usr/bin/env python3
"""Write a synthetic dataset to data/synthetic/contracts.jsonl (one IR per line).

    python scripts/make_synthetic_data.py [--n 240] [--seed 0]

Use this to inspect the input schema (DATA_SCHEMA.md) and to produce a file you
can diff against your own real-data export.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve_sc.data_synth import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic/contracts.jsonl")
    args = ap.parse_args()

    irs = generate_dataset(n=args.n, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for ir in irs:
            fh.write(json.dumps(ir) + "\n")
    print(f"wrote {len(irs)} synthetic contract IRs to {out}")


if __name__ == "__main__":
    main()
