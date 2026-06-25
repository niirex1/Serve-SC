#!/usr/bin/env python3
"""Convert SmartBugs Curated into the SERVE-SC IR schema (detection labels).

    python scripts/convert_smartbugs_to_ir.py \
        --dataset smartbugs-curated --out ir/smartbugs_ir.jsonl

SmartBugs Curated uses the DASP-10 taxonomy. The paper uses five classes. The
mapping below is an explicit modelling decision -- EDIT IT to match your paper.
Two honest caveats:

  * SmartBugs Curated has NO oracle_misuse category, so that paper class has no
    source here. You must supply oracle-misuse examples from elsewhere.
  * denial_of_service / short_addresses / other have no clean class and are
    DROPPED by default (use --keep-unmapped to emit them with all-zero labels).

The converter fills `labels` from the dataset annotations and a few APPROXIMATE
regex-derived static cues. It does NOT build the graph, transaction sequences,
service-context, exploit-preconditions, or impact label -- those need your
Solidity front-end and on-chain/DeFi annotations (see DATASETS.md). The output
therefore supports the Stage-1 detection task out of the box; Stage-2
prioritisation needs the additional fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# --- EDIT THIS to match your paper's taxonomy ------------------------------- #
CATEGORY_TO_CLASS = {
    "reentrancy": "reentrancy",
    "access_control": "access_control",
    "unchecked_low_level_calls": "unchecked_calls",
    "arithmetic": "logic_inconsistency",     # over/underflow -> logic
    "front_running": "logic_inconsistency",  # coarse; ordering-related
    "bad_randomness": "logic_inconsistency", # coarse; flawed logic
    "time_manipulation": "logic_inconsistency",
    "denial_of_service": None,               # no clean class -> dropped
    "short_addresses": None,                 # legacy ABI -> dropped
    "other": None,                           # dropped
    # NOTE: "oracle_misuse" is a paper class with NO SmartBugs source.
}
CLASSES = ["reentrancy", "access_control", "unchecked_calls",
           "oracle_misuse", "logic_inconsistency"]

FUNC_RE = re.compile(r"\bfunction\s+([A-Za-z_]\w*)")


def approximate_cues(src: str, n_func: int) -> dict:
    """Rough regex cues. APPROXIMATE -- replace with a Slither-based extractor."""
    low = len(re.findall(r"\.call\s*[({]|\.delegatecall|\.callcode|\.send\s*\(",
                         src))
    reentr = len(re.findall(r"\.call\.value|\.call\s*{\s*value", src))
    return {
        "reentrant_callsites": reentr,
        "low_level_calls": low,
        "write_after_call": 0,                       # needs dataflow; placeholder
        "authorization_checkpoints": len(re.findall(r"\brequire\s*\(", src)),
        "max_cross_contract_depth": 0,               # needs call graph
        "oracle_reads": 0,                           # no oracle category here
        "unchecked_returns": len(re.findall(r"\.send\s*\(", src)),
        "integer_arith": len(re.findall(r"[^=!<>]=[^=]|\+\+|--|\*|/", src)) // 5,
        "selfdestruct": int(bool(re.search(r"selfdestruct|suicide\s*\(", src))),
        "delegatecall": int("delegatecall" in src),
    }


def convert(dataset_dir: Path, vuln_json: Path, keep_unmapped: bool) -> list:
    entries = json.loads(vuln_json.read_text())
    out, dropped, by_class = [], 0, {c: 0 for c in CLASSES}
    for e in entries:
        cats = {v["category"] for v in e.get("vulnerabilities", [])}
        mapped = {CATEGORY_TO_CLASS.get(c) for c in cats}
        mapped.discard(None)
        if not mapped and not keep_unmapped:
            dropped += 1
            continue
        labels = {c: int(c in mapped) for c in CLASSES}
        for c in mapped:
            by_class[c] += 1

        sol_path = dataset_dir.parent / e["path"]
        src = sol_path.read_text(errors="replace") if sol_path.exists() else ""
        funcs = [f"f_{i}_{m.group(1)}" for i, m in enumerate(FUNC_RE.finditer(src))]
        if not funcs:
            funcs = ["f_0_fallback"]

        out.append({
            "id": Path(e["name"]).stem,
            "source_path": e["path"],
            "source_url": e.get("source"),
            "pragma": e.get("pragma"),
            "source_sha256": hashlib.sha256(src.encode()).hexdigest(),
            "smartbugs_categories": sorted(cats),
            "vulnerable_lines": sorted({ln for v in e["vulnerabilities"]
                                        for ln in v.get("lines", [])}),
            "functions": funcs,
            "state_vars": [], "ext_calls": [], "oracles": [],
            "edges": [], "tx_sequences": [],
            "static_cues": approximate_cues(src, len(funcs)),
            "labels": labels,
            # Stage-2 fields require your DeFi annotations (DATASETS.md):
            "phi": None, "service": None, "impact": None,
            "_partial": True,
            "_provenance": {"dataset": "smartbugs-curated",
                            "license": "Apache-2.0"},
        })
    return out, dropped, by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="smartbugs-curated")
    ap.add_argument("--out", default="ir/smartbugs_ir.jsonl")
    ap.add_argument("--keep-unmapped", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    dataset_dir = root / "dataset"
    vuln_json = root / "vulnerabilities.json"
    if not vuln_json.exists():
        raise SystemExit(f"vulnerabilities.json not found under {root}/")

    records, dropped, by_class = convert(dataset_dir, vuln_json, args.keep_unmapped)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {len(records)} IR records to {out}")
    print(f"dropped (unmapped category): {dropped}")
    print("per-class positive counts:")
    for c, n in by_class.items():
        tag = "  <-- NO SOURCE in SmartBugs" if c == "oracle_misuse" else ""
        print(f"  {c}: {n}{tag}")
    print("\nNote: labels are real; graph/tx/service/phi/impact are NOT populated "
          "(see DATASETS.md). Output supports Stage-1 detection as-is.")


if __name__ == "__main__":
    main()
