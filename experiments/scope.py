#!/usr/bin/env python3
"""RQ6 scope (descriptive): how much of real audit findings (Web3Bugs) and real
exploits (DeFi attack victims, Chaliasos et al.) falls within the 17 SWC types.

    python experiments/scope.py
Writes results/analysis/scope.json.
"""
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from specgnn.utils import load_config, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    paths = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]
    absp = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in paths.items()}
    out = {}
    w3 = os.path.join(ROOT, "reports", "web3bugs_scope.json")
    if os.path.exists(w3):
        with open(w3) as fh:
            d = json.load(fh)
        out["web3bugs"] = {"total_findings": d["total_bugs"], "in_scope": d["mapped_total"],
                           "share_in_scope": d["mapped_total"] / d["total_bugs"],
                           "in_scope_by_swc": d["per_swc"], "by_family": d["by_family"]}
    ch = os.path.join(absp["processed"], "chaliasos.jsonl")
    if os.path.exists(ch):
        with open(ch, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        titles = Counter(t for r in recs for t in r["meta"]["vulnerability_titles"])
        incidents_all = {i for r in recs for i in r["meta"]["incidents"]}
        incidents_in = {i for r in recs if r["meta"]["in_scope"] for i in r["meta"]["incidents"]}
        out["defi_attacks"] = {
            "victim_contracts": len(recs),
            "victim_contracts_in_scope": sum(r["meta"]["in_scope"] for r in recs),
            "incidents": len(incidents_all), "incidents_in_scope": len(incidents_in),
            "vulnerability_titles": dict(titles.most_common())}
    write_json(os.path.join(absp["results"], "analysis", "scope.json"), out)
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)})


if __name__ == "__main__":
    main()
