#!/usr/bin/env python3
"""Agreement between specification verdicts and ground truth (no model involved).

For every SWC type t:
  positives  = nodes annotated as vulnerable to t (line-level ground truth)
  negatives  = applicable nodes in files whose label for t is 0
and we report the verdict distribution on each, plus
  confirmed precision  = CONF on positives / all CONF   (ground-truth based)
  refuted precision    = REF on negatives / all REF
Negatives in audit datasets are "not reported", not "proven safe", so precision
values are lower bounds.

Usage: python scripts/03_eval_specs.py --dataset dappscan --splits test --out reports/spec_eval_dappscan.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import (
    RAW_SUBDIRS,  # noqa: E402
    SWC_TYPES,  # noqa: E402
)
from specgnn.data.normalize import read_source  # noqa: E402
from specgnn.graph.features import NODE_TYPES  # noqa: E402
from specgnn.spec.checker import FileChecker  # noqa: E402
from specgnn.spec.library import SPECS  # noqa: E402
from specgnn.training import load_samples  # noqa: E402

ROOTS = RAW_SUBDIRS
SCOPE_NODES = {"function": ("function",), "contract": ("contract",), "state_var": ("state_var",)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with the unzipped upstream releases")
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--dataset", default="dappscan")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    samples = load_samples(args.graphs, args.dataset,
                           set(args.splits) if args.splits else None)
    pos = defaultdict(lambda: defaultdict(int))
    neg = defaultdict(lambda: defaultdict(int))
    reasons = defaultdict(lambda: defaultdict(int))
    t0, checks = time.time(), 0
    for s in samples:
        g = s["graph"]
        fc = FileChecker(read_source(os.path.join(args.raw, ROOTS[args.dataset], s["rel_path"])))
        ntypes = [NODE_TYPES[t] for t in g["node_type"]]
        for j, t in enumerate(SWC_TYPES):
            if s["y_mask"][j] == 0:
                continue
            scope = SCOPE_NODES[SPECS[t].scope]
            if s["y"][j] == 1:
                nodes = [i for i in range(len(ntypes)) if s["node_y"][i, j] > 0]
                bucket = pos
            else:
                nodes = [i for i, nt in enumerate(ntypes) if nt in scope]
                bucket = neg
            for i in nodes:
                v = fc.check(g, i, t)
                bucket[t][v.verdict] += 1
                reasons[t][f"{'pos' if bucket is pos else 'neg'}:{v.verdict}:{v.reason}"] += 1
                checks += 1
    out = {"dataset": args.dataset, "splits": args.splits, "files": len(samples),
           "checks": checks, "seconds": round(time.time() - t0, 1), "per_type": {}}
    tot = defaultdict(int)
    for t in SWC_TYPES:
        p, n = pos[t], neg[t]
        cp, cn = p["CONFIRMED"], n["CONFIRMED"]
        rp, rn = p["REFUTED"], n["REFUTED"]
        row = {
            "positives": dict(p), "negatives": dict(n),
            "confirm_rate_pos": cp / max(sum(p.values()), 1),
            "refute_rate_pos": rp / max(sum(p.values()), 1),
            "confirm_rate_neg": cn / max(sum(n.values()), 1),
            "confirmed_precision": cp / (cp + cn) if cp + cn else None,
            "refuted_precision": rn / (rn + rp) if rn + rp else None,
            "exactness": SPECS[t].exactness, "atoms": SPECS[t].n_atoms,
            "reasons": dict(reasons[t]),
        }
        out["per_type"][t] = row
        for k, d in (("cp", cp), ("cn", cn), ("rp", rp), ("rn", rn),
                     ("np", sum(p.values())), ("nn", sum(n.values()))):
            tot[k] += d
    out["overall"] = {
        "confirm_rate_pos": tot["cp"] / max(tot["np"], 1),
        "refute_rate_pos": tot["rp"] / max(tot["np"], 1),
        "confirm_rate_neg": tot["cn"] / max(tot["nn"], 1),
        "confirmed_precision": tot["cp"] / max(tot["cp"] + tot["cn"], 1),
        "refuted_precision": tot["rn"] / max(tot["rn"] + tot["rp"], 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{args.dataset}: {len(samples)} files, {checks} checks in {out['seconds']}s")
    print(f"{'SWC':8s} {'kind':9s} {'pos':>5s} {'conf|pos':>8s} {'ref|pos':>8s} {'conf|neg':>8s} "
          f"{'precC':>6s} {'precR':>6s}")
    for t, r in out["per_type"].items():
        npos = sum(r["positives"].values())
        f = lambda v: "  -  " if v is None else f"{v:.3f}"
        print(f"{t:8s} {r['exactness']:9s} {npos:5d} {r['confirm_rate_pos']:8.3f} "
              f"{r['refute_rate_pos']:8.3f} {r['confirm_rate_neg']:8.3f} "
              f"{f(r['confirmed_precision']):>6s} {f(r['refuted_precision']):>6s}")
    o = out["overall"]
    print(f"overall  conf|pos {o['confirm_rate_pos']:.3f}  ref|pos {o['refute_rate_pos']:.3f}  "
          f"conf|neg {o['confirm_rate_neg']:.3f}  precC {o['confirmed_precision']:.3f}  "
          f"precR {o['refuted_precision']:.3f}")


if __name__ == "__main__":
    main()
