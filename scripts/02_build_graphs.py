#!/usr/bin/env python3
"""Parse every processed sample into a code graph with training targets.

Usage:
    python scripts/02_build_graphs.py --raw /path/to/raw --processed data/processed --out data/graphs
Writes data/graphs/<dataset>.pkl (list of dicts) and data/graphs/graph_stats.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import RAW_SUBDIRS  # noqa: E402
from specgnn.data.normalize import read_source  # noqa: E402
from specgnn.graph.builder import build_graph  # noqa: E402
from specgnn.graph.features import EDGE_TYPES, NODE_TYPES  # noqa: E402
from specgnn.graph.labels import graph_targets, node_targets  # noqa: E402

ROOTS = RAW_SUBDIRS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/graphs")
    ap.add_argument("--datasets", nargs="*", default=list(ROOTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for name in args.datasets:
        path = os.path.join(args.processed, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        root = os.path.join(args.raw, ROOTS[name])
        recs = [json.loads(line) for line in open(path, encoding="utf-8")]
        out, t0, errors = [], time.time(), 0
        for r in recs:
            g = build_graph(read_source(os.path.join(root, r["rel_path"])))
            errors += g["parse_error"]
            y, ym = graph_targets(r["labels"])
            ny, nm = node_targets(g, r["labels"], r["meta"].get("swc_annotations"))
            out.append({
                "uid": r["uid"], "dataset": name, "split": r["split"], "fold": r["fold"],
                "safe_rank": r["safe_rank"], "rel_path": r["rel_path"],
                "graph": g, "y": y, "y_mask": ym, "node_y": ny, "node_mask": nm,
            })
        dt = time.time() - t0
        n_nodes = np.array([len(o["graph"]["node_type"]) for o in out])
        types = np.concatenate([o["graph"]["node_type"] for o in out])
        edges = {e: int(sum(len(o["graph"]["edges"][e]) for o in out)) for e in EDGE_TYPES}
        localized = int(sum((o["node_y"].sum(0) > 0).sum() for o in out))
        positive_pairs = int(sum((o["y"] * o["y_mask"]).sum() for o in out))
        stats[name] = {
            "graphs": len(out), "parse_errors": int(errors),
            "seconds": round(dt, 1), "ms_per_file": round(1000 * dt / max(len(out), 1), 2),
            "nodes_mean": float(n_nodes.mean()), "nodes_median": float(np.median(n_nodes)),
            "nodes_max": int(n_nodes.max()),
            "node_types": {t: int((types == i).sum()) for i, t in enumerate(NODE_TYPES)},
            "edges": edges,
            "positive_file_type_pairs": positive_pairs,
            "localized_file_type_pairs": localized,
        }
        with open(os.path.join(args.out, f"{name}.pkl"), "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[graphs] {name}: {json.dumps(stats[name])}")
    with open(os.path.join(args.out, "graph_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
