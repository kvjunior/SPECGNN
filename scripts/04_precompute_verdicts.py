#!/usr/bin/env python3
"""Compute and cache the specification verdict of every (candidate node, SWC type)
pair of every graph (see specgnn/feedback/verdicts.py).

Usage:
    python scripts/04_precompute_verdicts.py --raw RAW --graphs data/graphs --out data/verdicts
Writes data/verdicts/<dataset>.pkl ({uid: int8 [nodes, 17]}) and verdict_stats.json, and
copies verdict_stats.json to --reports (read by analysis/make_assets.py for the checker cost).
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import RAW_SUBDIRS  # noqa: E402
from specgnn.data.normalize import read_source  # noqa: E402
from specgnn.feedback.verdicts import build_cache  # noqa: E402
from specgnn.spec.checker import CheckerConfig  # noqa: E402
from specgnn.training import load_samples  # noqa: E402
from specgnn.utils.config import load_config, section  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--out", default="data/verdicts")
    ap.add_argument("--checker-config", default="configs/spec/checker.yaml")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--datasets", nargs="*", default=list(RAW_SUBDIRS))
    args = ap.parse_args()
    cfg = section(load_config(args.checker_config), "spec", CheckerConfig)
    os.makedirs(args.out, exist_ok=True)
    stats_path = os.path.join(args.out, "verdict_stats.json")
    all_stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            all_stats = json.load(fh)
    for name in args.datasets:
        if not os.path.exists(os.path.join(args.graphs, f"{name}.pkl")):
            continue
        root = os.path.join(args.raw, RAW_SUBDIRS[name])
        samples = load_samples(args.graphs, name, safe_cap=None)     # every sample, any cap
        def source(s, root=root):
            return read_source(os.path.join(root, s["rel_path"]))

        cache, stats = build_cache(samples, source, cfg)
        with open(os.path.join(args.out, f"{name}.pkl"), "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        all_stats[name] = {**stats, "checker": {"max_depth": cfg.max_depth,
                                                "max_paths": cfg.max_paths}}
        with open(stats_path, "w") as fh:
            json.dump(all_stats, fh, indent=1)
        print(f"[verdicts] {name}: {json.dumps(stats)}")
    os.makedirs(args.reports, exist_ok=True)
    with open(os.path.join(args.reports, "verdict_stats.json"), "w") as fh:
        json.dump(all_stats, fh, indent=1)


if __name__ == "__main__":
    main()
