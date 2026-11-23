#!/usr/bin/env python3
"""Build code graphs and verdict tables for semantics-preserving transformations of
the DAppSCAN files (RQ6 robustness; specgnn/robustness/transforms.py).

    python scripts/07_build_transformed.py --raw RAW
Writes data/robust/<transform>.pkl ({uid: {"graph", "verdicts"}}) and robust_stats.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import RAW_SUBDIRS  # noqa: E402
from specgnn.data.normalize import read_source  # noqa: E402
from specgnn.feedback.verdicts import verdict_table  # noqa: E402
from specgnn.graph.builder import build_graph  # noqa: E402
from specgnn.robustness.transforms import TRANSFORMS  # noqa: E402
from specgnn.spec.checker import CheckerConfig  # noqa: E402
from specgnn.training import load_samples  # noqa: E402
from specgnn.utils.config import load_config, section  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--graphs", default="data/graphs")
    ap.add_argument("--out", default="data/robust")
    ap.add_argument("--dataset", default="dappscan")
    ap.add_argument("--safe-cap", type=int, default=2000)
    ap.add_argument("--transforms", nargs="*", default=list(TRANSFORMS))
    ap.add_argument("--checker-config", default="configs/spec/checker.yaml")
    args = ap.parse_args()
    cfg = section(load_config(args.checker_config), "spec", CheckerConfig)
    os.makedirs(args.out, exist_ok=True)
    root = os.path.join(args.raw, RAW_SUBDIRS[args.dataset])
    samples = load_samples(args.graphs, args.dataset, safe_cap=args.safe_cap)
    stats = {}
    for name in args.transforms:
        fn, t0, out, errors = TRANSFORMS[name], time.time(), {}, 0
        for s in samples:
            src = fn(read_source(os.path.join(root, s["rel_path"])))
            g = build_graph(src)
            errors += g["parse_error"] and not s["graph"]["parse_error"]
            out[s["uid"]] = {"graph": g, "verdicts": verdict_table(g, src, cfg)}
        with open(os.path.join(args.out, f"{name}.pkl"), "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        stats[name] = {"files": len(out), "new_parse_errors": int(errors),
                       "seconds": round(time.time() - t0, 1)}
        print(f"[robust] {name}: {stats[name]}")
    with open(os.path.join(args.out, "robust_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
