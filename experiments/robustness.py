#!/usr/bin/env python3
"""RQ6 robustness (analysis plan, Section 6): change in test macro-AP when the test
files are replaced by semantics-preserving transformations, for every mode, using
the saved DAppSCAN models of the primary runs (plain -> a, b; feedback -> c, d).

Pre-registered test (family RQ6-robust, one test): for the composite transformation
"all", robustness = AP(transformed) - AP(original); mode (c) vs mode (a), paired over
(fold, seed). Other transformations are descriptive.

    python experiments/robustness.py
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.matrix import FOLDS, SEEDS, all_jobs  # noqa: E402
from specgnn.eval.metrics import detection_metrics  # noqa: E402
from specgnn.eval.stats import compare_family, describe, supports_claim  # noqa: E402
from specgnn.training import cv_split, filtered_probs, load_model, load_samples, predict  # noqa: E402
from specgnn.utils import is_complete, load_config, write_json  # noqa: E402

PRIMARY = "all"
MODES = {"a": ("plain", False), "b": ("plain", True), "c": ("both", True), "d": ("both", False)}


def transformed(samples: list[dict], table: dict) -> tuple[list[dict], dict]:
    out, verdicts = [], {}
    for s in samples:
        t = table[s["uid"]]
        n = len(t["graph"]["node_type"])
        z = np.zeros((n, s["node_y"].shape[1]), dtype=np.float32)
        out.append({**s, "graph": t["graph"], "node_y": z, "node_mask": z})
        verdicts[s["uid"]] = t["verdicts"]
    return out, verdicts


def macro_ap(model, lcfg, samples, verdicts, filtered, tau):
    pred = predict(model, samples, lcfg)
    probs = filtered_probs(pred, verdicts, tau)[0] if filtered else pred["graph_probs"]
    return detection_metrics(probs, pred["y"], pred["mask"], tau)["macro"]["ap"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--robust", default="data/robust")
    args = ap.parse_args()
    paths = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]
    absp = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in paths.items()}
    robust_dir = args.robust if os.path.isabs(args.robust) else os.path.join(ROOT, args.robust)
    tables = {}
    for f in sorted(os.listdir(robust_dir)):
        if f.endswith(".pkl"):
            with open(os.path.join(robust_dir, f), "rb") as fh:
                tables[f[:-4]] = pickle.load(fh)
    with open(os.path.join(absp["verdicts"], "dappscan.pkl"), "rb") as fh:
        base_verdicts = pickle.load(fh)
    jobs = {(j.name.split("/")[1], j.fold, j.seed): j for j in all_jobs()
            if j.name in ("dappscan/plain", "dappscan/both")}
    samples = load_samples(absp["graphs"], "dappscan", safe_cap=2000)
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            dirs = {v: os.path.join(absp["results"], "runs", jobs[(v, fold, seed)].run_id())
                    for v in ("plain", "both")}
            if not all(is_complete(d) and os.path.exists(os.path.join(d, "model.pt"))
                       for d in dirs.values()):
                continue
            models = {v: load_model(os.path.join(d, "model.pt")) for v, d in dirs.items()}
            _, _, test = cv_split(samples, fold)
            for mode, (variant, filt) in MODES.items():
                model, lcfg = models[variant]
                tau = 0.5
                orig = macro_ap(model, lcfg, test, base_verdicts, filt, tau)
                for name, table in tables.items():
                    ts, tv = transformed(test, table)
                    new = macro_ap(model, lcfg, ts, tv, filt, tau)
                    rows.append({"fold": fold, "seed": seed, "mode": mode, "transform": name,
                                 "ap_original": orig, "ap_transformed": new, "delta": new - orig})
    out = {"rows": rows, "summary": {}, "test": None}
    for name in tables:
        out["summary"][name] = {m: describe([r["delta"] for r in rows
                                             if r["transform"] == name and r["mode"] == m])
                                for m in MODES if any(r["transform"] == name and r["mode"] == m
                                                      for r in rows)}
    key = lambda m: {(r["fold"], r["seed"]): r["delta"] for r in rows           # noqa: E731
                     if r["transform"] == PRIMARY and r["mode"] == m}
    dc, da = key("c"), key("a")
    pairs = sorted(set(dc) & set(da))
    if pairs:
        (c,) = compare_family({"c_vs_a": (np.array([dc[p] for p in pairs]),
                                          np.array([da[p] for p in pairs]))})
        out["test"] = {**c.as_dict(), "supports_claim": supports_claim(c), "transform": PRIMARY,
                       "complete": len(pairs) == len(SEEDS) * len(FOLDS)}
    write_json(os.path.join(absp["results"], "analysis", "robustness.json"), out)
    print(f"{len(rows)} measurements; primary test: {out['test']}")


if __name__ == "__main__":
    main()
