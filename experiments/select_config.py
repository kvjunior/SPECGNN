#!/usr/bin/env python3
"""Configuration selection, exactly as fixed in docs/analysis_plan.md Section 4.

Grid: hidden in {64, 128, 256} x focal alpha in {0.25, 0.75}; dropout 0.3, weight
decay 1e-3. Mode (a) on DAppSCAN, 5 outer folds, seed 42. Criterion: mean
validation macro-AP; ties (difference < 0.005) go to the smaller model. Test
folds are never read. Writes configs/selected.yaml and
results/selection/selection.json.

    python experiments/select_config.py [--set paths.raw=...]   # resumable
"""
import argparse
import itertools
import os
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from specgnn.utils import config_hash, list_runs, write_json  # noqa: E402
from specgnn.utils.config import identity_config, load_config  # noqa: E402

GRID = {"model.hidden": [64, 128, 256], "loss.alpha": [0.25, 0.75]}
FIXED = {"model.dropout": 0.3, "train.weight_decay": 0.001, "train.seed": 42,
         "data.dataset": "dappscan", "data.safe_cap": 2000, "data.evaluate_heldout": False}
FOLDS = range(5)
TIE = 0.005
CONFIG = os.path.join(ROOT, "configs", "experiments", "plain.yaml")


def overrides(point: dict) -> list[str]:
    return [f"{k}={v}" for k, v in {**FIXED, **point}.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides, e.g. paths")
    args = ap.parse_args()
    points = [dict(zip(GRID, values, strict=True)) for values in itertools.product(*GRID.values())]
    for point, fold in itertools.product(points, FOLDS):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "05_train.py"), "--config", CONFIG,
               "--fold", str(fold), "--allow-unselected", "--set", *overrides(point), *args.set]
        print("$", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    base = load_config(CONFIG, args.set)
    results_dir = os.path.join(ROOT, base["paths"]["results"]) \
        if not os.path.isabs(base["paths"]["results"]) else base["paths"]["results"]
    by_hash = {r["run_id"]: r for r in list_runs(results_dir, "dappscan.plain")}
    table = []
    for point in points:
        vals = []
        for fold in FOLDS:
            cfg = load_config(CONFIG, overrides(point) + args.set)
            cfg.setdefault("run", {})["fold"] = fold
            run_id = f"dappscan.plain-{config_hash(identity_config(cfg))}"
            vals.append(by_hash[run_id]["metrics"]["val"]["unfiltered"]["macro"]["ap"])
        table.append({**point, "val_macro_ap_per_fold": vals,
                      "val_macro_ap_mean": sum(vals) / len(vals)})
    best = max(r["val_macro_ap_mean"] for r in table)
    tied = [r for r in table if best - r["val_macro_ap_mean"] < TIE]
    chosen = min(tied, key=lambda r: (r["model.hidden"], -r["val_macro_ap_mean"]))
    write_json(os.path.join(results_dir, "selection", "selection.json"),
               {"grid": GRID, "fixed": FIXED, "criterion": "mean validation macro-AP, mode (a)",
                "tie_margin": TIE, "table": table, "chosen": chosen})
    selected = {"selection": {"status": "done", "criterion": "analysis plan Section 4",
                              "val_macro_ap_mean": round(chosen["val_macro_ap_mean"], 4)},
                "model": {"hidden": chosen["model.hidden"], "dropout": FIXED["model.dropout"]},
                "loss": {"alpha": chosen["loss.alpha"]},
                "train": {"weight_decay": FIXED["train.weight_decay"]}}
    with open(os.path.join(ROOT, "configs", "selected.yaml"), "w") as fh:
        fh.write("# Written by experiments/select_config.py (analysis plan, Section 4). "
                 "Do not edit by hand.\n")
        yaml.safe_dump(selected, fh, sort_keys=False)
    for r in table:
        mark = "  <- selected" if r is chosen else ""
        print(f"hidden {r['model.hidden']:3d}  alpha {r['loss.alpha']:.2f}  "
              f"val macro-AP {r['val_macro_ap_mean']:.4f}{mark}")


if __name__ == "__main__":
    main()
