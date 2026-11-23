#!/usr/bin/env python3
"""Hyperparameter selection for the neural baselines, by the rule of analysis-plan
Section 4: grid hidden in {64, 128, 256}; 5 outer folds, seed 42; mean validation
macro-AP; ties (< 0.005) go to the smaller model. Writes
configs/baselines/selected/<name>.yaml. Resumable.

    python experiments/select_baselines.py [--names drgcn ame] [--set paths.raw=...]
"""
import argparse
import os
import subprocess
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.matrix import FOLDS, NEURAL_BASELINES, Job  # noqa: E402
from specgnn.utils import is_complete, load_config, read_json, write_json  # noqa: E402

GRID = [64, 128, 256]
TIE = 0.005
FIXED = ["data.dataset=dappscan", "data.safe_cap=2000", "data.evaluate_heldout=false"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", default=list(NEURAL_BASELINES))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    results = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]["results"]
    results = results if os.path.isabs(results) else os.path.join(ROOT, results)
    for name in args.names:
        table = []
        for hidden in GRID:
            vals = []
            for fold in FOLDS:
                job = Job(f"select/{name}", f"baselines/{name}.yaml",
                          (*FIXED, f"baseline.params.hidden={hidden}"), fold, 42, 0, (),
                          "06_run_baseline.py")
                d = os.path.join(results, "runs", job.run_id(args.set))
                if not is_complete(d):
                    cmd = [sys.executable, os.path.join(ROOT, "scripts", job.script),
                           *job.command_args(), *args.set, "--allow-unselected"]
                    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
                vals.append(read_json(os.path.join(d, "metrics.json"))["val"]["unfiltered"]["macro"]["ap"])
            table.append({"hidden": hidden, "val_macro_ap_per_fold": vals,
                          "val_macro_ap_mean": sum(vals) / len(vals)})
        best = max(r["val_macro_ap_mean"] for r in table)
        chosen = min((r for r in table if best - r["val_macro_ap_mean"] < TIE),
                     key=lambda r: r["hidden"])
        write_json(os.path.join(results, "selection", f"baseline_{name}.json"),
                   {"grid": GRID, "table": table, "chosen": chosen})
        with open(os.path.join(ROOT, "configs", "baselines", "selected", f"{name}.yaml"), "w") as fh:
            fh.write("# Written by experiments/select_baselines.py (analysis plan, Section 4). "
                     "Do not edit by hand.\n")
            yaml.safe_dump({"selection": {"status": "done",
                                          "val_macro_ap_mean": round(chosen["val_macro_ap_mean"], 4)},
                            "baseline": {"params": {"hidden": chosen["hidden"]}}}, fh, sort_keys=False)
        print(f"{name}: hidden {chosen['hidden']} (val macro-AP {chosen['val_macro_ap_mean']:.4f})")


if __name__ == "__main__":
    main()
