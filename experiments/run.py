#!/usr/bin/env python3
"""Execute the run matrix (experiments/matrix.py) in priority order. Resumable:
complete runs are skipped by scripts/05_train.py.

    python experiments/run.py --dry-run                     # list what remains
    python experiments/run.py --max-priority 2              # primary analysis first
    python experiments/run.py --tags rq4 --set paths.raw=/data/raw
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.matrix import jobs_for  # noqa: E402
from specgnn.utils import is_complete, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--max-priority", type=int)
    ap.add_argument("--set", nargs="*", default=[], help="machine overrides, e.g. paths.raw=...")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = jobs_for(args.names, args.tags, args.max_priority)
    results = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]["results"]
    results = results if os.path.isabs(results) else os.path.join(ROOT, results)
    todo = [j for j in jobs if not is_complete(os.path.join(results, "runs", j.run_id(args.set)))]
    print(f"{len(jobs)} jobs selected, {len(jobs) - len(todo)} complete, {len(todo)} to run")
    if args.dry_run:
        by = {}
        for j in todo:
            by[j.name] = by.get(j.name, 0) + 1
        for name, n in by.items():
            print(f"  {n:3d}  {name}")
        return
    t0 = time.time()
    for i, j in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {j.name} fold {j.fold} seed {j.seed}  "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", j.script), *j.command_args(),
               *args.set]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
