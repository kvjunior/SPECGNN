#!/usr/bin/env python3
"""Export (or check) the released split assignment: data/splits/<dataset>.csv with
uid, split, fold, split_group, safe_rank. No source code or labels are included.

    python scripts/export_splits.py            # write data/splits/*.csv
    python scripts/export_splits.py --check    # verify a rebuild reproduces them exactly
"""
import argparse
import csv
import io
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ("dappscan", "resource2", "smartbugs_curated", "solidifi", "chaliasos")
FIELDS = ("uid", "split", "fold", "split_group", "safe_rank")


def render(processed: str, name: str) -> str:
    with open(os.path.join(processed, f"{name}.jsonl"), encoding="utf-8") as fh:
        recs = sorted((json.loads(line) for line in fh), key=lambda r: r["uid"])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FIELDS)
    for r in recs:
        w.writerow(["" if r[k] is None else r[k] for k in FIELDS])
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "splits"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bad = []
    for name in DATASETS:
        text = render(args.processed, name)
        path = os.path.join(args.out, f"{name}.csv")
        if args.check:
            with open(path, encoding="utf-8") as fh:
                same = fh.read() == text
            print(f"[splits] {name}: {'identical' if same else 'DIFFERENT'}")
            bad += [] if same else [name]
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"[splits] {name}: {text.count(chr(10)) - 1} rows -> {path}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
