#!/usr/bin/env python3
"""Build all SpecGNN datasets from the upstream releases and print Table III.

Usage:
    python scripts/01_prepare_data.py --raw /path/to/raw --out data/processed --reports reports

Expected layout under --raw (directory names as produced by unzipping the
upstream archives; see data/README.md for download links):
    dappscan/                     contracts/, SWCsource/, Audit_and_Repository_link.xlsx
    resource2/                    dataset_preprocessing_for_vulnerabilities (renamed)
    smartbugs-curated-main/
    SolidiFI-benchmark-master/
    sc-defi-security-main/dataset/
    Web3Bugs-main/
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data import config as C  # noqa: E402
from specgnn.data.build import build  # noqa: E402
from specgnn.data.loaders import web3bugs_scope  # noqa: E402
from specgnn.data.report import generate  # noqa: E402

LAYOUT = {
    "dappscan": "dappscan",
    "resource2": "resource2",
    "smartbugs_curated": "smartbugs-curated-main",
    "solidifi": "SolidiFI-benchmark-master",
    "chaliasos": os.path.join("sc-defi-security-main", "dataset"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--safe-cap", type=int, default=C.DEFAULT_SAFE_CAP,
                    help="DAppSCAN safe-sample cap used for Table III (-1 = no cap)")
    args = ap.parse_args()

    raw = {}
    for name, sub in LAYOUT.items():
        path = os.path.join(args.raw, sub)
        if os.path.isdir(path):
            raw[name] = path
        else:
            print(f"[warn] {name} not found at {path}; skipped")

    t0 = time.time()
    build(raw, args.out)
    w3 = None
    w3_root = os.path.join(args.raw, "Web3Bugs-main")
    if os.path.isdir(w3_root):
        w3 = web3bugs_scope(w3_root)
    cap = None if args.safe_cap < 0 else args.safe_cap
    print(generate(args.out, args.reports, w3, cap))
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
