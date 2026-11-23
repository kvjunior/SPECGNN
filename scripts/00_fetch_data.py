#!/usr/bin/env python3
"""Download the raw datasets at pinned upstream commits and verify their content.

    python scripts/00_fetch_data.py --raw data/raw            # download + verify
    python scripts/00_fetch_data.py --raw data/raw --verify   # verify only

GitHub sources are fetched as archives of the exact commit recorded in
data/sources.lock.json, and only the files the pipeline uses are extracted.
The Qian et al. dataset is distributed on Google Drive and must be downloaded by
hand (instructions are printed). Every dataset is then verified against the
content digest in the lock file; on a mismatch the per-file manifest in
data/manifests/ pinpoints the differing files.

Maintainers only: ``--write-lock`` recomputes digests and manifests from --raw.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from specgnn.data.manifest import SCOPES, build_manifest, diff, manifest_digest, read_manifest, write_manifest  # noqa: E402

LOCK = os.path.join(ROOT, "data", "sources.lock.json")
MANIFESTS = os.path.join(ROOT, "data", "manifests")

# Where each dataset comes from. "extract" maps archive paths (relative to the
# archive's top folder) to paths below --raw.
SOURCES = {
    "dappscan": {"repo": "InPlusLab/DAppSCAN", "commit": "66a56619c44770e05c2db600fa6468115ff0dcd5",
                 "extract": {"DAppSCAN-source/contracts/": "dappscan/contracts/",
                             "DAppSCAN-source/SWCsource/": "dappscan/SWCsource/",
                             "Audit_and_Repository_link.xlsx": "dappscan/Audit_and_Repository_link.xlsx"}},
    "smartbugs_curated": {"repo": "smartbugs/smartbugs-curated",
                          "commit": "230e649123477eff332742a59a1c7cc6dc286cab",
                          "extract": {"dataset/": "smartbugs-curated-main/dataset/",
                                      "vulnerabilities.json": "smartbugs-curated-main/vulnerabilities.json"}},
    "solidifi": {"repo": "DependableSystemsLab/SolidiFI-benchmark",
                 "commit": "4b0573e1b3f7031396de6f48f7f3e7380222ad3a",
                 "extract": {"buggy_contracts/": "SolidiFI-benchmark-master/buggy_contracts/"}},
    "chaliasos": {"repo": "StefanosChaliasos/sc-defi-security",
                  "commit": "70efe79e8e71eb904d86d1a3f8818dfad2c42562",
                  "extract": {"dataset/data/contracts/sources/":
                              "sc-defi-security-main/dataset/data/contracts/sources/",
                              "dataset/database/retro.db":
                              "sc-defi-security-main/dataset/database/retro.db"}},
    "web3bugs": {"repo": "ZhangZhuoSJTU/Web3Bugs", "commit": "fd8544e84f0d6cea4b4d6a44ee62d8f7623648f4",
                 "extract": {"results/bugs.csv": "Web3Bugs-main/results/bugs.csv"}},
    "resource2": {"manual": (
        "Download 'Dataset_preprocessing' (Resource 2) from the Google Drive link in "
        "https://github.com/Messi-Q/Smart-Contract-Dataset (README, commit "
        "7c152ffae158d2231a668ee6c35dadb1176ae427), unzip it, and rename the folder "
        "'dataset_preprocessing_for_vulnerabilities' to '<raw>/resource2'.")},
}


def fetch_github(name: str, spec: dict, raw: str, cache: str) -> None:
    url = f"https://codeload.github.com/{spec['repo']}/zip/{spec['commit']}"
    archive = os.path.join(cache, f"{name}-{spec['commit'][:12]}.zip")
    if not os.path.exists(archive):
        print(f"[fetch] {name}: {url}")
        tmp = archive + ".part"
        with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        os.replace(tmp, archive)
    with zipfile.ZipFile(archive) as z:
        top = z.namelist()[0].split("/", 1)[0] + "/"
        n = 0
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(top):
                continue
            inner = info.filename[len(top):]
            for src, dst in spec["extract"].items():
                if inner == src or (src.endswith("/") and inner.startswith(src)):
                    target = os.path.join(raw, dst + inner[len(src):] if src.endswith("/") else dst)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(info) as fin, open(target, "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                    n += 1
                    break
    print(f"[fetch] {name}: {n} files extracted")


def verify(name: str, raw: str, lock: dict) -> bool:
    if name not in lock:
        print(f"[verify] {name}: not in lock file")
        return False
    try:
        entries = build_manifest(raw, name)
    except FileNotFoundError as exc:
        print(f"[verify] {name}: MISSING ({exc})")
        return False
    ok = manifest_digest(entries) == lock[name]["digest"]
    print(f"[verify] {name}: {'OK' if ok else 'MISMATCH'} ({len(entries)} files)")
    if not ok:
        mpath = os.path.join(MANIFESTS, f"{name}.sha256.gz")
        if os.path.exists(mpath):
            d = diff(read_manifest(mpath), entries)
            for k, v in d.items():
                if v:
                    print(f"    {k}: {len(v)} e.g. {v[:3]}")
    return ok


def write_lock(raw: str) -> None:
    lock = {}
    for name in SCOPES:
        entries = build_manifest(raw, name)
        write_manifest(entries, os.path.join(MANIFESTS, f"{name}.sha256.gz"))
        src = {k: v for k, v in SOURCES[name].items() if k != "extract"}
        lock[name] = {**src, "files": len(entries), "digest": manifest_digest(entries)}
        print(f"[lock] {name}: {len(entries)} files {lock[name]['digest'][:16]}")
    with open(LOCK, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1, sort_keys=True)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--verify", action="store_true", help="only verify existing files")
    ap.add_argument("--write-lock", action="store_true", help="maintainers: recompute the lock")
    ap.add_argument("--datasets", nargs="*", default=list(SCOPES))
    args = ap.parse_args()
    if args.write_lock:
        return write_lock(args.raw)
    with open(LOCK, encoding="utf-8") as fh:
        lock = json.load(fh)
    if not args.verify:
        with tempfile.TemporaryDirectory() as cache:
            for name in args.datasets:
                spec = SOURCES[name]
                if "manual" in spec:
                    if not os.path.isdir(os.path.join(args.raw, SCOPES[name][0])):
                        print(f"[manual] {name}: {spec['manual'].replace('<raw>', args.raw)}")
                else:
                    fetch_github(name, spec, args.raw, cache)
    results = {name: verify(name, args.raw, lock) for name in args.datasets}
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
