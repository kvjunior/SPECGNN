"""Content manifests of the raw datasets.

A manifest lists ``<sha256>  <relative path>`` for every input file a dataset
uses, sorted by path; its digest is the SHA-256 of that listing. Verifying the
digest proves that the inputs are byte-identical to the ones used for the paper,
independently of how they were obtained (git clone, archive download, mirror).
Paths are compared after undoing the ``#Uxxxx`` escaping some unzip tools apply to
non-ASCII names, so the same content verifies on every platform.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re

# Files of each dataset that the pipeline reads, below the --raw directory.
SCOPES = {
    "dappscan": ("dappscan", ("contracts/", "SWCsource/", "Audit_and_Repository_link.xlsx")),
    "resource2": ("resource2", ("reentrancy/", "timestamp/", "Integeroverflow/", "delegatecall/")),
    "smartbugs_curated": ("smartbugs-curated-main", ("dataset/", "vulnerabilities.json")),
    "solidifi": ("SolidiFI-benchmark-master", ("buggy_contracts/",)),
    "chaliasos": ("sc-defi-security-main/dataset", ("data/contracts/sources/", "database/retro.db")),
    "web3bugs": ("Web3Bugs-main", ("results/bugs.csv",)),
}


def _norm(rel: str) -> str:
    rel = rel.replace(os.sep, "/")
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), rel)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(raw: str, dataset: str) -> list[tuple[str, str]]:
    """[(relative path, sha256)] of every file in the dataset's scope, sorted."""
    subdir, prefixes = SCOPES[dataset]
    root = os.path.join(raw, subdir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{dataset}: {root} not found")
    entries = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            rel = _norm(os.path.relpath(full, root))
            if any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in prefixes):
                entries.append((rel, file_sha256(full)))
    if not entries:
        raise FileNotFoundError(f"{dataset}: no files matched {prefixes} under {root}")
    return sorted(entries)


def manifest_digest(entries: list[tuple[str, str]]) -> str:
    listing = "".join(f"{sha}  {rel}\n" for rel, sha in entries)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def write_manifest(entries: list[tuple[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rel, sha in entries:
            fh.write(f"{sha}  {rel}\n")


def read_manifest(path: str) -> list[tuple[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [(line[66:].rstrip("\n"), line[:64]) for line in fh if line.strip()]


def diff(expected: list[tuple[str, str]], actual: list[tuple[str, str]]) -> dict:
    e, a = dict(expected), dict(actual)
    return {"missing": sorted(set(e) - set(a)), "unexpected": sorted(set(a) - set(e)),
            "changed": sorted(k for k in set(e) & set(a) if e[k] != a[k])}
