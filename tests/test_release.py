import gzip
import json
import os
import re
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.manifest import SCOPES, build_manifest, diff, manifest_digest, read_manifest, write_manifest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _tree(tmp, files):
    for rel, content in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def test_manifest_scope_digest_and_name_normalization(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _tree(a, {"web3/results/bugs.csv": b"x", "web3/other.txt": b"ignored"})
    # the same content with an escaped non-ASCII name must give the same manifest
    _tree(a, {"dap/contracts/P\u2013Q/C.sol": b"c", "dap/SWCsource/x.json": b"j",
              "dap/Audit_and_Repository_link.xlsx": b"s"})
    _tree(b, {"dap/contracts/P#U2013Q/C.sol": b"c", "dap/SWCsource/x.json": b"j",
              "dap/Audit_and_Repository_link.xlsx": b"s"})
    import specgnn.data.manifest as m
    old = dict(m.SCOPES)
    m.SCOPES.update({"t_web3": ("web3", ("results/bugs.csv",)),
                     "t_dap": ("dap", ("contracts/", "SWCsource/", "Audit_and_Repository_link.xlsx"))})
    try:
        e = build_manifest(str(a), "t_web3")
        assert [r for r, _ in e] == ["results/bugs.csv"]
        ma, mb = build_manifest(str(a), "t_dap"), build_manifest(str(b), "t_dap")
        assert ma == mb and manifest_digest(ma) == manifest_digest(mb)
    finally:
        m.SCOPES.clear()
        m.SCOPES.update(old)


def test_manifest_roundtrip_and_diff(tmp_path):
    e = [("a.sol", "0" * 64), ("b.sol", "1" * 64)]
    p = str(tmp_path / "m.sha256.gz")
    write_manifest(e, p)
    assert read_manifest(p) == e
    d = diff(e, [("a.sol", "2" * 64), ("c.sol", "3" * 64)])
    assert d == {"missing": ["b.sol"], "unexpected": ["c.sol"], "changed": ["a.sol"]}
    assert manifest_digest(e) != manifest_digest(e[:1])


def test_fetch_extracts_only_the_mapped_files(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    fetch = importlib.import_module("00_fetch_data")
    spec = {"repo": "x/y", "commit": "a" * 40,
            "extract": {"dataset/": "out/dataset/", "vuln.json": "out/vuln.json"}}
    cache = tmp_path / "cache"
    cache.mkdir()
    with zipfile.ZipFile(cache / f"t-{'a' * 12}.zip", "w") as z:
        z.writestr("y-aaaa/dataset/r/A.sol", "A")
        z.writestr("y-aaaa/vuln.json", "{}")
        z.writestr("y-aaaa/README.md", "not extracted")
    raw = tmp_path / "raw"
    fetch.fetch_github("t", spec, str(raw), str(cache))
    got = sorted(str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file())
    assert got == ["out/dataset/r/A.sol", "out/vuln.json"]


def test_lock_file_is_complete_and_well_formed():
    with open(os.path.join(ROOT, "data", "sources.lock.json")) as fh:
        lock = json.load(fh)
    assert set(lock) == set(SCOPES)
    for name, e in lock.items():
        assert re.fullmatch(r"[0-9a-f]{64}", e["digest"]), name
        assert e["files"] > 0
        assert ("commit" in e and re.fullmatch(r"[0-9a-f]{40}", e["commit"])) or "manual" in e, name
        with gzip.open(os.path.join(ROOT, "data", "manifests", f"{name}.sha256.gz"), "rt") as fh:
            assert sum(1 for _ in fh) == e["files"]


def test_released_splits_match_processed_data():
    path = os.path.join(ROOT, "data", "processed", "dappscan.jsonl")
    if not os.path.exists(path):
        pytest.skip("needs data/processed")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib
    exp = importlib.import_module("export_splits")
    for name in exp.DATASETS:
        with open(os.path.join(ROOT, "data", "splits", f"{name}.csv")) as fh:
            assert fh.read() == exp.render(os.path.join(ROOT, "data", "processed"), name), name


def test_release_files_and_pins():
    for f in ("LICENSE", "DATA_LICENSES.md", "CITATION.cff", ".zenodo.json", "REPRODUCE.md",
              "Dockerfile", "Dockerfile.gpu", ".dockerignore", "requirements.lock",
              ".github/workflows/tests.yml", ".github/workflows/data.yml"):
        assert os.path.exists(os.path.join(ROOT, f)), f
    pins = {re.split(r"==", line)[0].lower() for line in open(os.path.join(ROOT, "requirements.lock"))
            if "==" in line}
    import tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    for d in deps:
        assert re.split(r"[<>=]", d)[0].lower() in pins, d
