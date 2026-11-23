"""Static-analysis tool baselines: Slither and Mythril.

Each file is analysed in place (so relative imports resolve) with the newest
installed solc release that satisfies its pragma. Findings are mapped to the 17
SWC types (tables below); a type is scored 1 if any mapped finding is reported,
else 0. Every file's outcome is recorded: ok, no_compiler, compile_error,
timeout or crash. Failed files score 0 for every type and the failure rate is
reported with the results (docs/BASELINES.md).

Compilers are found in a solc-select artifacts directory
(``~/.solc-select/artifacts/solc-X.Y.Z/solc-X.Y.Z``) or any directory with the
same layout; install them with ``solc-select install all``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile

import numpy as np

from ..data.config import SWC_TYPES
from .base import Baseline, register

SLITHER_TO_SWC = {
    "reentrancy-eth": "SWC-107", "reentrancy-no-eth": "SWC-107", "reentrancy-benign": "SWC-107",
    "reentrancy-events": "SWC-107", "reentrancy-unlimited-gas": "SWC-107",
    "timestamp": "SWC-116", "tx-origin": "SWC-115",
    "unchecked-lowlevel": "SWC-104", "unchecked-send": "SWC-104",
    "controlled-delegatecall": "SWC-112", "delegatecall-loop": "SWC-112",
    "arbitrary-send-eth": "SWC-105", "weak-prng": "SWC-120",
    "solc-version": "SWC-102", "pragma": "SWC-103",
    "calls-loop": "SWC-113", "costly-loop": "SWC-128",
    "unused-state": "SWC-131", "redundant-statements": "SWC-135",
}
SLITHER_SUPPORTED = tuple(sorted(set(SLITHER_TO_SWC.values())))
MYTHRIL_SUPPORTED = ("SWC-101", "SWC-104", "SWC-105", "SWC-107", "SWC-112", "SWC-113",
                     "SWC-114", "SWC-115", "SWC-116", "SWC-120")


# ---------------------------------------------------------------- compiler choice
def _vt(v: str) -> tuple[int, int, int]:
    parts = [int(x) for x in re.findall(r"\d+", v)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def satisfies(version: str, spec: str) -> bool:
    """Does `version` satisfy a Solidity pragma constraint such as '^0.4.24',
    '>=0.6.0 <0.8.0', '0.8.19', '>0.5.0'? Disjunctions ('||') are supported."""
    v = _vt(version)
    for alt in spec.split("||"):
        ok = True
        for op, ver in re.findall(r"(\^|~|>=|<=|>|<|=)?\s*(\d+\.\d+(?:\.\d+)?)", alt):
            w = _vt(ver)
            if op == "^":
                ok &= v >= w and (v[0], v[1]) == (w[0], w[1]) if w[0] == 0 else v >= w and v[0] == w[0]
            elif op == "~":
                ok &= v >= w and (v[0], v[1]) == (w[0], w[1])
            elif op == ">=":
                ok &= v >= w
            elif op == "<=":
                ok &= v <= w
            elif op == ">":
                ok &= v > w
            elif op == "<":
                ok &= v < w
            else:
                ok &= v == w
        if ok and re.search(r"\d", alt):
            return True
    return False


def pragma_of(source: str) -> str | None:
    m = re.search(r"pragma\s+solidity\s+([^;]+);", source)
    return m.group(1).strip() if m else None


def installed_compilers(artifacts: str) -> dict[str, str]:
    out = {}
    if os.path.isdir(artifacts):
        for d in os.listdir(artifacts):
            m = re.fullmatch(r"solc-(\d+\.\d+\.\d+)", d)
            if m:
                binary = os.path.join(artifacts, d, d)
                if os.path.isfile(binary):
                    out[m.group(1)] = binary
    return out


def pick_compiler(source: str, compilers: dict[str, str]) -> tuple[str, str] | None:
    spec = pragma_of(source)
    versions = sorted(compilers, key=_vt, reverse=True)
    for v in versions:
        if spec is None or satisfies(v, spec):
            return v, compilers[v]
    return None


# ---------------------------------------------------------------- tool runners
class ToolBaseline(Baseline):
    kind = "tool"
    tool = "tool"

    def __init__(self, seed=42, source=None, timeout: int = 120,
                 artifacts: str = "~/.solc-select/artifacts", cache_dir: str = "results/tool_cache"):
        super().__init__(seed, source, timeout=timeout, artifacts=artifacts, cache_dir=cache_dir)
        self.path_of = None                       # set by the runner: sample -> file path
        self.compilers = installed_compilers(os.path.expanduser(artifacts))
        self.last_stats: dict = {}

    def fit(self, train, val) -> dict:
        return {}                                 # tools are not trained

    def run_one(self, path: str, version: str, binary: str) -> tuple[str, set[str]]:
        raise NotImplementedError

    def predict(self, samples: list[dict]) -> np.ndarray:
        if self.path_of is None or self.source is None:
            raise ValueError(f"{self.name}: path_of and source must be set by the runner")
        out = np.zeros((len(samples), len(SWC_TYPES)), dtype=np.float32)
        cache = os.path.join(self.params["cache_dir"], self.tool)
        os.makedirs(cache, exist_ok=True)
        stats: dict[str, int] = {}
        for i, s in enumerate(samples):
            src = self.source(s)
            choice = pick_compiler(src, self.compilers)
            if choice is None:
                status, found = "no_compiler", set()
            else:
                key = hashlib.sha256(f"{self.tool}|{choice[0]}|{src}".encode()).hexdigest()
                cpath = os.path.join(cache, key + ".json")
                if os.path.exists(cpath):
                    with open(cpath) as fh:
                        rec = json.load(fh)
                else:
                    status, found = self.run_one(self.path_of(s), *choice)
                    rec = {"status": status, "found": sorted(found), "solc": choice[0]}
                    with open(cpath, "w") as fh:
                        json.dump(rec, fh)
                status, found = rec["status"], set(rec["found"])
            stats[status] = stats.get(status, 0) + 1
            for t in found:
                if t in SWC_TYPES:
                    out[i, SWC_TYPES.index(t)] = 1.0
        self.last_stats = {"files": len(samples), **stats,
                           "failure_rate": 1 - stats.get("ok", 0) / max(len(samples), 1)}
        return self._check(out, len(samples))


@register
class SlitherBaseline(ToolBaseline):
    name = "slither"
    tool = "slither"
    supported_types = SLITHER_SUPPORTED

    def run_one(self, path, version, binary):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "out.json")
            cmd = ["slither", os.path.basename(path), "--solc", binary, "--json", out_json,
                   "--disable-color"]
            try:
                subprocess.run(cmd, cwd=os.path.dirname(path) or ".", capture_output=True,
                               timeout=self.params["timeout"], check=False)
            except subprocess.TimeoutExpired:
                return "timeout", set()
            except OSError:
                return "crash", set()
            if not os.path.exists(out_json):
                return "crash", set()
            with open(out_json) as fh:
                data = json.load(fh)
        if not data.get("success", False):
            return "compile_error", set()
        dets = (data.get("results") or {}).get("detectors") or []
        return "ok", {SLITHER_TO_SWC[d["check"]] for d in dets if d.get("check") in SLITHER_TO_SWC}


@register
class MythrilBaseline(ToolBaseline):
    name = "mythril"
    tool = "mythril"
    supported_types = MYTHRIL_SUPPORTED

    def run_one(self, path, version, binary):
        cmd = ["myth", "analyze", os.path.basename(path), "--solv", version, "-o", "json",
               "--execution-timeout", str(max(10, self.params["timeout"] - 20))]
        try:
            proc = subprocess.run(cmd, cwd=os.path.dirname(path) or ".", capture_output=True,
                                  text=True, timeout=self.params["timeout"], check=False)
        except subprocess.TimeoutExpired:
            return "timeout", set()
        except OSError:
            return "crash", set()
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return "compile_error", set()
        if not data.get("success", True) and not data.get("issues"):
            return "compile_error", set()
        found = set()
        for issue in data.get("issues", []):
            m = re.search(r"(\d+)", str(issue.get("swc-id", "")))
            if m:
                found.add(f"SWC-{int(m.group(1)):03d}")
        return "ok", found & set(MYTHRIL_SUPPORTED)
