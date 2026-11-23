"""Run registry.

Every experiment run owns one directory ``<results>/runs/<run_id>/``:

    config.yaml    fully resolved configuration (including seed and fold)
    env.json       software/hardware environment and git state (repro.environment)
    status.json    {"state": "running" | "complete" | "failed", timestamps, error}
    metrics.json   final metrics, written once at the end
    timing.json    named wall-clock sections (training, feedback, inference, ...)
    *              any further artifacts the run saves (predictions, model weights)

The run id is ``<kind>-<hash of the resolved config>``, so re-running the same
configuration maps to the same directory: complete runs are skipped, failed or
interrupted runs are restarted from scratch. All JSON files are written
atomically (write to a temporary file, then rename), so an interrupted run never
leaves a truncated file that later looks valid.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import traceback
from typing import Any

import yaml

from .config import config_hash, dump_yaml
from .repro import environment


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _sanitize(obj: Any) -> Any:
    """NaN/inf are not valid JSON; store them as null."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def write_json(path: str, obj: Any) -> None:
    """Atomic JSON write."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_sanitize(obj), fh, indent=1, default=_json_default, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Run:
    """One experiment run. Use :meth:`open`, then :meth:`finish` or :meth:`fail`."""

    def __init__(self, directory: str, run_id: str, config: dict):
        self.dir = directory
        self.id = run_id
        self.config = config

    @staticmethod
    def make_id(kind: str, config: dict) -> str:
        if not kind or any(c in kind for c in "/\\ "):
            raise ValueError(f"invalid run kind {kind!r}")
        return f"{kind}-{config_hash(config)}"

    @classmethod
    def open(cls, results_dir: str, kind: str, config: dict, overwrite: bool = False) -> Run | None:
        """Create (or restart) the run directory. Returns None if the run is complete
        and ``overwrite`` is False."""
        run_id = cls.make_id(kind, config)
        directory = os.path.join(results_dir, "runs", run_id)
        if is_complete(directory) and not overwrite:
            return None
        if os.path.isdir(directory):
            shutil.rmtree(directory)          # failed / interrupted / forced rerun
        os.makedirs(directory)
        dump_yaml(config, os.path.join(directory, "config.yaml"))
        write_json(os.path.join(directory, "env.json"), environment())
        write_json(os.path.join(directory, "status.json"),
                   {"state": "running", "started": time.time()})
        return cls(directory, run_id, config)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def save(self, name: str, obj: Any) -> str:
        p = self.path(name)
        write_json(p, obj)
        return p

    def finish(self, metrics: dict, timing: dict | None = None) -> None:
        if timing is not None:
            self.save("timing.json", timing)
        self.save("metrics.json", metrics)
        status = read_json(self.path("status.json"))
        status.update(state="complete", finished=time.time())
        self.save("status.json", status)

    def fail(self, exc: BaseException) -> None:
        status = read_json(self.path("status.json"))
        status.update(state="failed", finished=time.time(),
                      error="".join(traceback.format_exception(exc)))
        self.save("status.json", status)


def is_complete(directory: str) -> bool:
    try:
        return (read_json(os.path.join(directory, "status.json")).get("state") == "complete"
                and os.path.exists(os.path.join(directory, "metrics.json")))
    except (OSError, ValueError):
        return False


def list_runs(results_dir: str, kind: str | None = None, complete_only: bool = True) -> list[dict]:
    """Load config + metrics of all runs (optionally of one kind), for aggregation."""
    root = os.path.join(results_dir, "runs")
    out = []
    if not os.path.isdir(root):
        return out
    for run_id in sorted(os.listdir(root)):
        if kind is not None and not run_id.startswith(kind + "-"):
            continue
        d = os.path.join(root, run_id)
        if complete_only and not is_complete(d):
            continue
        with open(os.path.join(d, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        metrics = read_json(os.path.join(d, "metrics.json")) if is_complete(d) else None
        out.append({"run_id": run_id, "dir": d, "config": cfg, "metrics": metrics})
    return out
