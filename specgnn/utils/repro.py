"""Reproducibility: seeding, deterministic execution, and environment capture.

Every training run calls :func:`set_seed` before building data loaders or
models, and stores :func:`environment` next to its results so that any number in
the paper can be traced to the exact software and hardware that produced it.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from importlib import metadata

import numpy as np

# Packages whose versions affect results; recorded with every run.
TRACKED_PACKAGES = ("numpy", "torch", "scikit-learn", "tree-sitter", "tree-sitter-solidity",
                    "pyyaml", "openpyxl")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, and request deterministic kernels.

    Must be called before the first CUDA operation: the cuBLAS workspace
    setting is read when the CUDA context is created. Operations without a
    deterministic implementation emit a warning instead of failing
    (``warn_only=True``); their use is visible in the run log.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)       # affects subprocesses only
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:                             # data-only stages do not need torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_state(root: str | None = None) -> dict:
    """Commit hash and whether the working tree has uncommitted changes."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                 timeout=10, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment(root: str | None = None) -> dict:
    """Software and hardware facts recorded with every run."""
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": versions,
        "git": git_state(root),
        "cuda": None,
        "gpus": [],
    }
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return env
