"""Common interface for every RQ1 baseline.

All baselines are evaluated under one protocol (docs/BASELINES.md):
  * the same samples, cross-validation folds, seeds and label space (17 SWC types)
    as SpecGNN;
  * ``fit(train, val)`` may use the validation fold only for early stopping;
  * ``predict(samples)`` returns file-level scores [B, 17] in [0, 1]; types a
    method cannot report receive score 0 and are listed in ``supported_types``,
    so results are reported on the full label space and on the supported types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..data.config import SWC_TYPES

SourceReader = Callable[[dict], str]          # sample -> Solidity source text


class Baseline(ABC):
    name: str = "baseline"
    kind: str = "learned"                     # "learned" | "tool" | "llm"
    supported_types: tuple[str, ...] = tuple(SWC_TYPES)

    def __init__(self, seed: int = 42, source: SourceReader | None = None, **params):
        self.seed = seed
        self.source = source
        self.params = params

    @abstractmethod
    def fit(self, train: list[dict], val: list[dict]) -> dict:
        """Train; return a history/diagnostics dict (empty for tools)."""

    @abstractmethod
    def predict(self, samples: list[dict]) -> np.ndarray:
        """File-level scores [len(samples), 17]."""

    def supported_mask(self) -> np.ndarray:
        return np.array([t in self.supported_types for t in SWC_TYPES], dtype=bool)

    def _check(self, scores: np.ndarray, n: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (n, len(SWC_TYPES)):
            raise ValueError(f"{self.name}: expected scores of shape {(n, len(SWC_TYPES))}, "
                             f"got {scores.shape}")
        if np.any(~np.isfinite(scores)) or scores.min() < 0 or scores.max() > 1:
            raise ValueError(f"{self.name}: scores must be finite and in [0, 1]")
        scores[:, ~self.supported_mask()] = 0.0
        return scores


REGISTRY: dict[str, type[Baseline]] = {}


def register(cls: type[Baseline]) -> type[Baseline]:
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate baseline name {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def make_baseline(name: str, seed: int, source: SourceReader | None, params: dict) -> Baseline:
    # importing the modules populates REGISTRY
    from . import classic, llm, neural, tools  # noqa: F401
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](seed=seed, source=source, **params)
