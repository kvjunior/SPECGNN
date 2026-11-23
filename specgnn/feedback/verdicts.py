"""Verdict cache.

The specification checker is deterministic and does not depend on the model,
so the verdict for every (candidate node, SWC type) pair of every graph is
computed once and stored. Feedback rounds and the inference-time filter then
look verdicts up instead of re-running the checker; the checker's cost is
measured once, here, and reported separately.

Table layout per graph: int8 array [num_nodes, 17] with codes
    NA = 0         node cannot host a candidate (ext_call, event)
    CONFIRMED = 1
    REFUTED = 2
    UNKNOWN = 3
"""
from __future__ import annotations

import time

import numpy as np

from ..data.config import SWC_TYPES
from ..graph.features import CANDIDATE_NODE_TYPES, NODE_TYPES
from ..spec.checker import CONFIRMED, REFUTED, UNKNOWN, CheckerConfig, FileChecker

NA, CONF, REF, UNK = 0, 1, 2, 3
CODE = {CONFIRMED: CONF, REFUTED: REF, UNKNOWN: UNK}
_CAND = {NODE_TYPES.index(t) for t in CANDIDATE_NODE_TYPES}


def verdict_table(graph: dict, source: str, cfg: CheckerConfig) -> np.ndarray:
    """Verdicts for all candidate nodes and all 17 types of one graph."""
    fc = FileChecker(source, max_depth=cfg.max_depth, max_paths=cfg.max_paths)
    n = len(graph["node_type"])
    table = np.zeros((n, len(SWC_TYPES)), dtype=np.int8)
    for i in range(n):
        if int(graph["node_type"][i]) not in _CAND:
            continue
        for j, t in enumerate(SWC_TYPES):
            table[i, j] = CODE[fc.check(graph, i, t).verdict]
    return table


def build_cache(samples: list[dict], read_source, cfg: CheckerConfig) -> tuple[dict, dict]:
    """Return ({uid: table}, stats). ``read_source(sample) -> str``."""
    cache: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    counts = np.zeros(4, dtype=np.int64)
    for s in samples:
        tab = verdict_table(s["graph"], read_source(s), cfg)
        cache[s["uid"]] = tab
        counts += np.bincount(tab.ravel(), minlength=4)[:4]
    seconds = time.perf_counter() - t0
    checks = int(counts[1:].sum())
    stats = {"graphs": len(samples), "checks": checks, "seconds": round(seconds, 2),
             "ms_per_check": round(1000 * seconds / max(checks, 1), 4),
             "ms_per_graph": round(1000 * seconds / max(len(samples), 1), 2),
             "confirmed": int(counts[CONF]), "refuted": int(counts[REF]),
             "unknown": int(counts[UNK])}
    return cache, stats
