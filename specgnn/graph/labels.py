"""Training targets for one graph.

Graph level: y[t] in {0, 1} with mask[t] = 0 where the label is unknown.

Node level (candidate localization), for every SWC type t:
  * file label 0        -> every candidate node is a negative (known).
  * file label 1 with line annotations
                        -> nodes covering an annotated line are positives;
                           other nodes are left unknown, because audits often
                           report one instance of a weakness that occurs in
                           several places.
  * file label 1 without line annotations, or file label unknown
                        -> all nodes unknown; only the multiple-instance
                           (MIL) loss constrains them.
"""
from __future__ import annotations

import numpy as np

from ..data.config import SWC_TYPES
from .features import CANDIDATE_NODE_TYPES, NODE_TYPES

_CAND = np.array([NODE_TYPES.index(t) for t in CANDIDATE_NODE_TYPES])
_PRIORITY = [("function", "modifier"), ("state_var",), ("contract",)]


def graph_targets(labels: dict) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(len(SWC_TYPES), dtype=np.float32)
    m = np.zeros(len(SWC_TYPES), dtype=np.float32)
    for i, t in enumerate(SWC_TYPES):
        v = labels.get(t)
        if v is not None:
            y[i], m[i] = float(v), 1.0
    return y, m


def locate(graph: dict, start: int, end: int) -> list[int]:
    """Most specific candidate nodes overlapping the line range [start, end]."""
    nt = graph["node_type"]
    spans = graph["spans"]
    for group in _PRIORITY:
        ids = [NODE_TYPES.index(t) for t in group]
        hits = [i for i in range(len(nt)) if nt[i] in ids
                and spans[i][0] <= end and spans[i][1] >= start]
        if hits:
            # innermost: smallest span
            best = min(spans[i][1] - spans[i][0] for i in hits)
            return [i for i in hits if spans[i][1] - spans[i][0] == best]
    # outside every declaration (e.g. the pragma line): file-level -> contract nodes
    return [i for i in range(len(nt)) if nt[i] == NODE_TYPES.index("contract")]


def node_targets(graph: dict, labels: dict, annotations: list[dict] | None
                 ) -> tuple[np.ndarray, np.ndarray]:
    n = len(graph["node_type"])
    y = np.zeros((n, len(SWC_TYPES)), dtype=np.float32)
    m = np.zeros((n, len(SWC_TYPES)), dtype=np.float32)
    cand = np.isin(graph["node_type"], _CAND)
    by_type: dict[str, list[dict]] = {}
    for a in annotations or []:
        by_type.setdefault(a["swc"], []).append(a)
    func_ids = {name: i for i, name in enumerate(graph["names"])
                if graph["node_type"][i] == NODE_TYPES.index("function")}
    for j, t in enumerate(SWC_TYPES):
        v = labels.get(t)
        if v == 0:
            m[cand, j] = 1.0
        elif v == 1:
            hits: set[int] = set()
            for a in by_type.get(t, []):
                for s, e in a.get("lines") or []:
                    hits.update(locate(graph, s, e))
                fn = a.get("function")
                if fn and fn in func_ids:
                    hits.add(func_ids[fn])
            for i in hits:
                y[i, j], m[i, j] = 1.0, 1.0
    return y, m
