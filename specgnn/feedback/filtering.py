"""Post-hoc specification filter (modes b and c).

For each file and SWC type t, the inference-time candidates are the candidate
nodes with p_vt > tau, plus the single highest-scoring candidate node so that
every file has at least one candidate per type (this makes the filter defined
for every score, which threshold-free metrics such as AP require). If every
candidate of type t is REFUTED, the file-level prediction for t is suppressed
(score set to 0). CONFIRMED and UNKNOWN candidates leave the score unchanged.
"""
from __future__ import annotations

import numpy as np

from .pseudo_labels import select_candidates
from .verdicts import REF


def refuted_types(node_probs: np.ndarray, cand_nodes: np.ndarray, verdicts: np.ndarray,
                  tau: float) -> np.ndarray:
    """Boolean [T]: types whose inference-time candidates are all REFUTED."""
    n, t = node_probs.shape
    out = np.zeros(t, dtype=bool)
    if n == 0 or not cand_nodes.any():
        return out
    chosen = select_candidates(node_probs, cand_nodes, tau, top1=True)
    for j in range(t):
        rows = np.nonzero(chosen[:, j])[0]
        out[j] = rows.size > 0 and bool(np.all(verdicts[rows, j] == REF))
    return out


def apply_filter(graph_probs: np.ndarray, node_probs: list[np.ndarray], cand: list[np.ndarray],
                 verdicts: list[np.ndarray], tau: float) -> tuple[np.ndarray, dict]:
    """Filtered copy of graph_probs [B, T] and suppression statistics."""
    out = graph_probs.copy()
    suppressed = np.zeros(graph_probs.shape, dtype=bool)
    for b, (p, c, v) in enumerate(zip(node_probs, cand, verdicts, strict=True)):
        r = refuted_types(p, c, v, tau)
        out[b, r] = 0.0
        suppressed[b] = r
    above = graph_probs > tau
    stats = {"suppressed_pairs": int(suppressed.sum()),
             "suppressed_above_tau": int((suppressed & above).sum()),
             "predicted_above_tau": int(above.sum())}
    return out, stats
