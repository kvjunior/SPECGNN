"""Collate a list of graph samples into one disjoint-union batch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..graph.features import CANDIDATE_NODE_TYPES, EDGE_TYPES, NODE_TYPES

N_REL = 2 * len(EDGE_TYPES)          # each edge type plus its inverse
_CAND = [NODE_TYPES.index(t) for t in CANDIDATE_NODE_TYPES]


@dataclass
class Batch:
    x: torch.Tensor            # [N, F + n_node_types]
    batch: torch.Tensor        # [N] graph index
    src: list[torch.Tensor]    # per relation
    dst: list[torch.Tensor]
    norm: list[torch.Tensor]   # 1 / in-degree of dst under that relation
    cand: torch.Tensor         # [N] bool, node may host a candidate
    y: torch.Tensor            # [B, T]
    y_mask: torch.Tensor
    node_y: torch.Tensor       # [N, T]
    node_mask: torch.Tensor
    pl_target: torch.Tensor    # [N, T] pseudo-label targets (feedback rounds only)
    pl_weight: torch.Tensor    # [N, T] pseudo-label weights; 0 = no pseudo-label
    num_graphs: int

    def to(self, device) -> Batch:
        def mv(t: torch.Tensor) -> torch.Tensor:
            return t.to(device, non_blocking=True)
        return Batch(mv(self.x), mv(self.batch), [mv(s) for s in self.src],
                     [mv(d) for d in self.dst], [mv(n) for n in self.norm], mv(self.cand),
                     mv(self.y), mv(self.y_mask), mv(self.node_y), mv(self.node_mask),
                     mv(self.pl_target), mv(self.pl_weight), self.num_graphs)


def collate(samples: list[dict], node_feature_mask: np.ndarray | None = None) -> Batch:
    """Optional per-sample keys: ``extra_x`` [N, E] (appended to the node input,
    e.g. verdict features), ``pl_target`` / ``pl_weight`` [N, T] (pseudo-labels)."""
    xs, bs, cands, nys, nms, plt, plw = [], [], [], [], [], [], []
    src = [[] for _ in range(N_REL)]
    dst = [[] for _ in range(N_REL)]
    offset = 0
    n_types = len(NODE_TYPES)
    for b, s in enumerate(samples):
        g = s["graph"]
        n = len(g["node_type"])
        onehot = np.zeros((n, n_types), dtype=np.float32)
        onehot[np.arange(n), g["node_type"]] = 1.0
        x = g["x"] if node_feature_mask is None else g["x"] * node_feature_mask
        parts = [x, onehot] + ([s["extra_x"]] if "extra_x" in s else [])
        xs.append(np.concatenate(parts, axis=1).astype(np.float32))
        bs.append(np.full(n, b, dtype=np.int64))
        cands.append(np.isin(g["node_type"], _CAND))
        nys.append(s["node_y"])
        nms.append(s["node_mask"])
        zeros = np.zeros_like(s["node_y"])
        plt.append(s.get("pl_target", zeros))
        plw.append(s.get("pl_weight", zeros))
        for r, et in enumerate(EDGE_TYPES):
            e = g["edges"][et]
            if len(e):
                src[2 * r].append(e[:, 0] + offset)
                dst[2 * r].append(e[:, 1] + offset)
                src[2 * r + 1].append(e[:, 1] + offset)      # inverse relation
                dst[2 * r + 1].append(e[:, 0] + offset)
        offset += n
    N = offset

    def cat(lst, dtype=np.int64):
        if not lst:
            return torch.zeros(0, dtype=torch.long)
        return torch.from_numpy(np.concatenate(lst).astype(dtype))

    src_t = [cat(s) for s in src]
    dst_t = [cat(d) for d in dst]
    norm_t = []
    for d in dst_t:
        deg = torch.bincount(d, minlength=N).clamp(min=1).float()
        norm_t.append(1.0 / deg[d])
    return Batch(
        x=torch.from_numpy(np.concatenate(xs)),
        batch=torch.from_numpy(np.concatenate(bs)),
        src=src_t, dst=dst_t, norm=norm_t,
        cand=torch.from_numpy(np.concatenate(cands)),
        y=torch.from_numpy(np.stack([s["y"] for s in samples])),
        y_mask=torch.from_numpy(np.stack([s["y_mask"] for s in samples])),
        node_y=torch.from_numpy(np.concatenate(nys)),
        node_mask=torch.from_numpy(np.concatenate(nms)),
        pl_target=torch.from_numpy(np.concatenate(plt).astype(np.float32)),
        pl_weight=torch.from_numpy(np.concatenate(plw).astype(np.float32)),
        num_graphs=len(samples),
    )
