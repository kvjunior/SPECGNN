"""R-GCN candidate generator (Phase 1).

    h_i^(0)   = GELU(W_proj [x_i || t_i] + b)                         (Eq. 2)
    h_i^(l+1) = LN(h_i^(l) + Drop(GELU(W_self h_i^(l)
                 + sum_r sum_{j in N_i^r} 1/|N_i^r| W_r h_j^(l) + b)))  (Eq. 3, + residual)
    alpha^k   = masked per-graph softmax(w_k^T tanh(W_1 h_i^(L) + b_1)) (Eq. 4, K heads)
    g         = W_o [sum_i alpha_i^1 h_i || ... || sum_i alpha_i^K h_i]

Outputs, both over the 17 SWC types (multi-label, sigmoid):
    graph_logits [B, T]  from g     -> file-level detection
    node_logits  [N, T]  per node   -> candidates C = {(v, t, p_vt) : p_vt > tau}

Relations: 4 edge types and their inverses (8 relations).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.features import INPUT_DIM
from .batching import N_REL, Batch


@dataclass
class ModelConfig:
    input_dim: int = INPUT_DIM
    hidden: int = 256
    layers: int = 4
    heads: int = 8
    dropout: float = 0.1
    n_types: int = 17
    residual: bool = True


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    """Softmax of scores [N, K] within each segment given by index [N]."""
    k = scores.shape[1]
    idx = index.unsqueeze(1).expand(-1, k)
    mx = torch.full((num, k), float("-inf"), device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[index])
    den = torch.zeros((num, k), device=scores.device, dtype=scores.dtype).index_add_(0, index, ex)
    return ex / den[index].clamp(min=1e-12)


def segment_logsumexp(values: torch.Tensor, index: torch.Tensor, num: int,
                      mask: torch.Tensor) -> torch.Tensor:
    """log sum exp over the masked rows of each segment; -inf-safe. values [N, T]."""
    t = values.shape[1]
    v = values.masked_fill(~mask.unsqueeze(1), float("-inf"))
    idx = index.unsqueeze(1).expand(-1, t)
    mx = torch.full((num, t), float("-inf"), device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, idx, v, reduce="amax", include_self=True)
    safe_mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    ex = torch.exp(v - safe_mx[index]).masked_fill(~mask.unsqueeze(1), 0.0)
    s = torch.zeros((num, t), device=v.device, dtype=v.dtype).index_add_(0, index, ex)
    return torch.where(s > 0, torch.log(s.clamp(min=1e-30)) + safe_mx,
                       torch.full_like(s, -30.0))


class RGCNLayer(nn.Module):
    def __init__(self, dim: int, n_rel: int, dropout: float, residual: bool):
        super().__init__()
        self.self_loop = nn.Linear(dim, dim)
        self.rel = nn.Linear(dim, dim * n_rel, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.n_rel, self.dim, self.residual = n_rel, dim, residual
        nn.init.xavier_uniform_(self.rel.weight, gain=1.0 / n_rel ** 0.5)

    def forward(self, h: torch.Tensor, b: Batch) -> torch.Tensor:
        out = self.self_loop(h)
        hw = self.rel(h).view(-1, self.n_rel, self.dim)
        for r in range(self.n_rel):
            if b.src[r].numel():
                msg = hw[b.src[r], r] * b.norm[r].unsqueeze(1)
                out = out.index_add(0, b.dst[r], msg)
        out = self.drop(F.gelu(out))
        return self.norm(h + out if self.residual else out)


class AttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, heads, bias=False)
        self.out = nn.Linear(dim * heads, dim)
        self.heads = heads

    def forward(self, h: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
        alpha = segment_softmax(self.w2(torch.tanh(self.w1(h))), index, num)     # [N, K]
        weighted = alpha.unsqueeze(2) * h.unsqueeze(1)                            # [N, K, H]
        g = torch.zeros((num, self.heads, h.shape[1]), device=h.device, dtype=h.dtype)
        g = g.index_add(0, index, weighted)
        return self.out(g.flatten(1))


def _mlp(dim: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, out))


class SpecGNNGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.input_dim, cfg.hidden)
        self.layers = nn.ModuleList(RGCNLayer(cfg.hidden, N_REL, cfg.dropout, cfg.residual)
                                    for _ in range(cfg.layers))
        self.pool = AttentionPool(cfg.hidden, cfg.heads)
        self.graph_head = _mlp(cfg.hidden, cfg.n_types, cfg.dropout)
        self.node_head = _mlp(cfg.hidden, cfg.n_types, cfg.dropout)

    def encode(self, b: Batch) -> torch.Tensor:
        h = F.gelu(self.proj(b.x))
        for layer in self.layers:
            h = layer(h, b)
        return h

    def heads(self, h: torch.Tensor, b: Batch) -> dict[str, torch.Tensor]:
        g = self.pool(h, b.batch, b.num_graphs)
        node_logits = self.node_head(h)
        return {"graph_logits": self.graph_head(g), "node_logits": node_logits,
                "node_repr": h, "graph_repr": g}

    def forward(self, b: Batch) -> dict[str, torch.Tensor]:
        return self.heads(self.encode(b), b)
