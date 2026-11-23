"""Training objective of the candidate generator.

L = L_graph + lambda_node * L_node + lambda_mil * L_mil

L_graph  focal BCE on file-level labels (unknown labels masked out)
L_node   focal BCE on node-level targets where they are known
L_mil    focal BCE between the file label and a smooth maximum of the node
         logits of that type (multiple-instance learning): a vulnerable file
         must contain at least one high-scoring node, a safe file none. This is
         what trains candidate localization on datasets with contract-level
         labels only (Resource 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .rgcn import segment_logsumexp


@dataclass
class LossConfig:
    gamma: float = 2.0
    alpha: float = 0.25          # weight of the positive class in focal loss
    smoothing: float = 0.05
    lambda_node: float = 1.0
    lambda_mil: float = 0.5
    mil_temperature: float = 0.5


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
              cfg: LossConfig, weight: torch.Tensor | None = None) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    t = targets * (1 - cfg.smoothing) + 0.5 * cfg.smoothing
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = cfg.alpha * targets + (1 - cfg.alpha) * (1 - targets)
    loss = a_t * (1 - p_t).pow(cfg.gamma) * bce
    w = mask if weight is None else mask * weight
    return (loss * w).sum() / mask.sum().clamp(min=1.0)


def mil_logits(node_logits: torch.Tensor, batch, temperature: float) -> torch.Tensor:
    """Smooth maximum of candidate-node logits per graph and type: [B, T]."""
    return temperature * segment_logsumexp(node_logits / temperature, batch.batch,
                                           batch.num_graphs, batch.cand)


def generator_loss(out: dict, batch, cfg: LossConfig) -> tuple[torch.Tensor, dict]:
    lg = focal_bce(out["graph_logits"], batch.y, batch.y_mask, cfg)
    node_mask = batch.node_mask * batch.cand.unsqueeze(1).float()
    ln = focal_bce(out["node_logits"], batch.node_y, node_mask, cfg)
    lm = focal_bce(mil_logits(out["node_logits"], batch, cfg.mil_temperature),
                   batch.y, batch.y_mask, cfg)
    total = lg + cfg.lambda_node * ln + cfg.lambda_mil * lm
    return total, {"graph": lg.item(), "node": ln.item(), "mil": lm.item()}
