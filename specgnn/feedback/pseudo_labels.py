"""From specification verdicts to weighted pseudo-labels (Phase 3).

Candidates on a training graph are the (node, type) pairs whose node
probability exceeds ``cand_tau`` (the paper's C = {(v, t, p) : p > tau}) plus the
highest-scoring node of each type: the same candidate set the inference-time filter
judges (see select_candidates). Each
candidate's cached verdict becomes a pseudo-label according to the policy:

    policy          CONFIRMED            REFUTED              UNKNOWN
    both            target 1, w_pos      target 0, w_neg      discarded
    refute_only     discarded            target 0, w_neg      discarded
    confirm_only    target 1, w_pos      discarded            discarded

UNKNOWN verdicts never produce a pseudo-label, a correction signal, or an
embedding. Pseudo-labels are an additional, weighted loss term; they do not
overwrite ground-truth labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .verdicts import CONF, REF

POLICIES = ("both", "refute_only", "confirm_only")


@dataclass
class FeedbackConfig:
    enabled: bool = False
    policy: str = "both"
    every: int = 5               # K: feedback round every K supervised epochs
    w_pos: float = 1.5           # weight of CONFIRMED pseudo-labels
    w_neg: float = 2.0           # weight of REFUTED pseudo-labels
    lambda_pl: float = 1.0       # weight of the pseudo-label loss
    mini_epochs: int = 2         # mini-retraining epochs per round
    batch_size: int = 16
    lr_scale: float = 0.3        # mini-retraining learning rate = train.lr * lr_scale
    cand_tau: float = 0.5        # candidate threshold on node probabilities
    top1: bool = True            # also the top node per type (see select_candidates)
    correction: bool = True      # gated representation correction
    w_align: float = 0.3         # weight of the correction alignment loss

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if self.every < 1 or self.mini_epochs < 0 or self.batch_size < 1:
            raise ValueError("every >= 1, mini_epochs >= 0 and batch_size >= 1 are required")
        if min(self.w_pos, self.w_neg, self.lambda_pl, self.lr_scale, self.w_align) < 0:
            raise ValueError("weights and scales must be non-negative")
        if not 0.0 <= self.cand_tau < 1.0:
            raise ValueError("cand_tau must lie in [0, 1)")


def select_candidates(node_probs: np.ndarray, cand_nodes: np.ndarray, tau: float,
                      top1: bool = True) -> np.ndarray:
    """Boolean [N, T] candidate (node, type) pairs on candidate nodes: nodes with p > tau
    and, if ``top1``, the highest-scoring candidate node of every type.

    This one definition is used by the inference-time filter, by feedback rounds and by
    the verdict-reliability measurement, so filtering and feedback always act on the same
    candidates. The top-1 node keeps the definition meaningful when probabilities are not
    calibrated (with rare positives and focal loss no node may exceed tau).
    """
    chosen = (node_probs > tau) & cand_nodes[:, None]
    if top1 and cand_nodes.any() and node_probs.shape[0]:
        masked = np.where(cand_nodes[:, None], node_probs, -np.inf)
        chosen[masked.argmax(axis=0), np.arange(node_probs.shape[1])] = True
    return chosen


def pseudo_labels(candidates: np.ndarray, verdicts: np.ndarray, cfg: FeedbackConfig
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (target, weight), both float32 [N, T]; weight 0 = no pseudo-label."""
    if candidates.shape != verdicts.shape:
        raise ValueError(f"shape mismatch {candidates.shape} vs {verdicts.shape}")
    target = np.zeros(candidates.shape, dtype=np.float32)
    weight = np.zeros(candidates.shape, dtype=np.float32)
    if cfg.policy in ("both", "confirm_only"):
        conf = candidates & (verdicts == CONF)
        target[conf] = 1.0
        weight[conf] = cfg.w_pos
    if cfg.policy in ("both", "refute_only"):
        ref = candidates & (verdicts == REF)
        weight[ref] = cfg.w_neg          # target stays 0
    return target, weight
