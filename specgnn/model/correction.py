"""Gated representation correction (training-time only).

For a node i that carries pseudo-labels, a verdict context vector summarizes
which types were CONFIRMED and which were REFUTED on that node:

    v_i  = W_v [conf_i || ref_i]                    conf_i, ref_i in {0,1}^T
    c_i  = GELU(W_c [sg(h_i) || v_i] + b_c)          correction       (Eq. 6)
    g_i  = sigmoid(W_g [sg(h_i) || v_i] + b_g)       gate
    h~_i = sg(h_i) + g_i * c_i                       corrected representation

sg = stop-gradient. Two losses use it (specgnn/feedback/loop.py):

  teacher   pseudo-label loss on node_head(h~_i)  -> trains the correction
  align     || h_i - sg(h~_i) ||^2                 -> trains the encoder

The correction path sees the verdicts; the encoder does not. The encoder is
therefore trained to produce, from code alone, the representation the
verdict-informed path arrives at. At inference no verdicts exist and the module
is not used, so no verdict information can leak into test predictions. Nodes
whose candidates were all UNKNOWN get no pseudo-label and no correction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCorrection(nn.Module):
    def __init__(self, hidden: int, n_types: int):
        super().__init__()
        self.verdict = nn.Linear(2 * n_types, hidden)
        self.correct = nn.Linear(2 * hidden, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)
        nn.init.zeros_(self.correct.weight)          # starts as the identity map
        nn.init.zeros_(self.correct.bias)

    def forward(self, h: torch.Tensor, confirmed: torch.Tensor, refuted: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """h [M, H] (representations of pseudo-labeled nodes), confirmed/refuted [M, T].
        Returns (corrected representation [M, H], gate [M, H])."""
        hs = h.detach()
        v = self.verdict(torch.cat([confirmed, refuted], dim=1))
        x = torch.cat([hs, v], dim=1)
        g = torch.sigmoid(self.gate(x))
        return hs + g * F.gelu(self.correct(x)), g
