"""Neural baselines on the common code-graph input (docs/BASELINES.md).

Each model keeps the defining architectural idea of its paper; all consume the
same graphs, labels and splits as SpecGNN and are trained by one trainer
(focal loss on file labels, early stopping on validation macro-AP).

  drgcn      DR-GCN (Zhuang et al., IJCAI'20): degree-free GCN, homogeneous graph,
             max-pool readout
  ame        AME (Liu et al., IJCAI'21): GNN graph feature + expert-pattern feature,
             fused by attention
  cge        CGE (Liu et al., TKDE'23): GNN graph feature + expert-pattern feature,
             fused by concatenation
  efevd      EFEVD (Jiang et al., IJCAI'24): graph attention encoder trained
             multi-task on file and node labels
  scvhunter  SCVHunter (Luo et al., ICSE'24): heterogeneous graph attention with
             relation-level (semantic) attention
  mando_hgt  MANDO-HGT (Nguyen et al., MSR'23): heterogeneous graph transformer with
             node-type-specific projections and relation-specific attention
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.config import SWC_TYPES
from ..eval.metrics import detection_metrics
from ..graph.features import FEATURE_DIM, INPUT_DIM, NODE_TYPES, OFFSETS
from ..model.batching import N_REL, Batch, collate
from ..model.losses import LossConfig, focal_bce
from ..utils.repro import set_seed
from .base import Baseline, register

T = len(SWC_TYPES)
N_NODE_TYPES = len(NODE_TYPES)
FUNC = slice(OFFSETS["function"], OFFSETS["ext_call"])
CONTRACT = slice(OFFSETS["contract"], OFFSETS["event"])
EXPERT_DIM = 2 * (FUNC.stop - FUNC.start) + (CONTRACT.stop - CONTRACT.start)


# ---------------------------------------------------------------- segment operations
def seg_max(h: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    out = torch.full((num, h.shape[1]), float("-inf"), device=h.device, dtype=h.dtype)
    out = out.scatter_reduce(0, index.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def seg_mean(h: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    s = torch.zeros((num, h.shape[1]), device=h.device, dtype=h.dtype).index_add(0, index, h)
    c = torch.bincount(index, minlength=num).clamp(min=1).unsqueeze(1).to(h.dtype)
    return s / c


def edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num: int) -> torch.Tensor:
    """Softmax of edge scores [E] over the incoming edges of each destination."""
    mx = torch.full((num,), float("-inf"), device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(0, dst, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[dst])
    den = torch.zeros(num, device=scores.device, dtype=scores.dtype).index_add(0, dst, ex)
    return ex / den[dst].clamp(min=1e-12)


def all_edges(b: Batch, self_loops: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    src = torch.cat([s for s in b.src if s.numel()] or [torch.zeros(0, dtype=torch.long)])
    dst = torch.cat([d for d in b.dst if d.numel()] or [torch.zeros(0, dtype=torch.long)])
    src, dst = src.to(b.x.device), dst.to(b.x.device)
    if self_loops:
        n = torch.arange(b.x.shape[0], device=b.x.device)
        src, dst = torch.cat([src, n]), torch.cat([dst, n])
    return src, dst


def node_type_index(b: Batch) -> torch.Tensor:
    return b.x[:, FEATURE_DIM:FEATURE_DIM + N_NODE_TYPES].argmax(1)


def expert_patterns(b: Batch) -> torch.Tensor:
    """File-level security-pattern vector: max and mean of the function-block
    features over function/modifier nodes, and max of the contract-block features."""
    nt = node_type_index(b)
    fmask = (nt == NODE_TYPES.index("function")) | (nt == NODE_TYPES.index("modifier"))
    cmask = nt == NODE_TYPES.index("contract")
    f = b.x[:, FUNC] * fmask.unsqueeze(1)
    c = b.x[:, CONTRACT] * cmask.unsqueeze(1)
    f_sum = torch.zeros((b.num_graphs, f.shape[1]), device=f.device).index_add(0, b.batch, f)
    f_cnt = torch.zeros(b.num_graphs, device=f.device).index_add(0, b.batch, fmask.float())
    f_mean = f_sum / f_cnt.clamp(min=1).unsqueeze(1)          # mean over function nodes only
    return torch.cat([seg_max(f, b.batch, b.num_graphs), f_mean,
                      seg_max(c, b.batch, b.num_graphs)], dim=1)


def mlp(i: int, h: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


# ---------------------------------------------------------------- models
class DRGCN(nn.Module):
    """Degree-free GCN: h' = ReLU(W (h + sum_{j in N(i)} h_j)); max-pool readout."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(INPUT_DIM, hidden)
        self.convs = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.out = mlp(hidden, hidden, T, dropout)

    def encode(self, b: Batch) -> torch.Tensor:
        src, dst = all_edges(b)
        h = F.relu(self.inp(b.x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            m = torch.zeros_like(h).index_add(0, dst, h[src])
            h = norm(self.drop(F.relu(conv(h + m))))
        return h

    def forward(self, b: Batch) -> dict:
        return {"graph_logits": self.out(seg_max(self.encode(b), b.batch, b.num_graphs))}


class ExpertFusion(nn.Module):
    """AME (attention fusion) / CGE (concatenation fusion) of a DR-GCN graph feature
    and an expert-pattern feature."""

    def __init__(self, hidden: int, layers: int, dropout: float, fusion: str):
        super().__init__()
        self.gnn = DRGCN(hidden, layers, dropout)
        self.expert = mlp(EXPERT_DIM, hidden, hidden, dropout)
        self.fusion = fusion
        if fusion == "attention":
            self.att = nn.Linear(hidden, hidden)
            self.q = nn.Linear(hidden, 1, bias=False)
            self.out = mlp(hidden, hidden, T, dropout)
        else:
            self.out = mlp(2 * hidden, hidden, T, dropout)

    def forward(self, b: Batch) -> dict:
        g = seg_max(self.gnn.encode(b), b.batch, b.num_graphs)
        e = self.expert(expert_patterns(b))
        if self.fusion == "attention":
            parts = torch.stack([g, e], dim=1)                       # [B, 2, H]
            a = torch.softmax(self.q(torch.tanh(self.att(parts))), dim=1)
            fused = (a * parts).sum(1)
        else:
            fused = torch.cat([g, e], dim=1)
        return {"graph_logits": self.out(fused)}


class GATLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)
        self.a_src = nn.Linear(hidden, 1, bias=False)
        self.a_dst = nn.Linear(hidden, 1, bias=False)

    def forward(self, h, src, dst):
        wh = self.w(h)
        e = F.leaky_relu(self.a_src(wh)[src, 0] + self.a_dst(wh)[dst, 0], 0.2)
        alpha = edge_softmax(e, dst, h.shape[0])
        return torch.zeros_like(wh).index_add(0, dst, alpha.unsqueeze(1) * wh[src])


class EFEVDLike(nn.Module):
    """Graph attention encoder with file-level and node-level heads (multi-task)."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(INPUT_DIM, hidden)
        self.gats = nn.ModuleList(GATLayer(hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.graph_head = mlp(2 * hidden, hidden, T, dropout)
        self.node_head = mlp(hidden, hidden, T, dropout)

    def forward(self, b: Batch) -> dict:
        src, dst = all_edges(b, self_loops=True)
        h = F.elu(self.inp(b.x))
        for gat, norm in zip(self.gats, self.norms, strict=True):
            h = norm(h + self.drop(F.elu(gat(h, src, dst))))
        g = torch.cat([seg_mean(h, b.batch, b.num_graphs), seg_max(h, b.batch, b.num_graphs)], 1)
        return {"graph_logits": self.graph_head(g), "node_logits": self.node_head(h)}


class HANLike(nn.Module):
    """Relation-specific graph attention + per-node semantic attention over relations."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(INPUT_DIM, hidden)
        self.rel = nn.ModuleList(nn.ModuleList(GATLayer(hidden) for _ in range(N_REL))
                                 for _ in range(layers))
        self.sem = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.q = nn.ModuleList(nn.Linear(hidden, 1, bias=False) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.out = mlp(2 * hidden, hidden, T, dropout)

    def forward(self, b: Batch) -> dict:
        h = F.elu(self.inp(b.x))
        n = torch.arange(h.shape[0], device=h.device)
        for layer in range(len(self.rel)):
            zs = []
            for r in range(N_REL):
                src = torch.cat([b.src[r].to(h.device), n])
                dst = torch.cat([b.dst[r].to(h.device), n])
                zs.append(F.elu(self.rel[layer][r](h, src, dst)))
            z = torch.stack(zs, dim=1)                                     # [N, R, H]
            beta = torch.softmax(self.q[layer](torch.tanh(self.sem[layer](z))), dim=1)
            h = self.norms[layer](h + self.drop((beta * z).sum(1)))
        g = torch.cat([seg_mean(h, b.batch, b.num_graphs), seg_max(h, b.batch, b.num_graphs)], 1)
        return {"graph_logits": self.out(g)}


class HGTLike(nn.Module):
    """Heterogeneous graph transformer: node-type-specific K/Q/V, relation-specific
    attention and message matrices, softmax over all incoming edges."""

    def __init__(self, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(INPUT_DIM, hidden)
        self.layers_ = layers
        self.k = nn.ModuleList(nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(N_NODE_TYPES))
                               for _ in range(layers))
        self.q = nn.ModuleList(nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(N_NODE_TYPES))
                               for _ in range(layers))
        self.v = nn.ModuleList(nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(N_NODE_TYPES))
                               for _ in range(layers))
        self.w_att = nn.Parameter(torch.stack([torch.stack([torch.eye(hidden)] * N_REL)] * layers))
        self.w_msg = nn.Parameter(torch.stack([torch.stack([torch.eye(hidden)] * N_REL)] * layers))
        self.upd = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.out = mlp(2 * hidden, hidden, T, dropout)
        self.scale = 1.0 / math.sqrt(hidden)

    def _typed(self, mods, h, nt):
        out = torch.zeros_like(h)
        for t, m in enumerate(mods):
            sel = nt == t
            if sel.any():
                out[sel] = m(h[sel])
        return out

    def forward(self, b: Batch) -> dict:
        h = F.gelu(self.inp(b.x))
        nt = node_type_index(b)
        for layer in range(self.layers_):
            k = self._typed(self.k[layer], h, nt)
            q = self._typed(self.q[layer], h, nt)
            v = self._typed(self.v[layer], h, nt)
            srcs, dsts, scores, msgs = [], [], [], []
            for r in range(N_REL):
                s, d = b.src[r].to(h.device), b.dst[r].to(h.device)
                if not s.numel():
                    continue
                kr = k[s] @ self.w_att[layer, r]
                scores.append((kr * q[d]).sum(1) * self.scale)
                msgs.append(v[s] @ self.w_msg[layer, r])
                srcs.append(s)
                dsts.append(d)
            if dsts:
                dst = torch.cat(dsts)
                alpha = edge_softmax(torch.cat(scores), dst, h.shape[0])
                agg = torch.zeros_like(h).index_add(0, dst, alpha.unsqueeze(1) * torch.cat(msgs))
            else:
                agg = torch.zeros_like(h)
            h = self.norms[layer](h + self.drop(self.upd[layer](F.gelu(agg))))
        g = torch.cat([seg_mean(h, b.batch, b.num_graphs), seg_max(h, b.batch, b.num_graphs)], 1)
        return {"graph_logits": self.out(g)}


# ---------------------------------------------------------------- shared trainer
def _iterate(samples, bs, shuffle, rng):
    idx = list(range(len(samples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [samples[j] for j in idx[i:i + bs]]


class NeuralBaseline(Baseline):
    kind = "learned"
    defaults = {"hidden": 128, "layers": 3, "dropout": 0.3, "epochs": 100, "lr": 1e-3,
                "weight_decay": 1e-3, "patience": 15, "batch_size": 32, "alpha": 0.25,
                "gamma": 2.0, "smoothing": 0.05, "device": "auto"}

    def build(self, hidden: int, layers: int, dropout: float) -> nn.Module:
        raise NotImplementedError

    def __init__(self, seed=42, source=None, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        super().__init__(seed, source, **{**self.defaults, **params})
        p = self.params
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if p["device"] == "auto" \
            else p["device"]
        self.lcfg = LossConfig(gamma=p["gamma"], alpha=p["alpha"], smoothing=p["smoothing"])
        self.model = None

    @torch.no_grad()
    def predict(self, samples: list[dict]) -> np.ndarray:
        self.model.eval()
        out = []
        for chunk in _iterate(samples, 64, False, None):
            b = collate(chunk).to(self.device)
            out.append(torch.sigmoid(self.model(b)["graph_logits"]).cpu().numpy())
        scores = np.concatenate(out) if out else np.zeros((0, T), dtype=np.float32)
        return self._check(scores, len(samples))

    def fit(self, train: list[dict], val: list[dict]) -> dict:
        p = self.params
        set_seed(self.seed)
        rng = random.Random(self.seed)
        self.model = self.build(p["hidden"], p["layers"], p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        best, best_state, bad, history = -float("inf"), None, 0, []
        for epoch in range(1, p["epochs"] + 1):
            self.model.train()
            for chunk in _iterate(train, p["batch_size"], True, rng):
                b = collate(chunk).to(self.device)
                out = self.model(b)
                loss = focal_bce(out["graph_logits"], b.y, b.y_mask, self.lcfg)
                if "node_logits" in out:
                    m = b.node_mask * b.cand.unsqueeze(1).float()
                    loss = loss + focal_bce(out["node_logits"], b.node_y, m, self.lcfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            vs = self.predict(val)
            y = np.stack([s["y"] for s in val])
            m = np.stack([s["y_mask"] for s in val])
            ap = detection_metrics(vs, y, m)["macro"]["ap"]
            ap = -1.0 if np.isnan(ap) else ap
            history.append({"epoch": epoch, "val_macro_ap": ap})
            if ap > best:
                best, bad = ap, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"epochs": history, "best_val_macro_ap": best,
                "n_parameters": sum(x.numel() for x in self.model.parameters())}


@register
class DRGCNBaseline(NeuralBaseline):
    name = "drgcn"

    def build(self, hidden, layers, dropout):
        return DRGCN(hidden, layers, dropout)


@register
class AMEBaseline(NeuralBaseline):
    name = "ame"

    def build(self, hidden, layers, dropout):
        return ExpertFusion(hidden, layers, dropout, "attention")


@register
class CGEBaseline(NeuralBaseline):
    name = "cge"

    def build(self, hidden, layers, dropout):
        return ExpertFusion(hidden, layers, dropout, "concat")


@register
class EFEVDBaseline(NeuralBaseline):
    name = "efevd"

    def build(self, hidden, layers, dropout):
        return EFEVDLike(hidden, layers, dropout)


@register
class SCVHunterBaseline(NeuralBaseline):
    name = "scvhunter"

    def build(self, hidden, layers, dropout):
        return HANLike(hidden, layers, dropout)


@register
class MandoHGTBaseline(NeuralBaseline):
    name = "mando_hgt"

    def build(self, hidden, layers, dropout):
        return HGTLike(hidden, layers, dropout)
