"""Metrics used throughout the paper.

Detection is evaluated per SWC type over the labels that are known for that
type (masked entries are ignored), then aggregated:
  micro       TP/FP/FN summed over all (file, type) pairs
  macro       mean over types with at least one positive in the evaluation set
  any-vuln    file is "vulnerable" if any type is predicted; compared with
              "has at least one known positive label"
Average precision (AP) is threshold-free and reported next to F1, so results do
not hinge on the choice of tau.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from ..data.config import SWC_TYPES


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def detection_metrics(probs: np.ndarray, y: np.ndarray, mask: np.ndarray,
                      tau: float | np.ndarray = 0.5, types: list[str] = SWC_TYPES) -> dict:
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (probs.shape[1],))
    pred = probs > tau
    per_type, TP, FP, FN = {}, 0.0, 0.0, 0.0
    for j, t in enumerate(types):
        m = mask[:, j] > 0
        if not m.any():
            continue
        yt, pt, st = y[m, j] > 0.5, pred[m, j], probs[m, j]
        tp, fp, fn = float((yt & pt).sum()), float((~yt & pt).sum()), float((yt & ~pt).sum())
        p, r, f = _prf(tp, fp, fn)
        ap = float(average_precision_score(yt, st)) if yt.any() and (~yt).any() else float("nan")
        per_type[t] = {"precision": p, "recall": r, "f1": f, "ap": ap,
                       "support": int(yt.sum()), "n": int(m.sum())}
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    mp, mr, mf = _prf(TP, FP, FN)
    scored = [v for v in per_type.values() if v["support"] > 0]
    aps = [v["ap"] for v in scored if not np.isnan(v["ap"])]
    has_label = (mask > 0).any(1)
    y_any = ((y > 0.5) & (mask > 0)).any(1)[has_label]
    p_any = (pred & (mask > 0)).any(1)[has_label]
    ap_, ar_, af_ = _prf(float((y_any & p_any).sum()), float((~y_any & p_any).sum()),
                         float((y_any & ~p_any).sum()))
    return {
        "micro": {"precision": mp, "recall": mr, "f1": mf},
        "macro": {"precision": float(np.mean([v["precision"] for v in scored])) if scored else 0.0,
                  "recall": float(np.mean([v["recall"] for v in scored])) if scored else 0.0,
                  "f1": float(np.mean([v["f1"] for v in scored])) if scored else 0.0,
                  "ap": float(np.mean(aps)) if aps else float("nan")},
        "any_vulnerability": {"precision": ap_, "recall": ar_, "f1": af_,
                              "accuracy": float((y_any == p_any).mean()) if has_label.any() else 0.0},
        "per_type": per_type,
        "tau": tau.tolist(),
    }


def localization_hits(node_probs: list[np.ndarray], node_y: list[np.ndarray],
                      cand: list[np.ndarray], ks=(1, 3)) -> dict:
    """For every (file, type) with annotated nodes: is an annotated node among
    the top-k candidate nodes ranked by p_vt?"""
    hits = {k: 0 for k in ks}
    total = 0
    for p, yy, c in zip(node_probs, node_y, cand, strict=True):
        for j in range(yy.shape[1]):
            pos = np.where((yy[:, j] > 0) & c)[0]
            if not len(pos):
                continue
            total += 1
            order = np.argsort(-np.where(c, p[:, j], -1.0), kind="stable")
            for k in ks:
                hits[k] += bool(set(order[:k]) & set(pos))
    return {f"hit@{k}": (hits[k] / total if total else float("nan")) for k in ks} | {"pairs": total}


TAU_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))     # 0.05, 0.10, ..., 0.95


def calibrate_tau(probs: np.ndarray, y: np.ndarray, mask: np.ndarray, default: float = 0.5,
                  grid: tuple[float, ...] = TAU_GRID) -> np.ndarray:
    """Per-type threshold maximizing F1 on a validation set (default where the
    validation set has no positive of that type). Only used on validation data."""
    taus = np.full(probs.shape[1], default, dtype=float)
    for j in range(probs.shape[1]):
        m = mask[:, j] > 0
        yt = y[m, j] > 0.5
        if not yt.any():
            continue
        best, best_f = default, -1.0
        for t in grid:
            pt = probs[m, j] > t
            tp, fp, fn = (yt & pt).sum(), (~yt & pt).sum(), (yt & ~pt).sum()
            f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f > best_f:
                best, best_f = float(t), f
        taus[j] = best
    return taus
