"""Statistical analysis used in the paper (docs/analysis_plan.md, Section 5).

Unit of analysis: one (outer fold, seed) pair. Two methods evaluated on the same
5 folds x 3 seeds give 15 paired observations of a metric.

  * paired_test         two-sided Wilcoxon signed-rank test on the paired values
  * holm                Holm-Bonferroni step-down adjustment within one family
  * a12_paired          probability that method A beats method B on a random pair
                        (ties count 1/2); magnitude labels of Vargha & Delaney (2000)
  * a12                 the classic unpaired Vargha-Delaney A12
  * cluster_bootstrap_diff
                        percentile CI of a metric difference computed on pooled
                        test predictions, resampling *groups* (repositories /
                        near-duplicate clusters) so that dependent files are
                        resampled together
  * average_precision   exact re-implementation of sklearn's AP, ~20x faster,
                        so that bootstrap resampling of AP is affordable
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps

# Vargha & Delaney (2000), thresholds on |A - 0.5| expressed as A >= value.
A12_THRESHOLDS = ((0.71, "large"), (0.64, "medium"), (0.56, "small"))


def a12_magnitude(a: float) -> str:
    d = max(a, 1.0 - a)
    for threshold, label in A12_THRESHOLDS:
        if d >= threshold:
            return label
    return "negligible"


def a12(x: Sequence[float], y: Sequence[float]) -> float:
    """Unpaired Vargha-Delaney A12 = P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        raise ValueError("empty sample")
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (x.size * y.size))


def a12_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Paired A12: fraction of pairs where x beats y (ties count 1/2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired samples must be non-empty and of equal length")
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / x.size)


def paired_test(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples.

    Zero differences are dropped (Wilcoxon's original treatment, scipy's
    ``zero_method="wilcox"``); scipy uses the exact null distribution when there
    are no ties and n <= 50, the normal approximation otherwise. If every
    difference is zero there is no evidence of a difference and p = 1.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired samples must be non-empty and of equal length")
    d = x - y
    if np.all(d == 0):
        return 1.0
    return float(sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input, monotone, <= 1)."""
    p = np.asarray(pvalues, float)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1) | np.isnan(p)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


@dataclass
class Comparison:
    name: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    median_diff: float
    p: float
    p_holm: float | None
    a12: float
    magnitude: str
    wins: int
    ties: int
    losses: int

    def as_dict(self) -> dict:
        return asdict(self)


def compare(name: str, a: Sequence[float], b: Sequence[float]) -> Comparison:
    """Paired comparison of method A against method B on the same (fold, seed) pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    effect = a12_paired(a, b)
    return Comparison(
        name=name, n=int(a.size), mean_a=float(a.mean()), mean_b=float(b.mean()),
        mean_diff=float(d.mean()), median_diff=float(np.median(d)),
        p=paired_test(a, b), p_holm=None, a12=effect, magnitude=a12_magnitude(effect),
        wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def compare_family(pairs: dict[str, tuple[Sequence[float], Sequence[float]]]) -> list[Comparison]:
    """All comparisons of one hypothesis family, with Holm-adjusted p-values."""
    out = [compare(name, a, b) for name, (a, b) in pairs.items()]
    for c, p in zip(out, holm([c.p for c in out]), strict=True):
        c.p_holm = p
    return out


def supports_claim(c: Comparison, alpha: float = 0.05) -> bool:
    """Decision rule of the analysis plan: A is better than B only if the
    Holm-adjusted p-value is below alpha, the median difference is positive,
    and the effect is at least small (paired A12 >= 0.56)."""
    if c.p_holm is None:
        raise ValueError("decision requires a Holm-adjusted p-value (use compare_family)")
    return c.p_holm < alpha and c.median_diff > 0 and c.a12 >= A12_THRESHOLDS[-1][0]


# ---------------------------------------------------------------- metrics on pooled predictions
def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, identical to sklearn.metrics.average_precision_score
    (step-wise interpolation, tied scores share one threshold)."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(scores, float).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    distinct = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]     # last index of each score
    tp = tp[distinct]
    fp = distinct + 1 - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def macro_ap(probs: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean AP over types that have at least one known positive and one known negative."""
    aps = []
    for j in range(probs.shape[1]):
        m = mask[:, j] > 0
        yt = y[m, j] > 0.5
        if yt.any() and (~yt).any():
            aps.append(average_precision(yt, probs[m, j]))
    return float(np.mean(aps)) if aps else float("nan")


def cluster_bootstrap_diff(metric: Callable[[np.ndarray], float], groups: np.ndarray,
                           n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for ``metric(index_array)``, resampling whole groups.

    ``metric`` receives an array of sample indices (with repetitions) and returns
    the statistic of interest, typically metric(method A) - metric(method B) on
    those samples. Resamples on which the statistic is undefined (NaN, e.g. no
    positive drawn) are discarded and counted.
    """
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(uniq.size)]
    rng = np.random.default_rng(seed)
    estimate = metric(np.arange(groups.size))
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([members[g] for g in pick])
        v = metric(idx)
        if not math.isnan(v):
            values.append(v)
    values = np.asarray(values)
    if values.size == 0:
        return {"estimate": estimate, "low": float("nan"), "high": float("nan"),
                "n_valid": 0, "n_boot": n_boot}
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(estimate), "low": float(low), "high": float(high),
            "n_valid": int(values.size), "n_boot": n_boot,
            "p_le_zero": float((values <= 0).mean())}


def describe(values: Sequence[float]) -> dict:
    v = np.asarray(values, float)
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
