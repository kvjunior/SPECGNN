"""Leakage-free, label-balanced splitting.

Samples that share a group (same repository/project, same base contract, or the
same near-duplicate cluster) are always assigned to the same split. Within that
constraint, groups are distributed with iterative stratification (Sechidis et
al., ECML-PKDD 2011) adapted to groups: the group containing the rarest
remaining label is placed first, into the split that most needs that label.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .neardup import UnionFind

SAFE = "__safe__"


def connected_groups(records: list[dict]) -> list[int]:
    """Merge items sharing a dataset group key or a near-duplicate cluster."""
    uf = UnionFind(len(records))
    first: dict[tuple[str, str], int] = {}
    for i, r in enumerate(records):
        for key in (("g", r["group_key"]), ("c", str(r["cluster"]))):
            if key in first:
                uf.union(first[key], i)
            else:
                first[key] = i
    roots = [uf.find(i) for i in range(len(records))]
    remap: dict[int, int] = {}
    return [remap.setdefault(x, len(remap)) for x in roots]


def _label_set(r: dict) -> list[str]:
    pos = [t for t, v in r["labels"].items() if v == 1]
    return pos if pos else [SAFE]


def stratified_group_assign(records: list[dict], groups: list[int],
                            ratios: dict[str, float], seed: int) -> dict[int, str]:
    """Return {group id: split name}."""
    rng = random.Random(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    g_labels: dict[int, dict[str, int]] = {}
    totals: dict[str, int] = defaultdict(int)
    for g, idx in members.items():
        cnt: dict[str, int] = defaultdict(int)
        for i in idx:
            for lab in _label_set(records[i]):
                cnt[lab] += 1
        g_labels[g] = cnt
        for lab, n in cnt.items():
            totals[lab] += n

    splits = list(ratios)
    need = {s: {lab: ratios[s] * n for lab, n in totals.items()} for s in splits}
    need_size = {s: ratios[s] * len(records) for s in splits}

    order = list(members)
    rng.shuffle(order)
    # rarest label first; within that, larger groups first
    order.sort(key=lambda g: (min(totals[lab] for lab in g_labels[g]), -len(members[g])))

    # Greedy: put each group where it most reduces the normalized squared
    # deviation from the targets, summed over every label and the split size.
    # Normalizing by each label's total gives rare labels the same weight as
    # frequent ones; the size term keeps the 70/15/15 proportions.
    def delta(s: str, g: int) -> float:
        d = 0.0
        for lab, n in g_labels[g].items():
            t = totals[lab]
            d += ((need[s][lab] - n) ** 2 - need[s][lab] ** 2) / (t * t)
        size, t = len(members[g]), len(records)
        d += ((need_size[s] - size) ** 2 - need_size[s] ** 2) / (t * t)
        return d

    assign: dict[int, str] = {}
    for g in order:
        best = min(splits, key=lambda s: (delta(s, g), rng.random()))
        assign[g] = best
        for lab, n in g_labels[g].items():
            need[best][lab] -= n
        need_size[best] -= len(members[g])
    return assign


def assign_splits_and_folds(records: list[dict], ratios: dict[str, float],
                            n_folds: int, seed: int) -> None:
    """Annotate records in place with split, fold and split_group."""
    groups = connected_groups(records)
    split_of = stratified_group_assign(records, groups, ratios, seed)
    fold_ratios = {str(k): 1.0 / n_folds for k in range(n_folds)}
    fold_of = stratified_group_assign(records, groups, fold_ratios, seed + 1)
    for r, g in zip(records, groups, strict=True):
        r["split_group"] = g
        r["split"] = split_of[g]
        r["fold"] = int(fold_of[g])


def assign_safe_ranks(records: list[dict], seed: int) -> None:
    """Random but fixed rank for label-free ("safe") samples.

    A safe-sample cap C keeps the C lowest-ranked safe samples, so caps are
    nested (cap 1000 is a subset of cap 2000) and the split of every kept
    sample is unchanged -- which makes the cap-sensitivity study clean.
    """
    rng = random.Random(seed)
    safe = [r for r in records if is_safe(r)]
    rng.shuffle(safe)
    for rank, r in enumerate(safe):
        r["safe_rank"] = rank
    for r in records:
        r.setdefault("safe_rank", None)


def is_safe(r: dict) -> bool:
    """No in-scope positive and no annotation of any kind (DAppSCAN semantics)."""
    return (all(v == 0 for v in r["labels"].values())
            and not r.get("meta", {}).get("has_swc_annotation", False))


def apply_safe_cap(records: list[dict], cap: int | None) -> list[dict]:
    if cap is None:
        return list(records)
    return [r for r in records if r.get("safe_rank") is None or r["safe_rank"] < cap]
