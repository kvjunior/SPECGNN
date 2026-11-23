"""End-to-end dataset construction."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from . import config as C
from . import loaders
from .neardup import cluster
from .splits import assign_safe_ranks, assign_splits_and_folds, connected_groups

TRAINABLE = ("dappscan", "resource2")


def merge_exact_duplicates(records: list[dict]) -> tuple[list[dict], dict]:
    """Within one dataset, keep one record per normalized hash.

    Labels are merged with OR over annotated values (an audit that flagged a
    file outranks one that did not mention it); None never overrides a value.
    """
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_hash[r["norm_sha256"]].append(r)
    kept, conflicts, removed = [], 0, 0
    for group in by_hash.values():
        keep = min(group, key=lambda r: r["uid"])
        if len(group) > 1:
            removed += len(group) - 1
            for t in C.SWC_TYPES:
                vals = {r["labels"][t] for r in group if r["labels"][t] is not None}
                if len(vals) > 1:
                    conflicts += 1
                if vals:
                    keep["labels"][t] = max(vals)
                    if keep["label_quality"][t] == "unknown":
                        keep["label_quality"][t] = next(
                            r["label_quality"][t] for r in group if r["labels"][t] is not None)
            keep.setdefault("meta", {})["exact_duplicates"] = sorted(
                r["uid"] for r in group if r is not keep)
        kept.append(keep)
    return kept, {"removed": removed, "label_conflicts_resolved_by_OR": conflicts}


def collapse_cross_group_clusters(records: list[dict]) -> tuple[list[dict], dict]:
    """Remove near-duplicate copies that live in different groups (repositories).

    Shared library files (SafeMath, Address, ERC20, ...) are copied into many
    repositories. Left in place, they chain unrelated repositories into one giant
    split group; split across groups, they would leak near-identical code from
    train to test. We keep the copies inside ONE repository -- the one holding
    the most positive copies (ties: lexicographically smallest uid) -- and drop
    the rest. No labels are merged, so a fixed and a vulnerable version of the
    same code are never conflated.
    """
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_cluster[r["cluster"]].append(r)
    drop: set[int] = set()
    dropped_pos = 0
    n_clusters = 0
    for members in by_cluster.values():
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in members:
            groups[r["group_key"]].append(r)
        if len(groups) < 2:
            continue
        n_clusters += 1
        keep_key = min(groups, key=lambda g: (
            -sum(any(v == 1 for v in r["labels"].values()) for r in groups[g]),
            min(r["uid"] for r in groups[g])))
        for g, rs in groups.items():
            if g != keep_key:
                for r in rs:
                    drop.add(id(r))
                    dropped_pos += any(v == 1 for v in r["labels"].values())
    kept = [r for r in records if id(r) not in drop]
    return kept, {"clusters_spanning_groups": n_clusters, "files_dropped": len(drop),
                  "positive_files_dropped": dropped_pos}


def cross_dataset_dedup(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop copies of a cluster from lower-priority datasets."""
    prio = {d: i for i, d in enumerate(C.DEDUP_PRIORITY)}
    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_cluster[r["cluster"]].append(r)
    dropped = []
    keep_ids = set()
    for members in by_cluster.values():
        datasets = {r["dataset"] for r in members}
        if len(datasets) == 1:
            keep_ids.update(id(r) for r in members)
            continue
        winner = min(datasets, key=lambda d: prio[d])
        for r in members:
            if r["dataset"] == winner:
                keep_ids.add(id(r))
            else:
                dropped.append({"uid": r["uid"], "dataset": r["dataset"],
                                "kept_in": winner,
                                "was_positive": any(v == 1 for v in r["labels"].values())})
    return [r for r in records if id(r) in keep_ids], dropped


def build(raw: dict[str, str], out_dir: str, log=print) -> dict:
    """raw: dataset name -> raw root directory. Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    loader_fns = {
        "dappscan": loaders.load_dappscan,
        "resource2": loaders.load_resource2,
        "smartbugs_curated": loaders.load_smartbugs_curated,
        "solidifi": loaders.load_solidifi,
        "chaliasos": loaders.load_chaliasos,
    }
    all_records, exclusions, exact_stats = [], {}, {}
    loaded_counts = {}
    for name, fn in loader_fns.items():
        if name not in raw:
            continue
        recs, excl = fn(raw[name])
        # Process in uid order: filesystem order differs between machines and archive
        # tools, and clustering ids, seeded shuffles and safe ranks depend on order.
        recs.sort(key=lambda r: r["uid"])
        loaded_counts[name] = len(recs)
        log(f"[load] {name}: {len(recs)} files, {len(excl)} excluded")
        if name != "solidifi":   # SolidiFI copies differ only by injected bugs: keep all
            recs, exact_stats[name] = merge_exact_duplicates(recs)
        exclusions[name] = excl
        all_records.extend(recs)

    log(f"[dedup] clustering {len(all_records)} files ...")
    ids, cstats = cluster(
        [r["tokens"] for r in all_records], [r["norm_sha256"] for r in all_records],
        k=C.SHINGLE_SIZE, num_perm=C.MINHASH_PERMUTATIONS, bands=C.LSH_BANDS,
        threshold=C.NEAR_DUP_JACCARD, seed=C.SEED,
        ident_criterion={"set_threshold": C.IDENT_SET_JACCARD,
                         "multiset_threshold": C.IDENT_MULTISET_JACCARD,
                         "min_tokens": C.IDENT_MIN_TOKENS})
    for r, cid in zip(all_records, ids, strict=True):
        r["cluster"] = cid
    all_records, dropped = cross_dataset_dedup(all_records)
    log(f"[dedup] {cstats['clusters']} clusters; {len(dropped)} cross-dataset copies dropped")

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in all_records:
        by_ds[r["dataset"]].append(r)

    collapse_stats = {}
    for name in list(by_ds):
        if name in TRAINABLE:
            by_ds[name], collapse_stats[name] = collapse_cross_group_clusters(by_ds[name])
            log(f"[dedup] {name}: {collapse_stats[name]}")

    for name, recs in by_ds.items():
        if name in TRAINABLE:
            assign_splits_and_folds(recs, C.SPLIT_RATIOS, C.N_FOLDS, C.SEED)
        else:
            for r, g in zip(recs, connected_groups(recs), strict=True):
                r["split_group"], r["split"], r["fold"] = g, "test", None
        if name == "dappscan":
            assign_safe_ranks(recs, C.SEED)
        else:
            for r in recs:
                r["safe_rank"] = None

    # write one JSONL per dataset (no source text, no absolute paths, no tokens)
    for name, recs in by_ds.items():
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in sorted(recs, key=lambda r: r["uid"]):
                out = {k: v for k, v in r.items() if k not in ("tokens", "abs_path")}
                fh.write(json.dumps(out, sort_keys=True) + "\n")

    with open(os.path.join(out_dir, "exclusions.json"), "w") as fh:
        json.dump({d: [{"path": p, "reason": why, "was_positive": pos} for p, why, pos in e]
                   for d, e in exclusions.items()}, fh, indent=1)
    with open(os.path.join(out_dir, "cross_dataset_duplicates.json"), "w") as fh:
        json.dump(dropped, fh, indent=1)

    manifest = {
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
        "loaded_files": loaded_counts,
        "exclusion_counts": {d: dict(Counter(why for _, why, _ in e)) for d, e in exclusions.items()},
        "excluded_positives": {d: sum(1 for *_, pos in e if pos) for d, e in exclusions.items()},
        "within_dataset_exact_duplicates": exact_stats,
        "clustering": cstats,
        "cross_repository_near_duplicates": collapse_stats,
        "cross_dataset_dropped": {f"{a} -> kept in {b}": n for (a, b), n in
                                  Counter((x["dataset"], x["kept_in"]) for x in dropped).items()},
        "final_files": {d: len(v) for d, v in by_ds.items()},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    return manifest
