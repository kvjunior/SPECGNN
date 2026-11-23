#!/usr/bin/env python3
"""Pre-registered analyses for RQ2-RQ5 (docs/analysis_plan.md, Sections 5-6).

For every family: paired Wilcoxon tests over (fold, seed) pairs, Holm adjustment,
paired A12, the decision rule, and a 95% cluster-bootstrap CI of the difference
in pooled macro-AP (probabilities averaged over seeds, groups = split groups).
Writes results/analysis/<rq>_<dataset>.json.

    python experiments/analyze.py --dataset dappscan
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.matrix import (  # noqa: E402
    FEATURE_BASELINES,
    FOLDS,
    NEURAL_BASELINES,
    SEEDS,
    SENSITIVITY,
    UNTRAINED_BASELINES,
    all_jobs,
)
from specgnn.data.config import HELD_OUT_DATASETS, SWC_TYPES  # noqa: E402
from specgnn.eval.stats import cluster_bootstrap_diff, compare_family, describe, macro_ap, supports_claim  # noqa: E402
from specgnn.utils import is_complete, load_config, read_json, write_json  # noqa: E402

N_BOOT = 2000
# (variant, mode) pairs; mode "filtered" = with specification filter
FAMILIES = {
    "rq1": {f"c_vs_{b}": (("both", "filtered"), (f"baseline_{b}", "unfiltered"))
            for b in NEURAL_BASELINES + FEATURE_BASELINES},
    "rq2": {"c_vs_b": (("both", "filtered"), ("plain", "filtered")),
            "b_vs_a": (("plain", "filtered"), ("plain", "unfiltered")),
            "c_vs_a": (("both", "filtered"), ("plain", "unfiltered"))},
    "rq3": {"d_vs_a": (("both", "unfiltered"), ("plain", "unfiltered"))},
    "rq4": {"refute_only_vs_both": (("refute_only", "filtered"), ("both", "filtered")),
            "confirm_only_vs_both": (("confirm_only", "filtered"), ("both", "filtered")),
            "rules_features_vs_both": (("rules_features", "filtered"), ("both", "filtered"))},
    "rq5": {"correction_on_vs_off": (("both", "filtered"), ("both_nocorr", "filtered"))},
}
HELDOUT_FAMILY = {"c_vs_b": (("both", "filtered"), ("plain", "filtered")),
                  "c_vs_a": (("both", "filtered"), ("plain", "unfiltered")),
                  "d_vs_a": (("both", "unfiltered"), ("plain", "unfiltered"))}
SECONDARY = (("micro_f1", ("micro", "f1"), None), ("macro_f1", ("macro", "f1"), None),
             ("micro_f1_calibrated", ("micro", "f1"), "calibrated"),
             ("any_vuln_f1", ("any_vulnerability", "f1"), None))


class Registry:
    def __init__(self, results: str, dataset: str):
        self.results, self.dataset = results, dataset
        self.jobs = {}
        for j in all_jobs():
            if j.name.startswith(dataset + "/") and j.name.count("/") == 1:
                self.jobs.setdefault(j.name.split("/")[1], {})[(j.fold, j.seed)] = j
        self._cache = {}

    def run_dir(self, variant, fold, seed):
        job = self.jobs.get(variant, {}).get((fold, seed))
        if job is None:
            return None
        d = os.path.join(self.results, "runs", job.run_id())
        return d if is_complete(d) else None

    def metrics(self, variant, fold, seed):
        d = self.run_dir(variant, fold, seed)
        if d is None:
            return None
        if d not in self._cache:
            self._cache[d] = read_json(os.path.join(d, "metrics.json"))
        return self._cache[d]


def value(m: dict, mode: str, path=("macro", "ap"), sub=None, split="test"):
    if split not in m:
        return float("nan")
    node = m[split][mode]
    if sub:
        node = node[sub]
    for k in path:
        node = node[k]
    return float("nan") if node is None else float(node)


def paired(reg: Registry, a, b, **kw):
    xa, xb, keys = [], [], []
    for seed in SEEDS:
        for fold in FOLDS:
            ma, mb = reg.metrics(a[0], fold, seed), reg.metrics(b[0], fold, seed)
            if ma is None or mb is None:
                continue
            va, vb = value(ma, a[1], **kw), value(mb, b[1], **kw)
            if np.isnan(va) or np.isnan(vb):
                continue
            xa.append(va)
            xb.append(vb)
            keys.append((fold, seed))
    return np.array(xa), np.array(xb), keys


def pooled(reg: Registry, variant, mode, split="test"):
    """uid -> probabilities averaged over the available runs. On the test split each
    uid is predicted once per seed; on a held-out set, by every (fold, seed) model."""
    acc, cnt, y, mask = {}, {}, {}, {}
    key = "probs_filtered" if mode == "filtered" else "probs"
    for seed in SEEDS:
        for fold in FOLDS:
            d = reg.run_dir(variant, fold, seed)
            if d is None or not os.path.exists(os.path.join(d, f"pred_{split}.npz")):
                continue
            z = np.load(os.path.join(d, f"pred_{split}.npz"))
            for i, u in enumerate(z["uids"]):
                acc[u] = acc.get(u, 0) + z[key][i]
                cnt[u] = cnt.get(u, 0) + 1
                y[u], mask[u] = z["y"][i], z["mask"][i]
    return {u: (acc[u] / cnt[u], y[u], mask[u]) for u in acc}


def bootstrap(reg: Registry, a, b, groups: dict, split="test") -> dict:
    pa, pb = pooled(reg, *a, split=split), pooled(reg, *b, split=split)
    uids = sorted(set(pa) & set(pb))
    if not uids:
        return {"n_files": 0}
    A = np.stack([pa[u][0] for u in uids])
    B = np.stack([pb[u][0] for u in uids])
    Y = np.stack([pa[u][1] for u in uids])
    M = np.stack([pa[u][2] for u in uids])
    g = np.array([groups[u] for u in uids])

    def diff(idx):
        return macro_ap(A[idx], Y[idx], M[idx]) - macro_ap(B[idx], Y[idx], M[idx])

    return {"n_files": len(uids), **cluster_bootstrap_diff(diff, g, n_boot=N_BOOT, seed=0)}


def set_n_boot(n: int) -> None:
    """Bootstrap resamples (pre-registered: 2,000; smaller values only for tests)."""
    global N_BOOT
    N_BOOT = n


def family_result(reg: Registry, name: str, pairs: dict, groups: dict, split="test") -> dict:
    data, keys = {}, {}
    for cname, (a, b) in pairs.items():
        xa, xb, k = paired(reg, a, b, split=split)
        if len(xa):
            data[cname] = (xa, xb)
            keys[cname] = k
    comps = compare_family(data) if data else []
    out = {"family": name, "complete": all(len(k) == len(SEEDS) * len(FOLDS) for k in keys.values())
           and len(keys) == len(pairs), "comparisons": []}
    for c in comps:
        a, b = pairs[c.name]
        sec = {}
        for label, path, sub in SECONDARY:
            xa, xb, _ = paired(reg, a, b, path=path, sub=sub, split=split)
            if len(xa):
                sec[label] = {"mean_a": float(xa.mean()), "mean_b": float(xb.mean())}
        out["comparisons"].append({**c.as_dict(), "a": a, "b": b, "supports_claim": supports_claim(c),
                                   "bootstrap_pooled_macro_ap": bootstrap(reg, a, b, groups, split),
                                   "secondary": sec})
    missing = sorted(set(pairs) - set(data))
    if missing:
        out["missing"] = missing
    return out


def exact_only(reg: Registry, held: str, meta: dict) -> dict:
    """Pooled macro-AP per mode with label pairs from approximate category mappings
    removed (SmartBugs "access control" / "denial of service", DeFi cause mappings)."""
    out = {}
    for name, (variant, mode) in {"a": ("plain", "unfiltered"), "b": ("plain", "filtered"),
                                  "c": ("both", "filtered"), "d": ("both", "unfiltered")}.items():
        pool = pooled(reg, variant, mode, split=held)
        uids = sorted(pool)
        if not uids:
            continue
        P = np.stack([pool[u][0] for u in uids])
        Y = np.stack([pool[u][1] for u in uids])
        M = np.stack([pool[u][2] for u in uids]).astype(float)
        approx = np.array([[meta[u]["label_quality"][t] == "approx" for t in SWC_TYPES]
                           for u in uids])
        out[name] = {"all_labels": macro_ap(P, Y, M), "exact_labels": macro_ap(P, Y, M * ~approx),
                     "approx_pairs_removed": int((approx & (M > 0)).sum())}
    return out


def descriptive(reg: Registry, results: str, dataset: str) -> dict:
    """Per-variant means, verdict reliability on test candidates, cost, and sensitivity."""
    out = {"variants": {}, "sensitivity": {}}
    for variant in reg.jobs:
        ms = [reg.metrics(variant, f, s) for s in SEEDS for f in FOLDS]
        ms = [m for m in ms if m is not None]
        if not ms:
            continue
        row = {"runs": len(ms), "modes": {}}
        for mode in ("unfiltered", "filtered", "supported_types"):
            if not all(mode in m["test"] for m in ms):
                continue
            entry = {"macro_ap": describe([value(m, mode) for m in ms]),
                     # AP of a random ranking = positive rate (Saito & Rehmsmeier 2015):
                     # chance-level macro-AP = mean positive rate over the scored types
                     "chance_macro_ap": describe([float(np.mean(
                         [v["support"] / v["n"] for v in m["test"][mode]["per_type"].values()
                          if v["support"] > 0 and v["n"] > v["support"]] or [np.nan]))
                         for m in ms])}
            for label, path, sub in SECONDARY:
                vals = [value(m, mode, path=path, sub=sub) for m in ms
                        if sub is None or sub in m["test"][mode]]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    entry[label] = describe(vals)
            per_type = {}
            for t in SWC_TYPES:
                aps = [m["test"][mode]["per_type"].get(t, {}).get("ap") for m in ms]
                aps = [a for a in aps if a is not None and not np.isnan(a)]
                if aps:
                    sup = [m["test"][mode]["per_type"][t]["support"] for m in ms
                           if t in m["test"][mode]["per_type"]]
                    per_type[t] = {"ap": describe(aps), "support_mean": float(np.mean(sup))}
            entry["per_type"] = per_type
            row["modes"][mode] = entry
        rel = [m["test"].get("verdict_reliability") for m in ms if m["test"].get("verdict_reliability")]
        if rel:
            tot = {k: sum(r[k] for r in rel) for k in ("conf_tp", "conf_fp", "ref_tn", "ref_fn")}
            row["verdict_reliability"] = {
                **tot,
                "confirmed_precision": tot["conf_tp"] / max(tot["conf_tp"] + tot["conf_fp"], 1),
                "refuted_precision": tot["ref_tn"] / max(tot["ref_tn"] + tot["ref_fn"], 1)}
        timings = []
        for s in SEEDS:
            for f in FOLDS:
                d = reg.run_dir(variant, f, s)
                if d:
                    timings.append(read_json(os.path.join(d, "timing.json")))
        def sec(t, *names):
            return sum(t.get(n, {}).get("seconds", 0.0) for n in names)
        row["cost_seconds"] = {
            "supervised": describe([sec(t, "train.supervised", "train") for t in timings]),
            "feedback": describe([sec(t, "train.feedback.candidates", "train.feedback.retrain")
                                  for t in timings]),
            "total": describe([t["total_seconds"] for t in timings])}
        row["inference_ms_per_contract"] = describe([m["inference_ms_per_contract"] for m in ms])
        rounds: dict[int, dict[str, list]] = {}
        for s in SEEDS:
            for f in FOLDS:
                d = reg.run_dir(variant, f, s)
                hp = os.path.join(d, "history.json") if d else None
                if not hp or not os.path.exists(hp):
                    continue
                for k, r in enumerate(read_json(hp).get("feedback_rounds", []), start=1):
                    slot = rounds.setdefault(k, {})
                    for key in ("candidates", "confirmed", "refuted", "unknown", "pseudo_pos",
                                "pseudo_neg", "pseudo_label_accuracy"):
                        if r.get(key) is not None:
                            slot.setdefault(key, []).append(r[key])
        if rounds:
            row["feedback_dynamics"] = {k: {key: float(np.mean(v)) for key, v in slot.items()}
                                        | {"runs": len(slot.get("candidates", []))}
                                        for k, slot in sorted(rounds.items())}
        out["variants"][variant] = row
    if dataset == "dappscan":
        for j in all_jobs():
            if "sensitivity" not in j.tags:
                continue
            d = os.path.join(results, "runs", j.run_id())
            if not is_complete(d):
                continue
            m = read_json(os.path.join(d, "metrics.json"))
            e = out["sensitivity"].setdefault(j.name, {"folds": [], "filtered": [], "unfiltered": []})
            e["folds"].append(j.fold)
            e["filtered"].append(value(m, "filtered"))
            e["unfiltered"].append(value(m, "unfiltered"))
        out["sensitivity_factors"] = {k: v[0] for k, v in SENSITIVITY.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dappscan", choices=["dappscan", "resource2"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="bootstrap resamples (analysis plan: 2000; smaller only for testing)")
    args = ap.parse_args()
    set_n_boot(args.n_boot)
    paths = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]
    absp = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in paths.items()}
    groups = {}
    with open(os.path.join(absp["processed"], f"{args.dataset}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            groups[r["uid"]] = r["split_group"]
    reg = Registry(absp["results"], args.dataset)
    out_dir = os.path.join(absp["results"], "analysis")
    for rq, pairs in FAMILIES.items():
        res = family_result(reg, rq, pairs, groups)
        write_json(os.path.join(out_dir, f"{rq}_{args.dataset}.json"), res)
        state = "complete" if res["complete"] else "INCOMPLETE"
        print(f"\n{rq} ({args.dataset}, {state})")
        for c in res["comparisons"]:
            bs = c["bootstrap_pooled_macro_ap"]
            ci = f"[{bs['low']:+.3f}, {bs['high']:+.3f}]" if bs.get("n_valid") else "n/a"
            print(f"  {c['name']:24s} n={c['n']:2d}  dAP {c['mean_diff']:+.4f}  "
                  f"p_holm {c['p_holm']:.4f}  A12 {c['a12']:.2f} ({c['magnitude']})  "
                  f"CI {ci}  claim={c['supports_claim']}")
    # RQ6: held-out sets, evaluated by every DAppSCAN model
    if args.dataset == "dappscan":
        for held in HELD_OUT_DATASETS:
            meta = {}
            path = os.path.join(absp["processed"], f"{held}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    meta[r["uid"]] = r
            hgroups = {u: r["split_group"] for u, r in meta.items()}
            res = family_result(reg, f"rq6_{held}", HELDOUT_FAMILY, hgroups, split=held)
            res["exact_labels_only"] = exact_only(reg, held, meta)
            write_json(os.path.join(out_dir, f"rq6_{held}.json"), res)
            print(f"\nrq6 {held} ({'complete' if res['complete'] else 'INCOMPLETE'})")
            for c in res["comparisons"]:
                print(f"  {c['name']:8s} n={c['n']:2d}  dAP {c['mean_diff']:+.4f}  "
                      f"p_holm {c['p_holm']:.4f}  A12 {c['a12']:.2f}  claim={c['supports_claim']}")

    # RQ1 untrained baselines (tools, LLM): deterministic, 5 folds -> bootstrap CIs only
    untrained = {}
    for b in UNTRAINED_BASELINES:
        v = f"baseline_{b}"
        ms = [reg.metrics(v, f, 42) for f in FOLDS]
        ms = [m for m in ms if m is not None]
        if not ms:
            continue
        sup = ms[0].get("supported_types", [])
        scored = set()
        for m in ms:
            scored |= {t for t, v in m["test"]["unfiltered"]["per_type"].items()
                       if v.get("support", 0) > 0}
        untrained[b] = {
            "folds": len(ms),
            "n_supported_types": len(sup),
            "n_scored_types": len(scored),
            "macro_ap": describe([value(m, "unfiltered") for m in ms]),
            "macro_ap_supported_types": describe([value(m, "supported_types") for m in ms]),
            "tool_stats": [m.get("tool_stats") for m in ms],
            "bootstrap_c_minus_baseline": bootstrap(reg, ("both", "filtered"), (v, "unfiltered"), groups)}
    write_json(os.path.join(out_dir, f"rq1_untrained_{args.dataset}.json"), untrained)
    write_json(os.path.join(out_dir, f"descriptive_{args.dataset}.json"),
               descriptive(reg, absp["results"], args.dataset))


if __name__ == "__main__":
    main()
