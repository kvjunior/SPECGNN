"""Write a synthetic run registry for end-to-end tests of the analysis chain.

Every job of the run matrix for one dataset gets a complete run directory in the
exact format written by scripts/05_train.py or scripts/06_run_baseline.py
(config, env, status, metrics, timing, history, pred_*.npz), over the real
processed UIDs and labels. Scores are random with a per-variant signal strength,
and every metric is computed by the real metric functions, so the registry is
internally consistent. Nothing here is a result.
"""
from __future__ import annotations

import json
import os

import numpy as np

from experiments.matrix import all_jobs
from specgnn.data.config import HELD_OUT_DATASETS, SWC_TYPES
from specgnn.eval.metrics import detection_metrics
from specgnn.utils import Run
from specgnn.utils.config import identity_config

T = len(SWC_TYPES)
SIGNAL = {"plain": 0.6, "both": 0.8, "refute_only": 0.85, "confirm_only": 0.7,
          "rules_features": 0.75, "both_nocorr": 0.78}


def _records(processed: str, name: str) -> list[dict]:
    with open(os.path.join(processed, f"{name}.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _labels(recs):
    y = np.array([[1.0 if r["labels"][t] == 1 else 0.0 for t in SWC_TYPES] for r in recs])
    m = np.array([[0.0 if r["labels"][t] is None else 1.0 for t in SWC_TYPES] for r in recs])
    return y, m


def _scores(y, signal, rng):
    logit = signal * (3 * y - 1.5) + rng.normal(0, 1, y.shape) - 2.0
    return 1 / (1 + np.exp(-logit))


def _evaluate(probs, filtered, y, m, baseline: bool):
    out = {"n": len(y), "unfiltered": detection_metrics(probs, y, m, 0.5)}
    out["unfiltered"]["calibrated"] = detection_metrics(probs, y, m, 0.4)
    if baseline:
        out["supported_types"] = detection_metrics(probs, y, m, 0.5)
    else:
        out["filtered"] = detection_metrics(filtered, y, m, 0.5)
        out["filtered"]["calibrated"] = detection_metrics(filtered, y, m, 0.4)
        out["localization"] = {"hit@1": 0.4, "hit@3": 0.6, "pairs": 10}
        out["filter"] = {"suppressed_pairs": 5, "suppressed_above_tau": 2, "predicted_above_tau": 9}
        out["verdict_reliability"] = {"conf_tp": 3, "conf_fp": 30, "ref_tn": 300, "ref_fn": 2,
                                      "unknown": 40, "candidates": 375,
                                      "confirmed_precision": 0.09, "refuted_precision": 0.99}
    return out


def _emit(run, name, rs, signal, variant, baseline, rng):
    y, m = _labels(rs)
    p = _scores(y, signal, rng)
    f = p.copy()
    if not baseline:
        drop = (y == 0) & (rng.random(y.shape) < (0.3 if variant != "plain" else 0.2))
        f[drop] = 0.0
    np.savez_compressed(run.path(f"pred_{name}.npz"), uids=np.array([r["uid"] for r in rs]),
                        probs=p.astype(np.float32), probs_filtered=f.astype(np.float32),
                        y=y.astype(np.int8), mask=m.astype(np.int8))
    return _evaluate(p, f, y, m, baseline)


def build(results: str, processed: str, dataset: str = "dappscan", seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    recs = _records(processed, dataset)
    held = {h: _records(processed, h) for h in HELD_OUT_DATASETS
            if os.path.exists(os.path.join(processed, f"{h}.jsonl"))}
    n = 0
    for job in all_jobs():
        if not job.name.startswith(dataset + "/"):
            continue
        cfg = job.resolved()
        variant = job.name.split("/")[1]
        baseline = variant.startswith("baseline_")
        signal = 0.3 + 0.05 * (sum(map(ord, variant)) % 7) if baseline else SIGNAL[variant]   # stable
        cap = cfg["data"]["safe_cap"]
        test = [r for r in recs if r["fold"] == job.fold
                and (cap is None or r["safe_rank"] is None or r["safe_rank"] < cap)]
        kind = job.run_id().rsplit("-", 1)[0]
        run = Run.open(results, kind, identity_config(cfg))
        assert run is not None and run.id == job.run_id()

        args = (signal, variant, baseline, rng)
        metrics = {"kind": kind, "fold": job.fold, "seed": job.seed,
                   "test": _emit(run, "test", test, *args),
                   "inference_ms_per_contract": 1.0 + rng.random()}
        if baseline:
            metrics["tool_stats"] = {"files": len(test), "ok": len(test), "failure_rate": 0.0}
        if dataset == "dappscan" and cfg["data"].get("evaluate_heldout", False) and \
                "sensitivity" not in job.tags:
            for h, rs in held.items():
                metrics[h] = _emit(run, h, rs, *args)
        history = {"epochs": [], "feedback_rounds": []}
        if not baseline and variant != "plain" and variant != "rules_features":
            for k, ref in enumerate((5000, 300, 40), start=1):
                history["feedback_rounds"].append({"epoch": 5 * k, "candidates": ref + 700,
                                                   "confirmed": 600, "refuted": ref, "unknown": 100,
                                                   "pseudo_pos": 600, "pseudo_neg": ref,
                                                   "pseudo_label_accuracy": 0.95})
        run.save("history.json", history)
        timing = {"train.supervised": {"seconds": 60.0, "calls": 40}, "total_seconds": 80.0}
        if history["feedback_rounds"]:
            timing["train.feedback.retrain"] = {"seconds": 10.0, "calls": 8}
        run.finish(metrics, timing)
        n += 1
    return n
