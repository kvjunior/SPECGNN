#!/usr/bin/env python3
"""Train/run and evaluate one baseline for one (outer fold, seed).

    python scripts/06_run_baseline.py --config configs/baselines/drgcn.yaml --fold 0 \\
        --set baseline.seed=43

Same protocol, registry and output format as scripts/05_train.py: metrics on the
full 17-type label space and on the baseline's supported types, per-file
predictions for the bootstrap, held-out evaluation for DAppSCAN models.
"""
import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.baselines.base import make_baseline  # noqa: E402
from specgnn.data.config import HELD_OUT_DATASETS, RAW_SUBDIRS  # noqa: E402
from specgnn.data.normalize import read_source  # noqa: E402
from specgnn.eval.metrics import calibrate_tau, detection_metrics  # noqa: E402
from specgnn.training import DataConfig, cv_split, load_samples  # noqa: E402
from specgnn.utils import Run, Timer, load_config, section  # noqa: E402
from specgnn.utils.config import PathsConfig, identity_config  # noqa: E402


@dataclass
class BaselineConfig:
    name: str = "drgcn"
    params: dict = field(default_factory=dict)
    seed: int = 42


def evaluate(scores, samples, supported, tau=0.5, calibrated=None):
    y = np.stack([s["y"] for s in samples])
    m = np.stack([s["y_mask"] for s in samples])
    full = detection_metrics(scores, y, m, tau)
    if calibrated is not None:
        full["calibrated"] = detection_metrics(scores, y, m, calibrated)
    sm = m * supported[None, :]
    return {"n": len(samples), "unfiltered": full,
            "supported_types": detection_metrics(scores, y, sm, tau)}


def save_predictions(run, name, samples, scores):
    np.savez_compressed(run.path(f"pred_{name}.npz"), uids=np.array([s["uid"] for s in samples]),
                        probs=scores.astype(np.float32), probs_filtered=scores.astype(np.float32),
                        y=np.stack([s["y"] for s in samples]).astype(np.int8),
                        mask=np.stack([s["y_mask"] for s in samples]).astype(np.int8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-unselected", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    cfg.setdefault("run", {})["fold"] = args.fold
    paths, dcfg = section(cfg, "paths", PathsConfig), section(cfg, "data", DataConfig)
    bcfg = section(cfg, "baseline", BaselineConfig)
    status = cfg.get("selection", {}).get("status")
    if status not in ("done", "not_applicable") and not args.allow_unselected:
        raise SystemExit(f"{bcfg.name}: hyperparameter selection pending "
                         "(run experiments/select_baselines.py first)")
    unknown = set(cfg) - {"paths", "data", "baseline", "selection", "run"}
    if unknown:
        raise SystemExit(f"unknown config sections: {sorted(unknown)}")

    def root(ds):
        return os.path.join(paths.raw, RAW_SUBDIRS[ds])

    kind = f"{dcfg.dataset}.baseline_{bcfg.name}"
    run = Run.open(paths.results, kind, identity_config(cfg), overwrite=args.overwrite)
    if run is None:
        print(f"[skip] {kind} fold {args.fold}: complete run exists")
        return
    timer = Timer()
    try:
        current = {"ds": dcfg.dataset}

        def source(s):
            return read_source(os.path.join(root(current["ds"]), s["rel_path"]))

        baseline = make_baseline(bcfg.name, bcfg.seed, source, bcfg.params)
        if hasattr(baseline, "path_of"):
            baseline.path_of = lambda s: os.path.abspath(os.path.join(root(current["ds"]), s["rel_path"]))
        samples = load_samples(paths.graphs, dcfg.dataset, safe_cap=dcfg.safe_cap)
        train, val, test = cv_split(samples, args.fold)
        with timer("train"):
            history = baseline.fit(train, val)
        supported = baseline.supported_mask()
        calibrated, val_metrics = None, None
        if baseline.kind == "learned":            # tools and LLMs are not tuned on validation
            with timer("eval.validation"):
                vs = baseline.predict(val)
            calibrated = calibrate_tau(vs, np.stack([s["y"] for s in val]),
                                       np.stack([s["y_mask"] for s in val]))
            val_metrics = evaluate(vs, val, supported)
        with timer("inference.test"):
            ts = baseline.predict(test)
        metrics = {"kind": kind, "baseline": bcfg.name, "baseline_kind": baseline.kind,
                   "fold": args.fold, "seed": bcfg.seed,
                   "supported_types": list(baseline.supported_types),
                   "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "val": val_metrics,
                   "test": evaluate(ts, test, supported, calibrated=calibrated),
                   "inference_ms_per_contract": 1000 * timer.seconds["inference.test"] / max(len(test), 1),
                   "tool_stats": getattr(baseline, "last_stats", None)}
        save_predictions(run, "test", test, ts)
        if dcfg.evaluate_heldout and dcfg.dataset == "dappscan":
            for name in HELD_OUT_DATASETS:
                held = load_samples(paths.graphs, name, safe_cap=None)
                current["ds"] = name
                with timer(f"inference.{name}"):
                    hs = baseline.predict(held)
                metrics[name] = evaluate(hs, held, supported, calibrated=calibrated)
                metrics[name]["tool_stats"] = getattr(baseline, "last_stats", None)
                save_predictions(run, name, held, hs)
            current["ds"] = dcfg.dataset
        run.save("history.json", history)
        run.finish(metrics, timer.summary())
    except BaseException as exc:
        run.fail(exc)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
