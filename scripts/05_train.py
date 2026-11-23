#!/usr/bin/env python3
"""Train and evaluate one run = (experiment config, outer fold, seed).

    python scripts/05_train.py --config configs/experiments/plain.yaml --fold 0
    python scripts/05_train.py --config configs/experiments/feedback.yaml --fold 0 \\
        --set train.seed=43 data.dataset=resource2 data.safe_cap=null

A plain run yields modes (a) unfiltered and (b) filtered; a feedback run yields
modes (d) unfiltered and (c) filtered. Results go to the run registry
(results/runs/<kind>-<hash>/): config, environment, history, timing, metrics,
and per-file predictions for the bootstrap analysis. Complete runs are skipped.
"""
import argparse
import os
import pickle
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import HELD_OUT_DATASETS  # noqa: E402
from specgnn.feedback.pseudo_labels import FeedbackConfig  # noqa: E402
from specgnn.model.losses import LossConfig  # noqa: E402
from specgnn.model.rgcn import ModelConfig  # noqa: E402
from specgnn.spec.checker import CheckerConfig  # noqa: E402
from specgnn.training import (  # noqa: E402
    DataConfig,
    TrainConfig,
    calibrate,
    cv_split,
    evaluate,
    filtered_probs,
    load_samples,
    model_config_for,
    predict,
    resolve_device,
    train_model,
    with_verdict_features,
)
from specgnn.utils import Run, Timer, load_config, section  # noqa: E402
from specgnn.utils.config import PathsConfig, identity_config  # noqa: E402


def run_kind(dcfg: DataConfig, fcfg: FeedbackConfig) -> str:
    if fcfg.enabled:
        variant = f"fb_{fcfg.policy}" + ("" if fcfg.correction else "_nocorr")
    elif dcfg.verdict_features:
        variant = "rules_features"
    else:
        variant = "plain"
    return f"{dcfg.dataset}.{variant}"


def load_verdicts(directory: str, dataset: str) -> dict:
    path = os.path.join(directory, f"{dataset}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run scripts/04_precompute_verdicts.py first")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_predictions(run: Run, name: str, pred: dict, verdicts: dict, tau: float) -> None:
    filtered, _ = filtered_probs(pred, verdicts, tau)
    np.savez_compressed(run.path(f"pred_{name}.npz"), uids=np.array(pred["uids"]),
                        probs=pred["graph_probs"].astype(np.float32),
                        probs_filtered=filtered.astype(np.float32),
                        y=pred["y"].astype(np.int8), mask=pred["mask"].astype(np.int8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key.path=value")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--allow-unselected", action="store_true",
                    help="only for experiments/select_config.py (analysis plan, Section 4)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg.setdefault("run", {})["fold"] = args.fold
    paths = section(cfg, "paths", PathsConfig)
    mcfg, lcfg = section(cfg, "model", ModelConfig), section(cfg, "loss", LossConfig)
    tcfg, dcfg = section(cfg, "train", TrainConfig), section(cfg, "data", DataConfig)
    fcfg = section(cfg, "feedback", FeedbackConfig)
    section(cfg, "spec", CheckerConfig)          # validated; the checker ran in stage 04
    status = cfg.get("selection", {}).get("status")
    if status != "done" and not args.allow_unselected:
        raise SystemExit("configuration selection has not been run (configs/selected.yaml is "
                         "pending): run experiments/select_config.py first")
    unknown = set(cfg) - {"paths", "model", "loss", "train", "data", "feedback", "spec", "run",
                          "selection"}
    if unknown:
        raise SystemExit(f"unknown config sections: {sorted(unknown)}")

    kind = run_kind(dcfg, fcfg)
    run = Run.open(paths.results, kind, identity_config(cfg), overwrite=args.overwrite)
    if run is None:
        print(f"[skip] {kind} fold {args.fold}: complete run exists")
        return
    print(f"[run] {run.id}  fold {args.fold}  seed {tcfg.seed}  -> {run.dir}")
    timer = Timer()
    try:
        samples = load_samples(paths.graphs, dcfg.dataset, safe_cap=dcfg.safe_cap)
        verdicts = load_verdicts(paths.verdicts, dcfg.dataset)
        train, val, test = cv_split(samples, args.fold)
        if dcfg.verdict_features:
            train, val, test = (with_verdict_features(x, verdicts) for x in (train, val, test))
        model, history = train_model(train, val, model_config_for(mcfg, dcfg), lcfg, tcfg,
                                     fcfg, verdicts, timer)
        device = resolve_device(tcfg.device)
        with timer("eval.validation"):
            vpred = predict(model, val, lcfg, device)
        cal = calibrate(vpred, verdicts, tcfg.tau)
        with timer("inference.test"):
            tpred = predict(model, test, lcfg, device)
        metrics = {"kind": kind, "fold": args.fold, "seed": tcfg.seed,
                   "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "best_epoch": history.get("best_epoch"), "epochs_run": history["epochs_run"],
                   "val": evaluate(vpred, verdicts, tcfg.tau),
                   "test": evaluate(tpred, verdicts, tcfg.tau, cal),
                   "calibrated_tau": {k: v.tolist() for k, v in cal.items()},
                   "inference_ms_per_contract":
                       1000 * timer.seconds["inference.test"] / max(len(test), 1)}
        save_predictions(run, "test", tpred, verdicts, tcfg.tau)
        if dcfg.evaluate_heldout and dcfg.dataset == "dappscan":
            for name in HELD_OUT_DATASETS:
                held = load_samples(paths.graphs, name, safe_cap=None)
                hv = load_verdicts(paths.verdicts, name)
                if dcfg.verdict_features:
                    held = with_verdict_features(held, hv)
                with timer(f"inference.{name}"):
                    hpred = predict(model, held, lcfg, device)
                metrics[name] = evaluate(hpred, hv, tcfg.tau, cal)
                save_predictions(run, name, hpred, hv, tcfg.tau)
        run.save("history.json", history)
        if args.save_model:
            torch.save({"state_dict": model.state_dict(), "config": history["config"]},
                       run.path("model.pt"))
        run.finish(metrics, timer.summary())
        t = metrics["test"]
        line = (f"[test] unfiltered AP {t['unfiltered']['macro']['ap']:.3f} "
                f"microF1 {t['unfiltered']['micro']['f1']:.3f}")
        if "filtered" in t:
            line += (f" | filtered AP {t['filtered']['macro']['ap']:.3f} "
                     f"microF1 {t['filtered']['micro']['f1']:.3f}")
        print(line)
    except BaseException as exc:
        run.fail(exc)
        traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
