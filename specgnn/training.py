"""Training and evaluation of SpecGNN in every mode.

Modes (docs/analysis_plan.md, Section 6) are produced by two trainings:
    plain training     -> evaluated without filter = mode (a), with filter = mode (b)
    feedback training  -> evaluated with filter = mode (c), without filter = mode (d)

File-level score for type t:  p_t(G) = sigmoid((z_graph_t + z_mil_t) / 2), the
mean of the graph-head logit and the smooth maximum of the node logits. Both
heads are trained against file labels; the node term lets candidate-level
feedback act on file-level predictions directly.

One epoch:  supervised pass  ->  feedback round (every K epochs, feedback only)
            ->  validation (unfiltered macro-AP, used for early stopping).
"""
from __future__ import annotations

import math
import os
import pickle
import random
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .data.config import DEFAULT_SAFE_CAP, SWC_TYPES
from .eval.metrics import calibrate_tau, detection_metrics, localization_hits
from .feedback.filtering import apply_filter
from .feedback.pseudo_labels import FeedbackConfig, pseudo_labels, select_candidates
from .feedback.verdicts import CONF, REF, UNK
from .model.batching import collate
from .model.correction import GatedCorrection
from .model.losses import LossConfig, focal_bce, generator_loss, mil_logits
from .model.rgcn import ModelConfig, SpecGNNGenerator
from .utils.repro import set_seed
from .utils.timing import Timer

N_FOLDS = 5
VERDICT_FEATURE_DIM = 3 * len(SWC_TYPES)        # one-hot CONF / REF / UNK per type


@dataclass
class DataConfig:
    dataset: str = "dappscan"
    safe_cap: int | None = DEFAULT_SAFE_CAP
    evaluate_heldout: bool = True
    verdict_features: bool = False     # RQ4 "rules as features": verdicts as node inputs


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    clip: float = 1.0
    patience: int = 15
    seed: int = 42
    tau: float = 0.5
    device: str = "auto"                # "auto" = cuda if available, else cpu
    select_metric: str = "macro_ap"     # threshold-free, robust for rare types


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


# ---------------------------------------------------------------- data
def load_samples(graph_dir: str, dataset: str, splits=None, folds=None,
                 safe_cap: int | None = DEFAULT_SAFE_CAP) -> list[dict]:
    with open(os.path.join(graph_dir, f"{dataset}.pkl"), "rb") as fh:
        data = pickle.load(fh)
    out = []
    for s in data:
        if splits is not None and s["split"] not in splits:
            continue
        if folds is not None and s["fold"] not in folds:
            continue
        if safe_cap is not None and s["safe_rank"] is not None and s["safe_rank"] >= safe_cap:
            continue
        out.append(s)
    return out


def cv_split(samples: list[dict], fold: int, n_folds: int = N_FOLDS
             ) -> tuple[list[dict], list[dict], list[dict]]:
    """Outer fold k: test = k, validation = (k + 1) mod n, training = the rest."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds})")
    if any(s["fold"] is None for s in samples):
        raise ValueError("cross-validation requires samples with a fold (not held-out sets)")
    val_fold = (fold + 1) % n_folds
    train = [s for s in samples if s["fold"] not in (fold, val_fold)]
    val = [s for s in samples if s["fold"] == val_fold]
    test = [s for s in samples if s["fold"] == fold]
    return train, val, test


def verdict_features(table: np.ndarray) -> np.ndarray:
    """[N, 17] verdict codes -> [N, 51] one-hot (CONF, REF, UNK per type)."""
    return np.concatenate([(table == c).astype(np.float32) for c in (CONF, REF, UNK)], axis=1)


def with_verdict_features(samples: list[dict], cache: dict) -> list[dict]:
    return [{**s, "extra_x": verdict_features(cache[s["uid"]])} for s in samples]


def iterate(samples: list[dict], batch_size: int, shuffle: bool, rng: random.Random | None = None):
    idx = list(range(len(samples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [samples[j] for j in idx[i:i + batch_size]]


# ---------------------------------------------------------------- inference
def file_logits(out: dict, batch, lcfg: LossConfig) -> torch.Tensor:
    mil = mil_logits(out["node_logits"], batch, lcfg.mil_temperature)
    return 0.5 * (out["graph_logits"] + mil)


@torch.no_grad()
def predict(model: SpecGNNGenerator, samples: list[dict], lcfg: LossConfig,
            device: str = "cpu", batch_size: int = 64) -> dict:
    model.eval()
    gp, ys, ms, nps, cands = [], [], [], [], []
    for chunk in iterate(samples, batch_size, shuffle=False):
        b = collate(chunk).to(device)
        out = model(b)
        gp.append(torch.sigmoid(file_logits(out, b, lcfg)).cpu().numpy())
        ys.append(b.y.cpu().numpy())
        ms.append(b.y_mask.cpu().numpy())
        node_p = torch.sigmoid(out["node_logits"]).cpu().numpy()
        bidx = b.batch.cpu().numpy()
        cand = b.cand.cpu().numpy()
        for i in range(len(chunk)):
            sel = bidx == i
            nps.append(node_p[sel])
            cands.append(cand[sel])
    t = len(SWC_TYPES)
    return {"uids": [s["uid"] for s in samples],
            "graph_probs": np.concatenate(gp) if gp else np.zeros((0, t)),
            "y": np.concatenate(ys) if ys else np.zeros((0, t)),
            "mask": np.concatenate(ms) if ms else np.zeros((0, t)),
            "node_probs": nps, "cand": cands, "node_y": [s["node_y"] for s in samples],
            "node_mask": [s["node_mask"] for s in samples]}


def filtered_probs(pred: dict, verdict_cache: dict, tau: float) -> tuple[np.ndarray, dict]:
    tables = [verdict_cache[u] for u in pred["uids"]]
    return apply_filter(pred["graph_probs"], pred["node_probs"], pred["cand"], tables, tau)


def calibrate(pred: dict, verdict_cache: dict | None, tau: float) -> dict:
    """Per-type thresholds maximizing F1 on the validation fold, per mode."""
    out = {"unfiltered": calibrate_tau(pred["graph_probs"], pred["y"], pred["mask"], tau)}
    if verdict_cache is not None:
        fp, _ = filtered_probs(pred, verdict_cache, tau)
        out["filtered"] = calibrate_tau(fp, pred["y"], pred["mask"], tau)
    return out


def _metrics(probs: np.ndarray, pred: dict, tau: float, calibrated) -> dict:
    m = detection_metrics(probs, pred["y"], pred["mask"], tau)
    if calibrated is not None:
        m["calibrated"] = detection_metrics(probs, pred["y"], pred["mask"], calibrated)
    return m


def evaluate(pred: dict, verdict_cache: dict | None, tau: float,
             calibration: dict | None = None) -> dict:
    """Metrics without filter and, if verdicts are available, with filter.
    ``calibration`` is the output of :func:`calibrate` on the validation fold."""
    calibration = calibration or {}
    out = {"n": len(pred["uids"]),
           "unfiltered": _metrics(pred["graph_probs"], pred, tau, calibration.get("unfiltered")),
           "localization": localization_hits(pred["node_probs"], pred["node_y"], pred["cand"])}
    if verdict_cache is not None:
        filtered, stats = filtered_probs(pred, verdict_cache, tau)
        out["filtered"] = _metrics(filtered, pred, tau, calibration.get("filtered"))
        out["filter"] = stats
        out["verdict_reliability"] = verdict_reliability(pred, verdict_cache, tau)
    return out


def verdict_reliability(pred: dict, verdict_cache: dict, tau: float) -> dict:
    """RQ4 (descriptive): verdicts on the model's own candidates (node p > tau) whose
    node-level ground truth is known (line-annotated positives; nodes of files that
    are negative for the type). CONFIRMED precision = share of CONFIRMED candidates
    that are true positives; REFUTED precision = share of REFUTED candidates that are
    true negatives."""
    counts = {"conf_tp": 0, "conf_fp": 0, "ref_tn": 0, "ref_fn": 0, "unknown": 0, "candidates": 0}
    for uid, p, c, ny, s_mask in zip(pred["uids"], pred["node_probs"], pred["cand"],
                                     pred["node_y"], pred["node_mask"], strict=True):
        cand = select_candidates(p, c, tau) & (s_mask > 0)
        v = verdict_cache[uid]
        counts["candidates"] += int(cand.sum())
        counts["conf_tp"] += int((cand & (v == CONF) & (ny == 1)).sum())
        counts["conf_fp"] += int((cand & (v == CONF) & (ny == 0)).sum())
        counts["ref_tn"] += int((cand & (v == REF) & (ny == 0)).sum())
        counts["ref_fn"] += int((cand & (v == REF) & (ny == 1)).sum())
        counts["unknown"] += int((cand & (v == UNK)).sum())
    conf = counts["conf_tp"] + counts["conf_fp"]
    ref = counts["ref_tn"] + counts["ref_fn"]
    return {**counts,
            "confirmed_precision": counts["conf_tp"] / conf if conf else None,
            "refuted_precision": counts["ref_tn"] / ref if ref else None}


def _score(metrics: dict, name: str) -> float:
    if name == "macro_ap":
        v = metrics["macro"]["ap"]
        return -1.0 if math.isnan(v) else v
    if name == "macro_f1":
        return metrics["macro"]["f1"]
    if name == "micro_f1":
        return metrics["micro"]["f1"]
    raise ValueError(f"unknown select_metric {name!r}")


# ---------------------------------------------------------------- feedback
def feedback_round(model, train: list[dict], verdict_cache: dict, fcfg: FeedbackConfig,
                   lcfg: LossConfig, device: str) -> tuple[list[dict], dict]:
    """Generate candidates on the training graphs and turn verdicts into pseudo-labels.

    Returns the training samples that received at least one pseudo-label, and
    statistics, including the agreement of pseudo-labels with node ground truth
    where that truth is known.
    """
    pred = predict(model, train, lcfg, device)
    labelled, n_cand = [], 0
    counts = {"confirmed": 0, "refuted": 0, "unknown": 0}
    known = agree = pos = neg = 0
    for s, p, c in zip(train, pred["node_probs"], pred["cand"], strict=True):
        cand = select_candidates(p, c, fcfg.cand_tau, fcfg.top1)
        n_cand += int(cand.sum())
        v = verdict_cache[s["uid"]]
        counts["confirmed"] += int((cand & (v == CONF)).sum())
        counts["refuted"] += int((cand & (v == REF)).sum())
        counts["unknown"] += int((cand & (v == UNK)).sum())
        target, weight = pseudo_labels(cand, v, fcfg)
        has = weight > 0
        if has.any():
            k = has & (s["node_mask"] > 0)
            known += int(k.sum())
            agree += int((k & (target == s["node_y"])).sum())
            pos += int((has & (target == 1)).sum())
            neg += int((has & (target == 0)).sum())
            labelled.append({**s, "pl_target": target, "pl_weight": weight})
    stats = {"candidates": n_cand, **counts, "pseudo_pos": pos, "pseudo_neg": neg,
             "graphs_with_pseudo_labels": len(labelled),
             "pseudo_labels_with_known_truth": known,
             "pseudo_label_accuracy": (agree / known) if known else None}
    return labelled, stats


def mini_retrain(model: SpecGNNGenerator, correction: GatedCorrection | None, opt,
                 samples: list[dict], fcfg: FeedbackConfig, lcfg: LossConfig, clip: float,
                 device: str, rng: random.Random) -> dict:
    """Mini-retraining on pseudo-labelled graphs: supervised loss + weighted
    pseudo-label loss (+ teacher and alignment losses of the gated correction)."""
    model.train()
    params = list(model.parameters())
    if correction is not None:
        correction.train()
        params += list(correction.parameters())
    totals = {"supervised": 0.0, "pseudo": 0.0, "teacher": 0.0, "align": 0.0}
    steps = 0
    for _ in range(fcfg.mini_epochs):
        for chunk in iterate(samples, fcfg.batch_size, shuffle=True, rng=rng):
            b = collate(chunk).to(device)
            h = model.encode(b)
            out = model.heads(h, b)
            sup, _ = generator_loss(out, b, lcfg)
            pl_mask = (b.pl_weight > 0).float()
            pseudo = focal_bce(out["node_logits"], b.pl_target, pl_mask, lcfg, weight=b.pl_weight)
            loss = sup + fcfg.lambda_pl * pseudo
            teacher = align = torch.zeros((), device=h.device)
            if correction is not None:
                rows = pl_mask.sum(1) > 0
                if rows.any():
                    w, t = b.pl_weight[rows], b.pl_target[rows]
                    known = (w > 0).float()
                    conf = known * (t == 1).float()
                    ref = known * (t == 0).float()
                    h_corr, _ = correction(h[rows], conf, ref)
                    teacher = focal_bce(model.node_head(h_corr), t, known, lcfg, weight=w)
                    align = ((h[rows] - h_corr.detach()) ** 2).sum(1).mean()
                    loss = loss + fcfg.lambda_pl * teacher + fcfg.w_align * align
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            for key, val in (("supervised", sup), ("pseudo", pseudo),
                             ("teacher", teacher), ("align", align)):
                totals[key] += float(val.detach())
            steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()} | {"steps": steps}


# ---------------------------------------------------------------- training
def train_model(train: list[dict], val: list[dict], mcfg: ModelConfig, lcfg: LossConfig,
                tcfg: TrainConfig, fcfg: FeedbackConfig | None = None,
                verdict_cache: dict | None = None, timer: Timer | None = None,
                log=print) -> tuple[SpecGNNGenerator, dict]:
    """Train the generator; with ``fcfg.enabled`` also run feedback rounds."""
    fcfg = fcfg or FeedbackConfig()
    timer = timer or Timer()
    if fcfg.enabled and verdict_cache is None:
        raise ValueError("feedback training requires a verdict cache")
    if not train or not val:
        raise ValueError("training and validation sets must be non-empty")
    set_seed(tcfg.seed)
    rng = random.Random(tcfg.seed)
    device = resolve_device(tcfg.device)
    model = SpecGNNGenerator(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    steps_per_epoch = math.ceil(len(train) / tcfg.batch_size)
    total = steps_per_epoch * tcfg.epochs
    warm = max(1, int(tcfg.warmup_frac * total))

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    correction, fb_opt = None, None
    if fcfg.enabled:
        params = list(model.parameters())
        if fcfg.correction:
            correction = GatedCorrection(mcfg.hidden, mcfg.n_types).to(device)
            params += list(correction.parameters())
        fb_opt = torch.optim.AdamW(params, lr=tcfg.lr * fcfg.lr_scale,
                                   weight_decay=tcfg.weight_decay)

    history = {"epochs": [], "feedback_rounds": [],
               "config": {"model": asdict(mcfg), "loss": asdict(lcfg), "train": asdict(tcfg),
                          "feedback": asdict(fcfg)},
               "n_parameters": sum(p.numel() for p in model.parameters()),
               "n_correction_parameters": (sum(p.numel() for p in correction.parameters())
                                           if correction is not None else 0),
               "n_train": len(train), "n_val": len(val), "device": device}
    best, best_state, bad = -float("inf"), None, 0
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        parts = {"graph": 0.0, "node": 0.0, "mil": 0.0}
        n_batches = 0
        with timer("train.supervised"):
            for chunk in iterate(train, tcfg.batch_size, shuffle=True, rng=rng):
                b = collate(chunk).to(device)
                loss, comp = generator_loss(model(b), b, lcfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
                opt.step()
                sched.step()
                for k in parts:
                    parts[k] += comp[k]
                n_batches += 1
        if fcfg.enabled and epoch % fcfg.every == 0:
            with timer("train.feedback.candidates"):
                labelled, fstats = feedback_round(model, train, verdict_cache, fcfg, lcfg, device)
            with timer("train.feedback.retrain"):
                if labelled and fcfg.mini_epochs:
                    rstats = mini_retrain(model, correction, fb_opt, labelled, fcfg, lcfg,
                                          tcfg.clip, device, rng)
                else:
                    rstats = {"steps": 0}
            history["feedback_rounds"].append({"epoch": epoch, **fstats, **rstats})
            log(f"  feedback @ {epoch}: {fstats['candidates']} candidates, "
                f"{fstats['pseudo_pos']} pos / {fstats['pseudo_neg']} neg pseudo-labels, "
                f"accuracy {fstats['pseudo_label_accuracy']}")
        with timer("train.validation"):
            vpred = predict(model, val, lcfg, device)
            vm = detection_metrics(vpred["graph_probs"], vpred["y"], vpred["mask"], tcfg.tau)
        score = _score(vm, tcfg.select_metric)
        rec = {"epoch": epoch, **{f"loss_{k}": v / max(n_batches, 1) for k, v in parts.items()},
               "val_macro_ap": vm["macro"]["ap"], "val_macro_f1": vm["macro"]["f1"],
               "val_micro_f1": vm["micro"]["f1"]}
        history["epochs"].append(rec)
        log(f"epoch {epoch:3d}  loss g/n/m {rec['loss_graph']:.4f}/{rec['loss_node']:.4f}/"
            f"{rec['loss_mil']:.4f}  val AP {rec['val_macro_ap']:.3f}  "
            f"microF1 {rec['val_micro_f1']:.3f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            bad += 1
            if bad >= tcfg.patience:
                log(f"early stop at epoch {epoch} (best {history['best_epoch']})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["epochs_run"] = len(history["epochs"])
    return model, history


def load_model(path: str, device: str = "cpu") -> tuple[SpecGNNGenerator, LossConfig]:
    """Load a model saved by scripts/05_train.py --save-model."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SpecGNNGenerator(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, LossConfig(**cfg["loss"])


def model_config_for(mcfg: ModelConfig, dcfg: DataConfig) -> ModelConfig:
    """Input width grows by the verdict features in the rules-as-features variant."""
    if dcfg.verdict_features:
        return replace(mcfg, input_dim=mcfg.input_dim + VERDICT_FEATURE_DIM)
    return mcfg
