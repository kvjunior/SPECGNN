#!/usr/bin/env python3
"""RQ3 probing (exploratory, analysis plan Section 6): do node embeddings of a
feedback-trained model encode specification verdicts better than those of a plain
model?

For each outer fold (seed 42) and each SWC type, a logistic-regression probe is
trained on the frozen node embeddings of the training folds to predict whether the
specification CONFIRMS that type on the node, and evaluated (ROC-AUC) on the test
fold. Plain and feedback models are compared descriptively. Probes are linear and
regularized, so higher AUC means the information is linearly accessible, not that
the model uses it.

    python experiments/probing.py
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from experiments.matrix import FOLDS, all_jobs  # noqa: E402
from specgnn.data.config import SWC_TYPES  # noqa: E402
from specgnn.eval.stats import describe  # noqa: E402
from specgnn.feedback.verdicts import CONF  # noqa: E402
from specgnn.model.batching import collate  # noqa: E402
from specgnn.training import cv_split, iterate, load_model, load_samples  # noqa: E402
from specgnn.utils import is_complete, load_config, write_json  # noqa: E402

SEED = 42
MAX_TRAIN_NODES = 20000
MIN_POS_TRAIN, MIN_POS_TEST = 20, 5


@torch.no_grad()
def embeddings(model, samples, verdicts):
    """Candidate-node embeddings [M, H] and CONFIRMED indicators [M, 17]."""
    hs, ys = [], []
    for chunk in iterate(samples, 64, shuffle=False):
        b = collate(chunk)
        h = model.encode(b).numpy()
        cand = b.cand.numpy()
        idx = b.batch.numpy()
        for i, s in enumerate(chunk):
            sel = (idx == i) & cand
            hs.append(h[sel])
            ys.append(verdicts[s["uid"]][cand[idx == i]] == CONF)
    return np.concatenate(hs), np.concatenate(ys)


def probe(htr, ytr, hte, yte, rng):
    if len(htr) > MAX_TRAIN_NODES:
        keep = rng.choice(len(htr), MAX_TRAIN_NODES, replace=False)
        htr, ytr = htr[keep], ytr[keep]
    sc = StandardScaler().fit(htr)
    htr, hte = sc.transform(htr), sc.transform(hte)
    out = {}
    for j, t in enumerate(SWC_TYPES):
        if ytr[:, j].sum() < MIN_POS_TRAIN or yte[:, j].sum() < MIN_POS_TEST or yte[:, j].all():
            continue
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(htr, ytr[:, j])
        out[t] = float(roc_auc_score(yte[:, j], clf.decision_function(hte)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    paths = load_config(os.path.join(ROOT, "configs", "paths.yaml"), args.set)["paths"]
    absp = {k: v if os.path.isabs(v) else os.path.join(ROOT, v) for k, v in paths.items()}
    with open(os.path.join(absp["verdicts"], "dappscan.pkl"), "rb") as fh:
        verdicts = pickle.load(fh)
    samples = load_samples(absp["graphs"], "dappscan", safe_cap=2000)
    jobs = {(j.name.split("/")[1], j.fold): j for j in all_jobs()
            if j.name in ("dappscan/plain", "dappscan/both") and j.seed == SEED}
    rng = np.random.default_rng(SEED)
    per_fold = []
    for fold in FOLDS:
        dirs = {v: os.path.join(absp["results"], "runs", jobs[(v, fold)].run_id())
                for v in ("plain", "both")}
        if not all(is_complete(d) and os.path.exists(os.path.join(d, "model.pt"))
                   for d in dirs.values()):
            continue
        train, _, test = cv_split(samples, fold)
        row = {"fold": fold}
        for v, d in dirs.items():
            model, _ = load_model(os.path.join(d, "model.pt"))
            htr, ytr = embeddings(model, train, verdicts)
            hte, yte = embeddings(model, test, verdicts)
            row[v] = probe(htr, ytr, hte, yte, rng)
        per_fold.append(row)
    types = sorted({t for r in per_fold for v in ("plain", "both") for t in r[v]})
    summary = {}
    for t in types:
        pairs = [(r["both"][t], r["plain"][t]) for r in per_fold if t in r["both"] and t in r["plain"]]
        if pairs:
            d = [a - b for a, b in pairs]
            summary[t] = {"folds": len(pairs), "auc_feedback": describe([a for a, _ in pairs]),
                          "auc_plain": describe([b for _, b in pairs]), "difference": describe(d)}
    out = {"exploratory": True, "seed": SEED, "per_fold": per_fold, "summary": summary}
    write_json(os.path.join(absp["results"], "analysis", "probing.json"), out)
    print(f"probed {len(per_fold)} folds, {len(summary)} types")


if __name__ == "__main__":
    main()
