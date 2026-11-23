"""Feature-based baselines.

  contractward  ContractWard (Wang et al., TNSE'21) classifies opcode-bigram
                frequency vectors with ensemble learners. DAppSCAN source files
                often do not compile in isolation, so bigrams are taken over the
                normalized source-token stream instead of opcodes (documented
                adaptation); one random forest per SWC type.
  feature_lr    Logistic regression on per-node-type max/mean of the graph's node
                features: a strong, transparent non-neural reference.

Both train one binary classifier per type on the samples whose label for that
type is known; types without both classes in training predict 0.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.config import SWC_TYPES
from ..data.normalize import normalized_tokens
from ..graph.features import NODE_TYPES
from .base import Baseline, register

T = len(SWC_TYPES)


def graph_aggregate(sample: dict) -> np.ndarray:
    g = sample["graph"]
    x, nt = g["x"], g["node_type"]
    parts = []
    for t in range(len(NODE_TYPES)):
        sel = x[nt == t]
        parts.append(sel.max(0) if len(sel) else np.zeros(x.shape[1], dtype=np.float32))
        parts.append(sel.mean(0) if len(sel) else np.zeros(x.shape[1], dtype=np.float32))
    parts.append(np.array([np.log1p(len(nt))], dtype=np.float32))
    return np.concatenate(parts)


def bigrams(tokens: list[str]) -> list[str]:
    """Adjacent token pairs (n - 1 pairs for n tokens)."""
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class PerTypeClassifier(Baseline):
    kind = "learned"

    def features(self, samples: list[dict], fit: bool):
        raise NotImplementedError

    def make_classifier(self):
        raise NotImplementedError

    def fit(self, train: list[dict], val: list[dict]) -> dict:
        X = self.features(train, fit=True)
        Y = np.stack([s["y"] for s in train])
        M = np.stack([s["y_mask"] for s in train])
        self.models = {}
        fitted = []
        for j, t in enumerate(SWC_TYPES):
            rows = M[:, j] > 0
            y = Y[rows, j]
            if rows.sum() == 0 or y.min() == y.max():
                continue
            clf = self.make_classifier()
            clf.fit(X[rows], y.astype(int))
            self.models[j] = clf
            fitted.append(t)
        return {"fitted_types": fitted, "n_features": int(X.shape[1])}

    def predict(self, samples: list[dict]) -> np.ndarray:
        X = self.features(samples, fit=False)
        out = np.zeros((len(samples), T), dtype=np.float32)
        for j, clf in self.models.items():
            out[:, j] = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
        return self._check(out, len(samples))


@register
class ContractWardBaseline(PerTypeClassifier):
    name = "contractward"

    def __init__(self, seed=42, source=None, n_estimators: int = 300, min_df: int = 3,
                 max_features: int = 20000):
        if source is None:
            raise ValueError("contractward needs a source reader")
        super().__init__(seed, source, n_estimators=n_estimators, min_df=min_df,
                         max_features=max_features)
        self.vec = CountVectorizer(analyzer=bigrams, min_df=min_df, max_features=max_features)

    def features(self, samples, fit):
        docs = [normalized_tokens(self.source(s)) for s in samples]
        return self.vec.fit_transform(docs) if fit else self.vec.transform(docs)

    def make_classifier(self):
        return RandomForestClassifier(n_estimators=self.params["n_estimators"], n_jobs=1,
                                      class_weight="balanced_subsample", random_state=self.seed)


@register
class FeatureLRBaseline(PerTypeClassifier):
    name = "feature_lr"

    def __init__(self, seed=42, source=None, C: float = 0.5, max_iter: int = 2000):
        super().__init__(seed, source, C=C, max_iter=max_iter)
        self.scaler = StandardScaler()

    def features(self, samples, fit):
        X = np.stack([graph_aggregate(s) for s in samples])
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def make_classifier(self):
        return LogisticRegression(C=self.params["C"], max_iter=self.params["max_iter"],
                                  class_weight="balanced", random_state=self.seed)
