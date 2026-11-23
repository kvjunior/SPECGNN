import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import SWC_TYPES
from specgnn.feedback.filtering import apply_filter, refuted_types
from specgnn.feedback.pseudo_labels import FeedbackConfig, pseudo_labels, select_candidates
from specgnn.feedback.verdicts import CONF, NA, REF, UNK, verdict_table
from specgnn.graph.builder import build_graph
from specgnn.graph.labels import graph_targets, node_targets
from specgnn.model.correction import GatedCorrection
from specgnn.model.losses import LossConfig
from specgnn.model.rgcn import ModelConfig
from specgnn.spec.checker import CheckerConfig, FileChecker
from specgnn.training import (
    TrainConfig,
    cv_split,
    predict,
    train_model,
    verdict_features,
    with_verdict_features,
)

T = len(SWC_TYPES)
J107 = SWC_TYPES.index("SWC-107")

VULN = """pragma solidity ^0.4.24;
contract Bank { mapping(address => uint) balances;
  function withdraw() public { require(msg.sender.call.value(balances[msg.sender])());
                               balances[msg.sender] = 0; }
  function deposit() public payable { balances[msg.sender] += msg.value; } }"""
SAFE = """pragma solidity 0.8.19;
contract Vault { mapping(address => uint) balances;
  function withdraw() external { uint b = balances[msg.sender]; balances[msg.sender] = 0;
                                 payable(msg.sender).transfer(b); } }"""


# ---------------------------------------------------------------- policies
def test_policies():
    cand = np.array([[True, True, True, False]])
    verd = np.array([[CONF, REF, UNK, CONF]], dtype=np.int8)
    t, w = pseudo_labels(cand, verd, FeedbackConfig(policy="both", w_pos=1.5, w_neg=2.0))
    assert t.tolist() == [[1, 0, 0, 0]] and w.tolist() == [[1.5, 2.0, 0, 0]]
    t, w = pseudo_labels(cand, verd, FeedbackConfig(policy="refute_only"))
    assert w.tolist() == [[0, 2.0, 0, 0]] and t.sum() == 0
    t, w = pseudo_labels(cand, verd, FeedbackConfig(policy="confirm_only"))
    assert w.tolist() == [[1.5, 0, 0, 0]] and t.tolist() == [[1, 0, 0, 0]]


def test_unknown_never_produces_a_pseudo_label():
    cand = np.ones((3, T), dtype=bool)
    verd = np.full((3, T), UNK, dtype=np.int8)
    for policy in ("both", "refute_only", "confirm_only"):
        _, w = pseudo_labels(cand, verd, FeedbackConfig(policy=policy))
        assert w.sum() == 0


@pytest.mark.parametrize("kwargs", [{"policy": "all"}, {"every": 0}, {"w_pos": -1}, {"cand_tau": 1.0}])
def test_feedback_config_validation(kwargs):
    with pytest.raises(ValueError):
        FeedbackConfig(**kwargs)


def test_select_candidates_only_on_candidate_nodes():
    p = np.array([[0.9, 0.1], [0.8, 0.7]])
    c = select_candidates(p, np.array([True, False]), 0.5, top1=False)
    assert c.tolist() == [[True, False], [False, False]]
    # with top1 the best candidate node of every type is included even below tau
    c = select_candidates(p, np.array([True, False]), 0.5)
    assert c.tolist() == [[True, True], [False, False]]


def test_uncalibrated_probabilities_still_give_candidates():
    p = np.full((4, T), 0.01)
    p[2, :] = 0.03                                   # all far below tau
    c = select_candidates(p, np.array([True, True, True, False]), 0.5)
    assert c.sum() == T and c[2].all()


def test_filter_and_feedback_use_identical_candidates():
    from specgnn.feedback.filtering import refuted_types
    rng = np.random.default_rng(3)
    p = rng.random((6, T)) * 0.4
    cand = np.array([True, True, False, True, True, False])
    chosen = select_candidates(p, cand, 0.5)
    verd = np.full((6, T), CONF, dtype=np.int8)
    verd[chosen] = REF                               # refute exactly the candidate set
    assert refuted_types(p, cand, verd, 0.5).all()   # filter judges the same nodes


# ---------------------------------------------------------------- filter
def test_filter_suppresses_only_fully_refuted_types():
    p = np.array([[0.9, 0.2], [0.6, 0.1], [0.3, 0.05]])
    cand = np.array([True, True, False])
    verd = np.array([[REF, REF], [REF, CONF], [NA, NA]], dtype=np.int8)
    r = refuted_types(p, cand, verd, tau=0.5)
    # type 0: candidates rows 0,1 (both > tau) all REFUTED -> suppressed
    # type 1: none > tau, top-1 fallback = row 0 (REF) -> suppressed
    assert r.tolist() == [True, True]
    verd[1, 0] = CONF
    assert refuted_types(p, cand, verd, 0.5).tolist() == [False, True]


def test_apply_filter_never_increases_scores():
    rng = np.random.default_rng(0)
    gp = rng.random((4, T))
    nodes = [rng.random((5, T)) for _ in range(4)]
    cands = [np.array([True, True, False, True, True])] * 4
    verds = [rng.integers(0, 4, size=(5, T)).astype(np.int8) for _ in range(4)]
    out, stats = apply_filter(gp, nodes, cands, verds, 0.5)
    assert np.all(out <= gp) and np.all((out == gp) | (out == 0))
    assert stats["suppressed_pairs"] == int((out != gp).sum())


# ---------------------------------------------------------------- verdict cache
def test_verdict_table_matches_checker():
    g = build_graph(VULN)
    tab = verdict_table(g, VULN, CheckerConfig())
    fc = FileChecker(VULN)
    w = g["names"].index("withdraw")
    assert tab[w, J107] == CONF
    assert tab[g["names"].index("deposit"), J107] == REF
    assert fc.check(g, w, "SWC-107").verdict == "CONFIRMED"
    ext = [i for i, t in enumerate(g["node_type"]) if t == 4]       # ext_call nodes
    assert all((tab[i] == NA).all() for i in ext)


def test_verdict_features_one_hot():
    tab = np.array([[CONF, REF, UNK, NA] + [NA] * (T - 4)], dtype=np.int8)
    f = verdict_features(tab)
    assert f.shape == (1, 3 * T) and f.sum() == 3
    assert f[0, 0] == 1 and f[0, T + 1] == 1 and f[0, 2 * T + 2] == 1


# ---------------------------------------------------------------- correction
def test_correction_starts_as_identity_and_blocks_encoder_gradient():
    torch.manual_seed(0)
    corr = GatedCorrection(8, T)
    h = torch.randn(3, 8, requires_grad=True)
    out, g = corr(h, torch.zeros(3, T), torch.ones(3, T))
    assert torch.allclose(out, h.detach())                 # zero-initialized correction
    out.sum().backward()
    assert h.grad is None                                   # stop-gradient into the encoder
    assert g.shape == (3, 8)


# ---------------------------------------------------------------- CV and end-to-end
def _sample(src, positive, uid, fold):
    g = build_graph(src)
    labels = {t: (1 if (t == "SWC-107" and positive) else 0) for t in SWC_TYPES}
    y, ym = graph_targets(labels)
    ny, nm = node_targets(g, labels, None)
    return {"uid": uid, "graph": g, "y": y, "y_mask": ym, "node_y": ny, "node_mask": nm,
            "fold": fold, "split": "train", "safe_rank": None}


def test_cv_split_is_disjoint_and_rotates():
    s = [{"fold": k % 5, "uid": str(k)} for k in range(20)]
    for k in range(5):
        tr, va, te = cv_split(s, k)
        assert {x["fold"] for x in te} == {k} and {x["fold"] for x in va} == {(k + 1) % 5}
        assert not ({x["uid"] for x in tr} & {x["uid"] for x in va + te})
        assert len(tr) + len(va) + len(te) == 20
    with pytest.raises(ValueError):
        cv_split(s, 5)


@pytest.mark.parametrize("policy,correction", [("both", True), ("refute_only", False)])
def test_feedback_training_end_to_end(policy, correction):
    data = [_sample(VULN if i % 2 == 0 else SAFE, i % 2 == 0, f"u{i}", i % 5) for i in range(10)]
    cache = {s["uid"]: verdict_table(s["graph"], VULN if s["y"][J107] else SAFE, CheckerConfig())
             for s in data}
    tr, va, te = cv_split(data, 0)
    fcfg = FeedbackConfig(enabled=True, policy=policy, every=1, mini_epochs=1,
                          cand_tau=0.0, correction=correction)
    tcfg = TrainConfig(epochs=2, patience=5, batch_size=4, device="cpu")
    model, hist = train_model(tr, va, ModelConfig(hidden=16, layers=1, heads=2), LossConfig(),
                              tcfg, fcfg, cache, log=lambda *_: None)
    rounds = hist["feedback_rounds"]
    assert len(rounds) == 2 and all(r["steps"] > 0 for r in rounds)
    assert (hist["n_correction_parameters"] > 0) == correction
    # inference does not read verdicts: predictions are identical with or without pseudo-labels
    a = predict(model, te, LossConfig())["graph_probs"]
    b = predict(model, [{**s, "pl_target": s["node_y"], "pl_weight": s["node_y"]} for s in te],
                LossConfig())["graph_probs"]
    assert np.array_equal(a, b)


def test_feedback_requires_verdicts_and_rules_as_features_widen_input():
    data = [_sample(VULN, True, f"u{i}", i % 5) for i in range(5)]
    tr, va, _ = cv_split(data, 0)
    with pytest.raises(ValueError):
        train_model(tr, va, ModelConfig(hidden=8, layers=1, heads=1), LossConfig(),
                    TrainConfig(epochs=1, device="cpu"), FeedbackConfig(enabled=True), None)
    cache = {s["uid"]: verdict_table(s["graph"], VULN, CheckerConfig()) for s in data}
    wide = with_verdict_features(tr, cache)
    assert wide[0]["extra_x"].shape == (len(wide[0]["graph"]["node_type"]), 3 * T)


def test_verdict_reliability_counts():
    from specgnn.training import verdict_reliability
    p = np.zeros((3, T))
    p[:, J107] = [0.9, 0.8, 0.7]
    ny = np.zeros((3, T))
    ny[0, J107] = 1
    mask = np.ones((3, T))
    mask[2] = 0                                   # third node: truth unknown -> ignored
    verd = np.full((3, T), NA, dtype=np.int8)
    verd[:, J107] = [CONF, REF, CONF]
    pred = {"uids": ["a"], "node_probs": [p], "cand": [np.array([True, True, True])],
            "node_y": [ny], "node_mask": [mask]}
    r = verdict_reliability(pred, {"a": verd}, 0.5)
    assert (r["conf_tp"], r["conf_fp"], r["ref_tn"], r["ref_fn"]) == (1, 0, 1, 0)
    assert r["confirmed_precision"] == 1.0 and r["refuted_precision"] == 1.0
