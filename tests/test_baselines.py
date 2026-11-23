import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.baselines import llm as llm_mod
from specgnn.baselines.base import REGISTRY, make_baseline
from specgnn.baselines.classic import bigrams
from specgnn.baselines.llm import parse_findings
from specgnn.baselines.neural import NeuralBaseline
from specgnn.baselines.tools import ToolBaseline, pick_compiler, satisfies
from specgnn.data.config import SWC_TYPES
from specgnn.graph.builder import build_graph
from specgnn.graph.labels import graph_targets, node_targets
from specgnn.model.batching import collate

T = len(SWC_TYPES)
J107 = SWC_TYPES.index("SWC-107")
VULN = """pragma solidity ^0.4.24;
contract Bank { mapping(address => uint) balances;
  function withdraw() public { require(msg.sender.call.value(balances[msg.sender])());
                               balances[msg.sender] = 0; } }"""
SAFE = """pragma solidity 0.8.19;
contract Vault { uint total; function add(uint a) external { total += a; } }"""


def _sample(src, pos, uid):
    g = build_graph(src)
    labels = {t: (1 if t == "SWC-107" and pos else 0) for t in SWC_TYPES}
    y, ym = graph_targets(labels)
    ny, nm = node_targets(g, labels, None)
    return {"uid": uid, "graph": g, "y": y, "y_mask": ym, "node_y": ny, "node_mask": nm,
            "src": src, "fold": 0, "safe_rank": None}


DATA = [_sample(VULN if i % 2 == 0 else SAFE, i % 2 == 0, f"u{i}") for i in range(8)]


def test_registry_lists_all_baselines():
    make_baseline("feature_lr", 0, None, {})           # imports every module
    assert set(REGISTRY) == {"drgcn", "ame", "cge", "efevd", "scvhunter", "mando_hgt",
                             "contractward", "feature_lr", "slither", "mythril", "llm_auditor"}


# ---------------------------------------------------------------- compilers and tools
@pytest.mark.parametrize("version,spec,ok", [
    ("0.4.26", "^0.4.24", True), ("0.5.0", "^0.4.24", False), ("0.4.23", "^0.4.24", False),
    ("0.7.6", ">=0.6.0 <0.8.0", True), ("0.8.0", ">=0.6.0 <0.8.0", False),
    ("0.8.19", "0.8.19", True), ("0.8.20", "=0.8.19", False), ("0.6.12", ">0.5.0", True),
    ("0.4.24", "^0.4.0 || ^0.8.0", True), ("0.8.3", "^0.4.0 || ^0.8.0", True),
    ("0.6.1", "^0.4.0 || ^0.8.0", False), ("0.5.17", "~0.5.0", True)])
def test_pragma_satisfaction(version, spec, ok):
    assert satisfies(version, spec) is ok


def test_pick_compiler_prefers_newest_satisfying():
    comp = {"0.4.24": "a", "0.4.26": "b", "0.8.19": "c"}
    assert pick_compiler(VULN, comp) == ("0.4.26", "b")
    assert pick_compiler("pragma solidity ^0.6.0; contract A {}", comp) is None
    assert pick_compiler("contract A {}", comp) == ("0.8.19", "c")       # no pragma: newest


class FakeTool(ToolBaseline):
    name = "fake_tool"
    tool = "fake"
    supported_types = ("SWC-107", "SWC-103")
    calls = 0

    def run_one(self, path, version, binary):
        FakeTool.calls += 1
        if "Vault" in open(path).read():
            return "compile_error", set()
        return "ok", {"SWC-107", "SWC-999"}


def test_tool_mapping_caching_and_failures(tmp_path):
    files = {}
    for s in DATA[:2]:
        p = tmp_path / f"{s['uid']}.sol"
        p.write_text(s["src"])
        files[s["uid"]] = str(p)
    tool = FakeTool(source=lambda s: s["src"], cache_dir=str(tmp_path / "cache"),
                    artifacts=str(tmp_path / "none"))
    tool.compilers = {"0.4.26": "x", "0.8.19": "y"}
    tool.path_of = lambda s: files[s["uid"]]
    scores = tool.predict(DATA[:2])
    assert scores[0, J107] == 1 and scores[0].sum() == 1          # SWC-999 dropped
    assert scores[1].sum() == 0
    assert tool.last_stats["ok"] == 1 and tool.last_stats["compile_error"] == 1
    tool.predict(DATA[:2])
    assert FakeTool.calls == 2                                    # second pass fully cached


# ---------------------------------------------------------------- LLM
def test_parse_findings_robust():
    assert parse_findings('{"findings": [{"swc": "SWC-107", "confidence": 0.8}]}') == {"SWC-107": 0.8}
    assert parse_findings('Here: {"findings": [{"swc": "swc107", "confidence": 7}]}') == {"SWC-107": 1.0}
    assert parse_findings('{"findings": [{"swc": "SWC-999"}]}') == {}
    assert parse_findings("no json") == {}
    assert parse_findings('{"findings": []}') == {}


def test_llm_predict_with_mocked_api_and_cache(tmp_path, monkeypatch):
    calls = []

    def fake_call(self, prompt):
        calls.append(prompt)
        answer = '{"findings": [{"swc": "SWC-107", "confidence": 0.9}]}'
        return {"content": [{"type": "text", "text": answer}],
                "usage": {"input_tokens": 100, "output_tokens": 10}}

    monkeypatch.setattr(llm_mod.LLMAuditorBaseline, "_call", fake_call)
    b = make_baseline("llm_auditor", 0, lambda s: s["src"], {"cache_dir": str(tmp_path)})
    s1 = b.predict(DATA[:2])
    assert np.isclose(s1[0, J107], 0.9) and b.last_stats["called"] == 2
    s2 = b.predict(DATA[:2])
    assert np.array_equal(s1, s2) and b.last_stats["cached"] == 2 and len(calls) == 2


# ---------------------------------------------------------------- neural
NEURAL = ["drgcn", "ame", "cge", "efevd", "scvhunter", "mando_hgt"]


@pytest.mark.parametrize("name", NEURAL)
def test_neural_shapes_and_batch_isolation(name):
    b = make_baseline(name, 0, None, {"hidden": 16, "layers": 2, "device": "cpu"})
    torch.manual_seed(0)
    model = b.build(16, 2, 0.0).eval()
    out = model(collate(DATA[:3]))
    assert out["graph_logits"].shape == (3, T)
    alone = model(collate(DATA[:1]))["graph_logits"]
    assert torch.allclose(alone[0], out["graph_logits"][0], atol=1e-5)


@pytest.mark.parametrize("name", NEURAL)
def test_neural_fit_predict(name):
    b = make_baseline(name, 0, None, {"hidden": 16, "layers": 1, "epochs": 2, "device": "cpu"})
    hist = b.fit(DATA[:6], DATA[6:])
    scores = b.predict(DATA)
    assert scores.shape == (8, T) and len(hist["epochs"]) == 2
    assert np.all((scores >= 0) & (scores <= 1))


def test_unknown_neural_parameter_rejected():
    with pytest.raises(ValueError):
        make_baseline("drgcn", 0, None, {"hiden": 16})
    assert issubclass(REGISTRY["drgcn"], NeuralBaseline)


# ---------------------------------------------------------------- classic
def test_bigrams():
    assert bigrams(["a", "b", "c"]) == ["a b", "b c"] and bigrams(["a"]) == []


@pytest.mark.parametrize("name", ["feature_lr", "contractward"])
def test_classic_fit_predict(name):
    b = make_baseline(name, 0, lambda s: s["src"], {"n_estimators": 20, "min_df": 1}
                      if name == "contractward" else {})
    info = b.fit(DATA[:6], DATA[6:])
    assert info["fitted_types"] == ["SWC-107"]          # the only type with both classes
    scores = b.predict(DATA)
    assert scores[:, J107][0::2].mean() > scores[:, J107][1::2].mean()


def test_scores_validation_and_unsupported_types_zeroed():
    tool = make_baseline("slither", 0, None, {})
    s = tool._check(np.full((2, T), 0.5), 2)
    assert s[:, ~tool.supported_mask()].sum() == 0
    with pytest.raises(ValueError):
        tool._check(np.full((2, T), 1.5), 2)
