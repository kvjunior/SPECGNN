import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data.config import SWC_TYPES
from specgnn.graph.builder import build_graph
from specgnn.graph.features import NODE_TYPES, feature_index
from specgnn.graph.labels import graph_targets, node_targets
from specgnn.model.batching import collate
from specgnn.model.losses import LossConfig, generator_loss
from specgnn.model.rgcn import ModelConfig, SpecGNNGenerator, segment_softmax

SRC = """pragma solidity ^0.4.24;
// msg.sender.call.value(1)(); tx.origin  <- in a comment: must be ignored
contract Bank {
    mapping(address => uint) balances;
    uint unusedVar;
    function withdraw() public {
        uint bal = balances[msg.sender];
        require(msg.sender.call.value(bal)());
        balances[msg.sender] = 0;
    }
    function safeWithdraw() public {
        uint bal = balances[msg.sender];
        balances[msg.sender] = 0;
        msg.sender.transfer(bal);
    }
}
"""


def _node(g, name):
    return g["names"].index(name)


def _feat(g, node, block, name):
    return g["x"][node][feature_index(block, name)]


def test_comments_do_not_create_features():
    g = build_graph(SRC)
    assert not any(_feat(g, i, "function", "uses_tx_origin") for i in range(len(g["names"]))
                   if NODE_TYPES[g["node_type"][i]] == "function")


def test_call_before_write_is_captured_by_structure():
    g = build_graph(SRC)
    w, s = _node(g, "withdraw"), _node(g, "safeWithdraw")
    assert _feat(g, w, "function", "state_write_after_call") == 1
    assert _feat(g, s, "function", "state_write_after_call") == 0
    bal = _node(g, "balances")
    cf = {tuple(e) for e in g["edges"]["control_flow"]}
    call_nodes = [i for i in range(len(g["names"])) if NODE_TYPES[g["node_type"][i]] == "ext_call"]
    assert any((c, bal) in cf for c in call_nodes)          # call -> write after it


def test_renaming_identifiers_does_not_change_features():
    renamed = (SRC.replace("balances", "ledger").replace("withdraw", "takeOut")
               .replace("Bank", "Vault").replace("bal", "amt"))
    a, b = build_graph(SRC), build_graph(renamed)
    assert np.array_equal(a["x"], b["x"])
    assert all(np.array_equal(a["edges"][r], b["edges"][r]) for r in a["edges"])


def test_default_visibility_and_unused_state_var():
    g = build_graph(SRC)
    v = _node(g, "unusedVar")
    assert _feat(g, v, "state_var", "vis_missing") == 1
    assert _feat(g, v, "state_var", "never_used") == 1


def test_line_annotation_maps_to_function_node():
    g = build_graph(SRC)
    labels = {t: 0 for t in SWC_TYPES}
    labels["SWC-107"] = 1
    ny, nm = node_targets(g, labels, [{"swc": "SWC-107", "lines": [[8, 8]], "function": None}])
    j = SWC_TYPES.index("SWC-107")
    assert ny[_node(g, "withdraw"), j] == 1 and nm[_node(g, "withdraw"), j] == 1
    assert nm[_node(g, "safeWithdraw"), j] == 0          # unknown, not negative
    k = SWC_TYPES.index("SWC-101")
    assert nm[_node(g, "safeWithdraw"), k] == 1 and ny[_node(g, "safeWithdraw"), k] == 0


def _sample(src, label_types):
    g = build_graph(src)
    labels = {t: (1 if t in label_types else 0) for t in SWC_TYPES}
    y, ym = graph_targets(labels)
    ny, nm = node_targets(g, labels, None)
    return {"graph": g, "y": y, "y_mask": ym, "node_y": ny, "node_mask": nm}


def test_model_forward_backward_and_batch_isolation():
    torch.manual_seed(0)
    s1 = _sample(SRC, {"SWC-107"})
    s2 = _sample("contract A { function f() public { uint x = 1; x = x + 1; } }", set())
    model = SpecGNNGenerator(ModelConfig(hidden=32, layers=2, heads=2)).eval()
    b = collate([s1, s2])
    out = model(b)
    assert out["graph_logits"].shape == (2, len(SWC_TYPES))
    assert out["node_logits"].shape[0] == b.x.shape[0]
    # a graph's output must not depend on the other graphs in the batch
    alone = model(collate([s1]))["graph_logits"]
    assert torch.allclose(alone[0], out["graph_logits"][0], atol=1e-5)
    model.train()
    loss, parts = generator_loss(model(b), b, LossConfig())
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values())


def test_segment_softmax_sums_to_one_per_graph():
    s = torch.randn(7, 3)
    idx = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    a = segment_softmax(s, idx, 3)
    sums = torch.zeros(3, 3).index_add(0, idx, a)
    assert torch.allclose(sums, torch.ones(3, 3), atol=1e-6)


def test_member_name_is_not_a_state_variable_read():
    src = """pragma solidity ^0.8.0;
interface IRouter { function WETH() external view returns (address); }
contract A { address private WETH; IRouter r;
  function init() external { WETH = r.WETH(); } }"""
    g = build_graph(src)
    var_type = NODE_TYPES.index("state_var")
    w = next(i for i, n in enumerate(g["names"]) if n == "WETH" and g["node_type"][i] == var_type)
    # `r.WETH()` names the router's function, not the state variable: no read edge
    assert not any(int(s) == w for s, _ in g["edges"]["data_flow"])
    assert _feat(g, w, "state_var", "log_reads") == 0
