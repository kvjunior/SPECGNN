import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.graph.builder import _PARSER, build_graph
from specgnn.robustness.transforms import TRANSFORMS, dead_code, rename, reorder

SRC = """pragma solidity ^0.4.24;
// comment mentioning amount and owner must stay unchanged
contract Bank {
    mapping(address => uint) private balances;
    address public owner;
    uint64 internal level;
    event Paid(address who, uint amount);
    modifier onlyOwner() { require(msg.sender == owner); _; }
    function withdraw(uint amount) public {
        uint bal = balances[msg.sender];
        require(bal >= amount);
        require(msg.sender.call.value(amount)());
        balances[msg.sender] = bal - amount;
        emit Paid({who: msg.sender, amount: amount});
    }
    function median(uint a) private pure returns (uint) { return a; }
    function shadow(uint a) public view returns (uint) {
        uint median = median(a);            // initializer calls the function, not the local
        return median + level;
    }
    function kill() public onlyOwner { selfdestruct(owner); }
}
"""


def _same_graph(a, b):
    return np.array_equal(a["x"], b["x"]) and all(np.array_equal(a["edges"][r], b["edges"][r])
                                                   for r in a["edges"])


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transforms_parse_and_are_deterministic(name):
    out = TRANSFORMS[name](SRC)
    assert not _PARSER.parse(out.encode()).root_node.has_error
    assert out == TRANSFORMS[name](SRC)
    assert "// comment mentioning amount and owner must stay unchanged" in out
    assert "pragma solidity ^0.4.24;" in out


def test_rename_keeps_interface_and_members_and_graph():
    out = rename(SRC)
    for kept in ("function withdraw(", "address public owner", "uint64 internal level",
                 "event Paid(address who, uint amount)", "who: msg.sender", "onlyOwner",
                 "function shadow("):
        assert kept in out, kept
    assert "balances" not in out                      # private state variable renamed
    assert "uint bal" not in out                      # local renamed
    # the local `median` is ambiguous (its initializer calls the private function
    # `median`), so it is not renamed per function; the file-wide private renaming then
    # maps both to one new name, preserving the original shadowing exactly
    import re
    m = re.search(r"uint (__r\d+) = (__r\d+)\(__r\d+\);", out)
    assert m and m.group(1) == m.group(2)
    assert f"function {m.group(1)}(" in out
    assert _same_graph(build_graph(SRC), build_graph(out))


def test_dead_code_is_unreachable_and_reorder_preserves_nodes():
    out = dead_code(SRC)
    assert out.count("if (false) {") == 5             # 4 functions + 1 modifier
    g0, g1 = build_graph(SRC), build_graph(reorder(SRC))
    assert sorted(map(bytes, g0["x"])) == sorted(map(bytes, g1["x"]))
    assert reorder(SRC).index("function kill") < reorder(SRC).index("function withdraw")
