import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.graph.builder import build_graph
from specgnn.spec.checker import CONFIRMED, REFUTED, UNKNOWN, FileChecker
from specgnn.spec.library import SPECS

SRC = """pragma solidity ^0.4.24;
contract Bank {
    mapping(address => uint) balances;
    mapping(address => mapping(address => uint)) allowed;
    struct Acc { uint balance; }
    mapping(address => Acc) accs;
    address owner;
    uint unusedVar;
    bool locked;
    modifier noReentrant() { require(!locked); locked = true; _; locked = false; }
    modifier onlyOwner() { require(msg.sender == owner); _; }
    function withdraw() public {
        uint bal = balances[msg.sender];
        require(msg.sender.call.value(bal)());
        balances[msg.sender] = 0;
    }
    function withdrawSafe() public {
        uint bal = balances[msg.sender];
        balances[msg.sender] = 0;
        msg.sender.transfer(bal);
    }
    function withdrawGuarded() public noReentrant {
        msg.sender.call.value(balances[msg.sender])();
        balances[msg.sender] = 0;
    }
    function early(uint x) public {
        if (x > 0) { return; }
        msg.sender.call.value(1)();
        balances[msg.sender] = 0;
    }
    function branches(uint x) public {
        if (x > 0) { msg.sender.call.value(1)(); } else { balances[msg.sender] = 1; }
    }
    function viaPointer(uint am) public {
        var acc = accs[msg.sender];
        if (msg.sender.call.value(am)()) { acc.balance -= am; }
    }
    function lucky() public { if (block.timestamp % 2 == 0) { msg.sender.transfer(1); } }
    function auth() public { require(tx.origin == owner); selfdestruct(owner); }
    function proxy(address t, bytes d) public { t.delegatecall(d); }
    function pay(address[] users) public { for (uint i = 0; i < users.length; i++) { users[i].transfer(1); } }
    function approve(address s, uint v) public { allowed[msg.sender][s] = v; }
    function noop(uint a) public { a == 1; }
    function ignoreResult(address t) public { t.call.value(1)(); }
}
"""

CASES = [
    ("withdraw", "SWC-107", CONFIRMED), ("withdraw", "SWC-104", REFUTED),
    ("withdrawSafe", "SWC-107", REFUTED), ("withdrawGuarded", "SWC-107", REFUTED),
    ("withdrawGuarded", "SWC-104", CONFIRMED), ("early", "SWC-107", CONFIRMED),
    ("branches", "SWC-107", REFUTED), ("viaPointer", "SWC-107", CONFIRMED),
    ("lucky", "SWC-116", CONFIRMED), ("lucky", "SWC-120", CONFIRMED),
    ("auth", "SWC-115", CONFIRMED), ("withdraw", "SWC-115", REFUTED),
    ("proxy", "SWC-112", CONFIRMED), ("pay", "SWC-113", CONFIRMED),
    ("pay", "SWC-128", CONFIRMED), ("approve", "SWC-114", CONFIRMED),
    ("noop", "SWC-135", CONFIRMED), ("ignoreResult", "SWC-104", CONFIRMED),
    ("Bank", "SWC-102", CONFIRMED), ("Bank", "SWC-103", CONFIRMED),
    ("unusedVar", "SWC-108", CONFIRMED), ("unusedVar", "SWC-131", CONFIRMED),
]


@pytest.fixture(scope="module")
def setup():
    return build_graph(SRC), FileChecker(SRC)


@pytest.mark.parametrize("name,swc,expected", CASES)
def test_verdicts(setup, name, swc, expected):
    g, fc = setup
    v = fc.check(g, g["names"].index(name), swc)
    assert v.verdict == expected, (name, swc, v)


def test_confirmed_has_witness_lines(setup):
    g, fc = setup
    v = fc.check(g, g["names"].index("withdraw"), "SWC-107")
    assert v.witness == [14, 15]


def test_pinned_pragma_refutes_floating_pragma():
    src = "pragma solidity 0.8.19;\ncontract A { uint public x; }"
    g = build_graph(src)
    assert FileChecker(src).check(g, 0, "SWC-103").verdict == REFUTED


def test_unresolved_inherited_call_cannot_be_refuted():
    src = """pragma solidity 0.4.24;
import "./Base.sol";
contract A is Base { uint x;
  function f() public { helperFromBase(); x = 1; } }"""
    g = build_graph(src)
    v = FileChecker(src).check(g, g["names"].index("f"), "SWC-107")
    assert v.verdict == UNKNOWN


def test_library_has_17_specs():
    assert len(SPECS) == 17


CLASH = """pragma solidity 0.4.24;
contract B { function helper() internal { msg.sender.call.value(1)(); } }
contract A { uint x;
  function helper() private { x = 2; }
  function run() public { helper(); x = 1; } }"""


def test_internal_calls_resolve_within_the_callers_contract():
    g = build_graph(CLASH)
    fc = FileChecker(CLASH)
    run = g["names"].index("run")
    # A.helper only writes state: no external call reachable from A.run
    assert fc.check(g, run, "SWC-107").verdict == REFUTED
    helpers = [i for i, n in enumerate(g["names"]) if n == "helper"]
    a_helper = max(helpers)                         # declared after B.helper
    calls = {tuple(int(v) for v in e) for e in g["edges"]["call"]}
    assert (run, a_helper) in calls and (run, min(helpers)) not in calls


def test_inherited_function_is_resolved_through_the_base():
    src = CLASH.replace("contract A {", "contract A is B {").replace(
        "function helper() private { x = 2; }", "")
    g = build_graph(src)
    assert FileChecker(src).check(g, g["names"].index("run"), "SWC-107").verdict == CONFIRMED
