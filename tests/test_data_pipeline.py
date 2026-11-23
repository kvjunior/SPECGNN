import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.data import config as C
from specgnn.data.neardup import cluster
from specgnn.data.normalize import is_interface_only, normalized_hash, normalized_tokens, strip_comments
from specgnn.data.splits import apply_safe_cap, assign_safe_ranks, assign_splits_and_folds

SRC = """// SPDX-License-Identifier: MIT
pragma solidity ^0.4.24;
/* block comment
   spanning lines */
contract Bank {
    mapping(address => uint) balances;
    string url = "http://example.com"; // the // inside the string is not a comment
    function withdraw() public {
        uint bal = balances[msg.sender];
        require(msg.sender.call.value(bal)());
        balances[msg.sender] = 0;
    }
}
"""


def test_comments_removed_strings_kept():
    out = strip_comments(SRC)
    assert "block comment" not in out and "SPDX" not in out
    assert '"http://example.com"' in out
    assert out.count("\n") == SRC.count("\n")          # line numbers preserved


def test_hash_ignores_formatting_and_comments():
    reformatted = SRC.replace("    ", "\t").replace("// the", "/* x */ // y")
    assert normalized_hash(normalized_tokens(SRC)) == normalized_hash(normalized_tokens(reformatted))
    changed = SRC.replace("balances[msg.sender] = 0;", "balances[msg.sender] = 1;")
    assert normalized_hash(normalized_tokens(SRC)) != normalized_hash(normalized_tokens(changed))


def test_interface_detection():
    assert is_interface_only(normalized_tokens("interface IERC20 { function f() external; }"))
    assert not is_interface_only(normalized_tokens("interface I {} contract A is I {}"))


def test_near_duplicates_clustered_unrelated_not():
    base = normalized_tokens(SRC)
    near = normalized_tokens(SRC.replace("withdraw", "withdrawAll"))   # one identifier changed
    other = normalized_tokens("contract Token { uint total; function mint(uint a) public "
                              "{ total += a; } function burn(uint a) public { total -= a; } }")
    toks = [base, near, other]
    ids, _ = cluster(toks, [normalized_hash(t) for t in toks], k=3, num_perm=128,
                     bands=32, threshold=0.6, seed=1)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def _fake_records(n_groups=60, seed=0):
    rng = random.Random(seed)
    recs = []
    for g in range(n_groups):
        for k in range(rng.randint(1, 5)):
            labels = {t: 0 for t in C.SWC_TYPES}
            if rng.random() < 0.4:
                labels[rng.choice(C.SWC_TYPES[:5])] = 1
            recs.append({"uid": f"{g}-{k}", "group_key": f"repo{g}",
                         "cluster": len(recs) if rng.random() < 0.9 else 0,  # cluster 0 spans groups
                         "labels": labels, "label_quality": {}, "meta": {}})
    return recs


def test_no_group_or_cluster_spans_splits_or_folds():
    recs = _fake_records()
    assign_splits_and_folds(recs, C.SPLIT_RATIOS, 5, seed=42)
    for key in ("group_key", "cluster"):
        for attr in ("split", "fold"):
            seen = {}
            for r in recs:
                seen.setdefault(r[key], set()).add(r[attr])
            assert all(len(s) == 1 for s in seen.values()), (key, attr)


def test_split_is_deterministic():
    a, b = _fake_records(), _fake_records()
    assign_splits_and_folds(a, C.SPLIT_RATIOS, 5, seed=42)
    assign_splits_and_folds(b, C.SPLIT_RATIOS, 5, seed=42)
    assert [r["split"] for r in a] == [r["split"] for r in b]


def test_safe_caps_are_nested_and_keep_positives():
    recs = _fake_records(200)
    assign_safe_ranks(recs, seed=42)
    small = {r["uid"] for r in apply_safe_cap(recs, 20)}
    large = {r["uid"] for r in apply_safe_cap(recs, 80)}
    assert small <= large
    positives = {r["uid"] for r in recs if any(v == 1 for v in r["labels"].values())}
    assert positives <= small


def test_identifier_criterion_catches_edited_copies():
    from specgnn.data.neardup import identifier_pairs, identifier_tokens
    base = ("contract Token { mapping(address => uint256) balances; uint256 totalSupply; "
            "string name; uint8 decimals; event Transfer(address from, address to, uint256 amount); "
            "function transfer(address to, uint256 amount) public returns (bool) { "
            "require(balances[msg.sender] >= amount); balances[msg.sender] -= amount; "
            "balances[to] += amount; emit Transfer(msg.sender, to, amount); return true; } "
            "function balanceOf(address owner) public view returns (uint256) { return balances[owner]; } }")
    # reordered functions and changed formatting: breaks 5-gram shingles, not identifier sets
    edited = base.replace("function balanceOf", "function balanceOfX").replace("{ require", "{\n  require")
    other = ("contract Vault { uint256 locked; function lock(uint256 amount) public { locked += amount; } "
             "function unlock() public { locked = 0; } }")
    toks = [normalized_tokens(s) for s in (base, edited, other)]
    assert "balances" in identifier_tokens(toks[0]) and "function" not in identifier_tokens(toks[0])
    pairs = identifier_pairs(toks, set_threshold=0.8, multiset_threshold=0.7, min_tokens=5,
                             num_perm=128, bands=32, seed=0)
    assert (0, 1) in pairs and not any(2 in p for p in pairs)
