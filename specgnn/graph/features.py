"""Node and edge vocabulary of the SpecGNN code graph.

Every feature is computed from the tree-sitter syntax tree, never from regular
expressions over raw text, so comments and string contents cannot trigger a
feature and renaming identifiers does not change any feature value (no feature
depends on an identifier's spelling, except the Solidity built-ins listed in
`builder.py`, e.g. `block.timestamp`, `tx.origin`).

The feature vector of a node has one block per node type; blocks belonging to
other node types are zero. The node type itself is a separate one-hot vector.
"""
from __future__ import annotations

NODE_TYPES = ["contract", "function", "modifier", "state_var", "ext_call", "event"]

# Each relation also gets an inverse relation in the model (8 relations total).
EDGE_TYPES = [
    "control_flow",   # function -> its 1st external call -> next call ...; call -> state var written after it
    "data_flow",      # function -> state var it writes; state var -> function that reads it
    "call",           # function -> internal function / modifier / event it calls, invokes or emits
    "structure",      # contract -> member; contract -> base contract declared in the same file
]

FUNCTION_FEATURES = [
    # signature
    "vis_public", "vis_external", "vis_internal", "vis_private", "vis_missing",
    "mut_payable", "mut_view", "mut_pure", "is_constructor", "is_fallback_receive",
    "n_modifiers", "has_sender_guard_modifier", "has_lock_modifier",
    # body size
    "log_statements", "log_params",
    # environment reads
    "uses_block_timestamp", "uses_block_number", "uses_tx_origin", "uses_msg_value",
    "uses_blockhash_or_difficulty", "uses_gasleft",
    # external interaction
    "n_low_level_calls", "n_delegatecall", "n_send_transfer", "n_high_level_calls",
    "n_selfdestruct", "call_in_loop", "unchecked_call_result",
    # state
    "n_state_writes", "n_state_reads", "state_write_after_call",
    # arithmetic
    "n_arith_ops", "n_arith_ops_unchecked", "n_library_arith_calls",
    # checks and control flow
    "n_require", "n_assert", "n_revert", "n_if", "n_loops", "loop_over_dynamic_length",
    "sender_check", "n_emit", "uses_assembly", "uses_ecrecover", "uses_encode_packed",
    "n_no_effect_statements", "n_unused_locals",
]

CALL_FEATURES = [
    "kind_call", "kind_delegatecall", "kind_staticcall_callcode", "kind_send",
    "kind_transfer", "kind_selfdestruct", "kind_high_level",
    "sends_value", "result_checked", "in_loop", "relative_position", "state_write_after",
    "target_is_param_or_sender",
]

STATE_VAR_FEATURES = [
    "ty_mapping", "ty_address", "ty_integer", "ty_bool", "ty_string_bytes", "ty_array",
    "ty_user_defined", "vis_public", "vis_private", "vis_internal", "vis_missing",
    "is_constant", "is_immutable", "is_initialized", "log_reads", "log_writes", "never_used",
]

CONTRACT_FEATURES = [
    "kind_contract", "kind_library", "kind_interface", "kind_abstract",
    "pragma_floating", "pragma_lt_0_5", "pragma_lt_0_6", "pragma_lt_0_7", "pragma_lt_0_8",
    "pragma_missing", "uses_library_for_uint", "log_functions", "log_state_vars",
    "n_bases", "has_payable_entry",
]

EVENT_FEATURES = ["log_params", "n_indexed"]

BLOCKS = [
    ("function", FUNCTION_FEATURES),   # shared by function and modifier nodes
    ("ext_call", CALL_FEATURES),
    ("state_var", STATE_VAR_FEATURES),
    ("contract", CONTRACT_FEATURES),
    ("event", EVENT_FEATURES),
]

OFFSETS: dict[str, int] = {}
_o = 0
for _name, _feats in BLOCKS:
    OFFSETS[_name] = _o
    _o += len(_feats)
FEATURE_DIM = _o
INPUT_DIM = FEATURE_DIM + len(NODE_TYPES)

BLOCK_OF_TYPE = {"contract": "contract", "function": "function", "modifier": "function",
                 "state_var": "state_var", "ext_call": "ext_call", "event": "event"}

# Node types on which candidates (v, t, p) can be located.
CANDIDATE_NODE_TYPES = ("contract", "function", "modifier", "state_var")


def feature_index(block: str, name: str) -> int:
    feats = dict(BLOCKS)[block]
    return OFFSETS[block] + feats.index(name)
