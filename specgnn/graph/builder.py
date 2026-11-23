"""Build the heterogeneous code graph of one Solidity file.

Parsing uses tree-sitter-solidity, an error-tolerant parser that accepts every
Solidity version from 0.4 to 0.8 and partial files with unresolved imports.
Replacing the original regex extractor addresses two limitations raised in
review: keywords inside comments or strings no longer produce matches, and
graph structure does not depend on identifier names.

Scope: one file at a time. Inheritance and calls are resolved among the
contracts declared in the same file; imported contracts are not followed.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import tree_sitter as ts
import tree_sitter_solidity as tss

from .features import BLOCK_OF_TYPE, EDGE_TYPES, FEATURE_DIM, NODE_TYPES, feature_index

warnings.filterwarnings("ignore", message="int argument support is deprecated")
_PARSER = ts.Parser(ts.Language(tss.language()))

WRAPPERS = {"expression", "statement", "parenthesized_expression"}
LOW_LEVEL = {"call": "kind_call", "delegatecall": "kind_delegatecall",
             "staticcall": "kind_staticcall_callcode", "callcode": "kind_staticcall_callcode"}
ARITH_OPS = {"+", "-", "*", "/", "%", "**"}
ARITH_ASSIGN = {"+=", "-=", "*=", "/=", "%="}
LIB_ARITH = {"add", "sub", "mul", "div", "mod", "tryAdd", "trySub", "tryMul", "tryDiv"}
BUILTIN_MEMBERS = {"push", "pop", "encode", "encodePacked", "encodeWithSelector",
                   "encodeWithSignature", "encodeCall", "decode", "concat", "wrap", "unwrap",
                   "length", "selector"}
BUILTIN_OBJECTS = {"abi", "this", "super", "msg", "block", "tx", "type", "bytes", "string"}
BUILTIN_FUNCS = {"require", "assert", "revert", "selfdestruct", "suicide", "keccak256", "sha3",
                 "sha256", "ripemd160", "ecrecover", "blockhash", "gasleft", "addmod", "mulmod"}
LOOPS = {"for_statement", "while_statement", "do_while_statement"}
COMPARISONS = {"==", "!=", "<", ">", "<=", ">="}


def unwrap(n):
    while n is not None and n.type in WRAPPERS and n.named_child_count == 1:
        n = n.named_children[0]
    return n


def text(n) -> str:
    return n.text.decode("utf-8", errors="replace") if n is not None else ""


def walk(n):
    """Pre-order traversal = source order."""
    stack = [n]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.named_children))


def is_value_modifier_of_call(fn) -> bool:
    """True for the `x.call.value(v)` / `x.send.gas(g)` part of `x.call.value(v)(...)`:
    that inner call_expression only configures the outer call and is not a call itself."""
    if fn is None or fn.type != "member_expression":
        return False
    if text(fn.child_by_field_name("property")) not in ("value", "gas"):
        return False
    obj = unwrap(fn.child_by_field_name("object"))
    return obj is not None and obj.type == "member_expression" and \
        text(obj.child_by_field_name("property")) in ("call", "delegatecall", "callcode", "send", "transfer")


def storage_aliases(body, state_vars) -> set[str]:
    """Local variables that point into storage (e.g. `var acc = Acc[msg.sender];`,
    `Info storage i = infos[id];`): writes through them modify state."""
    out = set()
    if body is None:
        return out
    for n in walk(body):
        if n.type != "variable_declaration_statement":
            continue
        decl = next((c for c in n.named_children if c.type == "variable_declaration"), None)
        val = n.child_by_field_name("value")
        if decl is None or val is None:
            continue
        name = decl.child_by_field_name("name")
        ty = decl.child_by_field_name("type")
        base = unwrap(val)
        while base is not None and base.type in ("array_access", "member_expression"):
            base = unwrap(base.child_by_field_name("base") or base.child_by_field_name("object"))
        if base is None or base.type != "identifier" or text(base) not in state_vars:
            continue
        ty_text = text(ty).replace(" ", "") if ty is not None else ""
        if "storage" in text(decl).split() or ty_text in ("", "var") or not re.match(
                r"(u?int\d*|bool|address|addresspayable|bytes\d*|string)$", ty_text):
            out.add(text(name))
    return out


def alias_target(body, name: str) -> str | None:
    for n in walk(body):
        if n.type == "variable_declaration_statement":
            d = next((c for c in n.named_children if c.type == "variable_declaration"), None)
            if d is not None and text(d.child_by_field_name("name")) == name:
                b = unwrap(n.child_by_field_name("value"))
                while b is not None and b.type in ("array_access", "member_expression"):
                    b = unwrap(b.child_by_field_name("base") or b.child_by_field_name("object"))
                return text(b) if b is not None else None
    return None


def _is_member_name(n) -> bool:
    """True for `name` in `obj.name` and for field names in `{name: value}`: such
    identifiers name a member of another object, never a variable in scope."""
    p = n.parent
    if p is None:
        return False
    if p.type == "member_expression":
        prop = p.child_by_field_name("property")
        return prop is not None and prop.id == n.id
    if p.type == "struct_field_assignment":
        name = p.child_by_field_name("name")
        return name is not None and name.id == n.id
    return False


def _log(x: float) -> float:
    return math.log1p(max(x, 0.0)) / math.log(2.0) / 8.0   # ~[0, 1] for counts up to 255


@dataclass
class Graph:
    node_type: list[int] = field(default_factory=list)
    x: list[np.ndarray] = field(default_factory=list)
    spans: list[tuple[int, int]] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    owner: list[int] = field(default_factory=list)       # contract node of each node
    edges: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: {e: [] for e in EDGE_TYPES})

    def add(self, ntype: str, name: str, span, owner: int | None) -> int:
        i = len(self.node_type)
        self.node_type.append(NODE_TYPES.index(ntype))
        self.x.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        self.spans.append(span)
        self.names.append(name)
        self.owner.append(i if owner is None else owner)
        return i

    def set(self, node: int, name: str, value: float) -> None:
        ntype = NODE_TYPES[self.node_type[node]]
        self.x[node][feature_index(BLOCK_OF_TYPE[ntype], name)] = float(value)

    def get(self, node: int, name: str) -> float:
        ntype = NODE_TYPES[self.node_type[node]]
        return float(self.x[node][feature_index(BLOCK_OF_TYPE[ntype], name)])

    def edge(self, rel: str, s: int, d: int) -> None:
        self.edges[rel].append((s, d))

    def to_dict(self) -> dict:
        return {
            "node_type": np.asarray(self.node_type, dtype=np.int64),
            "x": (np.stack(self.x) if self.x else np.zeros((0, FEATURE_DIM), np.float32)),
            "spans": np.asarray(self.spans, dtype=np.int64).reshape(-1, 2),
            "names": list(self.names),
            "owner": np.asarray(self.owner, dtype=np.int64),
            "edges": {r: np.asarray(sorted(set(v)), dtype=np.int64).reshape(-1, 2)
                      for r, v in self.edges.items()},
        }


def _span(n) -> tuple[int, int]:
    return (n.start_point[0] + 1, n.end_point[0] + 1)


def _pragma_info(root) -> dict:
    for n in root.named_children:
        if n.type == "pragma_directive" and "solidity" in text(n):
            spec = text(n).split("solidity", 1)[1].strip(" ;")
            versions = re.findall(r"(\d+)\.(\d+)(?:\.(\d+))?", spec)
            minor = min((int(v[1]) for v in versions if v[0] == "0"), default=None)
            return {"floating": bool(re.search(r"[\^><*~]", spec)) or " - " in spec,
                    "minor": minor, "span": _span(n)}
    return {"floating": False, "minor": None, "span": None}


class _Analyzer:
    """Second pass over one function or modifier body."""

    def __init__(self, g: Graph, fnode: int, decl, state_vars: dict[str, int],
                 functions: dict[str, list[int]], events: dict[str, int],
                 modifiers: dict[str, int], libraries: set[str], has_using: bool):
        self.g, self.f, self.decl = g, fnode, decl
        self.state_vars, self.functions = state_vars, functions
        self.events, self.modifiers = events, modifiers
        self.libraries, self.has_using = libraries, has_using
        self.params = {text(p.child_by_field_name("name")) for p in decl.named_children
                       if p.type == "parameter" and p.child_by_field_name("name")}
        self.body = decl.child_by_field_name("body")
        self.aliases = {a: alias_target(self.body, a) for a in storage_aliases(self.body, state_vars)}
        self.locals: dict[str, int] = {}
        self.order: list[tuple[str, int]] = []       # ("call", node) | ("write", var node)
        self.calls: list[int] = []
        self.c: dict[str, float] = {}
        self.reads: set[int] = set()
        self.writes: set[int] = set()
        self.write_ids: set[int] = set()

    def inc(self, k: str, v: float = 1.0) -> None:
        self.c[k] = self.c.get(k, 0.0) + v

    # -- helpers -----------------------------------------------------------
    def _base_ident(self, n):
        n = unwrap(n)
        while n is not None and n.type in ("array_access", "member_expression"):
            n = unwrap(n.child_by_field_name("base") or n.child_by_field_name("object"))
        return n if n is not None and n.type == "identifier" else None

    def _state_var(self, ident) -> int | None:
        if ident is None:
            return None
        name = text(ident)
        if name in self.aliases:
            return self.state_vars.get(self.aliases[name] or "")
        if name in self.locals or name in self.params:
            return None
        return self.state_vars.get(name)

    def _in_loop(self, n) -> bool:
        p = n.parent
        while p is not None and p != self.body:
            if p.type in LOOPS:
                return True
            p = p.parent
        return False

    def _in_unchecked(self, n) -> bool:
        p = n.parent
        while p is not None and p != self.body:
            if p.type == "block_statement" and p.child_count and p.children[0].type == "unchecked":
                return True
            p = p.parent
        return False

    def _result_used(self, call) -> bool:
        p = call.parent
        while p is not None and p.type in WRAPPERS:
            p = p.parent
        return p is not None and p.type not in ("expression_statement", "block_statement")

    def _mark_write(self, target) -> None:
        ident = self._base_ident(target)
        sv = self._state_var(ident)
        if ident is not None:
            self.write_ids.add(ident.id)
        if sv is not None:
            self.writes.add(sv)
            self.order.append(("write", sv))

    # -- main ----------------------------------------------------------------
    def run(self) -> None:
        if self.body is None:
            return
        for n in walk(self.body):
            if n.type == "variable_declaration":
                nm = n.child_by_field_name("name")
                if nm is not None:
                    self.locals[text(nm)] = nm.id
        for n in walk(self.body):
            t = n.type
            if t == "call_expression":
                self._call(n)
            elif t == "emit_statement":
                ev = text(unwrap(n.child_by_field_name("name")))
                self.inc("n_emit")
                if ev in self.events:
                    self.g.edge("call", self.f, self.events[ev])
            elif t in ("assignment_expression", "augmented_assignment_expression"):
                self._mark_write(n.child_by_field_name("left"))
                op = next((text(c) for c in n.children if not c.is_named), "")
                if t == "augmented_assignment_expression" and op in ARITH_ASSIGN:
                    self._arith(n)
            elif t == "update_expression":
                self._mark_write(n.child_by_field_name("argument"))
                self._arith(n)
            elif t == "unary_expression" and text(n.child_by_field_name("operator")) == "delete":
                self._mark_write(n.child_by_field_name("argument"))
            elif t == "binary_expression":
                op = text(n.child_by_field_name("operator"))
                if op in ARITH_OPS:
                    self._arith(n)
                if op in ("==", "!=") and re.search(r"\b(msg\.sender|tx\.origin)\b", text(n)):
                    self.inc("sender_check")
            elif t == "member_expression":
                self._env(text(n))
            elif t == "identifier":
                name = text(n)
                if name == "now":
                    self.inc("uses_block_timestamp")
                if n.id not in self.write_ids and not _is_member_name(n):
                    sv = self._state_var(n)
                    if sv is not None and n.parent is not None and n.parent.type != "variable_declaration":
                        self.reads.add(sv)
            elif t == "if_statement":
                self.inc("n_if")
            elif t in LOOPS:
                self.inc("n_loops")
                cond = n.child_by_field_name("condition")
                if cond is not None and re.search(r"\.length\b", text(cond)):
                    self.inc("loop_over_dynamic_length")
            elif t == "revert_statement":
                self.inc("n_revert")
            elif t == "assembly_statement":
                self.inc("uses_assembly")
            elif t == "expression_statement":
                e = unwrap(n.named_children[0]) if n.named_child_count else None
                if e is not None and (e.type in ("identifier", "member_expression", "number_literal",
                                                 "boolean_literal", "string_literal")
                                      or (e.type == "binary_expression"
                                          and text(e.child_by_field_name("operator")) in COMPARISONS)):
                    if text(e) != "_":
                        self.inc("n_no_effect_statements")
        # unused locals: declared but never referenced again
        refs: dict[str, int] = {}
        for n in walk(self.body):
            if n.type == "identifier" and not _is_member_name(n):
                refs[text(n)] = refs.get(text(n), 0) + 1
        self.c["n_unused_locals"] = sum(1 for name in self.locals if refs.get(name, 0) <= 1)
        self._finish_calls()

    def _env(self, s: str) -> None:
        if s == "block.timestamp":
            self.inc("uses_block_timestamp")
        elif s == "block.number":
            self.inc("uses_block_number")
        elif s == "tx.origin":
            self.inc("uses_tx_origin")
        elif s == "msg.value":
            self.inc("uses_msg_value")
        elif s in ("block.difficulty", "block.prevrandao", "block.coinbase"):
            self.inc("uses_blockhash_or_difficulty")
        elif s in ("abi.encodePacked",):
            self.inc("uses_encode_packed")

    def _arith(self, n) -> None:
        self.inc("n_arith_ops")
        if self._in_unchecked(n):
            self.inc("n_arith_ops_unchecked")

    def _call(self, n) -> None:
        fn = unwrap(n.child_by_field_name("function"))
        nargs = sum(1 for c in n.named_children if c.type == "call_argument")
        sends_value = False
        # x.call{value: v}(...)
        if fn is not None and fn.type == "struct_expression":
            sends_value = any(text(c.child_by_field_name("name")) == "value"
                              for c in fn.named_children if c.type == "struct_field_assignment")
            fn = unwrap(fn.child_by_field_name("type"))
        # x.call.value(v)(...) / x.call.gas(g)(...)
        if fn is not None and fn.type == "call_expression":
            inner = unwrap(fn.child_by_field_name("function"))
            if inner is not None and inner.type == "member_expression" and \
                    text(inner.child_by_field_name("property")) in ("value", "gas"):
                sends_value = sends_value or text(inner.child_by_field_name("property")) == "value"
                fn = unwrap(inner.child_by_field_name("object"))
            else:
                return   # f()(...) -- handled when visiting the inner call
        if fn is None or is_value_modifier_of_call(fn):
            return
        if fn.type == "identifier":
            name = text(fn)
            if name == "require":
                self.inc("n_require")
            elif name == "assert":
                self.inc("n_assert")
            elif name == "revert":
                self.inc("n_revert")
            elif name in ("selfdestruct", "suicide"):
                self._ext(n, "kind_selfdestruct", False, None)
            elif name == "ecrecover":
                self.inc("uses_ecrecover")
            elif name in ("blockhash",):
                self.inc("uses_blockhash_or_difficulty")
            elif name == "gasleft":
                self.inc("uses_gasleft")
            elif name in self.events:                         # pre-0.4.21 event syntax
                self.inc("n_emit")
                self.g.edge("call", self.f, self.events[name])
            elif name in self.functions:
                for target in self.functions[name]:
                    if target != self.f:
                        self.g.edge("call", self.f, target)
            return
        if fn.type != "member_expression":
            return
        prop = text(fn.child_by_field_name("property"))
        obj = unwrap(fn.child_by_field_name("object"))
        obj_text = text(obj)
        if prop in LOW_LEVEL:
            self._ext(n, LOW_LEVEL[prop], sends_value, obj)
        elif prop == "send" and nargs == 1:
            self._ext(n, "kind_send", True, obj)
        elif prop == "transfer" and nargs == 1:
            self._ext(n, "kind_transfer", True, obj)
        elif obj_text in BUILTIN_OBJECTS or obj_text.split(".")[0] in BUILTIN_OBJECTS \
                or prop in BUILTIN_MEMBERS:
            if obj_text == "abi" and prop == "encodePacked":
                self.inc("uses_encode_packed")
        elif obj_text in self.libraries:
            if prop in LIB_ARITH:
                self.inc("n_library_arith_calls")
        elif prop in LIB_ARITH and self.has_using:
            self.inc("n_library_arith_calls")
        else:
            self._ext(n, "kind_high_level", sends_value, obj)

    def _ext(self, n, kind: str, sends_value: bool, obj) -> None:
        g = self.g
        node = g.add("ext_call", text(n)[:80], _span(n), g.owner[self.f])
        g.set(node, kind, 1)
        g.set(node, "sends_value", float(sends_value))
        g.set(node, "result_checked", float(self._result_used(n)))
        g.set(node, "in_loop", float(self._in_loop(n)))
        base = self._base_ident(obj) if obj is not None else None
        target_user = (base is not None and text(base) in self.params) or \
            (obj is not None and "msg.sender" in text(obj))
        g.set(node, "target_is_param_or_sender", float(target_user))
        self.calls.append(node)
        self.order.append(("call", node))
        if kind in ("kind_call", "kind_delegatecall", "kind_staticcall_callcode"):
            self.inc("n_low_level_calls")
        if kind == "kind_delegatecall":
            self.inc("n_delegatecall")
        if kind in ("kind_send", "kind_transfer"):
            self.inc("n_send_transfer")
        if kind == "kind_high_level":
            self.inc("n_high_level_calls")
        if kind == "kind_selfdestruct":
            self.inc("n_selfdestruct")
        if self._in_loop(n):
            self.c["call_in_loop"] = 1
        if kind in ("kind_call", "kind_delegatecall", "kind_send") and not self._result_used(n):
            self.c["unchecked_call_result"] = 1

    def _finish_calls(self) -> None:
        g = self.g
        prev = self.f
        for i, c in enumerate(self.calls):
            g.edge("control_flow", prev, c)
            prev = c
            g.set(c, "relative_position", (i + 1) / len(self.calls))
        current = None
        for kind, node in self.order:
            if kind == "call":
                current = node
            elif current is not None:
                g.edge("control_flow", current, node)
                g.set(current, "state_write_after", 1)
                self.c["state_write_after_call"] = 1
        for sv in self.writes:
            g.edge("data_flow", self.f, sv)
        for sv in self.reads:
            g.edge("data_flow", sv, self.f)


def build_graph(src: str | bytes) -> dict:
    data = src.encode("utf-8") if isinstance(src, str) else src
    tree = _PARSER.parse(data)
    root = tree.root_node
    g = Graph()
    pragma = _pragma_info(root)

    containers = [n for n in root.named_children
                  if n.type in ("contract_declaration", "library_declaration", "interface_declaration")]
    free_functions = [n for n in root.named_children if n.type == "function_definition"]

    contract_nodes: dict[str, int] = {}
    members: list[tuple[int, object, int]] = []        # (node, decl, contract node)
    contract_state: dict[int, dict[str, int]] = {}
    contract_bases: dict[int, list[str]] = {}
    # name registries per contract node; free functions are visible everywhere
    events: dict[int, dict[str, int]] = {}
    functions: dict[int, dict[str, list[int]]] = {}
    modifiers: dict[int, dict[str, int]] = {}
    free_funcs: dict[str, list[int]] = {}
    libraries: set[str] = set()
    uses_lib_uint: dict[int, bool] = {}

    def add_contract(decl, kind: str, name: str) -> int:
        cn = g.add("contract", name, _span(decl) if decl is not None else (1, 1), None)
        g.set(cn, {"contract": "kind_contract", "library": "kind_library",
                   "interface": "kind_interface", "abstract": "kind_abstract"}[kind], 1)
        contract_nodes.setdefault(name, cn)
        contract_state[cn], contract_bases[cn], uses_lib_uint[cn] = {}, [], False
        return cn

    for decl in containers:
        name = text(decl.child_by_field_name("name"))
        kind = {"library_declaration": "library",
                "interface_declaration": "interface"}.get(decl.type, "contract")
        if kind == "contract" and any(c.type == "abstract" or text(c) == "abstract" for c in decl.children):
            kind = "abstract"
        if kind == "library":
            libraries.add(name)
        cn = add_contract(decl, kind, name)
        for c in decl.children:
            if c.type == "inheritance_specifier":
                contract_bases[cn].append(re.split(r"[\s(]", text(c).strip(), maxsplit=1)[0])
        body = decl.child_by_field_name("body")
        for m in (body.named_children if body is not None else []):
            if m.type == "state_variable_declaration":
                nm = m.child_by_field_name("name") or next(
                    (c for c in m.named_children if c.type == "identifier"), None)
                vname = text(nm)
                v = g.add("state_var", vname, _span(m), cn)
                ty = text(m.child_by_field_name("type") or next(
                    (c for c in m.named_children if c.type == "type_name"), None)).replace(" ", "")
                if ty.startswith("mapping"):
                    g.set(v, "ty_mapping", 1)
                elif ty.endswith("]"):
                    g.set(v, "ty_array", 1)
                elif ty.startswith("address"):
                    g.set(v, "ty_address", 1)
                elif re.match(r"u?int\d*$", ty):
                    g.set(v, "ty_integer", 1)
                elif ty == "bool":
                    g.set(v, "ty_bool", 1)
                elif re.match(r"(string|bytes\d*)$", ty):
                    g.set(v, "ty_string_bytes", 1)
                else:
                    g.set(v, "ty_user_defined", 1)
                vis = next((text(c) for c in m.children if c.type == "visibility"), None)
                g.set(v, {"public": "vis_public", "private": "vis_private",
                          "internal": "vis_internal"}.get(vis, "vis_missing"), 1)
                g.set(v, "is_constant", float(any(c.type == "constant" for c in m.children)))
                g.set(v, "is_immutable", float(any(c.type == "immutable" for c in m.children)))
                g.set(v, "is_initialized", float(any(text(c) == "=" for c in m.children)))
                contract_state[cn][vname] = v
                g.edge("structure", cn, v)
            elif m.type == "event_definition":
                ename = text(m.child_by_field_name("name"))
                e = g.add("event", ename, _span(m), cn)
                params = [c for c in m.named_children if c.type == "event_parameter"]
                g.set(e, "log_params", _log(len(params)))
                g.set(e, "n_indexed", float(sum("indexed" in text(p).split() for p in params)))
                events.setdefault(cn, {}).setdefault(ename, e)
                g.edge("structure", cn, e)
            elif m.type == "modifier_definition":
                mname = text(m.child_by_field_name("name"))
                mn = g.add("modifier", mname, _span(m), cn)
                modifiers.setdefault(cn, {}).setdefault(mname, mn)
                members.append((mn, m, cn))
                g.edge("structure", cn, mn)
            elif m.type in ("function_definition", "constructor_definition",
                            "fallback_receive_definition"):
                if m.type == "function_definition":
                    fname = text(m.child_by_field_name("name"))
                elif m.type == "constructor_definition":
                    fname = "constructor"
                else:
                    fname = "receive" if text(m).lstrip().startswith("receive") else "fallback"
                fn = g.add("function", fname, _span(m), cn)
                functions.setdefault(cn, {}).setdefault(fname, []).append(fn)
                members.append((fn, m, cn))
                g.edge("structure", cn, fn)
            elif m.type == "using_directive":
                tn = next((c for c in m.named_children if c.type == "type_name"), None)
                if tn is not None and re.match(r"u?int", text(tn)):
                    uses_lib_uint[cn] = True

    if free_functions:
        cn = add_contract(None, "contract", "<file-level>")
        for m in free_functions:
            fname = text(m.child_by_field_name("name"))
            fn = g.add("function", fname, _span(m), cn)
            free_funcs.setdefault(fname, []).append(fn)
            members.append((fn, m, cn))
            g.edge("structure", cn, fn)

    if not contract_nodes:                       # empty or import-only file
        cn = add_contract(None, "contract", "<file-level>")

    # inheritance within the file
    for cn, bases in contract_bases.items():
        g.set(cn, "n_bases", float(len(bases)))
        for b in bases:
            if b in contract_nodes and contract_nodes[b] != cn:
                g.edge("structure", cn, contract_nodes[b])

    def visible(registry: dict, cn: int, seen=None) -> dict:
        """Names visible in contract cn: its bases' (declared in this file) first,
        then its own, so that a derived declaration overrides a base one
        (Solidity's resolution within the inheritance hierarchy)."""
        seen = seen or set()
        if cn in seen:
            return {}
        seen.add(cn)
        out: dict = {}
        for b in contract_bases.get(cn, []):
            if b in contract_nodes:
                out.update(visible(registry, contract_nodes[b], seen))
        out.update(registry.get(cn, {}))
        return out

    def visible_state(cn: int) -> dict[str, int]:
        return visible(contract_state, cn)

    def visible_functions(cn: int) -> dict[str, list[int]]:
        return {**free_funcs, **visible(functions, cn)}

    # modifiers first so their guard flags exist when functions are analysed
    has_using = any(uses_lib_uint.values())
    mod_flags: dict[int, tuple[bool, bool]] = {}
    ordered = sorted(members, key=lambda t: NODE_TYPES[g.node_type[t[0]]] != "modifier")
    analyses = {}
    for node, decl, cn in ordered:
        a = _Analyzer(g, node, decl, visible_state(cn), visible_functions(cn),
                      visible(events, cn), visible(modifiers, cn), libraries, has_using)
        a.run()
        analyses[node] = a
        if NODE_TYPES[g.node_type[node]] == "modifier":
            body_text = text(decl.child_by_field_name("body"))
            pre, _, post = body_text.partition("_;")
            lock = bool(a.writes) and bool(post.strip(" }\n\t")) and any(
                g.names[v] in pre and g.names[v] in post for v in a.writes)
            mod_flags[node] = (a.c.get("sender_check", 0) > 0, lock)

    for node, decl, cn in members:
        a = analyses[node]
        c = a.c
        is_fn = NODE_TYPES[g.node_type[node]] == "function"
        if is_fn:   # modifiers have no visibility
            vis = next((text(ch) for ch in decl.children if ch.type == "visibility"), None)
            g.set(node, {"public": "vis_public", "external": "vis_external",
                         "internal": "vis_internal", "private": "vis_private"}.get(vis, "vis_missing"), 1)
        mut = " ".join(text(ch) for ch in decl.children if ch.type == "state_mutability")
        for kw, feat in (("payable", "mut_payable"), ("view", "mut_view"), ("pure", "mut_pure")):
            if kw in mut.split() or (kw == "view" and "constant" in mut.split()):
                g.set(node, feat, 1)
        if is_fn:
            if decl.type == "constructor_definition" or g.names[node] == g.names[cn]:
                g.set(node, "is_constructor", 1)
            if decl.type == "fallback_receive_definition" or (decl.type == "function_definition"
                                                               and not decl.child_by_field_name("name")):
                g.set(node, "is_fallback_receive", 1)
        invoked = [text(ch).split("(")[0].strip() for ch in decl.children if ch.type == "modifier_invocation"]
        g.set(node, "n_modifiers", float(len(invoked)))
        mods_here = visible(modifiers, cn)
        for mname in invoked:
            if mname in mods_here:
                g.edge("call", node, mods_here[mname])
                sender, lock = mod_flags.get(mods_here[mname], (False, False))
                if sender:
                    g.set(node, "has_sender_guard_modifier", 1)
                if lock:
                    g.set(node, "has_lock_modifier", 1)
        body = decl.child_by_field_name("body")
        n_stmts = sum(1 for n in walk(body) if n.type == "statement") if body is not None else 0
        g.set(node, "log_statements", _log(n_stmts))
        g.set(node, "log_params", _log(len(a.params)))
        binary = {"uses_block_timestamp", "uses_block_number", "uses_tx_origin", "uses_msg_value",
                  "uses_blockhash_or_difficulty", "uses_gasleft", "call_in_loop",
                  "unchecked_call_result", "state_write_after_call", "loop_over_dynamic_length",
                  "sender_check", "uses_assembly", "uses_ecrecover", "uses_encode_packed"}
        for k, v in c.items():
            g.set(node, k, min(v, 1.0) if k in binary else _log(v))
        g.set(node, "n_state_writes", _log(len(a.writes)))
        g.set(node, "n_state_reads", _log(len(a.reads)))

    # state variable usage and contract summaries
    reads: dict[int, int] = {}
    writes: dict[int, int] = {}
    for a in analyses.values():
        for v in a.reads:
            reads[v] = reads.get(v, 0) + 1
        for v in a.writes:
            writes[v] = writes.get(v, 0) + 1
    for vars_ in contract_state.values():
        for v in vars_.values():
            g.set(v, "log_reads", _log(reads.get(v, 0)))
            g.set(v, "log_writes", _log(writes.get(v, 0)))
            g.set(v, "never_used", float(reads.get(v, 0) + writes.get(v, 0) == 0
                                         and not g.get(v, "vis_public")))
    for cn in contract_state:
        fns = [n for n, _, c in members if c == cn and NODE_TYPES[g.node_type[n]] == "function"]
        g.set(cn, "log_functions", _log(len(fns)))
        g.set(cn, "log_state_vars", _log(len(contract_state[cn])))
        g.set(cn, "uses_library_for_uint", float(uses_lib_uint[cn]))
        g.set(cn, "has_payable_entry", float(any(g.get(f, "mut_payable") for f in fns)))
        minor = pragma["minor"]
        if minor is None:
            g.set(cn, "pragma_missing", 1)
        else:
            for m, feat in ((5, "pragma_lt_0_5"), (6, "pragma_lt_0_6"), (7, "pragma_lt_0_7"),
                            (8, "pragma_lt_0_8")):
                if minor < m:
                    g.set(cn, feat, 1)
        g.set(cn, "pragma_floating", float(pragma["floating"]))

    out = g.to_dict()
    out["pragma_span"] = pragma["span"]
    out["parse_error"] = bool(root.has_error)
    return out
