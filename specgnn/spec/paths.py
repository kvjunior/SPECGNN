"""Path extraction for the specification checker (Phase 2).

A function body is turned into a set of execution paths. Each path is a list of
*states*; a state is one statement (or one condition) together with the atomic
propositions that hold there. Paths are enumerated by depth-first exploration
of the syntax tree:

  * sequence      -> concatenation
  * if / else     -> both branches (a missing else is an empty branch)
  * loops         -> 0 or 1 iteration (bounded unrolling); states inside a loop
                     carry the InLoop proposition
  * try / catch   -> try body or each catch clause
  * return, revert, throw, failing require -> the path ends (reverted paths have
                     no effect on state, so only the passing side of require is kept)
  * modifiers     -> modifier code before `_;`, the function body, code after `_;`
  * internal call -> one summary state carrying the callee's propositions
                     (computed recursively up to depth 2)

Exploration stops at `max_depth` states per path and `max_paths` paths; the
result is then marked *truncated*. Calls to functions not declared in the file
and inline assembly are marked *opaque*. Truncated or opaque results can still
CONFIRM a pattern but cannot REFUTE it (the checker returns UNKNOWN instead).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..graph.builder import (
    BUILTIN_FUNCS,
    BUILTIN_MEMBERS,
    BUILTIN_OBJECTS,
    LIB_ARITH,
    LOOPS,
    LOW_LEVEL,
    is_value_modifier_of_call,
    storage_aliases,
    text,
    unwrap,
    walk,
)

ATOMS = [
    # operational (per state)
    "ExtCall", "ValXfer", "LowLevelCall", "DelegateCall", "UserTarget", "RetChecked",
    "StateMod", "TsRead", "RandSrc", "ArithOp", "Unchecked", "TxOriginAuth", "SenderCheck",
    "SelfDestruct", "InLoop", "DynLoop", "NoEffect", "BalanceCheck",
    # structural (per function / contract / file)
    "ReGuard", "SafeMath", "Comp08", "FloatPragma", "OldComp", "DefaultVis", "Unused",
    "PublicEntry",
]
STATE_ATOMS = set(ATOMS[:18])

ARITH = {"+", "-", "*", "**"}
_CHAIN_ATTR = re.compile(r"\b(block\.(timestamp|number|difficulty|prevrandao|coinbase)|now|blockhash)\b")
_CHAIN_NODES = ("member_expression", "array_access", "call_expression")
_COMPARISONS = ("==", "!=", "<", ">", "<=", ">=")
ARITH_ASSIGN = {"+=", "-=", "*="}


@dataclass
class State:
    atoms: frozenset
    line: int
    code: str


@dataclass
class PathSet:
    paths: list[list[State]] = field(default_factory=list)
    truncated: bool = False
    opaque: bool = False


@dataclass
class Context:
    """Per-file facts the extractor needs."""
    state_vars: set[str]
    functions: dict[str, object]           # name -> declaration node (first)
    modifiers: dict[str, object]
    libraries: set[str]
    uses_lib_uint: bool
    pragma_minor: int | None
    events: set[str] = field(default_factory=set)


def _call_kind(n, ctx: Context):
    """Classify a call_expression: returns (kind, sends_value, obj_node) or None.
    kind in low-level names, 'send', 'transfer', 'selfdestruct', 'high', 'internal:<name>',
    'require', 'assert', 'builtin', 'library'."""
    fn = unwrap(n.child_by_field_name("function"))
    nargs = sum(1 for c in n.named_children if c.type == "call_argument")
    sends = False
    if fn is not None and fn.type == "struct_expression":
        sends = any(text(c.child_by_field_name("name")) == "value"
                    for c in fn.named_children if c.type == "struct_field_assignment")
        fn = unwrap(fn.child_by_field_name("type"))
    if fn is not None and fn.type == "call_expression":
        inner = unwrap(fn.child_by_field_name("function"))
        if inner is not None and inner.type == "member_expression" and \
                text(inner.child_by_field_name("property")) in ("value", "gas"):
            sends = sends or text(inner.child_by_field_name("property")) == "value"
            fn = unwrap(inner.child_by_field_name("object"))
        else:
            return None
    if fn is None or is_value_modifier_of_call(fn):
        return None
    if fn.type == "identifier":
        name = text(fn)
        if name in ("require", "assert"):
            return (name, False, None)
        if name in ("selfdestruct", "suicide"):
            return ("selfdestruct", True, None)
        if name in BUILTIN_FUNCS or name in ctx.events:
            return ("builtin", False, None)
        if name in ctx.functions:
            return ("internal:" + name, False, None)
        return ("unresolved", False, None)        # e.g. inherited from an imported file
    if fn.type != "member_expression":
        return ("builtin", False, None)
    prop = text(fn.child_by_field_name("property"))
    obj = unwrap(fn.child_by_field_name("object"))
    ot = text(obj)
    if prop in LOW_LEVEL:
        return (prop, sends, obj)
    if prop == "send" and nargs == 1:
        return ("send", True, obj)
    if prop == "transfer" and nargs == 1:
        return ("transfer", True, obj)
    if ot in ("this",):
        return ("internal:" + prop, False, None) if prop in ctx.functions else ("high", sends, obj)
    if ot == "super":
        return ("unresolved", False, None)
    if ot in BUILTIN_OBJECTS or ot.split(".")[0] in BUILTIN_OBJECTS or prop in BUILTIN_MEMBERS:
        return ("builtin", False, None)
    if ot in ctx.libraries or (prop in LIB_ARITH and ctx.uses_lib_uint):
        return ("library", False, None)
    return ("high", sends, obj)


class Extractor:
    def __init__(self, ctx: Context, max_depth: int = 50, max_paths: int = 100,
                 summary_depth: int = 2):
        self.ctx = ctx
        self.max_depth, self.max_paths, self.summary_depth = max_depth, max_paths, summary_depth
        self._summary_cache: dict[tuple[str, int], tuple[frozenset, bool]] = {}
        self._aliases: set[str] = set()

    # -- atoms of one expression/statement (not descending into sub-statements) --
    def atoms_of(self, node, params: set[str], locals_: set[str], in_loop: bool,
                 in_unchecked: bool, depth: int, result: PathSet) -> set[str]:
        a: set[str] = set()
        if in_loop:
            a.add("InLoop")
        if in_unchecked:
            a.add("Unchecked")
        stack = [node]
        while stack:
            n = stack.pop()
            t = n.type
            if t in ("block_statement", "if_statement", "for_statement", "while_statement",
                     "do_while_statement", "try_statement") and n is not node:
                continue                     # sub-statements become their own states
            if t == "call_expression":
                k = _call_kind(n, self.ctx)
                if k:
                    self._call_atoms(n, k, params, a, depth, result)
            elif t in ("assignment_expression", "augmented_assignment_expression",
                       "update_expression") or (t == "unary_expression" and
                                                text(n.child_by_field_name("operator")) == "delete"):
                tgt = n.child_by_field_name("left") or n.child_by_field_name("argument")
                base = tgt
                while base is not None and unwrap(base) is not None and \
                        unwrap(base).type in ("array_access", "member_expression"):
                    u = unwrap(base)
                    base = u.child_by_field_name("base") or u.child_by_field_name("object")
                name = text(unwrap(base)) if base is not None else ""
                if (name in self.ctx.state_vars and name not in locals_ and name not in params) \
                        or name in self._aliases:
                    a.add("StateMod")
                op = next((text(c) for c in n.children if not c.is_named), "")
                if t == "update_expression" or op in ARITH_ASSIGN:
                    a.add("ArithOp")
            elif t == "binary_expression":
                op = text(n.child_by_field_name("operator"))
                if op in ARITH and not self._constant(n):
                    a.add("ArithOp")
                if op in ("==", "!="):
                    s = text(n)
                    if re.search(r"\btx\.origin\b", s):
                        a.add("TxOriginAuth")
                    if re.search(r"\bmsg\.sender\b", s):
                        a.add("SenderCheck")
                if op == "%" and _CHAIN_ATTR.search(text(n)):
                    a.add("RandSrc")
            elif t == "member_expression":
                s = text(n)
                if s in ("block.timestamp", "block.number"):
                    a.add("TsRead")
            elif t == "identifier" and text(n) == "now":
                a.add("TsRead")
            elif t == "array_access":
                # balances[msg.sender] style read: ties a transfer amount to the caller
                if "msg.sender" in text(n.child_by_field_name("index") or n):
                    a.add("BalanceCheck")
            elif t == "assembly_statement":
                result.opaque = True
                continue
            stack.extend(n.named_children)
        if "TsRead" in a and re.search(r"keccak256|sha3|sha256|blockhash", text(node)):
            a.add("RandSrc")
        if re.search(r"\bblockhash\s*\(|block\.(difficulty|prevrandao)", text(node)):
            a.add("RandSrc")
        return a

    def _constant(self, n) -> bool:
        return all(unwrap(c) is not None and unwrap(c).type == "number_literal"
                   for c in (n.child_by_field_name("left"), n.child_by_field_name("right")))

    def _call_atoms(self, n, k, params, a, depth, result) -> None:
        kind, sends, obj = k
        if kind in ("require", "assert"):
            s = text(n)
            if re.search(r"\btx\.origin\b", s):
                a.add("TxOriginAuth")
            if re.search(r"\bmsg\.sender\b", s) and re.search(r"==|!=", s):
                a.add("SenderCheck")
            return
        if kind in ("builtin", "library"):
            return
        if kind == "unresolved":
            result.opaque = True
            return
        if kind.startswith("internal:"):
            summ, opaque = self.summary(kind.split(":", 1)[1], depth + 1)
            a.update(summ)
            result.opaque = result.opaque or opaque
            return
        a.add("ExtCall")
        if sends:
            a.add("ValXfer")
        if kind == "selfdestruct":
            a.add("SelfDestruct")
            a.discard("ExtCall")
            a.add("ValXfer")
        if kind in LOW_LEVEL or kind == "send":
            a.add("LowLevelCall")
        if kind == "delegatecall":
            a.add("DelegateCall")
        if obj is not None:
            base = obj
            while unwrap(base) is not None and unwrap(base).type in _CHAIN_NODES:
                u = unwrap(base)
                base = u.child_by_field_name("object") or u.child_by_field_name("base") or \
                    u.child_by_field_name("function")
                if base is None:
                    break
            if (base is not None and text(unwrap(base)) in params) or "msg.sender" in text(obj):
                a.add("UserTarget")
        p = n.parent
        while p is not None and p.type in ("expression", "parenthesized_expression"):
            p = p.parent
        if p is not None and p.type not in ("expression_statement",):
            a.add("RetChecked")

    def summary(self, fname: str, depth: int) -> tuple[frozenset, bool]:
        """Propositions an internal call may produce (union over its paths)."""
        key = (fname, depth)
        if key in self._summary_cache:
            return self._summary_cache[key]
        decl = self.ctx.functions.get(fname)
        if decl is None or depth > self.summary_depth:
            return frozenset(), True
        self._summary_cache[key] = (frozenset(), True)        # recursion guard
        ps = self.extract(decl, depth=depth, apply_modifiers=False)
        atoms = set()
        for p in ps.paths:
            for s in p:
                atoms |= s.atoms
        atoms &= {"ExtCall", "ValXfer", "LowLevelCall", "DelegateCall", "StateMod", "SelfDestruct"}
        out = (frozenset(atoms), ps.opaque or ps.truncated)
        self._summary_cache[key] = out
        return out

    # -- path enumeration ---------------------------------------------------------
    def extract(self, decl, depth: int = 0, apply_modifiers: bool = True) -> PathSet:
        result = PathSet()
        params = {text(p.child_by_field_name("name")) for p in decl.named_children
                  if p.type == "parameter" and p.child_by_field_name("name")}
        body = decl.child_by_field_name("body")
        locals_ = {text(v.child_by_field_name("name")) for v in walk(body)
                   if v.type == "variable_declaration" and v.child_by_field_name("name")} \
            if body is not None else set()
        saved_aliases = self._aliases
        self._aliases = storage_aliases(body, self.ctx.state_vars)
        env = (params, locals_, depth, result)
        stmts = list(body.named_children) if body is not None else []
        segments = [("stmts", stmts, False, False)]
        if apply_modifiers:
            for inv in [c for c in decl.children if c.type == "modifier_invocation"]:
                mname = text(inv).split("(")[0].strip()
                mdecl = self.ctx.modifiers.get(mname)
                if mdecl is None:
                    if mname not in ("public", "external", "internal", "private"):
                        result.opaque = True
                    continue
                mbody = mdecl.child_by_field_name("body")
                mst = list(mbody.named_children) if mbody is not None else []
                idx = next((i for i, s in enumerate(mst) if text(s).strip() == "_;"), len(mst))
                segments = [("stmts", mst[:idx], False, False)] + segments + \
                           [("stmts", mst[idx + 1:], False, False)]
        flat = []
        for _, s, _, _ in segments:
            flat.extend(s)
        finished, live = self._seq(flat, [[]], env, in_loop=False, in_unchecked=False)
        result.paths = finished + live
        self._aliases = saved_aliases
        return result

    def _seq(self, stmts, prefixes, env, in_loop, in_unchecked):
        live = [p for p in prefixes]
        finished = []
        for st in stmts:
            nxt = []
            for p in live:
                for q, ended in self._stmt(st, p, env, in_loop, in_unchecked):
                    (finished if ended else nxt).append(q)
                    if len(nxt) + len(finished) >= self.max_paths:
                        env[3].truncated = True
                        break
            live = nxt[: self.max_paths]
            if not live:
                break
        return finished, live

    def _state(self, node, env, in_loop, in_unchecked) -> State:
        params, locals_, depth, result = env
        atoms = self.atoms_of(node, params, locals_, in_loop, in_unchecked, depth, result)
        if node.type == "expression_statement" and node.named_child_count:
            e = unwrap(node.named_children[0])
            if e is not None and (e.type in ("identifier", "member_expression", "number_literal",
                                             "boolean_literal") or
                                  (e.type == "binary_expression" and
                                   text(e.child_by_field_name("operator")) in _COMPARISONS)):
                if text(e) != "_":
                    atoms.add("NoEffect")
        return State(frozenset(atoms), node.start_point[0] + 1, text(node)[:120])

    def _stmt(self, st, prefix, env, in_loop, in_unchecked):
        """Yield (path, ended) pairs."""
        u = unwrap(st)
        if u is None:
            return [(prefix, False)]
        if len(prefix) >= self.max_depth:
            env[3].truncated = True
            return [(prefix, True)]
        t = u.type
        if t == "block_statement":
            unchecked = in_unchecked or (u.child_count and u.children[0].type == "unchecked")
            fin, live = self._seq(list(u.named_children), [prefix], env, in_loop, unchecked)
            return [(p, True) for p in fin] + [(p, False) for p in live]
        if t == "if_statement":
            cond = u.child_by_field_name("condition")
            cs = self._state(cond, env, in_loop, in_unchecked) if cond is not None else None
            base = prefix + ([cs] if cs else [])
            body = u.child_by_field_name("body")
            branches = [c for c in u.named_children
                        if c.type in ("statement", "block_statement", "if_statement")
                        and (cond is None or c.id != cond.id)]
            alt = u.child_by_field_name("else")
            if alt is None and len(branches) > 1:
                alt = branches[-1] if body is None or branches[-1].id != body.id else None
            out = self._stmt(body, base, env, in_loop, in_unchecked) if body else [(base, False)]
            if alt is not None:
                out += self._stmt(alt, base, env, in_loop, in_unchecked)
            else:
                out.append((base, False))
            return out
        if t in LOOPS:
            cond = u.child_by_field_name("condition")
            cs = self._state(cond, env, True, in_unchecked) if cond is not None else None
            if cs is not None and re.search(r"\.length\b", text(cond)):
                cs = State(cs.atoms | {"DynLoop"}, cs.line, cs.code)
            init = u.child_by_field_name("initial")
            pre = prefix + ([self._state(init, env, in_loop, in_unchecked)] if init is not None else [])
            pre = pre + ([cs] if cs else [])
            body = u.child_by_field_name("body")
            once = self._stmt(body, pre, env, True, in_unchecked) if body is not None else [(pre, False)]
            return [(pre, False)] + once
        if t == "try_statement":
            out = []
            for c in u.named_children:
                if c.type in ("block_statement", "catch_clause"):
                    blk = c if c.type == "block_statement" else next(
                        (x for x in c.named_children if x.type == "block_statement"), None)
                    if blk is not None:
                        fin, live = self._seq(list(blk.named_children), [prefix], env,
                                              in_loop, in_unchecked)
                        out += [(p, True) for p in fin] + [(p, False) for p in live]
            return out or [(prefix, False)]
        s = self._state(u, env, in_loop, in_unchecked)
        ended = t in ("return_statement", "revert_statement") or \
            (t == "expression_statement" and re.match(r"\s*(throw|revert)\b", text(u)) is not None)
        return [(prefix + [s], ended)]
