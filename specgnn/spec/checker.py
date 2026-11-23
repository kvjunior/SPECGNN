"""Specification checker: verdict for a candidate (node v, SWC type t).

Verdict rules (identical for all 17 specifications):

  CONFIRMED  the formula holds; a witness (states with line numbers) is returned.
  REFUTED    the formula does not hold AND the evidence is complete:
               - the defining construct of the pattern is absent from every path
                 (e.g. no external call at all for reentrancy), or
               - for "pattern" specs: path extraction finished without
                 truncation and without opaque calls/assembly.
  UNKNOWN    otherwise: truncated or opaque extraction, a "heuristic" spec whose
             defining construct is present but whose formula fails, missing
             pragma for compiler-dependent specs, or a candidate node the
             specification does not apply to.

Candidates on a contract node with a function-scope spec are checked on every
function of that contract (CONFIRMED if any function confirms, REFUTED if all
refute); candidates on a state variable are checked on the functions that read
or write it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..graph.builder import _PARSER, _is_member_name, text, walk
from ..graph.features import NODE_TYPES
from .library import SPECS, evaluate
from .paths import Context, Extractor, PathSet

CONFIRMED, REFUTED, UNKNOWN = "CONFIRMED", "REFUTED", "UNKNOWN"
FUNC_DECLS = ("function_definition", "constructor_definition", "fallback_receive_definition",
              "modifier_definition")


@dataclass
class CheckerConfig:
    max_depth: int = 50      # states per path
    max_paths: int = 100     # paths per function


@dataclass
class Verdict:
    verdict: str
    reason: str
    witness: list[int] = field(default_factory=list)     # line numbers


def _span(n) -> tuple[int, int]:
    return (n.start_point[0] + 1, n.end_point[0] + 1)


class FileChecker:
    """Parses one file once and answers verdict queries for its candidates."""

    def __init__(self, source: str | bytes, max_depth: int = 50, max_paths: int = 100):
        data = source.encode("utf-8") if isinstance(source, str) else source
        self.root = _PARSER.parse(data).root_node
        self.decl_at: dict[tuple[str, int, int], object] = {}
        self.contract_of: dict[int, object] = {}
        state_vars, functions, modifiers, libraries, events = set(), {}, {}, set(), set()
        self.var_decls: dict[str, object] = {}
        uses_lib_uint = False
        for n in walk(self.root):
            t = n.type
            if t in ("contract_declaration", "library_declaration", "interface_declaration"):
                self.decl_at[("contract",) + _span(n)] = n
                if t == "library_declaration":
                    libraries.add(text(n.child_by_field_name("name")))
            elif t == "state_variable_declaration":
                nm = n.child_by_field_name("name") or next(
                    (c for c in n.named_children if c.type == "identifier"), None)
                state_vars.add(text(nm))
                self.var_decls[text(nm)] = n
                self.decl_at[("state_var",) + _span(n)] = n
            elif t in FUNC_DECLS:
                kind = "modifier" if t == "modifier_definition" else "function"
                self.decl_at[(kind,) + _span(n)] = n
                name = text(n.child_by_field_name("name")) if n.child_by_field_name("name") else t
                (modifiers if kind == "modifier" else functions).setdefault(name, n)
            elif t == "event_definition":
                events.add(text(n.child_by_field_name("name")))
            elif t == "using_directive":
                tn = next((c for c in n.named_children if c.type == "type_name"), None)
                if tn is not None and re.match(r"u?int", text(tn)):
                    uses_lib_uint = True
        self.pragma_minor, self.floating = None, False
        for n in self.root.named_children:
            if n.type == "pragma_directive" and "solidity" in text(n):
                spec = text(n).split("solidity", 1)[1].strip(" ;")
                minors = [int(m) for a, m in re.findall(r"(\d+)\.(\d+)", spec) if a == "0"]
                self.pragma_minor = min(minors) if minors else None
                self.floating = bool(re.search(r"[\^><*~]", spec))
                break
        self.ctx = Context(state_vars, functions, modifiers, libraries, uses_lib_uint,
                           self.pragma_minor, events)
        self.identifier_counts: dict[str, int] = {}
        for n in walk(self.root):
            if n.type == "identifier" and not _is_member_name(n):
                self.identifier_counts[text(n)] = self.identifier_counts.get(text(n), 0) + 1
        self._paths: dict[int, PathSet] = {}
        self._max = (max_depth, max_paths)
        self._build_scopes()

    # ---------------------------------------------------------------- name scopes
    CONTAINERS = ("contract_declaration", "library_declaration", "interface_declaration")

    def _build_scopes(self) -> None:
        """Per-contract registries; names resolve within the contract and its bases
        declared in this file (derived declarations override base ones)."""
        self.scope: dict[str, dict] = {}
        free: dict[str, object] = {}
        for c in self.root.named_children:
            if c.type == "function_definition" and c.child_by_field_name("name") is not None:
                free.setdefault(text(c.child_by_field_name("name")), c)
            if c.type not in self.CONTAINERS:
                continue
            name = text(c.child_by_field_name("name"))
            entry = {"node": c, "bases": [], "functions": {}, "modifiers": {}, "state": set(),
                     "events": set()}
            for ch in c.children:
                if ch.type == "inheritance_specifier":
                    entry["bases"].append(re.split(r"[\s(]", text(ch).strip(), maxsplit=1)[0])
            body = c.child_by_field_name("body")
            for m in (body.named_children if body is not None else []):
                if m.type == "state_variable_declaration":
                    nm = m.child_by_field_name("name") or next(
                        (x for x in m.named_children if x.type == "identifier"), None)
                    entry["state"].add(text(nm))
                elif m.type == "modifier_definition":
                    entry["modifiers"].setdefault(text(m.child_by_field_name("name")), m)
                elif m.type in FUNC_DECLS and m.child_by_field_name("name") is not None:
                    entry["functions"].setdefault(text(m.child_by_field_name("name")), m)
                elif m.type == "event_definition":
                    entry["events"].add(text(m.child_by_field_name("name")))
            self.scope[name] = entry
        self._free = free
        self._contexts: dict[int, tuple[Context, Extractor, dict]] = {}

    def _visible(self, name: str, key: str, seen=None):
        seen = seen or set()
        if name in seen or name not in self.scope:
            return {} if key in ("functions", "modifiers") else set()
        seen.add(name)
        e = self.scope[name]
        if key in ("functions", "modifiers"):
            out: dict = {}
            for b in e["bases"]:
                out.update(self._visible(b, key, seen))
            out.update(e[key])
            return out
        out_s: set = set()
        for b in e["bases"]:
            out_s |= self._visible(b, key, seen)
        return out_s | e[key]

    def _owner(self, decl):
        p = decl.parent
        while p is not None and p.type not in self.CONTAINERS:
            p = p.parent
        return p

    def _context(self, decl) -> tuple[Context, Extractor, dict]:
        owner = self._owner(decl)
        key = owner.id if owner is not None else -1
        if key not in self._contexts:
            name = text(owner.child_by_field_name("name")) if owner is not None else None
            funcs = {**self._free, **(self._visible(name, "functions") if name else {})}
            mods = self._visible(name, "modifiers") if name else {}
            state = self._visible(name, "state") if name else set()
            events = self._visible(name, "events") if name else set()
            ctx = Context(state, funcs, mods, self.ctx.libraries, self.ctx.uses_lib_uint,
                          self.pragma_minor, events)
            ext = Extractor(ctx, max_depth=self._max[0], max_paths=self._max[1])
            flags = {m: self._modifier_flags(d, state) for m, d in mods.items()}
            self._contexts[key] = (ctx, ext, flags)
        return self._contexts[key]

    # ---------------------------------------------------------------- facts
    @staticmethod
    def _modifier_flags(m, state_vars) -> tuple[bool, bool]:
        body = text(m.child_by_field_name("body"))
        sender = bool(re.search(r"msg\.sender\s*[!=]=|[!=]=\s*msg\.sender", body))
        pre, _, post = body.partition("_;")
        if not post.strip(" }\n\t"):
            return sender, False
        lock = any(re.search(rf"\b{re.escape(v)}\b\s*=", pre)
                   and re.search(rf"\b{re.escape(v)}\b\s*=", post) for v in state_vars)
        return sender, lock

    def function_facts(self, decl) -> set[str]:
        facts = self.file_facts()
        vis = next((text(c) for c in decl.children if c.type == "visibility"), None)
        is_ctor = decl.type == "constructor_definition"
        if decl.type == "function_definition" and vis is None and decl.child_by_field_name("name"):
            facts.add("DefaultVis")
        if vis in (None, "public", "external") and not is_ctor and decl.type != "modifier_definition":
            facts.add("PublicEntry")
        flags = self._context(decl)[2]
        for inv in (c for c in decl.children if c.type == "modifier_invocation"):
            sender, lock = flags.get(text(inv).split("(")[0].strip(), (False, False))
            if sender:
                facts.add("SenderGuard")
            if lock:
                facts.add("ReGuard")
        body = decl.child_by_field_name("body")
        if body is not None:
            for n in walk(body):
                if n.type == "assignment_expression":
                    left = text(n.child_by_field_name("left")).replace(" ", "")
                    if re.match(r"\w+\[msg\.sender\]\[\w+\]$", left):
                        facts.add("ApproveLike")
        return facts

    def file_facts(self) -> set[str]:
        f = set()
        if self.ctx.uses_lib_uint:
            f.add("SafeMath")
        if self.pragma_minor is not None:
            if self.pragma_minor >= 8:
                f.add("Comp08")
            if self.pragma_minor < 6:
                f.add("OldComp")
            if self.pragma_minor < 5:
                f.add("Pre05")
        if self.floating:
            f.add("FloatPragma")
        return f

    def var_facts(self, decl) -> set[str]:
        f = set()
        if not any(c.type == "visibility" for c in decl.children):
            f.add("DefaultVis")
        nm = decl.child_by_field_name("name") or next(
            (c for c in decl.named_children if c.type == "identifier"), None)
        public = any(text(c) == "public" for c in decl.children)
        if self.identifier_counts.get(text(nm), 0) <= 1 and not public:
            f.add("Unused")
        return f

    # ---------------------------------------------------------------- checking
    def paths(self, decl) -> PathSet:
        if decl.id not in self._paths:
            self._paths[decl.id] = self._context(decl)[1].extract(decl)
        return self._paths[decl.id]

    def check_function(self, decl, swc: str) -> Verdict:
        spec = SPECS[swc]
        facts = self.function_facts(decl)
        ps = self.paths(decl)
        holds, wit = evaluate(spec.formula, ps, facts)
        if holds:
            return Verdict(CONFIRMED, "formula satisfied", sorted({s.line for s in wit})[:10])
        present = {a for p in ps.paths for s in p for a in s.atoms}
        complete = not (ps.truncated or ps.opaque)
        if spec.trigger and not (spec.trigger & present) and complete:
            return Verdict(REFUTED, "defining construct absent")
        if spec.trigger and not (spec.trigger & present) and not complete:
            return Verdict(UNKNOWN, "construct absent but extraction incomplete")
        if spec.exactness == "heuristic":
            return Verdict(UNKNOWN, "heuristic specification not satisfied")
        if not complete:
            return Verdict(UNKNOWN, "truncated or opaque paths")
        if swc in ("SWC-101", "SWC-100") and self.pragma_minor is None:
            return Verdict(UNKNOWN, "compiler version unknown")
        return Verdict(REFUTED, "formula contradicted on all extracted paths")

    def check_structural(self, swc: str, decl=None) -> Verdict:
        spec = SPECS[swc]
        if spec.scope == "contract":
            if self.pragma_minor is None and swc in ("SWC-102", "SWC-103"):
                return Verdict(UNKNOWN, "no pragma")
            facts = self.file_facts()
        else:
            facts = self.var_facts(decl)
        holds, _ = evaluate(spec.formula, PathSet(), facts)
        if holds:
            return Verdict(CONFIRMED, "structural fact holds", [_span(decl)[0]] if decl is not None else [])
        if spec.exactness == "heuristic":
            return Verdict(UNKNOWN, "heuristic specification not satisfied")
        return Verdict(REFUTED, "structural fact does not hold")

    def _aggregate(self, verdicts: list[Verdict]) -> Verdict:
        if not verdicts:
            return Verdict(UNKNOWN, "no applicable declaration")
        for v in verdicts:
            if v.verdict == CONFIRMED:
                return v
        if all(v.verdict == REFUTED for v in verdicts):
            return Verdict(REFUTED, "refuted in every applicable declaration")
        return Verdict(UNKNOWN, "no confirmation, not all refuted")

    def check(self, graph: dict, node: int, swc: str) -> Verdict:
        """Verdict for candidate (graph node `node`, SWC type `swc`)."""
        spec = SPECS[swc]
        ntype = NODE_TYPES[graph["node_type"][node]]
        span = tuple(int(x) for x in graph["spans"][node])
        decl = self.decl_at.get((ntype,) + span)
        if decl is None and ntype in ("function", "modifier"):
            decl = self.decl_at.get(("function",) + span) or self.decl_at.get(("modifier",) + span)
        if spec.scope == "contract":
            return self.check_structural(swc)
        if spec.scope == "state_var":
            if ntype == "state_var" and decl is not None:
                return self.check_structural(swc, decl)
            owner = int(graph["owner"][node])
            vars_ = [self.decl_at.get(("state_var",) + tuple(int(x) for x in graph["spans"][i]))
                     for i in range(len(graph["node_type"]))
                     if NODE_TYPES[graph["node_type"][i]] == "state_var" and int(graph["owner"][i]) == owner]
            return self._aggregate([self.check_structural(swc, d) for d in vars_ if d is not None])
        # function-scope specification
        if ntype in ("function", "modifier") and decl is not None:
            return self.check_function(decl, swc)
        if ntype == "contract":
            members = [i for i in range(len(graph["node_type"]))
                       if NODE_TYPES[graph["node_type"][i]] == "function" and int(graph["owner"][i]) == node]
        elif ntype == "state_var":
            members = sorted({int(s) for s, d in graph["edges"]["data_flow"] if d == node} |
                             {int(d) for s, d in graph["edges"]["data_flow"] if s == node})
        else:
            return Verdict(UNKNOWN, f"specification does not apply to {ntype} nodes")
        decls = [self.decl_at.get(("function",) + tuple(int(x) for x in graph["spans"][i]))
                 for i in members]
        return self._aggregate([self.check_function(d, swc) for d in decls if d is not None])
