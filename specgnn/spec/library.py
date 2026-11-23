"""Specification library: 17 CTL-inspired pattern formulas.

Formulas are evaluated over the finite set of paths extracted from a function
(see paths.py). Operators:

    A(p)        atomic proposition p holds in the current state
    G(p)        structural (path-independent) fact p holds
    EF(f)       some path has some state satisfying state formula f
    AF(f)       every path has some state satisfying f
    AG(f)       every state of every path satisfies f
    EU(f, g)    some path where f holds in every state before the first g-state
    Seq(f, g)   some path with an f-state followed later by a g-state
    And / Or / Not

These are pattern rules over syntax-derived paths; they are not a model checker
and give no soundness guarantee. Each formula was written by hand from the SWC
registry definition (see `SPECS[...].source`) and is documented together with
its known false-positive and false-negative sources in the supplementary table.
"""
from __future__ import annotations

from dataclasses import dataclass

from .paths import PathSet, State


# ---------------------------------------------------------------- formula AST
class F:
    def __and__(self, o): return And(self, o)
    def __or__(self, o): return Or(self, o)
    def __invert__(self): return Not(self)


@dataclass(frozen=True)
class A(F):            # state atom
    name: str


@dataclass(frozen=True)
class G(F):            # structural fact
    name: str


@dataclass(frozen=True)
class Not(F):
    f: F


@dataclass(frozen=True)
class And(F):
    a: F
    b: F


@dataclass(frozen=True)
class Or(F):
    a: F
    b: F


@dataclass(frozen=True)
class EF(F):
    f: F


@dataclass(frozen=True)
class AF(F):
    f: F


@dataclass(frozen=True)
class AG(F):
    f: F


@dataclass(frozen=True)
class EU(F):
    f: F
    g: F


@dataclass(frozen=True)
class Seq(F):
    f: F
    g: F


def atoms_in(f: F) -> set[str]:
    if isinstance(f, (A, G)):
        return {f.name}
    out = set()
    for v in f.__dict__.values():
        if isinstance(v, F):
            out |= atoms_in(v)
    return out


# ---------------------------------------------------------------- evaluation
def _state_holds(f: F, s: State, facts: set[str]) -> bool:
    if isinstance(f, A):
        return f.name in s.atoms
    if isinstance(f, G):
        return f.name in facts
    if isinstance(f, Not):
        return not _state_holds(f.f, s, facts)
    if isinstance(f, And):
        return _state_holds(f.a, s, facts) and _state_holds(f.b, s, facts)
    if isinstance(f, Or):
        return _state_holds(f.a, s, facts) or _state_holds(f.b, s, facts)
    raise TypeError(f"temporal operator {type(f).__name__} inside a state formula")


def evaluate(f: F, ps: PathSet, facts: set[str]) -> tuple[bool, list[State]]:
    """Return (holds, witness states)."""
    if isinstance(f, G):
        return f.name in facts, []
    if isinstance(f, Not):
        v, _ = evaluate(f.f, ps, facts)
        return not v, []
    if isinstance(f, And):
        va, wa = evaluate(f.a, ps, facts)
        if not va:
            return False, []
        vb, wb = evaluate(f.b, ps, facts)
        return vb, (wa + wb if vb else [])
    if isinstance(f, Or):
        va, wa = evaluate(f.a, ps, facts)
        if va:
            return True, wa
        return evaluate(f.b, ps, facts)
    if isinstance(f, EF):
        for p in ps.paths:
            for s in p:
                if _state_holds(f.f, s, facts):
                    return True, [s]
        return False, []
    if isinstance(f, AF):
        ok = bool(ps.paths) and all(any(_state_holds(f.f, s, facts) for s in p) for p in ps.paths)
        return ok, []
    if isinstance(f, AG):
        return all(_state_holds(f.f, s, facts) for p in ps.paths for s in p), []
    if isinstance(f, EU):
        for p in ps.paths:
            for i, s in enumerate(p):
                if _state_holds(f.g, s, facts):
                    if all(_state_holds(f.f, x, facts) for x in p[:i]):
                        return True, p[: i + 1][-10:]
                    break
        return False, []
    if isinstance(f, Seq):
        for p in ps.paths:
            for i, s in enumerate(p):
                if _state_holds(f.f, s, facts):
                    for x in p[i + 1:]:
                        if _state_holds(f.g, x, facts):
                            return True, [s, x]
        return False, []
    if isinstance(f, A):
        return evaluate(EF(f), ps, facts)
    raise TypeError(type(f))


# ---------------------------------------------------------------- the library
@dataclass(frozen=True)
class Spec:
    swc: str
    name: str
    formula: F
    scope: str               # "function" | "contract" | "state_var"
    exactness: str           # "pattern" | "heuristic"
    source: str              # SWC registry definition the formula encodes
    trigger: frozenset       # atoms whose total absence lets the checker refute early

    @property
    def n_atoms(self) -> int:
        return len(atoms_in(self.formula))


def _s(swc, name, formula, scope, exactness, source, trigger=()):
    return Spec(swc, name, formula, scope, exactness, source, frozenset(trigger))


SPECS: dict[str, Spec] = {s.swc: s for s in [
    _s("SWC-100", "Function default visibility", G("DefaultVis") & G("Pre05"),
       "function", "pattern", "function without explicit visibility (default public before 0.5)"),
    _s("SWC-101", "Integer overflow/underflow",
       EF(A("ArithOp") & ~G("Comp08")) | EF(A("ArithOp") & A("Unchecked")),
       "function", "pattern",
       "raw arithmetic operator (not a SafeMath-style library call) on a compiler without "
       "checked arithmetic (<0.8), or inside an unchecked block", ("ArithOp",)),
    _s("SWC-102", "Outdated compiler version", G("OldComp"), "contract", "pattern",
       "pragma admits a compiler older than 0.6"),
    _s("SWC-103", "Floating pragma", G("FloatPragma"), "contract", "pattern",
       "pragma does not pin one compiler version"),
    _s("SWC-104", "Unchecked call return value", EF(A("LowLevelCall") & ~A("RetChecked")),
       "function", "pattern", "result of call/send/delegatecall is discarded", ("LowLevelCall",)),
    _s("SWC-105", "Unprotected ether withdrawal",
       G("PublicEntry") & EF(A("ValXfer") & ~A("BalanceCheck")) & ~EF(A("SenderCheck"))
       & ~G("SenderGuard"),
       "function", "heuristic",
       "publicly callable value transfer not tied to the caller's balance and without "
       "an access check", ("ValXfer",)),
    _s("SWC-107", "Reentrancy", Seq(A("ExtCall"), A("StateMod")) & ~G("ReGuard"),
       "function", "pattern",
       "external call followed by a state modification on some path, without a "
       "reentrancy lock", ("ExtCall",)),
    _s("SWC-108", "State variable default visibility", G("DefaultVis"), "state_var", "pattern",
       "state variable without explicit visibility"),
    _s("SWC-112", "Delegatecall to untrusted callee", EF(A("DelegateCall") & A("UserTarget")),
       "function", "pattern", "delegatecall whose target comes from a parameter or the caller",
       ("DelegateCall",)),
    _s("SWC-113", "DoS with failed call",
       EF(A("ExtCall") & A("InLoop")) | EF(A("ValXfer") & A("RetChecked") & A("InLoop")),
       "function", "pattern",
       "external call or checked value transfer inside a loop: one failing callee "
       "blocks every other iteration", ("ExtCall", "ValXfer")),
    _s("SWC-114", "Transaction order dependence",
       G("PublicEntry") & G("ApproveLike") & EF(A("StateMod") & ~A("SenderCheck")),
       "function", "heuristic",
       "publicly callable function that overwrites a per-caller allowance without first "
       "checking its current value (ERC20 approve race)", ("StateMod",)),
    _s("SWC-115", "Authorization through tx.origin", EF(A("TxOriginAuth")), "function",
       "pattern", "tx.origin compared in an authorization check", ("TxOriginAuth",)),
    _s("SWC-116", "Block values as a proxy for time",
       EF(A("TsRead") & (A("StateMod") | A("ValXfer")))
       | Seq(A("TsRead") & ~A("NoEffect"), A("StateMod") | A("ValXfer")),
       "function", "pattern",
       "block.timestamp/number used in a computation or condition that is followed by a "
       "state change or value transfer", ("TsRead",)),
    _s("SWC-120", "Weak randomness from chain attributes", EF(A("RandSrc")), "function",
       "pattern", "block attributes hashed or reduced modulo n as a random source", ("RandSrc",)),
    _s("SWC-128", "DoS with block gas limit", EF(A("DynLoop")) & G("PublicEntry"), "function",
       "pattern", "publicly reachable loop bounded by a dynamic array length", ("DynLoop",)),
    _s("SWC-131", "Presence of unused variables", G("Unused"), "state_var", "heuristic",
       "state or local variable that is never read or written"),
    _s("SWC-135", "Code with no effects", EF(A("NoEffect")), "function", "heuristic",
       "expression statement whose value is discarded (e.g. `x == 1;`)", ("NoEffect",)),
]}

assert len(SPECS) == 17
