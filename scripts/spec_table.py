#!/usr/bin/env python3
"""Print the specification table (Markdown) directly from the library."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from specgnn.spec.library import AF, AG, EF, EU, SPECS, A, And, G, Not, Or, Seq  # noqa: E402


def show(f) -> str:
    if isinstance(f, (A, G)):
        return f.name
    if isinstance(f, Not):
        return "¬" + show(f.f)
    if isinstance(f, And):
        return f"({show(f.a)} ∧ {show(f.b)})"
    if isinstance(f, Or):
        return f"({show(f.a)} ∨ {show(f.b)})"
    for op in (EF, AF, AG):
        if isinstance(f, op):
            return f"{op.__name__}({show(f.f)})"
    if isinstance(f, EU):
        return f"E[{show(f.f)} U {show(f.g)}]"
    if isinstance(f, Seq):
        return f"Seq({show(f.f)}, {show(f.g)})"
    return str(f)


print("| SWC | Name | Formula | Scope | Kind | Atoms | Encodes |")
print("|---|---|---|---|---|---|---|")
for s in SPECS.values():
    print(f"| {s.swc} | {s.name} | `{show(s.formula)}` | {s.scope} | {s.exactness} | "
          f"{s.n_atoms} | {s.source} |")
