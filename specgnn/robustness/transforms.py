"""Semantics-preserving transformations of Solidity source (RQ6 robustness).

Each transformation edits the syntax tree and changes neither the contract's
external interface nor its storage layout nor its behaviour:

  rename     Parameters and local variables are renamed within their function;
             *private* state variables and *private* functions are renamed
             file-wide. Public/external/internal names are never touched (they
             are part of the ABI or visible to derived contracts in other
             files), nor are member names (``x.name``) or named arguments.
  dead_code  Every function, constructor and modifier body starts with an
             unreachable block ``if (false) { uint256 __dead = 1; __dead = __dead + 1; }``.
  reorder    Function definitions inside each contract are emitted in reverse
             order (function order affects neither selectors nor storage).
  all        rename, then dead_code, then reorder.

Edits are applied on byte offsets of the tree-sitter parse tree, never by text
search, so comments and strings are left untouched.
"""
from __future__ import annotations

from ..graph.builder import _PARSER, text, walk

FUNC_DECLS = ("function_definition", "constructor_definition", "fallback_receive_definition",
              "modifier_definition")
DEAD = " if (false) { uint256 __dead = 1; __dead = __dead + 1; } "


def _apply(data: bytes, edits: list[tuple[int, int, bytes]]) -> bytes:
    """Apply non-overlapping (start, end, replacement) byte edits."""
    out, pos = [], 0
    for start, end, rep in sorted(edits):
        if start < pos:
            raise ValueError("overlapping edits")
        out += [data[pos:start], rep]
        pos = end
    out.append(data[pos:])
    return b"".join(out)


def _is_member_property(n) -> bool:
    p = n.parent
    return p is not None and p.type == "member_expression" and \
        p.child_by_field_name("property") is not None and p.child_by_field_name("property").id == n.id


def _is_field_name(n) -> bool:
    p = n.parent
    if p is None:
        return False
    if p.type == "struct_field_assignment" and p.child_by_field_name("name") is not None:
        return p.child_by_field_name("name").id == n.id
    return p.type in ("struct_member", "enum_value", "event_parameter")


def _is_call_target(n) -> bool:
    p = n.parent
    while p is not None and p.type == "expression":
        child, p = p, p.parent
        if p is not None and p.type == "call_expression":
            fn = p.child_by_field_name("function")
            return fn is not None and fn.id == child.id
    return False


def _renamable(n) -> bool:
    return n.type == "identifier" and not _is_member_property(n) and not _is_field_name(n)


def rename(src: str) -> str:
    data = src.encode("utf-8")
    root = _PARSER.parse(data).root_node
    edits: dict[int, tuple[int, int, bytes]] = {}
    counter = 0

    def fresh() -> bytes:
        nonlocal counter
        counter += 1
        return f"__r{counter}".encode()

    # file-wide: private state variables and private functions
    private: dict[str, bytes] = {}
    for n in walk(root):
        if n.type == "state_variable_declaration":
            if any(c.type == "visibility" and text(c) == "private" for c in n.children):
                nm = n.child_by_field_name("name") or next(
                    (c for c in n.named_children if c.type == "identifier"), None)
                if nm is not None and text(nm) not in private:
                    private[text(nm)] = fresh()
        elif n.type == "function_definition":
            vis = [text(c) for c in n.children if c.type == "visibility"]
            special = any(c.type in ("override_specifier", "virtual") for c in n.children)
            nm = n.child_by_field_name("name")
            if vis == ["private"] and not special and nm is not None and text(nm) not in private:
                private[text(nm)] = fresh()
    # per function: parameters and locals, renamed only where every occurrence of the
    # name inside the function unambiguously denotes that variable
    scopes = []
    for f in walk(root):
        if f.type not in FUNC_DECLS:
            continue
        declared: dict[str, int] = {}             # name -> byte offset where its scope starts
        for p in f.named_children:
            if p.type == "parameter" and p.child_by_field_name("name") is not None:
                declared.setdefault(text(p.child_by_field_name("name")), 0)
        body = f.child_by_field_name("body")
        if body is not None:
            for v in walk(body):
                if v.type == "variable_declaration" and v.child_by_field_name("name") is not None:
                    stmt = v.parent
                    while stmt is not None and stmt.type not in ("variable_declaration_statement",
                                                                  "for_statement", "catch_clause",
                                                                  "try_statement"):
                        stmt = stmt.parent
                    end = stmt.end_byte if stmt is not None and stmt.type == \
                        "variable_declaration_statement" else v.end_byte
                    name = text(v.child_by_field_name("name"))
                    if name in declared:          # declared twice in one function: ambiguous
                        declared[name] = -1
                    else:
                        declared[name] = end
        ambiguous = {n for n, start in declared.items() if start < 0 or n == ""}
        decl_ids = {v.child_by_field_name("name").id for v in walk(f)
                    if v.type in ("variable_declaration", "parameter")
                    and v.child_by_field_name("name") is not None}
        for n in walk(f):
            if n.type != "identifier" or not _renamable(n) or text(n) not in declared:
                continue
            name = text(n)
            if _is_call_target(n):                  # a variable is never called: outer function
                ambiguous.add(name)
            elif n.id not in decl_ids and n.start_byte < declared[name]:
                ambiguous.add(name)                 # used before its scope starts: outer entity
        names = sorted(set(declared) - ambiguous)
        if names:
            scopes.append((f, {nm: fresh() for nm in names}))
    for f, mapping in scopes:
        for n in walk(f):
            if _renamable(n) and text(n) in mapping:
                edits[n.start_byte] = (n.start_byte, n.end_byte, mapping[text(n)])
    for n in walk(root):
        if _renamable(n) and text(n) in private and n.start_byte not in edits:
            edits[n.start_byte] = (n.start_byte, n.end_byte, private[text(n)])
    return _apply(data, list(edits.values())).decode("utf-8")


def dead_code(src: str) -> str:
    data = src.encode("utf-8")
    root = _PARSER.parse(data).root_node
    edits = []
    for f in walk(root):
        if f.type in FUNC_DECLS:
            body = f.child_by_field_name("body")
            if body is not None and body.child_count and text(body.children[0]) == "{":
                pos = body.children[0].end_byte
                edits.append((pos, pos, DEAD.encode()))
    return _apply(data, edits).decode("utf-8")


def reorder(src: str) -> str:
    data = src.encode("utf-8")
    root = _PARSER.parse(data).root_node
    edits = []
    for c in walk(root):
        if c.type not in ("contract_declaration", "library_declaration"):
            continue
        body = c.child_by_field_name("body")
        funcs = [m for m in (body.named_children if body is not None else [])
                 if m.type == "function_definition"]
        if len(funcs) < 2:
            continue
        texts = [data[f.start_byte:f.end_byte] for f in funcs]
        for f, t in zip(funcs, reversed(texts), strict=True):
            edits.append((f.start_byte, f.end_byte, t))
    return _apply(data, edits).decode("utf-8")


def all_transforms(src: str) -> str:
    return reorder(dead_code(rename(src)))


TRANSFORMS = {"rename": rename, "dead_code": dead_code, "reorder": reorder, "all": all_transforms}
