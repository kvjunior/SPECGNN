"""Solidity source normalization used for hashing and near-duplicate detection.

Normalization removes everything that does not change program meaning for the
purpose of duplicate detection: comments, whitespace, and the SPDX header.
String literals are preserved (comment markers inside strings are not comments).
"""
from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(
    r"""
    "(?:\\.|[^"\\])*"          # double-quoted string
  | '(?:\\.|[^'\\])*'          # single-quoted string
  | 0x[0-9a-fA-F]+             # hex literal
  | \d+(?:\.\d+)?(?:e\d+)?     # number
  | [A-Za-z_$][A-Za-z0-9_$]*   # identifier / keyword
  | ==|!=|<=|>=|&&|\|\||\+\+|--|\+=|-=|\*=|/=|=>|<<|>>|\*\*
  | [^\s]                      # any other single symbol
    """,
    re.VERBOSE,
)


def read_source(path: str) -> str:
    """Read a source file robustly (some upstream files are not valid UTF-8)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.decode("utf-8", errors="replace")


def strip_comments(src: str) -> str:
    """Remove // and /* */ comments, leaving string literals untouched."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            q = c
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j          # keep the newline
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            end = n if j == -1 else j + 2
            out.append("\n" * src.count("\n", i, end))  # preserve line numbers
            i = end
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tokenize(src: str) -> list[str]:
    """Tokenize comment-free Solidity source."""
    return _TOKEN_RE.findall(src)


def normalized_tokens(src: str) -> list[str]:
    return tokenize(strip_comments(src))


def normalized_hash(tokens: list[str]) -> str:
    """SHA-256 of the token stream: identical for formatting/comment-only edits."""
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def raw_hash(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def is_interface_only(tokens: list[str]) -> bool:
    """True if the file declares interfaces but no contract or library."""
    kinds = {tokens[i] for i in range(len(tokens) - 1)
             if tokens[i] in ("contract", "library", "interface")
             and re.match(r"[A-Za-z_$]", tokens[i + 1])}
    return "interface" in kinds and not ({"contract", "library"} & kinds)


def pragma_version(src: str) -> str | None:
    m = re.search(r"pragma\s+solidity\s+([^;]+);", src)
    return m.group(1).strip() if m else None
