"""Near-duplicate clustering of source files.

Two files are near-duplicates when the Jaccard similarity of their token
k-shingle sets is at least `threshold`. Candidates come from MinHash LSH; every
candidate pair is then verified with the exact Jaccard similarity, so the LSH
step only affects speed, never the final decision for a verified pair.
Clusters are connected components (union-find) of verified pairs plus exact
hash matches.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np

_MERSENNE = np.uint64((1 << 61) - 1)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def shingles(tokens: list[str], k: int) -> np.ndarray:
    """Unique 32-bit hashes of token k-grams (rolling combination of token hashes)."""
    if not tokens:
        return np.zeros(0, dtype=np.uint64)
    th = np.fromiter((zlib.crc32(t.encode("utf-8")) for t in tokens),
                     dtype=np.uint64, count=len(tokens))
    if len(th) < k:
        k = len(th)
    acc = np.zeros(len(th) - k + 1, dtype=np.uint64)
    for j in range(k):
        acc = acc * np.uint64(1_000_003) + th[j:len(th) - k + 1 + j]
    return np.unique(acc & np.uint64(0xFFFFFFFF))


def minhash_signatures(shingle_sets: list[np.ndarray], num_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 61) - 1, size=num_perm, dtype=np.uint64)
    sigs = np.full((len(shingle_sets), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(shingle_sets):
        if len(s) == 0:
            continue
        # (a*x + b) mod p, computed with wraparound then reduced; adequate for LSH.
        h = (np.outer(s, a) + b) % _MERSENNE
        sigs[i] = h.min(axis=0)
    return sigs


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (len(a) + len(b) - inter)


_KEYWORDS = frozenset(
    "pragma solidity contract library interface function modifier event returns return if "
    "else for while do public private internal external view pure payable memory storage "
    "calldata uint uint256 address bool string bytes mapping emit require assert revert new "
    "delete this msg sender value true false constructor import is using struct enum override "
    "virtual constant immutable indexed".split())


def identifier_tokens(tokens: list[str]) -> list[str]:
    """Identifiers and literals (SourcererCC/Allamanis fingerprint), keywords removed."""
    return [t for t in tokens
            if (t[0].isalpha() or t[0] in "_$\"'" or t[0].isdigit()) and t not in _KEYWORDS]


def multiset_jaccard(a: list[str], b: list[str]) -> float:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 1.0


def identifier_pairs(token_lists: list[list[str]], *, set_threshold: float,
                     multiset_threshold: float, min_tokens: int, num_perm: int, bands: int,
                     seed: int) -> list[tuple[int, int]]:
    """Pairs satisfying the Allamanis criterion (MinHash LSH candidates, verified exactly)."""
    idents = [identifier_tokens(t) for t in token_lists]
    sets = [np.unique(np.fromiter((zlib.crc32(x.encode("utf-8")) for x in t), dtype=np.uint64,
                                  count=len(t))) if len(t) >= min_tokens else np.zeros(0, np.uint64)
            for t in idents]
    sigs = minhash_signatures(sets, num_perm, seed + 1)
    rows = num_perm // bands
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * rows:(band + 1) * rows]
        for i, s in enumerate(sets):
            if len(s):
                buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 500:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))
    return sorted((i, j) for i, j in candidates
                  if jaccard(sets[i], sets[j]) >= set_threshold
                  and multiset_jaccard(idents[i], idents[j]) >= multiset_threshold)


def cluster(token_lists: list[list[str]], exact_hashes: list[str], *, k: int,
            num_perm: int, bands: int, threshold: float, seed: int,
            ident_criterion: dict | None = None) -> tuple[list[int], dict]:
    """Return a cluster id per item and summary statistics. ``ident_criterion`` (keys
    set_threshold, multiset_threshold, min_tokens) adds the Allamanis criterion."""
    n = len(token_lists)
    uf = UnionFind(n)

    # 1) exact duplicates
    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(exact_hashes):
        by_hash[h].append(i)
    exact_pairs = 0
    for idx in by_hash.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
            exact_pairs += 1

    # 2) near duplicates among one representative per exact-hash group
    reps = [idx[0] for idx in by_hash.values()]
    sh = [shingles(token_lists[i], k) for i in reps]
    sigs = minhash_signatures(sh, num_perm, seed)
    rows = num_perm // bands
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sigs[:, band * rows:(band + 1) * rows]
        for r in range(len(reps)):
            if len(sh[r]):
                buckets[block[r].tobytes()].append(r)
        for members in buckets.values():
            if 1 < len(members) <= 500:   # skip degenerate mega-buckets
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidates.add((members[x], members[y]))
    verified = 0
    for x, y in candidates:
        if jaccard(sh[x], sh[y]) >= threshold:
            uf.union(reps[x], reps[y])
            verified += 1

    ident_pairs = 0
    if ident_criterion is not None:
        for i, j in identifier_pairs(token_lists, num_perm=num_perm, bands=bands, seed=seed,
                                     **ident_criterion):
            uf.union(i, j)
            ident_pairs += 1

    roots = [uf.find(i) for i in range(n)]
    remap: dict[int, int] = {}
    ids = [remap.setdefault(r, len(remap)) for r in roots]
    stats = {
        "identifier_criterion_pairs": ident_pairs,
        "items": n,
        "exact_duplicate_links": exact_pairs,
        "lsh_candidate_pairs": len(candidates),
        "verified_near_duplicate_pairs": verified,
        "clusters": len(remap),
    }
    return ids, stats
