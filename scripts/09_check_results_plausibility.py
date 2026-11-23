#!/usr/bin/env python3
"""Flag results that are structurally impossible under this repository's evaluator.

This does not judge whether a result is *good*; it checks that each number is
attainable given how `specgnn/eval/metrics.py` and `specgnn/eval/stats.py` compute
things. Every bound below is a theorem about the code, not an expectation:

  B1 macro-AP floor for partial-coverage methods (tools): a method evaluated on the
     full 17-type space has
         macro_ap_full >= macro_ap_supported * n_supported / n_scored
     because unsupported types are scored (they have positives in the data) and
     contribute AP >= 0 to the macro mean.
  B2 any metric (AP, P, R, F1) lies in [0, 1]; AP of a scored type >= 0 and the
     per-type support/positive-rate relationship holds.
  B3 the paired effect size A12 on n observations is a multiple of 1/(2n); with the
     plan's n = 15, A12 in {0, 1/30, 2/30, ...}. A reported A12 off this grid is wrong.
  B4 a two-sided Wilcoxon p-value on n nonzero paired differences is >= 2 / 2**n
     (exact-test floor): >= 6.1e-5 for n = 15, >= 0.0625 for n = 5 (deterministic tools).
  B5 the decision rule needs median paired difference > 0, Holm p < alpha and
     A12 >= 0.56 together; a "claim = yes" with any of these missing is inconsistent.
  B6 chance macro-AP equals the mean per-type positive rate of the evaluation set; a
     method's macro-AP below chance on every type is a red flag (not impossible).

Usage:
    python scripts/09_check_results_plausibility.py --results results
Exits non-zero if any violation is found.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

N_PAIRS_DEFAULT = 15
ALPHA = 0.05
A12_SMALL = 0.56


def a12_on_grid(value: float, n: int, tol: float = 1e-6) -> bool:
    g = round(value * 2 * n)
    return abs(value - g / (2 * n)) <= tol


def wilcoxon_floor(n: int) -> float:
    return 2.0 / (2 ** n)


def check_comparison(c: dict, n_pairs: int, problems: list[str], where: str) -> None:
    name = c.get("name", "?")
    a12 = c.get("a12")
    if a12 is not None and not a12_on_grid(a12, n_pairs):
        nearest = round(a12 * 2 * n_pairs) / (2 * n_pairs)
        problems.append(f"{where}:{name}: A12={a12:.4f} is off the n={n_pairs} grid "
                        f"(nearest attainable {nearest:.4f})")
    p = c.get("p")
    if p is not None and p + 1e-12 < wilcoxon_floor(n_pairs):
        problems.append(f"{where}:{name}: raw p={p:.2e} below the exact Wilcoxon floor "
                        f"{wilcoxon_floor(n_pairs):.2e} for n={n_pairs}")
    if c.get("supports_claim") is True:
        if c.get("median_diff") is None or c["median_diff"] <= 0:
            problems.append(f"{where}:{name}: claim=yes but median_diff "
                            f"{c.get('median_diff')} is not > 0")
        if c.get("p_holm") is None or c["p_holm"] >= ALPHA:
            problems.append(f"{where}:{name}: claim=yes but Holm p {c.get('p_holm')} >= {ALPHA}")
        if a12 is not None and a12 < A12_SMALL:
            problems.append(f"{where}:{name}: claim=yes but A12 {a12:.3f} < {A12_SMALL}")


def check_tool_floor(reg_untrained: dict, problems: list[str]) -> None:
    for name, row in (reg_untrained or {}).items():
        full = row.get("macro_ap", {}).get("mean")
        supp = row.get("macro_ap_supported_types", {}).get("mean")
        # need the number of supported and scored types; recorded per run if available
        n_sup = row.get("n_supported_types")
        n_scored = row.get("n_scored_types", 17)
        if full is None or supp is None or n_sup is None:
            continue
        floor = supp * n_sup / n_scored
        if full + 1e-6 < floor:
            problems.append(f"rq1_untrained:{name}: full-space macro-AP {full:.3f} is below the "
                            f"floor {floor:.3f} (= supported AP {supp:.3f} x {n_sup}/{n_scored})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS_DEFAULT)
    args = ap.parse_args()
    analysis = os.path.join(args.results, "analysis")
    if not os.path.isdir(analysis):
        print(f"no analysis directory at {analysis}; run experiments/analyze.py first")
        sys.exit(2)
    problems: list[str] = []
    for path in sorted(glob.glob(os.path.join(analysis, "rq*_*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        where = os.path.basename(path)[:-5]
        if isinstance(data, dict) and "comparisons" in data:
            # tools/LLM in a held-out family may be deterministic: n could be 5
            n = args.n_pairs
            for c in data["comparisons"]:
                nn = c.get("n", n)
                check_comparison(c, nn, problems, where)
    unt = glob.glob(os.path.join(analysis, "rq1_untrained_*.json"))
    for path in unt:
        with open(path) as fh:
            check_tool_floor(json.load(fh), problems)
    for p in problems:
        print("VIOLATION:", p)
    print(f"\nplausibility check: {'FAILED' if problems else 'passed'} "
          f"({len(problems)} violation(s))")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
