import os
import sys

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.eval.stats import (
    a12,
    a12_magnitude,
    a12_paired,
    average_precision,
    cluster_bootstrap_diff,
    compare,
    compare_family,
    describe,
    holm,
    macro_ap,
    paired_test,
    supports_claim,
)


# ---------------------------------------------------------------- average precision
@pytest.mark.parametrize("seed", range(20))
def test_average_precision_matches_sklearn_including_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 400))
    y = rng.random(n) < rng.uniform(0.02, 0.6)
    if not y.any():
        y[0] = True
    # coarse rounding creates many tied scores
    s = np.round(rng.random(n), int(rng.integers(1, 4)))
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_average_precision_edge_cases():
    assert np.isnan(average_precision([0, 0, 0], [0.1, 0.2, 0.3]))
    assert average_precision([1, 1], [0.3, 0.3]) == 1.0
    assert average_precision([1, 0], [0.9, 0.1]) == 1.0
    assert average_precision([0, 1], [0.9, 0.1]) == 0.5


def test_macro_ap_skips_undefined_types_and_masks():
    probs = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]])
    y = np.array([[1, 1], [0, 1], [0, 1]])
    mask = np.array([[1, 1], [1, 1], [0, 1]])        # third row unknown for type 0
    # type 1 has no negative -> skipped; type 0 uses rows 0-1 only -> AP 1.0
    assert macro_ap(probs, y, mask) == 1.0


# ---------------------------------------------------------------- tests and corrections
def test_paired_test_matches_scipy_and_handles_zeros():
    rng = np.random.default_rng(1)
    a, b = rng.random(15), rng.random(15)
    assert paired_test(a, b) == pytest.approx(sps.wilcoxon(a, b).pvalue)
    assert paired_test(a, a) == 1.0
    # 15 distinct positive differences (no ties) -> exact two-sided p = 2 / 2**15
    assert paired_test(a + np.linspace(0.1, 1.5, 15), a) == pytest.approx(2 / 2 ** 15)
    with pytest.raises(ValueError):
        paired_test([1, 2], [1])


def test_holm_hand_computed():
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == [1.0, 1.0]      # 0.5*2 = 1.0 carries over (monotone)
    assert holm([]) == []
    with pytest.raises(ValueError):
        holm([1.2])


def test_a12_values_and_magnitudes():
    assert a12([1, 2, 3], [1, 2, 3]) == 0.5
    assert a12([4, 5], [1, 2]) == 1.0
    assert a12([1, 2], [4, 5]) == 0.0
    assert a12_paired([2, 2, 1, 1], [1, 1, 1, 2]) == pytest.approx((2 + 0.5) / 4)
    assert a12_magnitude(0.5) == "negligible"
    assert a12_magnitude(0.56) == "small"
    assert a12_magnitude(0.30) == "medium"      # symmetric: 1 - 0.30 = 0.70
    assert a12_magnitude(0.71) == "large"


def test_compare_family_applies_holm_and_decision_rule():
    rng = np.random.default_rng(0)
    base = rng.random(15)
    fam = compare_family({"big": (base + 0.2, base), "none": (base, base)})
    big, none = fam
    assert big.p_holm == pytest.approx(min(1.0, 2 * big.p))      # smallest p times 2
    assert supports_claim(big) and not supports_claim(none)
    assert (big.wins, big.ties, big.losses) == (15, 0, 0)
    with pytest.raises(ValueError):
        supports_claim(compare("x", base + 0.2, base))           # no Holm value yet


def test_decision_rule_rejects_negative_direction():
    base = np.linspace(0, 1, 15)
    (worse,) = compare_family({"w": (base - 0.2, base)})
    assert worse.p_holm < 0.05 and not supports_claim(worse)


# ---------------------------------------------------------------- bootstrap
def test_cluster_bootstrap_resamples_whole_groups_and_is_deterministic():
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    seen = []

    def metric(idx):
        g = groups[idx]
        # every group appears with both of its members, or not at all
        for k in np.unique(g):
            assert (g == k).sum() % 2 == 0
        seen.append(len(idx))
        return float(idx.mean())

    r1 = cluster_bootstrap_diff(metric, groups, n_boot=200, seed=3)
    r2 = cluster_bootstrap_diff(metric, groups, n_boot=200, seed=3)
    assert r1 == r2 and r1["n_valid"] == 200
    assert r1["low"] <= r1["estimate"] <= r1["high"]
    assert set(seen) == {8}


def test_cluster_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(7)
    n = 600
    y = rng.random(n) < 0.2
    good = y + rng.normal(0, 0.6, n)          # informative scores
    bad = rng.random(n)                        # uninformative scores
    groups = np.arange(n) // 3

    def diff(idx):
        return average_precision(y[idx], good[idx]) - average_precision(y[idx], bad[idx])

    r = cluster_bootstrap_diff(diff, groups, n_boot=500, seed=0)
    assert r["low"] > 0 and r["p_le_zero"] == 0.0


def test_describe():
    d = describe([1.0, 2.0, 3.0])
    assert d["mean"] == 2.0 and d["median"] == 2.0 and d["std"] == pytest.approx(1.0)
