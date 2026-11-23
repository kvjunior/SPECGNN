import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.analyze import FAMILIES, family_result, paired
from experiments.matrix import FOLDS, SEEDS, all_jobs, jobs_for


def test_matrix_counts_and_priorities():
    jobs = all_jobs()
    per = lambda name: sum(j.name == name for j in jobs)          # noqa: E731
    for name in ("dappscan/plain", "dappscan/both", "resource2/plain", "resource2/both",
                 "dappscan/refute_only", "dappscan/both_nocorr", "resource2/rules_features"):
        assert per(name) == len(SEEDS) * len(FOLDS), name
    assert {j.priority for j in jobs_for(max_priority=1)} == {1}
    assert len(jobs_for(max_priority=1)) == 30
    sens = [j for j in jobs if "sensitivity" in j.tags]
    assert all(j.seed == 42 for j in sens) and len(sens) == (13 + 8) * len(FOLDS)


def test_baseline_jobs():
    jobs = all_jobs()
    for ds in ("dappscan", "resource2"):
        assert sum(j.name == f"{ds}/baseline_drgcn" for j in jobs) == 15
        assert sum(j.name == f"{ds}/baseline_slither" for j in jobs) == 5     # deterministic
    b = next(j for j in jobs if j.name == "dappscan/baseline_cge")
    assert b.script == "06_run_baseline.py" and b.run_id().startswith("dappscan.baseline_cge-")
    assert "baseline.seed=" in " ".join(b.command_args())


def test_run_ids_are_unique_and_variant_named():
    jobs = all_jobs()
    ids = [j.run_id() for j in jobs]
    assert len(ids) == len(set(ids))
    j = next(j for j in jobs if j.name == "dappscan/both_nocorr")
    assert j.run_id().startswith("dappscan.fb_both_nocorr-")
    j = next(j for j in jobs if j.name == "resource2/rules_features")
    assert j.run_id().startswith("resource2.rules_features-")


class FakeRegistry:
    """Metrics for two variants: 'both' beats 'plain' by 0.05 in filtered AP on every pair."""

    def __init__(self, missing=()):
        self.missing = set(missing)

    def metrics(self, variant, fold, seed):
        if (variant, fold, seed) in self.missing or variant not in ("plain", "both"):
            return None
        base = 0.1 + 0.01 * fold + 0.001 * seed
        ap_f = base + (0.05 if variant == "both" else 0.0)
        mk = lambda ap: {"macro": {"ap": ap, "f1": ap}, "micro": {"f1": ap},   # noqa: E731
                         "any_vulnerability": {"f1": ap}, "calibrated": {"micro": {"f1": ap}}}
        return {"test": {"filtered": mk(ap_f), "unfiltered": mk(base)}}

    def run_dir(self, *_):
        return None


def test_family_result_detects_consistent_improvement():
    res = family_result(FakeRegistry(), "rq2", FAMILIES["rq2"], groups={})
    by = {c["name"]: c for c in res["comparisons"]}
    assert res["complete"]
    assert by["c_vs_b"]["n"] == 15 and by["c_vs_b"]["supports_claim"]
    assert np.isclose(by["c_vs_b"]["mean_diff"], 0.05)
    assert not by["b_vs_a"]["supports_claim"]        # plain filtered == plain unfiltered here


def test_incomplete_family_is_flagged_and_pairs_align():
    reg = FakeRegistry(missing={("both", 0, 42)})
    xa, xb, keys = paired(reg, ("both", "filtered"), ("plain", "filtered"))
    assert len(xa) == 14 and (0, 42) not in keys
    assert not family_result(reg, "rq2", FAMILIES["rq2"], groups={})["complete"]


def test_primary_dappscan_runs_save_models():
    jobs = all_jobs()
    prim = [j for j in jobs if j.name in ("dappscan/plain", "dappscan/both")]
    assert len(prim) == 30 and all("--save-model" in j.command_args() for j in prim)
    assert not any("--save-model" in j.command_args() for j in jobs if j.name == "resource2/both")


class HeldoutRegistry(FakeRegistry):
    def metrics(self, variant, fold, seed):
        m = super().metrics(variant, fold, seed)
        if m is not None:
            m["smartbugs_curated"] = m["test"]
        return m


def test_heldout_family_reads_the_heldout_split():
    from experiments.analyze import HELDOUT_FAMILY
    res = family_result(HeldoutRegistry(), "rq6_smartbugs_curated", HELDOUT_FAMILY, {},
                        split="smartbugs_curated")
    by = {c["name"]: c for c in res["comparisons"]}
    assert res["complete"] and by["c_vs_b"]["supports_claim"]
    # a registry without the held-out split yields no pairs rather than wrong numbers
    res2 = family_result(FakeRegistry(), "rq6_x", HELDOUT_FAMILY, {}, split="smartbugs_curated")
    assert res2["comparisons"] == [] and not res2["complete"]


def test_transformed_samples_replace_graph_and_verdicts():
    from experiments.robustness import transformed
    s = {"uid": "u", "node_y": np.ones((3, 17)), "node_mask": np.ones((3, 17)), "y": np.zeros(17),
         "graph": {"node_type": np.zeros(3)}}
    g = {"node_type": np.zeros(5)}
    out, v = transformed([s], {"u": {"graph": g, "verdicts": "V"}})
    assert out[0]["graph"] is g and out[0]["node_y"].shape == (5, 17) and v == {"u": "V"}
    assert out[0]["y"] is s["y"]                       # file labels unchanged


def test_probe_detects_linearly_encoded_information():
    from experiments.probing import probe
    rng = np.random.default_rng(0)
    h = rng.normal(size=(600, 8))
    y = np.zeros((600, 17), dtype=bool)
    y[:, 0] = h[:, 0] > 0.5                            # encoded in dimension 0
    out = probe(h[:400], y[:400], h[400:], y[400:], rng)
    assert out["SWC-100"] > 0.95 and len(out) == 1


def test_plausibility_checker_catches_impossible_values(tmp_path):
    import importlib
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    chk = importlib.import_module("09_check_results_plausibility")
    # A12 grid (n=15): 0.74 is off-grid, 0.733 is on-grid
    assert not chk.a12_on_grid(0.74, 15) and chk.a12_on_grid(11 / 15, 15)
    assert chk.wilcoxon_floor(15) < 1e-4 and abs(chk.wilcoxon_floor(5) - 0.0625) < 1e-9
    probs = []
    chk.check_comparison({"name": "x", "a12": 0.74, "p": 1e-6, "supports_claim": True,
                          "median_diff": 0.01, "p_holm": 0.01}, 15, probs, "t")
    assert any("off the n=15 grid" in p for p in probs)
    probs2 = []
    chk.check_comparison({"name": "y", "a12": 11 / 15, "p": 0.01, "supports_claim": True,
                          "median_diff": -0.01, "p_holm": 0.01}, 15, probs2, "t")
    assert any("median_diff" in p for p in probs2)
    probs3 = []
    chk.check_tool_floor({"slither": {"macro_ap": {"mean": 0.058},
                                      "macro_ap_supported_types": {"mean": 0.141},
                                      "n_supported_types": 13, "n_scored_types": 17}}, probs3)
    assert any("below the floor" in p for p in probs3)
