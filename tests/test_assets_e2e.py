"""End-to-end: synthetic registry -> analyze.py -> make_assets.py -> LaTeX -> consistency check."""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
PROCESSED = os.path.join(ROOT, "data", "processed")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.path.exists(os.path.join(PROCESSED, "dappscan.jsonl")),
                                 reason="needs data/processed (run scripts/01_prepare_data.py)")]


def _run(*args):
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from tests.fixtures.synthetic_results import build
    tmp = tmp_path_factory.mktemp("e2e")
    results = str(tmp / "results")
    n = build(results, PROCESSED, "dappscan")
    over = [f"paths.results={results}", f"paths.processed={PROCESSED}"]
    _run("experiments/analyze.py", "--dataset", "dappscan", "--n-boot", "30", "--set", *over)
    analysis = os.path.join(results, "analysis")
    with open(os.path.join(analysis, "robustness.json"), "w") as fh:     # needs saved models
        json.dump({"summary": {"all": {m: {"mean": -0.01, "std": 0.005} for m in "abcd"}},
                   "test": {"mean_diff": 0.004, "p_holm": 0.03, "supports_claim": True}}, fh)
    _run("experiments/scope.py", "--set", *over)
    out = str(tmp / "generated")
    log = _run("analysis/make_assets.py", "--results", results, "--out", out,
               "--reports", os.path.join(ROOT, "reports"))
    return {"tmp": tmp, "results": results, "out": out, "log": log, "runs": n}


def test_analysis_outputs_exist(pipeline):
    a = os.path.join(pipeline["results"], "analysis")
    for f in ("rq1_dappscan.json", "rq2_dappscan.json", "rq3_dappscan.json", "rq4_dappscan.json",
              "rq5_dappscan.json", "rq6_smartbugs_curated.json", "descriptive_dappscan.json",
              "rq1_untrained_dappscan.json", "scope.json"):
        assert os.path.exists(os.path.join(a, f)), f
    with open(os.path.join(a, "rq2_dappscan.json")) as fh:
        rq2 = json.load(fh)
    assert rq2["complete"] and len(rq2["comparisons"]) == 3
    assert all(c["n"] == 15 for c in rq2["comparisons"])


def test_assets_written(pipeline):
    t = os.path.join(pipeline["out"], "tables")
    for f in ("tab_datasets", "tab_modes_dappscan", "tab_modes_tests_dappscan", "tab_rq1_dappscan",
              "tab_rq4_dappscan", "tab_cost_dappscan", "tab_rq6_heldout", "tab_rq6_robustness",
              "tab_per_type_dappscan"):
        assert os.path.exists(os.path.join(t, f + ".tex")), f
    for f in ("fig_modes_dappscan", "fig_sensitivity_dappscan", "fig_dynamics_dappscan"):
        assert os.path.exists(os.path.join(pipeline["out"], "figures", f + ".pdf")), f


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_generated_latex_compiles(pipeline):
    doc = pipeline["tmp"] / "doc"
    doc.mkdir()
    tables = sorted(os.listdir(os.path.join(pipeline["out"], "tables")))
    body = "\n".join(f"\\input{{{pipeline['out']}/tables/{t}}}\\clearpage" for t in tables)
    figs = "\n".join(f"\\includegraphics[width=0.4\\linewidth]{{{pipeline['out']}/figures/{f}}}"
                     for f in sorted(os.listdir(os.path.join(pipeline["out"], "figures"))))
    (doc / "main.tex").write_text(
        "\\documentclass{article}\\usepackage{booktabs,graphicx,amsmath}\n"
        f"\\input{{{pipeline['out']}/numbers.tex}}\n\\begin{{document}}\n"
        "Feedback vs filtering: \\val{dappscan.c-vs-b.diff} (p = \\val{dappscan.c-vs-b.p}).\n"
        f"{body}\n{figs}\n\\end{{document}}\n")
    out = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                         cwd=doc, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:]
    assert (doc / "main.pdf").exists()


def test_undefined_value_fails_latex(pipeline):
    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not installed")
    doc = pipeline["tmp"] / "doc2"
    doc.mkdir()
    (doc / "main.tex").write_text(
        f"\\documentclass{{article}}\\input{{{pipeline['out']}/numbers.tex}}\n"
        "\\begin{document}\\val{no.such.key}\\end{document}\n")
    out = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                         cwd=doc, capture_output=True, text=True)
    assert out.returncode != 0 and "undefined value" in out.stdout


def test_consistency_check_flags_literals_and_unknown_keys(pipeline):
    from analysis.consistency_check import check
    paper = pipeline["tmp"] / "paper"
    (paper / "generated").mkdir(parents=True)
    shutil.copy(os.path.join(pipeline["out"], "numbers.tex"), paper / "generated" / "numbers.tex")
    (paper / "allowed_numbers.txt").write_text("0.5  # tau\n")
    (paper / "main.tex").write_text(
        "We use $\\tau=0.5$ and 5 folds \\cite{x2024}. % 0.999 in a comment is fine\n"
        "Feedback adds \\val{dappscan.c-vs-b.diff} macro-AP.\n"
        "Our F1 is 0.879 on 2,901 files (12.4\\%).\n"
        "See \\val{missing.key}.\n")
    problems = check(str(paper))
    joined = "\n".join(problems)
    assert "'0.879'" in joined and "'2,901'" in joined and "'12.4\\%'" in joined
    assert "missing.key" in joined
    assert not any("0.5" in p or "0.999" in p or "2024" in p for p in problems)
    assert len(problems) == 4


def test_every_interval_contains_its_printed_estimate(pipeline):
    import re
    t = os.path.join(pipeline["out"], "tables", "tab_modes_tests_dappscan.tex")
    cells = re.findall(r"([+-]\d\.\d{3}) \[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]", open(t).read())
    assert cells, "no pooled estimates found"
    for est, lo, hi in cells:
        assert float(lo) <= float(est) <= float(hi), (est, lo, hi)
