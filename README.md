# SpecGNN (revision, TSE-2026-05-0412)

Code for "Teaching Neural Detectors with Symbolic Rules", rebuilt for the major
revision. Status of each component:

| Component | Code | Docs | Status |
|---|---|---|---|
| Data pipeline (5 datasets, dedup, grouped splits) | `specgnn/data`, `scripts/01_prepare_data.py` | `data/README.md` | done |
| Code graphs (tree-sitter, 6 node / 4 edge types) | `specgnn/graph`, `scripts/02_build_graphs.py` | `specgnn/graph/features.py` | done |
| R-GCN candidate generator (mode a) | `specgnn/model`, `specgnn/training.py`, `scripts/train_generator.py` | docstrings | done, tuning |
| Specification checker (17 specs) | `specgnn/spec`, `scripts/03_eval_specs.py` | `docs/SPECIFICATIONS.md`, `docs/spec_table.md` | done |
| Shared infrastructure (config, seeding, run registry, timing) | `specgnn/utils`, `configs/` | docstrings | done |
| Feedback loop, modes a–d, verdict cache | `specgnn/feedback`, `specgnn/training.py` | `docs/FEEDBACK.md` | done |
| Experiment matrix, runner, RQ2–RQ5 analysis | `experiments/` | `docs/EXPERIMENTS.md` | done; runs pending |
| Baselines (RQ1): 8 learned, 2 tools, LLM | `specgnn/baselines`, `scripts/06_run_baseline.py` | `docs/BASELINES.md` | done; runs pending |
| Paper assets: tables, figures, `numbers.tex`, consistency check | `analysis/` | `docs/EXPERIMENTS.md` §4 | done |
| Release: pinned data, manifests, splits, Docker, CI, citation | `scripts/00_fetch_data.py`, `data/`, `Dockerfile*`, `.github/` | `REPRODUCE.md`, `DATA_LICENSES.md` | done |
| Results plausibility checker (evaluator bounds) | `scripts/09_check_results_plausibility.py` | docstring | done |
| Literature digest and calibrations | `docs/LITERATURE.md`, `scripts/08_label_rule_agreement.py` | — | done |
| Generalization and robustness (RQ6), probes (RQ3) | `specgnn/robustness`, `scripts/07_build_transformed.py`, `experiments/{robustness,scope,probing}.py` | docstrings | done; runs pending |
| Baselines, RQ scripts | – | – | later |

Start with **[REPRODUCE.md](REPRODUCE.md)**: setup (Docker or local), data download
with content verification, experiments, analysis, and a map from every paper element
to the command and file that produce it.

```bash
python scripts/00_fetch_data.py --raw data/raw   # pinned download + SHA-256 verification
make data graphs RAW=data/raw                     # datasets, splits, code graphs
python scripts/export_splits.py --check           # splits identical to the released ones
make test                                         # all tests (make test-fast: unit tests only)
```

| Path | Contents |
|---|---|
| `specgnn/` | the package: data, graphs, model, specification checker, feedback, baselines, robustness, evaluation, utilities |
| `scripts/` | pipeline stages `00`–`07` |
| `experiments/` | configuration selection, run matrix and runner, pre-registered analyses |
| `analysis/` | paper tables, figures, `numbers.tex`, consistency check |
| `configs/` | every experiment as YAML (inheritance, command-line overrides) |
| `data/` | pinned sources, content manifests, released splits (no dataset content: see `DATA_LICENSES.md`) |
| `docs/` | analysis plan (pre-registered), specifications, feedback design, baselines, experiments |
| `tests/` | unit and end-to-end tests |

License: MIT (`LICENSE`). Citation: `CITATION.cff`.
