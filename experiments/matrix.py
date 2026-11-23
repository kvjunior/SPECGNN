"""The complete run matrix of the analysis plan (Sections 4 and 6).

Every job is (experiment config, overrides, fold, seed). Run ids are computed
exactly as scripts/05_train.py computes them, so results can be located without
scanning configurations. Priorities order execution: the primary analysis
(RQ2/RQ3 on DAppSCAN) first, descriptive sensitivity runs last.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from specgnn.feedback.pseudo_labels import FeedbackConfig
from specgnn.training import DataConfig
from specgnn.utils import config_hash, load_config, section
from specgnn.utils.config import identity_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = os.path.join(ROOT, "configs")
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))
SENSITIVITY_SEED = 42

DATASET_OVERRIDES = {
    "dappscan": [],
    "resource2": ["data.dataset=resource2", "data.safe_cap=null", "data.evaluate_heldout=false"],
}

# variant -> (experiment file, extra overrides)
VARIANTS = {
    "plain": ("plain.yaml", []),
    "both": ("both.yaml", []),
    "refute_only": ("refute_only.yaml", []),
    "confirm_only": ("confirm_only.yaml", []),
    "rules_features": ("rules_features.yaml", []),
    "both_nocorr": ("both.yaml", ["feedback.correction=false"]),
}

# RQ5 one-factor-at-a-time sensitivity around the defaults (mode c, seed 42)
SENSITIVITY = {
    "every": ("feedback.every", [1, 3, 10]),
    "w_pos": ("feedback.w_pos", [0.5, 1.0, 2.0]),
    "w_neg": ("feedback.w_neg", [1.0, 1.5, 3.0]),
    "tau": ("train.tau", [0.3, 0.4, 0.6, 0.7]),
}
SAFE_CAPS = [500, 1000, 4000, "null"]       # DAppSCAN only, plain and both


@dataclass(frozen=True)
class Job:
    name: str                     # e.g. "dappscan/both", "dappscan/baseline_drgcn"
    config: str                   # path relative to configs/
    overrides: tuple[str, ...]
    fold: int
    seed: int
    priority: int
    tags: tuple[str, ...] = field(default=())
    script: str = "05_train.py"
    save_model: bool = False      # weights needed for robustness and probing (RQ6, RQ3)

    @property
    def seed_key(self) -> str:
        return "baseline.seed" if self.script == "06_run_baseline.py" else "train.seed"

    def resolved(self, extra: list[str] | None = None) -> dict:
        cfg = load_config(os.path.join(CONFIGS, self.config),
                          [*self.overrides, f"{self.seed_key}={self.seed}", *(extra or [])])
        cfg.setdefault("run", {})["fold"] = self.fold
        return cfg

    def run_id(self, extra: list[str] | None = None) -> str:
        cfg = self.resolved(extra)
        dcfg = section(cfg, "data", DataConfig)
        if self.script == "06_run_baseline.py":
            name = cfg["baseline"]["name"]
            return f"{dcfg.dataset}.baseline_{name}-{config_hash(identity_config(cfg))}"
        fcfg = section(cfg, "feedback", FeedbackConfig)
        if fcfg.enabled:
            variant = f"fb_{fcfg.policy}" + ("" if fcfg.correction else "_nocorr")
        elif dcfg.verdict_features:
            variant = "rules_features"
        else:
            variant = "plain"
        return f"{dcfg.dataset}.{variant}-{config_hash(identity_config(cfg))}"

    def command_args(self) -> list[str]:
        flags = ["--save-model"] if self.save_model else []
        return [*flags, "--config", os.path.join(CONFIGS, self.config), "--fold", str(self.fold),
                "--set", *self.overrides, f"{self.seed_key}={self.seed}"]


def _grid(name, config, overrides, priority, tags, seeds=SEEDS, script="05_train.py",
          save_model=False) -> list[Job]:
    return [Job(name, config, tuple(overrides), f, s, priority, tuple(tags), script, save_model)
            for s in seeds for f in FOLDS]


NEURAL_BASELINES = ("drgcn", "ame", "cge", "efevd", "scvhunter", "mando_hgt")
FEATURE_BASELINES = ("contractward", "feature_lr")
UNTRAINED_BASELINES = ("slither", "mythril", "llm_auditor")


def all_jobs() -> list[Job]:
    jobs: list[Job] = []
    # priority 1-2: RQ2 / RQ3 (and RQ6, which reuses these runs' held-out predictions)
    for p, ds in ((1, "dappscan"), (2, "resource2")):
        for v in ("plain", "both"):
            cfg, extra = VARIANTS[v]
            jobs += _grid(f"{ds}/{v}", f"experiments/{cfg}", DATASET_OVERRIDES[ds] + extra, p,
                           ("rq2", "rq3", "rq6"), save_model=(ds == "dappscan"))
    # priority 3-4: RQ4 variants and the RQ5 correction ablation
    for p, ds in ((3, "dappscan"), (4, "resource2")):
        for v in ("refute_only", "confirm_only", "rules_features", "both_nocorr"):
            cfg, extra = VARIANTS[v]
            tag = "rq5" if v == "both_nocorr" else "rq4"
            jobs += _grid(f"{ds}/{v}", f"experiments/{cfg}", DATASET_OVERRIDES[ds] + extra, p, (tag,))
    # priority 5: RQ5 descriptive sensitivity (DAppSCAN, seed 42)
    for factor, (key, values) in SENSITIVITY.items():
        for val in values:
            jobs += _grid(f"dappscan/both/{factor}={val}", "experiments/both.yaml", [f"{key}={val}"], 5,
                          ("rq5", "sensitivity"), seeds=(SENSITIVITY_SEED,))
    for cap in SAFE_CAPS:
        for v in ("plain", "both"):
            jobs += _grid(f"dappscan/{v}/safe_cap={cap}", f"experiments/{VARIANTS[v][0]}",
                          [f"data.safe_cap={cap}"],
                          5, ("rq5", "sensitivity"), seeds=(SENSITIVITY_SEED,))
    # priority 6: RQ1 baselines (same folds; untrained tools and the LLM run once per fold)
    for ds in ("dappscan", "resource2"):
        for b in NEURAL_BASELINES + FEATURE_BASELINES:
            jobs += _grid(f"{ds}/baseline_{b}", f"baselines/{b}.yaml", DATASET_OVERRIDES[ds], 6,
                          ("rq1",), script="06_run_baseline.py")
        for b in UNTRAINED_BASELINES:
            jobs += _grid(f"{ds}/baseline_{b}", f"baselines/{b}.yaml", DATASET_OVERRIDES[ds], 6,
                          ("rq1", "untrained"), seeds=(SENSITIVITY_SEED,), script="06_run_baseline.py")
    return sorted(jobs, key=lambda j: (j.priority, j.name, j.seed, j.fold))


def jobs_for(names: list[str] | None = None, tags: list[str] | None = None,
             max_priority: int | None = None) -> list[Job]:
    out = all_jobs()
    if names:
        out = [j for j in out if j.name in names]
    if tags:
        out = [j for j in out if set(tags) & set(j.tags)]
    if max_priority is not None:
        out = [j for j in out if j.priority <= max_priority]
    return out
