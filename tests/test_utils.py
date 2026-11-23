import dataclasses
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specgnn.feedback.pseudo_labels import FeedbackConfig
from specgnn.model.losses import LossConfig
from specgnn.model.rgcn import ModelConfig
from specgnn.spec.checker import CheckerConfig
from specgnn.training import DataConfig, TrainConfig
from specgnn.utils import (
    ConfigError,
    Run,
    Timer,
    config_hash,
    is_complete,
    list_runs,
    load_config,
    read_json,
    section,
    set_seed,
    to_dataclass,
    write_json,
)
from specgnn.utils.config import PathsConfig, apply_overrides, deep_merge, identity_config

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXPERIMENTS = os.path.join(ROOT, "configs", "experiments")


# ---------------------------------------------------------------- config
def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_deep_merge_is_recursive_and_pure():
    a = {"x": {"y": 1, "z": 2}, "k": [1]}
    b = {"x": {"y": 9}, "k": [2]}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 9, "z": 2}, "k": [2]}
    assert a == {"x": {"y": 1, "z": 2}, "k": [1]}          # inputs untouched


def test_base_chain_and_override_order(tmp_path):
    _write(tmp_path, "a.yaml", "train: {epochs: 10, lr: 0.1}\n")
    _write(tmp_path, "b.yaml", "_base_: a.yaml\ntrain: {lr: 0.2}\n")
    c = _write(tmp_path, "c.yaml", "_base_: [b.yaml]\ntrain: {seed: 7}\n")
    cfg = load_config(c, ["train.epochs=3"])
    assert cfg == {"train": {"epochs": 3, "lr": 0.2, "seed": 7}}


def test_circular_base_is_rejected(tmp_path):
    _write(tmp_path, "a.yaml", "_base_: b.yaml\n")
    b = _write(tmp_path, "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(b)


def test_override_parsing():
    cfg = apply_overrides({}, ["a.b=3", "a.c=0.5", "a.d=null", "a.e=true", "f=text"])
    assert cfg == {"a": {"b": 3, "c": 0.5, "d": None, "e": True}, "f": "text"}
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a.b=2"])                  # a is not a mapping


def test_unknown_key_and_wrong_type_are_errors():
    with pytest.raises(ConfigError, match="unknown"):
        to_dataclass(TrainConfig, {"epoch": 3})                # typo
    with pytest.raises(ConfigError, match="integer"):
        to_dataclass(TrainConfig, {"epochs": 2.5})
    with pytest.raises(ConfigError, match="integer"):
        to_dataclass(TrainConfig, {"epochs": True})            # bool is not an int
    assert to_dataclass(TrainConfig, {"lr": 1}).lr == 1.0      # int accepted as float
    assert to_dataclass(DataConfig, {"safe_cap": None}).safe_cap is None


@pytest.mark.parametrize("name", sorted(os.listdir(EXPERIMENTS)))
def test_every_experiment_config_builds(name):
    cfg = load_config(os.path.join(EXPERIMENTS, name))
    section(cfg, "model", ModelConfig)
    section(cfg, "loss", LossConfig)
    section(cfg, "train", TrainConfig)
    section(cfg, "data", DataConfig)
    section(cfg, "spec", CheckerConfig)
    section(cfg, "feedback", FeedbackConfig)
    section(cfg, "paths", PathsConfig)
    assert set(cfg) <= {"paths", "model", "loss", "train", "data", "spec", "feedback", "selection"}


def test_yaml_defaults_match_dataclass_defaults():
    """The YAML files document the defaults; they must not drift from the code."""
    # base files only: configs/selected.yaml legitimately overrides them after selection
    cfg = {}
    for rel in ("paths.yaml", "model/rgcn.yaml", "train/generator.yaml", "spec/checker.yaml",
                "feedback/both.yaml"):
        cfg = deep_merge(cfg, load_config(os.path.join(ROOT, "configs", rel)))
    for name, cls in (("model", ModelConfig), ("loss", LossConfig), ("spec", CheckerConfig),
                      ("paths", PathsConfig)):
        assert section(cfg, name, cls) == cls(), name
    train = dataclasses.asdict(section(cfg, "train", TrainConfig))
    assert train == dataclasses.asdict(TrainConfig())
    assert section(cfg, "feedback", FeedbackConfig) == FeedbackConfig(enabled=True)


def test_run_identity_ignores_machine_paths():
    a = {"paths": {"raw": "/a"}, "train": {"seed": 1}}
    b = {"paths": {"raw": "/b"}, "train": {"seed": 1}}
    assert config_hash(identity_config(a)) == config_hash(identity_config(b))


def test_config_hash_is_stable():
    a = {"x": 1.0, "y": {"b": [1, 2], "a": "s"}}
    b = {"y": {"a": "s", "b": [1, 2]}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": {"a": "s", "b": [1, 2]}})
    assert config_hash(ModelConfig()) == config_hash(dataclasses.asdict(ModelConfig()))


# ---------------------------------------------------------------- runs
def test_write_json_handles_numpy_and_nan(tmp_path):
    p = str(tmp_path / "m.json")
    write_json(p, {"a": np.float32(0.5), "b": np.int64(3), "c": np.arange(2), "d": math.nan})
    assert read_json(p) == {"a": 0.5, "b": 3, "c": [0, 1], "d": None}
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]


def test_run_lifecycle_skip_and_restart(tmp_path):
    cfg = {"train": {"seed": 1}}
    run = Run.open(str(tmp_path), "gen", cfg)
    assert run is not None and os.path.exists(run.path("env.json"))
    assert read_json(run.path("status.json"))["state"] == "running"
    assert not is_complete(run.dir)
    # an interrupted run is restarted from scratch
    with open(run.path("partial.txt"), "w") as fh:
        fh.write("x")
    run = Run.open(str(tmp_path), "gen", cfg)
    assert not os.path.exists(run.path("partial.txt"))
    run.finish({"f1": 0.5}, {"train": {"seconds": 1.0, "calls": 1}})
    assert is_complete(run.dir)
    assert Run.open(str(tmp_path), "gen", cfg) is None               # complete -> skipped
    assert Run.open(str(tmp_path), "gen", cfg, overwrite=True) is not None
    other = Run.open(str(tmp_path), "gen", {"train": {"seed": 2}})
    assert other.id != run.id                                         # config decides identity


def test_failed_run_records_traceback_and_is_listed_separately(tmp_path):
    run = Run.open(str(tmp_path), "gen", {"a": 1})
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        run.fail(exc)
    status = read_json(run.path("status.json"))
    assert status["state"] == "failed" and "boom" in status["error"]
    done = Run.open(str(tmp_path), "gen", {"a": 2})
    done.finish({"f1": 1.0})
    runs = list_runs(str(tmp_path), "gen")
    assert [r["metrics"] for r in runs] == [{"f1": 1.0}]
    assert len(list_runs(str(tmp_path), "gen", complete_only=False)) == 2


def test_invalid_run_kind():
    with pytest.raises(ValueError):
        Run.make_id("bad kind", {})


# ---------------------------------------------------------------- timing, seeding
def test_timer_accumulates_and_survives_exceptions():
    t = Timer()
    for _ in range(2):
        with t("train"):
            pass
    with pytest.raises(KeyError):
        with t("check"):
            raise KeyError
    s = t.summary()
    assert s["train"]["calls"] == 2 and s["check"]["calls"] == 1
    assert s["total_seconds"] >= s["train"]["seconds"]
    json.dumps(s)


def test_set_seed_makes_torch_and_numpy_repeatable():
    import torch
    set_seed(5)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(5)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])
