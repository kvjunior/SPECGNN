"""Experiment configuration.

Configurations are YAML files. A file may inherit from others:

    _base_: [../model/rgcn.yaml, ../train/generator.yaml]
    train:
      epochs: 50            # overrides the inherited value

Bases are merged left to right, then the file itself; mappings merge
recursively, every other value replaces. Command-line overrides use dotted
keys (``train.seed=43``, ``model.hidden=128``) and are parsed as YAML scalars.

``to_dataclass`` rejects unknown keys, so a typo in a config file is an error,
not a silently ignored setting. ``config_hash`` gives a stable identifier of the
fully resolved configuration, used to name and deduplicate runs.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import types
import typing
from dataclasses import dataclass
from typing import Any

import yaml

BASE_KEY = "_base_"


class ConfigError(ValueError):
    """Invalid configuration file, override, or value."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged into ``base`` (mappings recursively)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str, _stack: tuple[str, ...] = ()) -> dict:
    """Load a YAML file and resolve its ``_base_`` chain."""
    path = os.path.abspath(path)
    if path in _stack:
        chain = " -> ".join((*_stack, path))
        raise ConfigError(f"circular _base_ reference: {chain}")
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    bases = data.pop(BASE_KEY, [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        merged = deep_merge(merged, load_yaml(base_path, (*_stack, path)))
    return deep_merge(merged, data)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if not all(parts):
            raise ConfigError(f"empty key segment in override {item!r}")
        node = out
        for p in parts[:-1]:
            if not isinstance(node.get(p, {}), dict):
                raise ConfigError(f"override {item!r}: {p!r} is not a mapping")
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    return apply_overrides(load_yaml(path), overrides)


# ---------------------------------------------------------------- dataclasses
def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Check (and minimally convert) a YAML value against a type annotation."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is Any:
        return value
    if origin is typing.Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(value, a, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: {value!r} matches none of {args}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}")
        return to_dataclass(tp, value, where)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        inner = args[0] if args else Any
        items = [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
        return items if origin is list else tuple(items)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        return dict(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def to_dataclass(cls: type, data: dict | None, where: str = "") -> Any:
    """Build ``cls`` from a mapping; unknown keys and wrong types are errors."""
    data = data or {}
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown key(s) {unknown}; "
                          f"allowed: {sorted(fields)}")
    kwargs = {k: _coerce(v, hints[k], f"{where or cls.__name__}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def section(cfg: dict, name: str, cls: type) -> Any:
    """Build the dataclass for one top-level section (missing section = defaults)."""
    return to_dataclass(cls, cfg.get(name), name)


@dataclass
class PathsConfig:
    """Machine-specific locations; excluded from run identity (see identity_config)."""
    raw: str = "data/raw"
    processed: str = "data/processed"
    graphs: str = "data/graphs"
    verdicts: str = "data/verdicts"
    results: str = "results"


def identity_config(cfg: dict) -> dict:
    """The part of a resolved configuration that defines an experiment: everything
    except machine-specific paths. Two machines running the same experiment
    produce the same run id."""
    return {k: v for k, v in cfg.items() if k != "paths"}


# ---------------------------------------------------------------- identity
def _canonical(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)       # 1.0 and 1 hash identically
    return obj


def config_hash(cfg: Any, length: int = 12) -> str:
    """Stable hash of a resolved configuration (key order and 1 vs 1.0 do not matter)."""
    blob = json.dumps(_canonical(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def dump_yaml(cfg: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_canonical(cfg), fh, sort_keys=True, allow_unicode=True)
