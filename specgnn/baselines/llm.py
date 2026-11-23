"""LLM auditor baseline (Reviewer 1: LLM-based detectors).

A general-purpose LLM is asked, with a fixed and versioned prompt, which of the
17 SWC weaknesses a file contains and how confident it is. Settings are
deterministic where the API allows it (temperature 0); every response is cached
by (model, prompt version, file hash), so re-running reproduces the results
exactly and costs nothing. Input/output tokens are recorded for cost reporting.
Files longer than ``max_chars`` are truncated (counted in the statistics).

Requires the environment variable ANTHROPIC_API_KEY (or a populated cache).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np

from ..data.config import SWC_NAMES, SWC_TYPES
from .base import Baseline, register

PROMPT_VERSION = "v1"
SYSTEM = ("You are a smart-contract security auditor. You answer only with JSON and never "
          "invent weaknesses that are not supported by the code.")
TEMPLATE = """Audit the Solidity file below for the following weaknesses (SWC registry):
{catalogue}

Report only weaknesses that are present in this file. Answer with JSON only, exactly:
{{"findings": [{{"swc": "SWC-XXX", "confidence": <number between 0 and 1>}}]}}
Use an empty list if none is present.

```solidity
{source}
```"""


def build_prompt(source: str, max_chars: int) -> tuple[str, bool]:
    truncated = len(source) > max_chars
    catalogue = "\n".join(f"- {t}: {SWC_NAMES[t]}" for t in SWC_TYPES)
    return TEMPLATE.format(catalogue=catalogue, source=source[:max_chars]), truncated


def parse_findings(text: str) -> dict[str, float]:
    """Extract {SWC id: confidence} from a model answer; malformed answers give {}."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out: dict[str, float] = {}
    for f in data.get("findings", []) if isinstance(data, dict) else []:
        swc = str(f.get("swc", "")).strip().upper() if isinstance(f, dict) else ""
        mm = re.fullmatch(r"SWC-?(\d{3})", swc)
        if not mm:
            continue
        swc = f"SWC-{mm.group(1)}"
        try:
            conf = float(f.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if swc in SWC_TYPES:
            out[swc] = max(out.get(swc, 0.0), min(max(conf, 0.0), 1.0))
    return out


@register
class LLMAuditorBaseline(Baseline):
    name = "llm_auditor"
    kind = "llm"

    def __init__(self, seed=42, source=None, model: str = "claude-sonnet-4-6",
                 max_chars: int = 60000, max_tokens: int = 1024, cache_dir: str = "results/llm_cache",
                 retries: int = 3):
        super().__init__(seed, source, model=model, max_chars=max_chars, max_tokens=max_tokens,
                         cache_dir=cache_dir, retries=retries)
        self.last_stats: dict = {}

    def fit(self, train, val) -> dict:
        return {}

    def _call(self, prompt: str) -> dict:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set and the response is not cached")
        body = json.dumps({"model": self.params["model"], "max_tokens": self.params["max_tokens"],
                           "temperature": 0, "system": SYSTEM,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
            "content-type": "application/json", "x-api-key": key,
            "anthropic-version": "2023-06-01"})
        for attempt in range(self.params["retries"]):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.params["retries"] - 1:
                    raise
                time.sleep(2 ** attempt * 5)
        raise RuntimeError("unreachable")

    def predict(self, samples: list[dict]) -> np.ndarray:
        if self.source is None:
            raise ValueError("llm_auditor needs a source reader")
        cache = os.path.join(self.params["cache_dir"], f"{self.params['model']}-{PROMPT_VERSION}")
        os.makedirs(cache, exist_ok=True)
        out = np.zeros((len(samples), len(SWC_TYPES)), dtype=np.float32)
        stats = {"files": len(samples), "cached": 0, "called": 0, "truncated": 0,
                 "unparsable": 0, "input_tokens": 0, "output_tokens": 0}
        for i, s in enumerate(samples):
            prompt, truncated = build_prompt(self.source(s), self.params["max_chars"])
            stats["truncated"] += truncated
            key = hashlib.sha256(prompt.encode()).hexdigest()
            path = os.path.join(cache, key + ".json")
            if os.path.exists(path):
                with open(path) as fh:
                    resp = json.load(fh)
                stats["cached"] += 1
            else:
                resp = self._call(prompt)
                with open(path, "w") as fh:
                    json.dump(resp, fh)
                stats["called"] += 1
            usage = resp.get("usage", {})
            stats["input_tokens"] += int(usage.get("input_tokens", 0))
            stats["output_tokens"] += int(usage.get("output_tokens", 0))
            text = "".join(b.get("text", "") for b in resp.get("content", []) if isinstance(b, dict))
            findings = parse_findings(text)
            if not findings and '"findings"' not in text:
                stats["unparsable"] += 1
            for t, c in findings.items():
                out[i, SWC_TYPES.index(t)] = c
        self.last_stats = stats
        return self._check(out, len(samples))
