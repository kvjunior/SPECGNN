"""Wall-clock accounting for cost reporting (Reviewer 2: training and retraining time).

    timer = Timer()
    with timer("train"):
        ...
    with timer("feedback.check"):     # dotted names group sections in reports
        ...
    timer.summary()  ->  {"train": {"seconds": 12.3, "calls": 5}, ...,
                          "total_seconds": ...}

On GPU, pending kernels are synchronized before reading the clock so that
asynchronous execution is not attributed to the wrong section.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._created = time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0
            self.calls[name] = self.calls.get(name, 0) + 1

    def summary(self) -> dict:
        out: dict = {k: {"seconds": round(v, 3), "calls": self.calls[k]}
                     for k, v in sorted(self.seconds.items())}
        out["total_seconds"] = round(time.perf_counter() - self._created, 3)
        return out
