"""Per-tick stage timing for the closed-loop eval loop.

The eval loop's cost is serial per-tick Python interleaved with engine sim, and
the only way to steer optimization work is to know which stage owns the
milliseconds. ``py-spy`` gives a sampled call-tree but cannot separate "waiting
for the engine" from "running Python" on the receive path (both are inside one
``read``-bound frame), and it needs ptrace, which the eval containers do not
always grant. So the loop carries its own explicit stage accounting, gated on
``QNN_EVAL_PROFILE`` so a production eval pays nothing for it.

Usage::

    prof = StageProfile.from_env()
    with prof.stage("act"):
        ...
    prof.tick()                     # once per macro-step
    prof.write(path, lanes=..., tick_hz=...)

When disabled, :meth:`stage` returns a shared no-op whose ``__enter__`` /
``__exit__`` are empty, so the instrumented loop is the SAME code path in
production — there is no second, un-instrumented copy of the tick loop to drift
against (the repo's no-legacy-paths rule applied to instrumentation).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Dict


class _NullStage:
    """Disabled stage: both hooks are empty, so the cost is one method call."""

    __slots__ = ()

    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


_NULL_STAGE = _NullStage()


class _Stage:
    """One named accumulator. Reused across ticks (no per-tick allocation)."""

    __slots__ = ("_prof", "_name", "_t0")

    def __init__(self, prof: "StageProfile", name: str) -> None:
        self._prof = prof
        self._name = name
        self._t0 = 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = perf_counter()
        return self

    def __exit__(self, *exc: object) -> bool:
        dt = perf_counter() - self._t0
        prof = self._prof
        prof.total[self._name] = prof.total.get(self._name, 0.0) + dt
        prof.count[self._name] = prof.count.get(self._name, 0) + 1
        return False


class StageProfile:
    """Accumulates wall time per named stage of the eval tick loop.

    Stages may NEST (``metrics`` contains ``ruler_lead``); the report keeps both
    the raw totals and each stage's share of the measured loop wall, so a nested
    child is never mistaken for an independent slice. ``lane_steps`` counts
    per-lane env steps (the realtime denominator); ``macro_steps`` counts tick
    iterations.
    """

    __slots__ = ("enabled", "total", "count", "_stages", "macro_steps",
                 "lane_steps", "_t0", "_loop_wall")

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.total: Dict[str, float] = {}
        self.count: Dict[str, int] = {}
        self._stages: Dict[str, _Stage] = {}
        self.macro_steps = 0
        self.lane_steps = 0
        self._t0 = perf_counter()
        self._loop_wall = 0.0

    @classmethod
    def from_env(cls, var: str = "QNN_EVAL_PROFILE") -> "StageProfile":
        """Enabled when ``QNN_EVAL_PROFILE`` is set to anything but 0/false/''."""
        raw = (os.environ.get(var, "") or "").strip().lower()
        return cls(enabled=raw not in ("", "0", "false", "no"))

    def stage(self, name: str):
        if not self.enabled:
            return _NULL_STAGE
        st = self._stages.get(name)
        if st is None:
            st = self._stages[name] = _Stage(self, name)
        return st

    def mark(self) -> float:
        """Timestamp for the manual ``mark``/``add`` form.

        Used where a ``with`` block would force reindenting a long existing
        block (the tick loop's Pass A): ``t = prof.mark()`` … ``prof.add(name,
        t)``. Returns 0.0 when disabled so ``add`` stays cheap.
        """
        return perf_counter() if self.enabled else 0.0

    def add(self, name: str, t0: float) -> None:
        """Close a manual stage opened with :meth:`mark`."""
        if not self.enabled:
            return
        dt = perf_counter() - t0
        self.total[name] = self.total.get(name, 0.0) + dt
        self.count[name] = self.count.get(name, 0) + 1

    def tick(self, lanes: int = 0) -> None:
        """Close out one macro-step (``lanes`` = active lane count this tick)."""
        if not self.enabled:
            return
        self.macro_steps += 1
        self.lane_steps += int(lanes)

    def loop_start(self) -> None:
        """Start the loop-wall clock (excludes env construction / model load,
        which are one-time costs and would otherwise dilute every share)."""
        if self.enabled:
            self._t0 = perf_counter()

    def loop_done(self) -> None:
        """Mark the end of the tick loop (excludes setup/teardown/summary)."""
        if self.enabled and not self._loop_wall:
            self._loop_wall = perf_counter() - self._t0

    def report(self, tick_hz: float = 0.0) -> Dict[str, object]:
        wall = self._loop_wall or (perf_counter() - self._t0)
        rows = []
        for name in sorted(self.total, key=lambda k: -self.total[k]):
            tot = self.total[name]
            n = self.count[name]
            rows.append({
                "stage": name,
                "total_s": round(tot, 4),
                "calls": n,
                "ms_per_call": round(1e3 * tot / n, 4) if n else 0.0,
                "ms_per_macro_step": (round(1e3 * tot / self.macro_steps, 4)
                                      if self.macro_steps else 0.0),
                "pct_loop_wall": round(100.0 * tot / wall, 2) if wall else 0.0,
            })
        out: Dict[str, object] = {
            "loop_wall_s": round(wall, 3),
            "macro_steps": self.macro_steps,
            "lane_steps": self.lane_steps,
            "ms_per_macro_step": (round(1e3 * wall / self.macro_steps, 4)
                                  if self.macro_steps else 0.0),
            "ms_per_lane_step": (round(1e3 * wall / self.lane_steps, 4)
                                 if self.lane_steps else 0.0),
            "stages": rows,
        }
        if tick_hz and self.lane_steps and wall:
            # ×-realtime: simulated lane-seconds per wall second. Aggregate over
            # lanes (the quantity an eval's wall clock actually depends on).
            out["x_realtime_aggregate"] = round(
                (self.lane_steps / float(tick_hz)) / wall, 3)
            out["mean_lanes"] = (round(self.lane_steps / self.macro_steps, 3)
                                 if self.macro_steps else 0.0)
        return out

    def write(self, path: str | Path, tick_hz: float = 0.0) -> Path | None:
        """Write the report as JSON. No-op (None) when disabled."""
        if not self.enabled:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.report(tick_hz=tick_hz), indent=2) + "\n")
        return p
