"""Geometry-mediated behavior metrics from the closed-loop pose stream.

Tier 2 of the spatial-substrate validation: offline BC metrics are
near-blind to spatial quality outside jump, so substrates are compared
on what geometry perception is FOR — moving through the world without
colliding, stalling, or falling. All metrics derive from the per-tick
pose exported by the QNN_POSE_TAIL worker channel (origin_xyz +
view_yaw; see qnn_io.h), so they need no engine metric plumbing and no
decode fit — they are decode-calibration-free by construction.

Metrics (per episode, aggregated by the caller):
- traversal_dist: horizontal path length (units).
- net_displacement: start→end horizontal distance (exploration proxy).
- mean_speed: horizontal path length / wall time (u/s).
- stuck_frac: fraction of ticks whose trailing 1 s window moved < 8 u
  horizontally (the "pushing a wall / wedged on geometry" signature).
- big_falls: count of continuous descents > 200 u (ledge/void falls;
  fall damage in Quake starts well below this — 200 u only counts
  clearly unintended drops or deliberate long drops, matched A/B).
- fall_dist: total distance lost in those descents.
- turn_rate_mean: mean |yaw delta| per second (sanity/context, not a
  quality axis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STUCK_WINDOW_S = 1.0
STUCK_DIST_U = 8.0
BIG_FALL_U = 200.0


@dataclass
class GeometryAccumulator:
    tick_hz: int = 20
    poses: list[np.ndarray] = field(default_factory=list)

    def add(self, pose: np.ndarray) -> None:
        """pose = [x, y, z, view_yaw_deg] for one tick."""
        self.poses.append(np.asarray(pose, dtype=np.float64))

    def finish(self) -> dict[str, float]:
        pose = np.stack(self.poses) if self.poses else np.zeros((0, 4))
        n = len(pose)
        if n < 2:
            return {"ticks": float(n)}
        xy = pose[:, :2]
        z = pose[:, 2]
        yaw = pose[:, 3]

        step_xy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        # Respawn/teleport teleports the pose; a >280 u single-tick jump
        # (14x max run speed at 20 Hz) is a discontinuity, not movement.
        moved = step_xy[step_xy < 280.0]
        traversal = float(moved.sum())
        seconds = (n - 1) / self.tick_hz

        window = max(1, int(round(STUCK_WINDOW_S * self.tick_hz)))
        cum = np.concatenate([[0.0], np.cumsum(step_xy)])
        win_dist = cum[window:] - cum[:-window]
        stuck_frac = float(np.mean(win_dist < STUCK_DIST_U)) if len(win_dist) else 0.0

        # Continuous-descent runs on z (teleport steps excluded).
        dz = np.diff(z)
        dz = np.where(np.abs(dz) < 280.0, dz, 0.0)
        falls: list[float] = []
        run = 0.0
        for step in dz:
            if step < -0.5:
                run += -step
            else:
                if run > BIG_FALL_U:
                    falls.append(run)
                run = 0.0
        if run > BIG_FALL_U:
            falls.append(run)

        dyaw = np.abs((np.diff(yaw) + 180.0) % 360.0 - 180.0)
        return {
            "ticks": float(n),
            "traversal_dist": traversal,
            "net_displacement": float(np.linalg.norm(xy[-1] - xy[0])),
            "mean_speed": traversal / seconds if seconds > 0 else 0.0,
            "stuck_frac": stuck_frac,
            "big_falls": float(len(falls)),
            "fall_dist": float(sum(falls)),
            "turn_rate_mean": float(dyaw.mean() * self.tick_hz),
        }


def aggregate(episodes: list[dict[str, float]]) -> dict[str, float]:
    """Tick-weighted means over per-episode metric dicts."""
    if not episodes:
        return {}
    keys = {k for e in episodes for k in e if k != "ticks"}
    weights = np.asarray([e.get("ticks", 0.0) for e in episodes])
    out: dict[str, float] = {"episodes": float(len(episodes)),
                             "ticks": float(weights.sum())}
    for k in sorted(keys):
        vals = np.asarray([e.get(k, 0.0) for e in episodes])
        out[k] = float((vals * weights).sum() / max(weights.sum(), 1.0))
    return out
