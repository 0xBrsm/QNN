"""Contiguous per-tick align-hbw trace — schema + writer for ``intercept_trace.npz``.

``intercept_events.npz`` and ``intercept_windows.npz`` are both DISCHARGE-
ANCHORED: they only exist where the policy pulled the trigger. At SG cadence
discharges sit ~13 ticks apart, so those artifacts are full of holes, and the
holes are exactly the ticks the policy DECLINED to fire on. Any question of the
form "could a different trigger have done better on this same aim trajectory?"
needs those declined ticks, which is what this instrument records: ONE ROW PER
LANE-TICK, discharge or not, LOS or not.

ONE LAW, NO SECOND COPY OF THE GEOMETRY. The ``hbw`` column is the SAME
``_LeadRuler.hbw`` array the eval already computes once per tick for every lane
(``qnn.eval.run._lead_ruler_batched`` → ``_intercept_hbw_array`` over the
engine's v3 lead geometry) — the very array whose values the window instrument
samples into ``hbw_win``. The trace therefore reproduces
``intercept_windows.npz`` by construction rather than by agreement, which is
the property ``qnn.diag.crest_frontier`` relies on to recompute the crest
denominator under a counterfactual fire set.

Weapon keying is FREE on this path, unlike the PPO-side
``CrestRewardShaper.hbw_all_los`` (ab7d8ec0), which had to assume a hitscan
impulse because the live obs carries no engine-equipped-weapon field. The engine's v3
lead geometry ships ``lead_weapon_id`` = the CURRENTLY-EQUIPPED weapon on EVERY
tick (``src/engine/nq/qnn_reward.c``: ``_lead_weapon_id = snapshot->weapon_id``),
firing or not, so the trace records the real equipped weapon and the RL exclusion
that blocks the PPO path does not apply here.

NaN in ``hbw`` means "no participating in-LOS actor this tick" (the eval's
``lead_valid = 0``) — the tick is still recorded, because a cooldown
simulation has to advance through it.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

INTERCEPT_TRACE_NPZ = "intercept_trace.npz"     # under <run>/metrics/eval/
TRACE_SCHEMA_VERSION = 1

# Row cap. A 240-episode / 8-lane / 1800-tick SG judge is ~432k rows (~9 MB
# on disk), so this is ~10x the standard surface and still bounded. Past the
# cap rows stop and the npz is stamped truncated=True — a truncated trace has
# torn lane-episodes, so consumers must refuse to draw cadence conclusions
# from one.
TRACE_CAP = 4_000_000

# Column-parallel schema. Deliberately narrow and integer-keyed: this is the
# widest artifact the eval writes, so every column is sized to what it holds
# rather than to the convenient dtype (memory: cost every consumer).
TRACE_FIELDS = ("env_idx", "episode", "tick", "hbw", "fired", "attacked",
                "weapon_id", "scenario_idx")
_TRACE_DTYPES: dict[str, Any] = {
    "env_idx": np.int16,        # lane index
    "episode": np.int32,        # episode ordinal WITHIN the lane (0-based)
    "tick": np.int32,           # eval macro-step (global, ~54k for a judge)
    "hbw": np.float32,          # align-hbw; NaN = no in-LOS actor this tick
    "fired": np.bool_,          # OPERATIVE discharge (engine shots_fired > 0)
    "attacked": np.int8,        # decoded attack class (0 = no attack, 1..8)
    "weapon_id": np.int8,       # raw EQUIPPED weapon id 1..8 (0 = none/unknown)
    "scenario_idx": np.int16,   # index into the scenario_ids codebook
}


def write_intercept_trace(
    output_dir: str | Path,
    cols: Mapping[str, Sequence[np.ndarray]],
    scenario_ids: Sequence[str],
    truncated: bool = False,
) -> Path | None:
    """Write the per-tick trace to ``output_dir/intercept_trace.npz``.

    ``cols`` maps each :data:`TRACE_FIELDS` name to the list of PER-MACRO-STEP
    array chunks the eval appended (one chunk per tick, one entry per active
    lane) — chunks, not scalars, because the eval already has every column as a
    lane-aligned array and appending them whole keeps the hot path free of
    per-lane Python. ``scenario_ids`` is the codebook ``scenario_idx`` indexes.

    Zero rows ⇒ no file and ``None`` (absence means "instrument off / no ticks",
    the same contract ``_write_intercept_events`` uses). The write goes through
    a temp sibling + ``os.replace`` so a reader never sees a torn file.
    """
    missing = set(TRACE_FIELDS) - set(cols)
    if missing:
        raise ValueError(f"intercept trace missing column(s) {sorted(missing)}")
    arrays = {
        k: (np.concatenate([np.asarray(c).reshape(-1) for c in cols[k]])
            if len(cols[k]) else np.empty(0)).astype(_TRACE_DTYPES[k])
        for k in TRACE_FIELDS
    }
    lens = {k: int(v.size) for k, v in arrays.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"intercept trace columns are ragged: {lens}")
    if lens["hbw"] == 0:
        return None

    path = Path(output_dir) / INTERCEPT_TRACE_NPZ
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.stem}.{os.getpid()}-{threading.get_ident()}.tmp.npz")
    np.savez_compressed(
        tmp,
        schema_version=np.int64(TRACE_SCHEMA_VERSION),
        truncated=np.bool_(truncated),
        scenario_ids=np.asarray(list(scenario_ids), dtype="U64"),
        **arrays,
    )
    os.replace(tmp, path)
    return path


def load_intercept_trace(run_dir: str | Path) -> dict[str, np.ndarray]:
    """Read a trace npz back. Accepts a run dir or a direct npz path.

    FAILS LOUD on a truncated trace: truncation tears lane-episodes mid-flight,
    and every consumer of this artifact reasons about contiguous time.
    """
    p = Path(run_dir)
    if p.is_dir():
        p = p / "metrics" / "eval" / INTERCEPT_TRACE_NPZ
    if not p.exists():
        raise FileNotFoundError(f"no intercept trace at {p}")
    with np.load(p, allow_pickle=False) as z:
        missing = set(TRACE_FIELDS) - set(z.files)
        if missing:
            raise ValueError(f"{p}: trace missing column(s) {sorted(missing)}")
        if bool(z["truncated"]):
            raise ValueError(
                f"{p}: trace is TRUNCATED — lane-episodes are torn, so tick "
                "contiguity is not guaranteed and no cadence conclusion may "
                "be drawn from it")
        out = {k: np.asarray(z[k]) for k in TRACE_FIELDS}
        out["scenario_ids"] = np.asarray(z["scenario_ids"])
    return out
