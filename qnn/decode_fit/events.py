"""Per-discharge INTERCEPT event tables — the fit's sample unit.

v2's foundational contract: **every operative discharge is one likelihood
sample**, not a cell median. The eval writes one row per discharge to
``metrics/eval/intercept_events.npz`` (see ``qnn.eval.run``; schema pinned
here); this module loads those rows and annotates them with the decode
operating point their lane ran at (gain/α/tremor/tms + weapon + range pin),
producing the flat table ``response.py`` fits.

A ``legacy_grid_table`` adapter turns v1 cell-median grid JSONs
(``_aim_grid_lead_*.json`` etc.) into weighted pseudo-rows (``is_median``,
``weight = n_attacks``) so the MLE machinery can be validated retroactively on
the seed43 grids (Phase 1 acceptance) before any new compute is spent.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from qnn.decode_fit.context import FRIKBOT_TO_PIN, MODELNAME_TO_ABBR, read_json

# ── the eval-side npz schema (written by qnn.eval.run) ────────────────────────
EVENT_SCHEMA_VERSION = 1
INTERCEPT_EVENTS_NPZ = "intercept_events.npz"          # under <run>/metrics/eval/
# npz arrays, all row-aligned (one row per operative discharge):
#   scenario_id : unicode  — the lane's scenario id (cell key within a wave)
#   weapon      : unicode  — abbr of the DISCHARGING weapon (from lead_weapon_id)
#   hbw         : f32      — discharge alignment in hitbox-half-widths (lower=better)
#   range_u     : f32      — target range at the shot (engine units)
#   episode     : i32      — episode ordinal within the lane (bootstrap cluster)
#   env_idx     : i32      — lane index
#   tick        : i64      — eval macro-step when recorded
# plus a scalar `schema_version`.
RAW_FIELDS = ("scenario_id", "weapon", "hbw", "range_u", "episode", "env_idx", "tick")

# The decode keys a lane's operating point is described by (the per-lane
# override key plus the shared substrate values).
OP_KEYS = ("gain", "alpha", "tremor", "tms")
_SWEPT_KEY_TO_OP = {
    "look.aim_prior_gain": "gain",
    "look.aim_mag_gain": "alpha",
    "look.aim_degrade_tremor_mag": "tremor",
    "look.turn_mag_scale": "tms",
}


@dataclass
class EventTable:
    """Flat row store: dict of equal-length numpy arrays. Columns beyond the
    raw eval fields: the OP_KEYS operating point, ``pin`` (frikbot range-pin
    tag, "" when unpinned/free-play), ``cluster`` (bootstrap cluster id,
    stable across concatenation), ``weight`` (rows summarized; 1 for raw
    events) and ``is_median`` (legacy cell-median pseudo-rows)."""
    cols: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return 0 if not self.cols else len(next(iter(self.cols.values())))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.cols[name]

    def filter(self, mask: np.ndarray) -> "EventTable":
        return EventTable({k: v[mask] for k, v in self.cols.items()})

    def where(self, **eq: Any) -> "EventTable":
        mask = np.ones(len(self), dtype=bool)
        for k, v in eq.items():
            mask &= (self.cols[k] == v)
        return self.filter(mask)

    @staticmethod
    def concat(tables: "Iterable[EventTable]") -> "EventTable":
        tables = [t for t in tables if len(t)]
        if not tables:
            return EventTable()
        keys = set(tables[0].cols)
        for t in tables[1:]:
            if set(t.cols) != keys:
                raise ValueError(f"EventTable column mismatch: {sorted(keys)} vs "
                                 f"{sorted(t.cols)}")
        return EventTable({k: np.concatenate([t.cols[k] for t in tables])
                           for k in keys})


def _cluster_ids(run_name: str, env_idx: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """Stable bootstrap-cluster id per (run, lane, episode) — clustered
    resampling must respect within-episode correlation (consecutive shots at
    one opponent trajectory are not independent)."""
    base = zlib.crc32(run_name.encode()) & 0xFFFF
    return (np.int64(base) << 40) + (env_idx.astype(np.int64) << 20) \
        + np.maximum(episode.astype(np.int64), 0)


def load_run_events(run_dir: Path) -> dict[str, np.ndarray] | None:
    """The raw per-discharge arrays for one eval run, or None when the run
    predates event logging / recorded no discharges."""
    p = Path(run_dir) / "metrics" / "eval" / INTERCEPT_EVENTS_NPZ
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        if int(z.get("schema_version", 0)) != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"{p}: event schema {z.get('schema_version')} != "
                f"{EVENT_SCHEMA_VERSION} — re-run the eval on this checkout")
        return {k: z[k] for k in RAW_FIELDS}


def _lane_ops(wave_dir: Path) -> dict[str, dict[str, Any]]:
    """scenario_id → the lane's full operating point + cell identity, from the
    wave's row-aligned (scenarios, per_env_decode_overrides) + the shared
    substrate decode. Every lane gets ALL OP_KEYS resolved (substrate value
    unless its swept key overrides it)."""
    wave_dir = Path(wave_dir)
    sc = read_json(wave_dir / "config" / "scenario.json")["scenarios"]
    train = read_json(wave_dir / "config" / "train.json")
    overrides = train.get("eval_per_env_decode_overrides") or [{}] * len(sc)
    if len(overrides) != len(sc):
        raise ValueError(f"{wave_dir}: overrides not row-aligned with scenarios "
                         f"({len(overrides)} vs {len(sc)})")
    sub_params: dict[str, Any] = {}
    dec = wave_dir / "decode.json"
    if dec.exists():
        sub_params = read_json(dec).get("params", {})

    def _sub_scalar(key: str, default: float) -> float:
        v = sub_params.get(key, default)
        # per-impulse vectors on the substrate collapse per lane via the forced
        # model weapon below; scalar substrates pass through.
        return v if isinstance(v, list) else float(v)

    out: dict[str, dict[str, Any]] = {}
    for scen, ov in zip(sc, overrides):
        opts = scen.get("options") or {}
        mw = (opts.get("inventory") or {}).get("selected_weapon")
        pin = FRIKBOT_TO_PIN.get(str(opts.get("bot_weapon_pin", "")), "")
        op = {
            "gain": _sub_scalar("look.aim_prior_gain", 0.0),
            "alpha": _sub_scalar("look.aim_mag_gain", 0.0),
            "tremor": _sub_scalar("look.aim_degrade_tremor_mag", 0.0),
            "tms": _sub_scalar("look.turn_mag_scale", 1.0),
        }
        # substrate per-impulse vectors → this lane's forced weapon slot
        from qnn.decode_fit.context import WEAPON_IMPULSE
        abbr = MODELNAME_TO_ABBR.get(str(mw), "")
        for k in ("gain", "alpha", "tremor"):
            if isinstance(op[k], list):
                imp = WEAPON_IMPULSE.get(abbr)
                op[k] = float(op[k][imp]) if imp is not None else 0.0
        for key, val in ov.items():
            slot = _SWEPT_KEY_TO_OP.get(key)
            if slot is None:
                raise ValueError(f"{wave_dir}: unknown per-lane override {key!r}")
            op[slot] = float(val)
        out[scen["scenario_id"]] = {**op, "pin": pin, "model_weapon": abbr}
    return out


def wave_event_table(wave_dir: Path) -> EventTable:
    """One wave run-dir → the annotated EventTable (raw discharges joined to
    their lane operating points). Empty table when the wave logged no events."""
    wave_dir = Path(wave_dir)
    raw = load_run_events(wave_dir)
    if raw is None or len(raw["hbw"]) == 0:
        return EventTable()
    ops = _lane_ops(wave_dir)
    n = len(raw["hbw"])
    cols: dict[str, np.ndarray] = {
        "weapon": raw["weapon"].astype("U8"),
        "hbw": raw["hbw"].astype(np.float64),
        "range_u": raw["range_u"].astype(np.float64),
        "cluster": _cluster_ids(wave_dir.name, raw["env_idx"], raw["episode"]),
        "weight": np.ones(n, dtype=np.float64),
        "is_median": np.zeros(n, dtype=bool),
    }
    for k in OP_KEYS + ("pin",):
        default = "" if k == "pin" else np.nan
        vals = [ops.get(str(sid), {}).get(k, default) for sid in raw["scenario_id"]]
        cols[k] = (np.array(vals, dtype="U8") if k == "pin"
                   else np.array(vals, dtype=np.float64))
    unknown = ~np.isin(raw["scenario_id"], list(ops))
    if unknown.any():
        raise ValueError(
            f"{wave_dir}: {int(unknown.sum())} events carry scenario_ids absent "
            "from config/scenario.json — wave dir is inconsistent, rebuild it")
    return EventTable(cols)


def load_waves(wave_dirs: Iterable[Path]) -> EventTable:
    return EventTable.concat([wave_event_table(d) for d in wave_dirs])


# ── legacy v1 grid adapter (Phase-1 retrofit only) ────────────────────────────

def legacy_grid_table(grid_json: Path) -> EventTable:
    """v1 cell-median grid JSON → weighted pseudo-rows. Each ok cell becomes ONE
    row: ``hbw`` = the cell median, ``weight`` = n_attacks, ``is_median`` =
    True (response.py widens the variance of median rows accordingly). The
    sweep mode routes the swept value to its OP column; non-swept OP values
    come from the substrate decode when resolvable, else the v1 conventions
    (gain sweeps ran aim-off substrates: α=0, tremor=0)."""
    doc = read_json(Path(grid_json))
    mode = {"g": "gain", "a": "alpha", "t": "tremor", "s": "tms"}[doc.get("sweep") or "g"]
    sub_params: dict[str, Any] = {}
    sub = doc.get("substrate_decode")
    if sub and Path(sub).exists():
        sub_params = read_json(Path(sub)).get("params", {})
    sub_tms = float(sub_params.get("look.turn_mag_scale", 1.0))
    sub_gain = sub_params.get("look.aim_prior_gain", 0.0)

    rows: dict[str, list] = {k: [] for k in
                             ("weapon", "pin", "gain", "alpha", "tremor", "tms",
                              "hbw", "range_u", "cluster", "weight", "is_median")}
    for i, x in enumerate(doc.get("results", [])):
        if not x.get("ok") or x.get("intercept_hbw") is None:
            continue
        abbr = MODELNAME_TO_ABBR.get(str(x.get("model_weapon", "")), "")
        if not abbr:
            continue
        val = float(x["gain"])                      # the sweep's generic slot
        op = {"gain": 0.0, "alpha": 0.0, "tremor": 0.0, "tms": sub_tms}
        if mode == "alpha" and isinstance(sub_gain, list):
            from qnn.decode_fit.context import WEAPON_IMPULSE
            op["gain"] = float(sub_gain[WEAPON_IMPULSE[abbr]])
        op[mode] = val
        rows["weapon"].append(abbr)
        rows["pin"].append(FRIKBOT_TO_PIN.get(str(x.get("frikbot_weapon", "")), ""))
        for k in OP_KEYS:
            rows[k].append(op[k])
        rows["hbw"].append(float(x["intercept_hbw"]))
        rows["range_u"].append(np.nan)
        rows["cluster"].append(i)
        rows["weight"].append(float(x.get("n_attacks", 1) or 1))
        rows["is_median"].append(True)
    if not rows["hbw"]:
        return EventTable()
    return EventTable({
        "weapon": np.array(rows["weapon"], dtype="U8"),
        "pin": np.array(rows["pin"], dtype="U8"),
        **{k: np.array(rows[k], dtype=np.float64) for k in OP_KEYS},
        "hbw": np.array(rows["hbw"], dtype=np.float64),
        "range_u": np.array(rows["range_u"], dtype=np.float64),
        "cluster": np.array(rows["cluster"], dtype=np.int64),
        "weight": np.array(rows["weight"], dtype=np.float64),
        "is_median": np.array(rows["is_median"], dtype=bool),
    })
