"""Closed-loop instrument waves — build, launch, collect (the fit's eval arm).

The library port of the wave machinery in
``scripts/analysis/aim_grid_closedloop.py`` (packed ≤64-lane process waves,
warm single-weapon ``qnn_arena8`` servers, per-lane decode overrides, the
substrate/env staleness resume check, the collision-free port counter, the
``EVAL_TIMEOUT_S`` router subprocess, the worker pool) plus the free-play
validation launcher from ``qnn.eval.decode_fit_pipeline`` — turned into plain
FUNCTIONS with no module-global mode flags and no argparse.

v2 contract deltas vs the v1 script:

  * wave dirs live under ``ctx.waves_dir`` (``runs/decode_fit/<model>/waves``),
    never ``runs/eval/``; the config skeleton still comes from the pilot
    template run (``DEFAULT_TEMPLATE``) with the checkpoint's OWN
    ``look_grid.json`` + ``model.json`` copied over it (the v1
    ``RUN_DIR_OVERRIDE`` logic, always on — decode/guards don't transfer
    across archs);
  * every lane pins its FULL operating point: the per-lane override dict
    carries ALL FOUR decode keys (gain/α/tremor/tms) with explicit values,
    uniform key set across lanes — a lane's op never depends on what the
    substrate happened to bake, and ``qnn.decode_fit.events._lane_ops``
    resolves it back off the override row;
  * a wave is a heterogeneous list of ``Cell`` (model_weapon, frikbot_pin, op)
    entries, not one swept key: scenario ids carry a cell ORDINAL
    (``m<w>_f<p>_c<idx>``), never a value to parse back out;
  * botpin intercept waves take the arena backend (one warm single-weapon
    ``qnn_arena8`` server, ≤8 cells); the face-away ACQUISITION instrument
    (spawn_face_away/fov) stays on the process backend exactly like v1
    ``_use_arena()``; ``DECODEFIT_NO_ARENA=1`` forces process.

Collection round-trips through ``qnn.decode_fit.events`` (per-discharge
tables) and ``qnn.human.acquisition`` (per-cell Fitts throughput on the HUMAN
effective width from ``ctx.acq_path``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from qnn.decode_fit.context import (
    _REPO,
    _git_sha,
    INSTRUMENT_WEAPONS,
    MODELNAME_TO_ABBR,
    read_json,
    rel_to_repo,
)
from qnn.decode_fit.events import (
    EventTable,
    OP_KEYS,
    TRACKING_WINDOWS_NPZ,
    _SWEPT_KEY_TO_OP,
)

# The working bot-pin pilot whose config/ is the wave skeleton (v1 BASE_RUN).
# Tests inject a fabricated template via the build functions' parameter.
DEFAULT_TEMPLATE = _REPO / "runs" / "eval" / "pilot_botpin_lg_vs_sg_s1_g0015"

# PACKED process waves: one batched model.act(B=N) steps every lane; PPO proves
# N=64 engines in one shared forward is the reliable cpu-service concurrency.
LANE_CAP = 64

# ARENA waves: one warm qnn_arena8 server hosts ≤8 single-weapon 1v1 matches.
ARENA_MAP = "qnn_arena8"
ARENA_MATCHES_PER_SERVER = 8
ARENA_BOT_SKILL_DEFAULT = 3

# Per-wave server port: base + counter*stride, allocated by the ONE process
# that plans a round's waves (before the pool dispatch), so concurrent waves
# never collide — the same scheme as qnn.ppo.arena_backend's server_id (the v1
# hash-of-wave-name ports collided under concurrency). Concurrent SEPARATE
# fits should pick different bases, same operator responsibility as PPO.
ARENA_PORT_BASE = 29000
ARENA_PORT_STRIDE = 16
_arena_port_counter = 0

# Backstop per-wave timeout: the frikbot bridge can hang unbounded; the router
# subprocess is killed on expiry so a stray hang can't block the pool.
EVAL_TIMEOUT_S = 1800
FREEPLAY_TIMEOUT_S = 2400

# Arena startup can lose a bind race to a recently retired server even though
# every wave in this process has a distinct planned port.  Re-run only the
# failed tail after the concurrent pool has drained; at that point no sibling
# wave can still own the port.  Keep this to one attempt so a deterministic
# engine/config failure still stops promptly and visibly.
WAVE_RETRY_ATTEMPTS = 1

# Default wave concurrency: fill the box (Brian 2026-07-16: target ≥90%
# saturation — decode-fit wall clock was dominated by a magic 4 on a 32-core
# host). Each concurrent wave costs ~1-1.5 cores (serial per-tick Python;
# the eval-batching port raises that toward 1 full core each), so 3/4 of
# the visible cores keeps headroom for the router/OS without idling the
# pool. DECODEFIT_MAX_WORKERS still overrides.
_DEFAULT_MAX_WORKERS = max(4, (os.cpu_count() or 8) * 3 // 4)

# Short weapon tags for run-dir / scenario-id names — derived from the package
# vocabulary (never a second hand-maintained copy).
WTAG = {name: abbr.lower() for name, abbr in MODELNAME_TO_ABBR.items()}

# op slot ("gain"/"alpha"/"tremor"/"tms") → the dotted decode key its per-lane
# override rides (the inverse of the events-side loader map, so the two can
# never drift apart).
OP_TO_DECODE_KEY = {op: key for key, op in _SWEPT_KEY_TO_OP.items()}


@dataclass(frozen=True)
class Cell:
    """One instrument cell: the model's forced weapon, the opponent frikbot's
    pinned weapon (the engagement-range knob), and the FULL decode operating
    point the cell's lane runs at (all four OP_KEYS, explicit)."""
    model_weapon: str          # "shotgun" | "super_nailgun" | "rocket_launcher" | "lightning"
    frikbot_pin: str           # same vocabulary (the opponent weapon pin)
    op: dict[str, float]       # {"gain","alpha","tremor","tms"} — ALL FOUR present


@dataclass(frozen=True)
class WaveSpec:
    """One planned wave: a named chunk of cells bound to a backend. Arena
    waves are single-matchup by construction and carry their server port.
    ``shard``/``episodes`` carry the episode-sharding split (a cell's episode
    budget spread over several waves with offset eval seeds — parallelism =
    wave count, and each wave is a single-core router process)."""
    name: str
    cells: tuple[Cell, ...]
    arena: bool
    base_port: int | None = None
    shard: int = 0
    episodes: int | None = None      # per-cell episodes override for this wave
    seed_extra: int = 0              # replicate salt folded into the eval seed


def _next_arena_port() -> int:
    global _arena_port_counter
    port = ARENA_PORT_BASE + _arena_port_counter * ARENA_PORT_STRIDE
    _arena_port_counter += 1
    return port


def _use_arena(spawn_face_away: int = 0, fov: int = 0) -> bool:
    """Route through the arena backend unless the wave needs a scenario-ENV
    feature the arena server does not plumb (spawn-yaw offset / perception
    cone — the face-away ACQUISITION instrument), or the DECODEFIT_NO_ARENA=1
    kill switch forces the process backend."""
    if os.environ.get("DECODEFIT_NO_ARENA"):
        return False
    return not (spawn_face_away or fov)


def _max_workers(param: int | None) -> int:
    if param:
        return max(1, int(param))
    return max(1, int(os.environ.get("DECODEFIT_MAX_WORKERS",
                                     str(_DEFAULT_MAX_WORKERS))))


def _done(d: Path) -> bool:
    eval_dir = d / "metrics" / "eval"
    if not (eval_dir / "eval_summary.json").exists():
        return False
    # Every content-keyed instrument wave declares its aim statistic in the
    # env signature. A summary without the matching tracking-window payload is
    # incomplete even when the wave legitimately produced zero discharges;
    # rebuilding on the current writer emits an explicit empty schema file.
    sig_path = d / "_sweep_env.json"
    if sig_path.exists():
        try:
            aim_stat = str(read_json(sig_path).get("aim_stat", ""))
        except Exception:
            return False
        if aim_stat.startswith("tracking-window-"):
            return (eval_dir / TRACKING_WINDOWS_NPZ).exists()
    return True


def _validate_cells(cells: Sequence[Cell]) -> None:
    if not cells:
        raise ValueError("no cells to run")
    for i, c in enumerate(cells):
        if c.model_weapon not in INSTRUMENT_WEAPONS:
            raise ValueError(f"cell {i}: model_weapon {c.model_weapon!r} not in "
                             f"{INSTRUMENT_WEAPONS}")
        if c.frikbot_pin not in INSTRUMENT_WEAPONS:
            raise ValueError(f"cell {i}: frikbot_pin {c.frikbot_pin!r} not in "
                             f"{INSTRUMENT_WEAPONS}")
        if set(c.op) != set(OP_KEYS):
            raise ValueError(
                f"cell {i}: op keys {sorted(c.op)} != {sorted(OP_KEYS)} — every "
                "cell pins its FULL operating point (all four keys, explicit)")


def _cells_hash(cells: Sequence[Cell]) -> str:
    """Content key for a wave's cell chunk: same cells → same wave name → the
    done dir resume-SKIPs; changed cells → a new name → built fresh (the v1
    swept-tuple-in-the-name semantics, generalized to heterogeneous cells)."""
    blob = json.dumps([[c.model_weapon, c.frikbot_pin,
                        [[k, float(c.op[k])] for k in OP_KEYS]] for c in cells],
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def _cell_scenario_id(cell: Cell, idx: int) -> str:
    """UNIQUE per-cell scenario id WITHIN a wave: matchup tags + the cell
    ORDINAL. Nothing parses values back out of the id — the lane's op rides
    the row-aligned override (resolved by events._lane_ops)."""
    return f"m{WTAG[cell.model_weapon]}_f{WTAG[cell.frikbot_pin]}_c{idx:03d}"


# ── planning (pure; unit-testable) ────────────────────────────────────────────

def plan_botpin_waves(cells: Sequence[Cell], *, tag: str,
                      arena: bool) -> list[WaveSpec]:
    """Chunk heterogeneous cells into waves. ARENA: group by (model_weapon,
    frikbot_pin) matchup — a warm qnn_arena8 server is single-weapon — then
    ≤8 cells per server, each wave drawing its own collision-free port.
    PROCESS: ≤64 cells per packed wave, input order preserved."""
    _validate_cells(cells)
    specs: list[WaveSpec] = []
    if arena:
        # ONE CELL PER WAVE (a25rc3c box-saturation redesign): parallelism =
        # wave count and each wave is a single-core router process, so the old
        # ≤8-cells-per-server packing capped a round at #matchup-groups cores
        # (the 16-cells→6-waves→6-cores edge round). A warm server per cell
        # costs ~nothing next to the serial per-tick loop it unblocks.
        groups: dict[tuple[str, str], list[Cell]] = {}
        for c in cells:
            groups.setdefault((c.model_weapon, c.frikbot_pin), []).append(c)
        for mw, fw in sorted(groups):
            for wi, cell in enumerate(groups[(mw, fw)]):
                chunk = (cell,)
                name = (f"{tag}_A_m{WTAG[mw]}_f{WTAG[fw]}_w{wi:02d}"
                        f"_{_cells_hash(chunk)}")
                specs.append(WaveSpec(name=name, cells=chunk, arena=True,
                                      base_port=_next_arena_port()))
    else:
        for wi, lo in enumerate(range(0, len(cells), LANE_CAP)):
            chunk = tuple(cells[lo:lo + LANE_CAP])
            specs.append(WaveSpec(
                name=f"{tag}_P_w{wi:02d}_{_cells_hash(chunk)}",
                cells=chunk, arena=False))
    return specs


def plan_acq_waves(tms_values: Sequence[float], *, tag: str,
                   matchups: Sequence[tuple[str, str]] | None = None
                   ) -> list[WaveSpec]:
    """The face-away ACQUISITION cells at each turn_mag_scale value, op =
    {gain:0, alpha:0, tremor:0, tms:v} — always on the PROCESS backend
    (spawn_face_away/fov are not plumbed to the arena server). Throughput is
    target-free and GLOBAL, so the matchup matrix buys nothing: the default is
    the design's 4 diverse matchups per tms (design.ACQ_MATCHUPS — the v1
    sweep burned the full 16-cell matrix per tms for no extra information)."""
    vals = sorted({float(v) for v in tms_values})
    if not vals:
        raise ValueError("no tms values to sweep")
    if matchups is None:
        from qnn.decode_fit.design import ACQ_MATCHUPS
        matchups = ACQ_MATCHUPS
    cells = [Cell(mw, fw, {"gain": 0.0, "alpha": 0.0, "tremor": 0.0, "tms": v})
             for v in vals for mw, fw in matchups]
    return plan_botpin_waves(cells, tag=tag, arena=False)


# ── wave dir build (resume/staleness-aware) ───────────────────────────────────

def _scenario_options(base_opts: dict, mw: str, fw: str, *,
                      spawn_face_away: int, fov: int) -> dict:
    """Full options for one cell: base inventory with the model weapon forced
    and the bot weapon pinned. inventory is copied whole (scenario options
    REPLACE the base nested dict, not deep-merge) so ammo/armor/health stay
    inherited. spawn_face_away rotates the MODEL's spawn yaw (the acquisition
    instrument); fov sets the perception cone via post_map_commands."""
    opts = json.loads(json.dumps(base_opts))   # deep copy
    opts["inventory"]["weapons"] = [mw]
    opts["inventory"]["selected_weapon"] = mw
    # Pin episodes are 90s of near-continuous fire: spawn ammo alone runs the
    # high-consumption weapons dry in seconds (LG: 100 cells = 10s; NG: 200
    # nails = 20s) and the dry-weapon forced switch corrupts the rate/mass
    # measurement. The QuakeC per-frame top-up holds BOTH players at their
    # spawn loadout for the whole episode.
    opts["inventory"]["infinite_ammo"] = 1
    opts["bot_weapon_pin"] = fw
    if spawn_face_away:
        opts["spawn_face_away"] = int(spawn_face_away)
    if fov:
        pmc = opts.get("post_map_commands", "")
        opts["post_map_commands"] = (pmc + ("\n" if pmc else "")
                                     + f"qnn_fov {int(fov)}")
    return opts


def _env_sig(spawn_face_away: int, fov: int, episodes_per_cell: int) -> dict:
    """The scenario-ENV signature the decode params do NOT capture: perception
    cone, spawn-yaw offset, and the per-cell episode budget. A done wave built
    at a different signature must NOT be reused — the resume check compares
    this alongside the baked substrate params."""
    return {"fov": int(fov), "spawn_face_away": int(spawn_face_away),
            "episodes_per_cell": int(episodes_per_cell),
            # constant, but IN the signature: every wave built before the
            # infinite-ammo regime landed measured ammo-starved fire rates
            # for the high-consumption weapons and must not be resumed.
            "infinite_ammo": 1,
            # per-wave content-derived eval seed (see build_wave_dir): waves
            # built on the old global seed-42 anchor must not be resumed.
            "seed_scheme": "cells-hash-v1",
            # the aim statistic the fit consumes: window-sampled tracking at
            # the frozen k (events.TRACKING_K), captured on the frozen
            # asymmetric window (events.TRACKING_WINDOW_ENV). Waves without the
            # window npz (pre-instrument) or on the old symmetric ±4 capture
            # must not be resumed.
            "aim_stat": "tracking-window-k16pre4post",
            # constant, but IN the signature: every wave built before the
            # E9/E10 look-applier fix (04ee17de — atan2 pitch + increment
            # apply replaced by the exact basis inverse) ran the bot through
            # a broken pitch actuator; every closed-loop aim measurement from
            # that regime is contaminated (a26-superiority-decomposition.md
            # E10 blast radius) and must not be resumed.
            "look_applier": "e9e10-exact-inverse"}


# Each PROCESS shard spawns its own full engine set (eval_num_envs pairs), so
# process fan-out is capped hard — 2× halves the acq tail without entering the
# 30-wide process-spawn regime that hangs. Arena shards are single-core router
# processes and shard freely.
PROCESS_SHARD_CAP = 2


def _shard_specs(specs: list, episodes_per_cell: int, workers: int) -> list:
    """Episode-shard small rounds: when a round has fewer waves than workers,
    split each cell's episode budget across shards with OFFSET eval seeds
    (identical seeds would replay identical episodes — duplicated, not
    independent, data). Arena shards get their own ports; ≥4 episodes per
    shard; process shards cap at PROCESS_SHARD_CAP."""
    if not specs or len(specs) >= workers:
        return specs
    arena = specs[0].arena
    S = min(max(workers // len(specs), 1), max(int(episodes_per_cell) // 4, 1))
    if not arena:
        S = min(S, PROCESS_SHARD_CAP)
    if S <= 1:
        return specs
    out = []
    for sp in specs:
        base, rem = divmod(int(episodes_per_cell), S)
        for si in range(S):
            e = base + (1 if si < rem else 0)
            if e <= 0:
                continue
            out.append(dataclasses.replace(
                sp, name=(sp.name if si == 0 else f"{sp.name}_s{si}"),
                shard=si, episodes=e,
                base_port=(sp.base_port if si == 0 or not arena
                           else _next_arena_port())))
    return out


def build_wave_dir(spec: WaveSpec, *, waves_dir: Path, template_run_dir: Path,
                   run_dir: Path, checkpoint_path: Path | str, git_commit: str,
                   substrate: dict, episodes_per_cell: int,
                   spawn_face_away: int = 0, fov: int = 0) -> Path:
    """Materialize one wave run-dir under ``waves_dir`` (resume-aware).

    Layout: config/ cloned from the template with the checkpoint's OWN
    look_grid.json + model.json copied over it; decode.json = the shared
    substrate baked AS-IS (per-lane values ride the overrides); scenario.json
    with one scenario per cell; train.json with the row-aligned full-op
    per-lane overrides + batched forward + both stream logs; machine.json env
    counts (+ arena keys); run.json provenance; _sweep_env.json signature.

    Resume: a DONE dir with matching substrate params + env signature is
    returned untouched; a not-done or stale dir is rebuilt."""
    if "params" not in substrate:
        raise ValueError("substrate decode dict has no 'params' key")
    n = len(spec.cells)
    if spec.arena:
        if n > ARENA_MATCHES_PER_SERVER:
            raise ValueError(f"wave {spec.name}: {n} cells > "
                             f"{ARENA_MATCHES_PER_SERVER}/arena server")
        if len({(c.model_weapon, c.frikbot_pin) for c in spec.cells}) != 1:
            raise ValueError(f"wave {spec.name}: arena waves are single-weapon "
                             "(one matchup per warm server)")
        if spec.base_port is None:
            raise ValueError(f"wave {spec.name}: arena wave has no base_port")
    elif n > LANE_CAP:
        raise ValueError(f"wave {spec.name}: {n} cells > LANE_CAP {LANE_CAP}")

    template_run_dir = Path(template_run_dir)
    dst = Path(waves_dir) / spec.name
    eps_cell = int(spec.episodes if spec.episodes is not None else episodes_per_cell)
    sig = _env_sig(spawn_face_away, fov, eps_cell)
    if spec.shard:
        sig["shard"] = int(spec.shard)
    if dst.exists() and not _done(dst):
        shutil.rmtree(dst)                       # stale/failed — rebuild
    elif dst.exists():
        # SUBSTRATE/ENV-AWARE resume: never silently reuse a done wave built on
        # a different substrate or scenario env (the cells themselves are
        # pinned by the content hash in the dir name).
        existing = existing_env = None
        try:
            existing = read_json(dst / "decode.json").get("params")
        except Exception:
            pass
        try:
            existing_env = read_json(dst / "_sweep_env.json")
        except Exception:
            pass
        if existing != substrate["params"] or existing_env != sig:
            print(f"  [stale] {spec.name}: substrate/env changed -> rebuild",
                  flush=True)
            shutil.rmtree(dst)
    if dst.exists():
        return dst

    shutil.copytree(template_run_dir / "config", dst / "config")
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    (dst / "metrics").mkdir(exist_ok=True)
    # The checkpoint's OWN look grid (load-bearing — the corpus-fit look-grid
    # trap) + its model.json/probe.json WHEN PRESENT. Bench head-probe runs
    # carry probe.json only, and the graph spec is embedded in the checkpoint
    # itself (self-describing — qnn.model.bench.runner), so a missing
    # model.json is fine; a missing look grid is not.
    lg = Path(run_dir) / "config" / "look_grid.json"
    if not lg.exists():
        raise FileNotFoundError(
            f"{lg} missing — the wave MUST pin the checkpoint's own look grid")
    shutil.copy(lg, dst / "config" / "look_grid.json")
    for cfg in ("model.json", "probe.json"):
        src = Path(run_dir) / "config" / cfg
        if src.exists():
            shutil.copy(src, dst / "config" / cfg)

    base_sc = read_json(dst / "config" / "scenario.json")
    base_opts, base_map = base_sc["options"], base_sc["map_id"]
    scenarios, overrides = [], []
    for idx, c in enumerate(spec.cells):
        scenarios.append({
            "scenario_id": _cell_scenario_id(c, idx),
            "map_id": ARENA_MAP if spec.arena else base_map,
            "options": _scenario_options(base_opts, c.model_weapon,
                                         c.frikbot_pin,
                                         spawn_face_away=spawn_face_away,
                                         fov=fov),
        })
        # The FULL operating point, every lane, uniform key set — never a
        # single swept key (v2 contract; events._lane_ops resolves these).
        overrides.append({OP_TO_DECODE_KEY[k]: float(c.op[k]) for k in OP_KEYS})

    # scenario.json: one scenario per cell, row-aligned with the overrides —
    # the eval pins scenario i -> lane i (strict lane fill).
    base_sc["scenarios"] = scenarios
    if spec.arena:
        base_sc["map_id"] = ARENA_MAP
    (dst / "config" / "scenario.json").write_text(
        json.dumps(base_sc, indent=2) + "\n")
    (dst / "_sweep_env.json").write_text(json.dumps(sig) + "\n")

    # decode.json: the shared substrate, baked AS-IS. Per-lane op values ride
    # the overrides, so the substrate never carries a swept value.
    (dst / "decode.json").write_text(json.dumps(substrate, indent=2) + "\n")

    t = read_json(dst / "config" / "train.json")
    t["eval_decode_regime"] = str((dst / "decode.json").resolve())
    t.pop("eval_look_aim_prior_gain", None)     # the retired scalar widener
    t["eval_batched_forward"] = True
    t["eval_per_env_decode_overrides"] = overrides
    # per-tick action streams -> human-band scoring; per-tick entity obs
    # streams -> the acquisition (Fitts-throughput) axis. Both axes fall out
    # of every wave.
    t["eval_log_action_streams"] = True
    t["eval_log_acq_streams"] = True
    # Content-derived per-wave seed salt: without it every wave anchored on
    # the one global eval seed (42), so the whole fit was a single-seed
    # sample dressed as many waves. The salt rides the same cells hash that
    # names the dir, so resume stays content-keyed (same cell → same seed)
    # while distinct cells — every (weapon, pin, op) — sample distinct RNG
    # streams; the cluster bootstrap then covers seed variation.
    _seed_off = (spec.shard * 7919 + spec.seed_extra * 104729
                 + (int(_cells_hash(spec.cells), 16) % 99991) * 13)
    if _seed_off:
        # episode shards / seed replicates MUST diverge: identical seeds
        # replay identical episodes (duplicated, not independent, data)
        t["eval_seed"] = int(t.get("eval_seed", 42)) + _seed_off
    (dst / "config" / "train.json").write_text(json.dumps(t, indent=1) + "\n")

    mc = read_json(dst / "config" / "machine.json")
    mc["eval_num_envs"] = n
    mc["eval_num_episodes"] = n * eps_cell
    if spec.arena:
        mc["eval_env_backend"] = "arena_grid"
        mc["arena_map_id"] = ARENA_MAP
        mc["matches_per_server"] = min(ARENA_MATCHES_PER_SERVER, n)
        mc["eval_arena_base_port"] = int(spec.base_port)
        mc["arena_bot_skill"] = int((base_sc.get("options") or {}).get(
            "skill", ARENA_BOT_SKILL_DEFAULT))
        mc.setdefault("arena_server_binary", "assets/bin/ppo_arena_server")
        mc.setdefault("arena_client_binary", "assets/bin/ppo_arena_client")
    (dst / "config" / "machine.json").write_text(json.dumps(mc, indent=2) + "\n")

    run = {"name": spec.name, "run_id": f"decodefit-{spec.name}", "mode": "eval",
           "runtime_scale": "live", "resume": False,
           "description": f"{'arena' if spec.arena else 'process'} decode-fit "
                          f"wave: {n} cells, full-op per-lane overrides",
           "checkpoint_path": rel_to_repo(Path(checkpoint_path)),
           "config": {"train": "config/train.json",
                      "scenario": "config/scenario.json",
                      "reward": "config/reward.json",
                      "machine": "config/machine.json",
                      "model": "config/model.json"},
           "output": {"checkpoints": "checkpoints/", "metrics": "metrics/",
                      "logs": "logs/"},
           "git_commit": git_commit}
    (dst / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return dst


# ── launch ────────────────────────────────────────────────────────────────────

def _run_wave(dst: Path) -> str:
    """Run one wave via the router; DONE waves resume-skip. The subprocess is
    killed at EVAL_TIMEOUT_S so a stray bridge hang can't block the pool.
    Every pin wave runs the tracking-window instrument (the trigger-free aim
    statistic the fit consumes — events.TRACKING_K, part of the frozen
    instrument definition; the env signature's aim_stat pins it) at the frozen
    asymmetric capture geometry (events.TRACKING_WINDOW_ENV)."""
    from qnn.decode_fit.events import TRACKING_WINDOW_ENV
    if _done(dst):
        return f"skip (done): {dst.name}"
    log = dst / "logs" / "decodefit.log"
    try:
        with open(log, "w") as lf:
            subprocess.run(
                [sys.executable, "-m", "qnn.run.router", "--run-dir", str(dst)],
                stdout=lf, stderr=subprocess.STDOUT, timeout=EVAL_TIMEOUT_S,
                cwd=_REPO,
                env={**os.environ,
                     "QNN_EVAL_INTERCEPT_WINDOW": TRACKING_WINDOW_ENV})
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {dst.name}"
    return f"{'done' if _done(dst) else 'FAILED'}: {dst.name}"


def _launch_waves(wave_dirs: Sequence[Path], max_workers: int | None) -> None:
    """Run the pending waves concurrently (waves are independent; each is one
    python driver + its engines). A failed tail gets one sequential retry after
    the pool drains, which clears transient arena-port bind races without
    concealing deterministic failures. Raises RuntimeError naming every wave
    that still has no eval summary."""
    pend = [d for d in wave_dirs if not _done(d)]
    if not pend:
        return
    cap = min(len(pend), _max_workers(max_workers))
    statuses: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=cap) as ex:
        futs = {ex.submit(_run_wave, d): d for d in pend}
        for fut in as_completed(futs):
            d = futs[fut]
            statuses[d] = fut.result()
            print(f"  {statuses[d]}", flush=True)

    bad_dirs = [d for d in pend if not _done(d)]
    for attempt in range(1, WAVE_RETRY_ATTEMPTS + 1):
        if not bad_dirs:
            break
        print(f"  retrying {len(bad_dirs)} failed wave(s) sequentially "
              f"({attempt}/{WAVE_RETRY_ATTEMPTS})", flush=True)
        retry_bad = []
        for d in bad_dirs:
            statuses[d] = _run_wave(d)
            print(f"  retry {attempt}: {statuses[d]}", flush=True)
            if not _done(d):
                retry_bad.append(d)
        bad_dirs = retry_bad

    if bad_dirs:
        bad = sorted(statuses[d] for d in bad_dirs)
        raise RuntimeError("decode-fit waves failed after retry:\n"
                           + "\n".join(bad))


# ── public entry points ───────────────────────────────────────────────────────

def run_botpin_wave_groups(ctx, groups: "list[dict]", substrate: dict, *,
                           max_workers: int | None = None,
                           template_run_dir: Path = DEFAULT_TEMPLATE
                           ) -> dict[str, list[Path]]:
    """Launch SEVERAL botpin cell groups in ONE worker pool (one wall-time
    round-trip): each group is ``{"cells", "episodes_per_cell", "tag"}`` +
    optional ``"seed_extra"`` (replicate salt — offsets every wave's eval seed
    so seed replicates sample fresh episodes). Returns ``{tag: wave_dirs}``.
    Raises RuntimeError listing failed/timeout waves."""
    git = _git_sha() or ctx.git_commit
    out: dict[str, list[Path]] = {}
    all_dirs: list[Path] = []
    for g in groups:
        cells, eps, tag = g["cells"], int(g["episodes_per_cell"]), g["tag"]
        salt = int(g.get("seed_extra", 0))
        specs = plan_botpin_waves(cells, tag=tag, arena=_use_arena())
        specs = _shard_specs(specs, eps, _max_workers(max_workers))
        if salt:
            specs = [dataclasses.replace(s, seed_extra=salt) for s in specs]
        dirs = [build_wave_dir(s, waves_dir=ctx.waves_dir,
                               template_run_dir=template_run_dir,
                               run_dir=ctx.run_dir,
                               checkpoint_path=ctx.checkpoint,
                               git_commit=git, substrate=substrate,
                               episodes_per_cell=eps)
                for s in specs]
        print(f"[botpin] {tag}: {len(cells)} cells -> {len(dirs)} "
              f"{'arena' if specs[0].arena else 'process'} wave(s), "
              f"{eps} eps/cell"
              + (f", seed_extra={salt}" if salt else ""), flush=True)
        out[tag] = dirs
        all_dirs += dirs
    _launch_waves(all_dirs, max_workers)
    return out


def run_botpin_waves(ctx, cells: list[Cell], substrate: dict, *,
                     episodes_per_cell: int = 12, tag: str = "fit",
                     max_workers: int | None = None,
                     template_run_dir: Path = DEFAULT_TEMPLATE) -> list[Path]:
    """Chunk cells into waves (≤8/arena server or ≤64/process wave), build wave
    dirs under ctx.waves_dir, launch each via qnn.run.router (resume-skip done
    waves), return the wave dirs. ``substrate`` is the shared decode.json dict
    (params WITHOUT the per-lane keys baked; the per-lane overrides carry the
    full op). Raises RuntimeError listing failed/timeout waves."""
    return run_botpin_wave_groups(
        ctx, [{"cells": cells, "episodes_per_cell": episodes_per_cell,
               "tag": tag}],
        substrate, max_workers=max_workers,
        template_run_dir=template_run_dir)[tag]




def run_acq_waves(ctx, tms_values: list[float], substrate: dict, *,
                  episodes_per_cell: int = 12, spawn_face_away: int = 180,
                  fov: int = 0, tag: str = "acq", seed_extra: int = 0,
                  max_workers: int | None = None,
                  template_run_dir: Path = DEFAULT_TEMPLATE) -> list[Path]:
    """Face-away acquisition instrument on the PROCESS backend: cells = every
    (model_weapon × frikbot_pin) × tms value (like the v1 tms sweep), op =
    {"gain":0,"alpha":0,"tremor":0,"tms":v}. ``seed_extra`` salts the eval
    seeds for a replicate round (pair it with a fresh ``tag`` so the wave
    dirs don't resume-skip onto the base round's)."""
    specs = plan_acq_waves(tms_values, tag=tag)
    if seed_extra:
        specs = [dataclasses.replace(s, seed_extra=int(seed_extra))
                 for s in specs]
    specs = _shard_specs(specs, episodes_per_cell, _max_workers(max_workers))
    git = _git_sha() or ctx.git_commit
    dirs = [build_wave_dir(s, waves_dir=ctx.waves_dir,
                           template_run_dir=template_run_dir,
                           run_dir=ctx.run_dir, checkpoint_path=ctx.checkpoint,
                           git_commit=git, substrate=substrate,
                           episodes_per_cell=episodes_per_cell,
                           spawn_face_away=spawn_face_away, fov=fov)
            for s in specs]
    print(f"[acq] {len(tms_values)} tms values -> {len(dirs)} process wave(s), "
          f"face-away {spawn_face_away} deg, fov {fov or 'default'}", flush=True)
    _launch_waves(dirs, max_workers)
    return dirs


# ── collect ───────────────────────────────────────────────────────────────────

def collect_events(wave_dirs: list[Path]) -> EventTable:
    """The pooled per-discharge EventTable over the wave dirs (each lane's rows
    annotated with its full operating point + pin via events._lane_ops).
    At-discharge rows: the world-results report card + crest-capture numerator
    — the FIT consumes ``collect_tracking`` (trigger-free) instead."""
    from qnn.decode_fit.events import load_waves
    return load_waves([Path(d) for d in wave_dirs])


def collect_tracking(wave_dirs: list[Path]) -> EventTable:
    """The pooled WINDOW-SAMPLED tracking EventTable over the wave dirs — the
    trigger-free aim statistic the response fits and the placement gate ride
    (decode-fit-v2 addendum 2026-07-18). Fails loud on any wave missing the
    window npz (pre-instrument waves must rebuild, never silently thin the
    sample)."""
    from qnn.decode_fit.events import load_waves_tracking
    return load_waves_tracking([Path(d) for d in wave_dirs])


def collect_forced_attack_rate(wave_dirs: list[Path],
                               abbr: str) -> dict[str, float]:
    """Conditional attack-pulse rate for one forced-weapon pin cell.

    Weapon attribution is BY THE PIN: the cell forces ``abbr`` with infinite
    ammo, so every tick of the wave is that weapon's exposure by construction
    — no held-weapon lane is read. (The gate ``weapon`` column is the intent
    channel on a26 and the fired attack-with class — zero on holds — on a27+;
    neither is "held", and the human reference this rate is fit against
    (qnn.human.op_attack) attributes by attack context, never held weapon.)
    Denominator = keep ∧ engaged ticks, the model-side mirror of the human
    engaged-LOS conditional."""
    fires = 0
    engaged = 0
    tick_hz: float | None = None
    for wave_dir in map(Path, wave_dirs):
        p = wave_dir / "metrics" / "eval" / "move_streams_sampled.npz"
        if not p.exists():
            raise FileNotFoundError(f"forced cadence stream missing: {p}")
        with np.load(p) as z:
            hz = float(np.asarray(z["tick_hz"]).reshape(-1)[0])
            if tick_hz is not None and abs(hz - tick_hz) > 1e-9:
                raise ValueError(
                    f"forced cadence tick_hz mismatch: {tick_hz} vs {hz} in {p}")
            tick_hz = hz
            if "engaged" not in z:
                raise ValueError(
                    f"forced cadence stream {p} lacks the band-v5 'engaged' "
                    "lane (pre-v5 wave) — rebuild the wave")
            keep = np.asarray(z["keep"]).astype(bool)
            eng = np.asarray(z["engaged"]).astype(bool)
            attack = np.asarray(z["attack"]).astype(bool)
            mask = keep & eng
            engaged += int(mask.sum())
            fires += int((attack & mask).sum())
    if tick_hz is None or engaged == 0:
        raise ValueError(f"forced cadence cell for {abbr} has no engaged ticks")
    return {"fires": float(fires), "engaged_ticks": float(engaged),
            "tick_hz": tick_hz,
            "rate_per_s": float(fires / engaged * tick_hz)}


def collect_acq_throughput(ctx, wave_dirs: list[Path]) -> list[dict]:
    """Per-cell acquisition throughput rows: ``[{tms, throughput_bits_per_s,
    n_flicks, n_settled, wave, scenario_id, model_weapon, pin}]``.

    Every cell rides the HUMAN ISO effective width
    (``human_refs.acquisition_band(ctx.acq_path)["effective_width_deg"]``) so
    model and human throughput share one instrument — v1's `_human_acq_we`
    passthrough, sourced from the collect-keyed baseline (never a global
    copy). Waves whose acq npz is missing contribute no rows (same contract
    as ``collect_events`` on a wave without an events npz)."""
    from qnn.decode_fit.events import _lane_ops
    from qnn.decode_fit.human_refs import acquisition_band
    from qnn.human import acquisition as acq

    we = float(acquisition_band(ctx.acq_path)["effective_width_deg"])
    rows: list[dict] = []
    for d in wave_dirs:
        d = Path(d)
        cand = sorted((d / "metrics" / "eval").glob("acq_streams_*.npz"))
        if not cand:
            continue
        npz = cand[0]
        for sid, op in _lane_ops(d).items():
            cell = acq.cell_acquisition_throughput(npz, sid, we)
            rows.append({
                "tms": float(op["tms"]),
                "throughput_bits_per_s": (cell or {}).get("throughput"),
                "n_flicks": int((cell or {}).get("n_flick_events") or 0),
                "n_settled": int((cell or {}).get("n_settled") or 0),
                "wave": d.name,
                "scenario_id": sid,
                "model_weapon": op["model_weapon"],
                "pin": op["pin"],
            })
    return rows


# ── free play (deployment-report instrument) ──────────────────────────────────

def run_freeplay(ctx, decode_config: Path, template_run_dir: Path, *,
                 tag: str, episodes: int | None = None,
                 num_envs: int | None = None) -> Path | None:
    """Free-play arena eval at an emitted decode config (the v1
    ``_launch_validation_eval`` port): clone the template eval run-dir into
    ``ctx.waves_dir/freeplay_<tag>``, point ``eval_decode_regime`` at
    ``decode_config``, force the action + acq stream logs, run the router
    (timeout ``FREEPLAY_TIMEOUT_S``), and return
    ``metrics/eval/move_streams_sampled.npz`` — or None on failure/timeout.
    A completed dir (npz present) is reused without relaunching.

    ``episodes``/``num_envs`` optionally override the template's own
    ``machine.json`` ``eval_num_episodes``/``eval_num_envs`` (budget-
    conscious sample sizing — e.g. the weapon-switch-evidence phase-2 fit's
    per-candidate round count, or a smoke test's tiny sample). ``None``
    (the default, every pre-existing call site) leaves the template's own
    values untouched and the dir name unchanged. When either is set it rides
    the CONTENT key (dir-name suffix) too — otherwise a later full-size call
    reusing the same tag+config sha would silently resume the SMALLER
    override sample as if it were the real one."""
    template = Path(template_run_dir)
    # CONTENT-keyed dir: the trim mutates the decode config between
    # iterations under reused tags — resuming on npz existence alone would
    # silently rescore a STALE config (first a25rc3d re-run hazard). The sha
    # keys the eval to the exact config content; crash-resume still reuses an
    # identical iteration. The gate-stream schema version rides in the key so
    # an instrument/writer bump re-produces the wave instead of handing the
    # scorer a stale-schema npz it can only refuse.
    from qnn.decode_fit.context import sha256_file
    from qnn.schema import GATE_STREAM_SCHEMA_VERSION
    override_suffix = ""
    if episodes is not None:
        override_suffix += f"_e{int(episodes)}"
    if num_envs is not None:
        override_suffix += f"_w{int(num_envs)}"
    dst = Path(ctx.waves_dir) / (
        f"freeplay_{tag}_{sha256_file(Path(decode_config))[:8]}"
        f"_g{GATE_STREAM_SCHEMA_VERSION}{override_suffix}")
    npz_out = dst / "metrics" / "eval" / "move_streams_sampled.npz"
    if npz_out.exists():
        return npz_out
    (dst / "config").mkdir(parents=True, exist_ok=True)
    for f in (template / "config").glob("*.json"):
        shutil.copy(f, dst / "config" / f.name)
    # Pin the checkpoint's own look grid (mandatory — the corpus-fit look-grid
    # trap) + model.json/probe.json when present, same rule as the wave build
    # (bench checkpoints are self-describing; qnn.model.bench.runner).
    lg = Path(ctx.run_dir) / "config" / "look_grid.json"
    if not lg.exists():
        raise FileNotFoundError(
            f"{lg} missing — free play MUST pin the checkpoint's own look grid")
    shutil.copy(lg, dst / "config" / "look_grid.json")
    for cfg in ("model.json", "probe.json"):
        src = Path(ctx.run_dir) / "config" / cfg
        if src.exists():
            shutil.copy(src, dst / "config" / cfg)
    for d in ("logs", "metrics", "checkpoints"):
        (dst / d).mkdir(exist_ok=True)

    t = read_json(dst / "config" / "train.json")
    t["eval_decode_regime"] = str(Path(decode_config).resolve())
    t["eval_log_action_streams"] = True
    t["eval_log_acq_streams"] = True
    t["eval_batched_forward"] = True
    (dst / "config" / "train.json").write_text(json.dumps(t, indent=1) + "\n")

    if episodes is not None or num_envs is not None:
        mc = read_json(dst / "config" / "machine.json")
        if episodes is not None:
            mc["eval_num_episodes"] = int(episodes)
        if num_envs is not None:
            mc["eval_num_envs"] = int(num_envs)
        (dst / "config" / "machine.json").write_text(
            json.dumps(mc, indent=2) + "\n")

    run = read_json(template / "run.json")
    run["name"] = dst.name
    run["run_id"] = f"decodefit-freeplay-{tag}"
    run["mode"] = "eval"
    run["checkpoint_path"] = rel_to_repo(Path(ctx.checkpoint))
    run["description"] = (f"decode-fit free-play eval ({tag}) for "
                          f"{Path(decode_config).name}")
    (dst / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    log = dst / "logs" / "freeplay.log"
    print(f"[freeplay] launching {dst.name}", flush=True)
    try:
        with open(log, "w") as lf:
            subprocess.run(
                [sys.executable, "-m", "qnn.run.router", "--run-dir", str(dst)],
                stdout=lf, stderr=subprocess.STDOUT,
                timeout=FREEPLAY_TIMEOUT_S, cwd=_REPO)
    except subprocess.TimeoutExpired:
        print(f"[freeplay] TIMEOUT: {dst.name}", flush=True)
        return None
    return npz_out if npz_out.exists() else None
