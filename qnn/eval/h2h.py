"""Head-to-head closed-loop eval: two decode-fitted checkpoints fight each other.

    python -m qnn.eval.h2h \
        --side-a a26rc1b=runs/head_probe/head_probe_atlas_awposw_seed43:runs/decode_fit/head_probe_atlas_awposw_seed43/decode.a26rc1b.json \
        --side-b a26rc2b=runs/head_probe/head_probe_atlas24x11_awposw_seed43:runs/decode_fit/head_probe_atlas24x11_awposw_seed43/decode.a26rc2b.json \
        --rounds 160 --matches 8 --out runs/eval/h2h_rc1b_vs_rc2b/arm0 [--swap]

The engine has supported two externally-driven seats per match since the
self-play arena landed (``qnn_arena_selfplay``: action index ``i`` maps onto
``match i//2`` seat ``i%2``; virtual observers mirror both seats). What never
existed is the Python half: every driver funnels ALL lanes through one
``policy.act``. This module is that missing router — side A owns the even env
ids (seat 0), side B the odd (seat 1), each side runs ITS OWN batched forward
per tick (`qnn.eval.run._select_actions_batched`, states/RNG fully isolated
per side), and the interleaved actions go down the existing
:class:`~qnn.ppo.arena_backend.ArenaGridBackend` in one grouped step.

ROUND PROTOCOL. A round is one bout in one sealed match: it ends when either
seat dies (arena QC ends the bout for both seats) or on the engine timeout
(``max_steps_per_episode`` → truncated, backend auto-resets the match). The
winner is the surviving seat; a suicide loses without crediting the opponent
(``win_reason="suicide"``); both seats dying on one tick is a draw ("double");
timeout is a draw ("timeout"). Every match plays the SAME number of rounds
(no length bias from early stopping). Seat spawns within a cell are fixed, so
a fair pairing runs TWO arms — ``--swap`` gives side B seat 0 — and the caller
aggregates.

MEASUREMENT. Three layers, all per side:
  * rounds.jsonl — one record per round: winner/reason/duration + both seats'
    pre-death context (held weapon impulse, movement_id [1 = airborne], health,
    engaged, shots fired in the round, damage dealt/taken).
  * side_<name>_streams.npz — the band-v5 flat gate-stream schema, one
    "episode" per (lane, round) segment, byte-compatible with what
    ``qnn.eval.run`` writes under ``eval_log_action_streams`` (same field
    construction, lifted verbatim), so the human-band scorer
    (qnn.eval.humanlikeness.human_band) and rate-fidelity ruler run unchanged.
  * summary.json — win ratio + Wilson CI, per-side kills/deaths/suicides,
    op-shot fires/s per weapon, damage per round, stand-still fraction,
    jump-press rate, airborne-at-death rate.

The per-tick stream fields must stay IDENTICAL to qnn.eval.run's
``_log_streams`` (band scores are only comparable to prior fits if the
construction matches); the logic is duplicated here deliberately and pinned by
tests/test_h2h.py rather than shared, because run.py builds it inside a
closure over eval-loop state.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qnn.env.reward import RewardWeights
from qnn.eval.run import (_apply_decode_config_params, _commit_reset_lanes,
                          _EpisodeState, _install_decode_regime,
                          _load_checkpoint, _seed_attack_rng,
                          _select_actions_batched)
from qnn.model.policy import QNNPolicy
from qnn.ppo.arena_backend import ArenaGridBackend
from qnn.schema import GATE_STREAM_SCHEMA_VERSION
from qnn.vocab import TOKEN_ACTOR, self_weapon_id_to_impulse

_log = lambda m: print(f"[h2h] {m}", flush=True)  # noqa: E731

TICK_HZ = 20
MAX_STEPS_PER_ROUND = 1800  # 90 s — the arena timeout the backend enforces


# ── model loading ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SideSpec:
    """One combatant: a trained run + the decode config that made it an rc."""

    name: str
    run_dir: Path
    decode_config: Path

    @classmethod
    def parse(cls, text: str) -> "SideSpec":
        """``name=run_dir:decode_config.json``"""
        name, _, rest = text.partition("=")
        run_dir, _, decode = rest.partition(":")
        if not (name and run_dir and decode):
            raise ValueError(
                f"side spec must be name=run_dir:decode_config.json, got {text!r}")
        return cls(name=name, run_dir=Path(run_dir), decode_config=Path(decode))

    def checkpoint(self) -> Path:
        cks = (sorted((self.run_dir / "checkpoints").glob("best_*.pth"))
               or sorted((self.run_dir / "checkpoints").glob("bc_best_model.pth")))
        if not cks:
            raise FileNotFoundError(f"{self.run_dir}: no best_*.pth checkpoint")
        return cks[0]

    def look_grid(self) -> dict:
        p = self.run_dir / "config" / "look_grid.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: h2h needs the run's pinned look grid")
        return json.loads(p.read_text())


def _install_shared_look_grid(specs: Sequence[SideSpec]) -> None:
    """One process-global polar grid serves both models — REQUIRE the two runs'
    grids to agree on every numeric field (they are corpus fits; metadata like
    git_commit/created may differ). A silent mismatch would decode one side's
    look head on the other side's magnitude ladder."""
    from qnn.model.look_bins import install_polar_grid

    grids = [s.look_grid() for s in specs]
    numeric = [{k: v for k, v in g.items()
                if isinstance(v, (int, float, list))} for g in grids]
    if numeric[0] != numeric[1]:
        diff = [k for k in numeric[0] if numeric[0].get(k) != numeric[1].get(k)]
        raise ValueError(
            f"look grids differ between sides on numeric fields {diff} — "
            "two models with different grids cannot share a process")
    g = grids[0]
    install_polar_grid(
        torch.tensor(g["mag_centers_rad"], dtype=torch.float32),
        torch.tensor(g["dir_centers_rad"], dtype=torch.float32)
        if "dir_centers_rad" in g else None,
        deadzone_rad=g.get("deadzone_rad"),
    )


def load_side(spec: SideSpec) -> QNNPolicy:
    """Checkpoint + decode config → ready policy, via the SAME loader chain the
    eval runner uses (load → install regime → apply the DECODE_PARAMS registry),
    so an h2h side runs exactly the decode its rc letter shipped."""
    model = _load_checkpoint(str(spec.checkpoint()), device="cpu")
    resolved = _install_decode_regime(model, str(spec.decode_config))
    # _apply_decode_config_params mirrors one provenance scalar onto the eval
    # config; a namespace stands in for the EvalConfig we don't have.
    shim = SimpleNamespace(look_aim_prior_gain=None)
    _apply_decode_config_params(shim, model, resolved)
    return model


# ── per-tick stream logging (band-v5; lifted from qnn.eval.run._log_streams) ─

def _log_lane_tick(st: _EpisodeState, act: Mapping[str, Any]) -> None:
    """Append this tick's move/gate stream rows for one lane.

    Field-for-field the ``_log_streams`` construction in qnn.eval.run (move
    classes as mv+1; turn from the look vector; keep = any-recency actor;
    engaged = live LOS actor; discharge = attack & attack_finished expired;
    weapon_imp from self_weapon_id). Pinned by tests/test_h2h.py."""
    mv = act["move"]
    st.move_trace.append((int(mv[0]) + 1, int(mv[1]) + 1, int(mv[2]) + 1))
    lk = act.get("look")
    if lk is not None:
        turn = float(np.degrees(np.arctan2(
            np.hypot(float(lk[1]), float(lk[2])), float(lk[0]))))
    else:
        turn = 0.0
    keep = 0
    engaged = 0
    obs = st.obs
    if isinstance(obs, dict) and "entity_types" in obs:
        et = np.asarray(obs["entity_types"]).reshape(-1)
        keep = int((et == TOKEN_ACTOR).any())
        if "entity_recency" in obs and "entity_count" in obs:
            n = int(np.asarray(obs["entity_count"]).reshape(-1)[0])
            rec = np.asarray(obs["entity_recency"], dtype=np.float64).reshape(-1)[:n]
            engaged = int(((et[:n] == TOKEN_ACTOR) & (rec <= 0.0)).any())
    attack = int(act.get("attack", 0) or 0)
    af = obs.get("attack_finished") if isinstance(obs, dict) else None
    disch = int(bool(attack) and af is not None
                and float(np.asarray(af).reshape(-1)[0]) <= 1e-6)
    wimp = 0
    if isinstance(obs, dict) and "self_weapon_id" in obs:
        wimp = int(self_weapon_id_to_impulse(
            int(np.asarray(obs["self_weapon_id"]).reshape(-1)[0])))
    st.gate_trace.append(
        (attack, int(act.get("weapon", 0) or 0), turn, keep, disch, wimp, engaged))


def _lane_ctx(st: _EpisodeState) -> dict[str, Any]:
    """The pre-step context snapshot used as a seat's death/kill context."""
    obs = st.obs
    ctx: dict[str, Any] = {}
    if isinstance(obs, dict):
        if "self_movement_id" in obs:
            ctx["movement_id"] = int(np.asarray(obs["self_movement_id"]).reshape(-1)[0])
        if "health" in obs:
            ctx["health"] = float(np.asarray(obs["health"]).reshape(-1)[0])
        if "self_weapon_id" in obs:
            ctx["weapon_imp"] = int(self_weapon_id_to_impulse(
                int(np.asarray(obs["self_weapon_id"]).reshape(-1)[0])))
    if st.gate_trace:
        ctx["engaged"] = int(st.gate_trace[-1][6])
    return ctx


# ── the match driver ─────────────────────────────────────────────────────────

@dataclass
class _LaneLedger:
    """Per-lane per-round counters folded from the step infos."""

    shots: int = 0
    hits: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0

    def fold(self, info: Mapping[str, Any]) -> None:
        self.shots += int(info.get("shots_fired", 0) or 0)
        self.hits += int(info.get("hit_count", 0) or 0)
        self.damage_dealt += float(info.get("damage_direct", 0.0) or 0.0)
        self.damage_dealt += float(info.get("damage_splash", 0.0) or 0.0)
        self.damage_taken += float(info.get("damage_taken_other", 0.0) or 0.0)

    def snapshot(self) -> dict[str, float]:
        return {"shots": self.shots, "hits": self.hits,
                "damage_dealt": round(self.damage_dealt, 1),
                "damage_taken": round(self.damage_taken, 1)}


@dataclass
class _SideStreams:
    """Round-segmented band-v5 stream accumulation for one side."""

    move_rows: list[tuple] = field(default_factory=list)
    gate_rows: list[tuple] = field(default_factory=list)
    offsets: list[int] = field(default_factory=lambda: [0])

    def close_segment(self, st: _EpisodeState) -> None:
        if st.move_trace:
            self.move_rows.extend(st.move_trace)
            self.gate_rows.extend(st.gate_trace)
            self.offsets.append(len(self.move_rows))
        st.move_trace = []
        st.gate_trace = []

    def write(self, path: Path) -> None:
        mv = np.asarray(self.move_rows, dtype=np.int64).reshape(-1, 3)
        gt = np.asarray(self.gate_rows, dtype=np.float64).reshape(-1, 7)
        np.savez_compressed(
            path,
            tick_hz=np.asarray([float(TICK_HZ)]),
            episode_offsets=np.asarray(self.offsets, dtype=np.int64),
            fb=mv[:, 0], lr=mv[:, 1], ud=mv[:, 2],
            attack=gt[:, 0].astype(np.int8),
            weapon=gt[:, 1].astype(np.int8),
            turn_deg=gt[:, 2].astype(np.float32),
            keep=gt[:, 3].astype(bool),
            discharge=gt[:, 4].astype(np.int8),
            weapon_imp=gt[:, 5].astype(np.int8),
            engaged=gt[:, 6].astype(bool),
            gate_stream_schema=np.asarray([GATE_STREAM_SCHEMA_VERSION]),
        )


def _fresh_state(model: QNNPolicy, obs_row: dict[str, np.ndarray],
                 seed: int) -> _EpisodeState:
    return _EpisodeState(
        episode_index=0,
        obs=obs_row,
        scenario_id="h2h",
        attack_rng=_seed_attack_rng(seed),
        hidden=model.zero_hidden(1)[0].copy(),
        move_commit=np.asarray(_commit_reset_lanes(), dtype=np.float32),
    )


def _obs_row(obs: Mapping[str, np.ndarray], lane: int) -> dict[str, np.ndarray]:
    return {k: np.asarray(v[lane]).copy() for k, v in obs.items()}


def _wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_h2h(side_a: SideSpec, side_b: SideSpec, *, out_dir: Path,
            rounds: int, matches: int = 8, map_id: str = "qnn_arena8",
            swap: bool = False, seed: int = 43, base_port: int = 29000,
            reward_json: Path | None = None,
            server_executable: str = "assets/bin/ppo_arena_server",
            client_executable: str = "assets/bin/ppo_arena_client",
            basedir: str = "assets") -> dict[str, Any]:
    """Play ``rounds`` bouts (split evenly across ``matches`` sealed 1v1s) and
    write rounds.jsonl + per-side stream npz + summary.json into ``out_dir``."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seat0, seat1 = (side_b, side_a) if swap else (side_a, side_b)
    _install_shared_look_grid((seat0, seat1))
    models = {0: load_side(seat0), 1: load_side(seat1)}
    names = {0: seat0.name, 1: seat1.name}
    _log(f"seat0={seat0.name} seat1={seat1.name} map={map_id} "
         f"matches={matches} rounds={rounds} swap={swap}")

    rw = RewardWeights.from_json(
        reward_json if reward_json else
        Path("runs/eval/seed43_stage6_template/config/reward.json"))
    num_lanes = 2 * matches
    backend = ArenaGridBackend(
        num_lanes=num_lanes,
        server_executable=server_executable,
        client_executable=client_executable,
        basedir=basedir, workdir=None, map_id=map_id,
        matches_per_server=matches, seat_mode="self_play",
        base_port=base_port, bot_skill=0,
        max_steps_per_episode=MAX_STEPS_PER_ROUND,
        fixed_tick_hz=TICK_HZ, reward_weights=rw,
        direct_actions=True, observer_mode="virtual",
        scenario_id=f"h2h-{map_id}",
    )
    lanes_of_seat = {s: [e for e in range(num_lanes) if e % 2 == s] for s in (0, 1)}
    match_of_lane = [lane // 2 for lane in range(num_lanes)]

    rounds_per_match = max(1, rounds // matches)
    rounds_done = [0] * matches
    round_start_tick = [0] * matches
    match_live = [True] * matches
    ledgers = {lane: _LaneLedger() for lane in range(num_lanes)}
    streams = {0: _SideStreams(), 1: _SideStreams()}
    round_records: list[dict[str, Any]] = []
    wins = {seat0.name: 0, seat1.name: 0, "draw": 0}

    rng = {s: torch.Generator(device="cpu") for s in (0, 1)}
    for s in (0, 1):
        rng[s].manual_seed(seed + 7919 * s + (100003 if swap else 0))

    try:
        obs = backend.reset()
        states: list[_EpisodeState] = [
            _fresh_state(models[lane % 2], _obs_row(obs, lane),
                         seed * 1_000_003 + lane)
            for lane in range(num_lanes)
        ]

        tick = 0
        t0 = time.perf_counter()
        while any(match_live):
            # One batched forward per SIDE; states/RNG stay side-isolated.
            actions_by_lane: dict[int, Mapping[str, Any]] = {}
            for s in (0, 1):
                lanes = lanes_of_seat[s]
                sts = [states[lane] for lane in lanes]
                acts, next_hidden = _select_actions_batched(
                    models[s], "sampled", sts, rng[s])
                for i, lane in enumerate(lanes):
                    actions_by_lane[lane] = acts[i]
                    states[lane].hidden = next_hidden[i]

            # Pre-step context snapshots + stream rows (pre-step obs semantics,
            # same as run.py's overlap window).
            ctx = [_lane_ctx(states[lane]) for lane in range(num_lanes)]
            for lane in range(num_lanes):
                _log_lane_tick(states[lane], actions_by_lane[lane])

            move = np.zeros((num_lanes, 3), dtype=np.float32)
            look = np.zeros((num_lanes, 3), dtype=np.float32)
            look[:, 0] = 1.0
            attack = np.zeros(num_lanes, dtype=np.int64)
            weapon = np.zeros(num_lanes, dtype=np.int64)
            for lane, act in actions_by_lane.items():
                move[lane] = np.asarray(act["move"], dtype=np.float32)
                look[lane] = np.asarray(act["look"], dtype=np.float32)
                attack[lane] = int(act.get("attack", 0))
                weapon[lane] = int(act.get("weapon", 0))
            batch = backend.step_many(
                {"move": move, "look": look, "attack": attack, "weapon": weapon})
            tick += 1

            for lane in range(num_lanes):
                states[lane].obs = _obs_row(batch.obs, lane)
                states[lane].step_count += 1
                ledgers[lane].fold(batch.infos[lane])

            # Round boundaries, one per match.
            for m in range(matches):
                if not match_live[m]:
                    continue
                l0, l1 = 2 * m, 2 * m + 1
                died = {lane: bool(batch.infos[lane].get("player_died"))
                        for lane in (l0, l1)}
                timed_out = bool(batch.truncated[l0] or batch.truncated[l1])
                if not (died[l0] or died[l1] or timed_out):
                    continue

                if died[l0] and died[l1]:
                    winner, reason = None, "double"
                elif died[l0] or died[l1]:
                    victim = l0 if died[l0] else l1
                    killer = l1 if victim == l0 else l0
                    suicide = bool(batch.infos[victim].get("player_suicide"))
                    winner = killer
                    reason = "suicide" if suicide else "frag"
                else:
                    winner, reason = None, "timeout"

                rec = {
                    "match": m, "round": rounds_done[m],
                    "ticks": tick - round_start_tick[m],
                    "winner": None if winner is None else names[winner % 2],
                    "winner_seat": None if winner is None else winner % 2,
                    "reason": reason,
                    "seats": {},
                }
                for lane in (l0, l1):
                    rec["seats"][names[lane % 2]] = {
                        "seat": lane % 2, "died": died[lane],
                        **ledgers[lane].snapshot(), **ctx[lane]}
                round_records.append(rec)
                wins["draw" if winner is None else names[winner % 2]] += 1
                rounds_done[m] += 1
                round_start_tick[m] = tick

                for lane in (l0, l1):
                    streams[lane % 2].close_segment(states[lane])
                    ledgers[lane] = _LaneLedger()

                if rounds_done[m] >= rounds_per_match:
                    match_live[m] = False
                    continue
                # Timeout already reset engine-side (backend receive() returned
                # the live spawn rows, which states[].obs now hold); a death
                # needs the explicit match reset the eval adapter also does.
                if not timed_out:
                    obs = backend.reset_lanes((l0,))
                for lane in (l0, l1):
                    states[lane] = _fresh_state(
                        models[lane % 2],
                        _obs_row(obs, lane) if not timed_out else states[lane].obs,
                        seed * 1_000_003 + 977 * lane + 7 * rounds_done[m])

            if tick % 600 == 0:
                done = sum(rounds_done)
                rate = tick / max(time.perf_counter() - t0, 1e-9)
                _log(f"tick {tick}  rounds {done}/{rounds_per_match * matches}  "
                     f"{rate:.0f} ticks/s  wins={wins}")
    finally:
        backend.close()

    total = sum(rounds_done)
    decided = wins[seat0.name] + wins[seat1.name]
    summary: dict[str, Any] = {
        "sides": {"seat0": seat0.name, "seat1": seat1.name},
        "specs": {seat0.name: {"run_dir": str(seat0.run_dir),
                               "decode_config": str(seat0.decode_config)},
                  seat1.name: {"run_dir": str(seat1.run_dir),
                               "decode_config": str(seat1.decode_config)}},
        "map": map_id, "matches": matches, "swap": swap, "seed": seed,
        "rounds": total, "wins": wins,
        "decided_rounds": decided,
    }
    for s, spec in ((0, seat0), (1, seat1)):
        w = wins[spec.name]
        lo, hi = _wilson(w, decided) if decided else (0.0, 1.0)
        side_rounds = [r for r in round_records]
        deaths = sum(1 for r in side_rounds if r["seats"][spec.name]["died"])
        suicides = sum(1 for r in side_rounds
                       if r["winner"] and r["winner"] != spec.name
                       and r["reason"] == "suicide")
        air_deaths = sum(1 for r in side_rounds
                         if r["seats"][spec.name]["died"]
                         and r["seats"][spec.name].get("movement_id") == 1)
        gt = np.asarray(streams[s].gate_rows, dtype=np.float64).reshape(-1, 7)
        mv = np.asarray(streams[s].move_rows, dtype=np.int64).reshape(-1, 3)
        ticks_total = max(len(gt), 1)
        per_weapon = {}
        if len(gt):
            eng = gt[:, 6] > 0
            for imp in range(1, 9):
                sel = (gt[:, 5] == imp)
                n_disch = int((gt[sel, 4] > 0).sum())
                n_eng = int((sel & eng).sum())
                if sel.sum():
                    per_weapon[str(imp)] = {
                        "ticks_held": int(sel.sum()),
                        "discharges": n_disch,
                        "fires_per_s_engaged": round(
                            float((gt[sel & eng, 4] > 0).sum())
                            / max(n_eng / TICK_HZ, 1e-9), 3),
                    }
        summary[spec.name] = {
            "win_rate_decided": round(w / decided, 4) if decided else None,
            "win_rate_ci95": [round(lo, 4), round(hi, 4)],
            "deaths": deaths, "deaths_by_suicide": suicides,
            "airborne_death_frac": round(air_deaths / max(deaths, 1), 4),
            "standstill_frac": round(
                float(((mv[:, 0] == 1) & (mv[:, 1] == 1)).mean()), 4) if len(mv) else None,
            "jump_press_rate_per_s": round(
                float((mv[:, 2] == 2).mean()) * TICK_HZ, 3) if len(mv) else None,
            "engaged_frac": round(float((gt[:, 6] > 0).mean()), 4) if len(gt) else None,
            "discharges_per_s": round(
                float((gt[:, 4] > 0).sum()) / (ticks_total / TICK_HZ), 3),
            "damage_dealt_per_round": round(sum(
                r["seats"][spec.name]["damage_dealt"] for r in side_rounds)
                / max(total, 1), 1),
            "per_weapon": per_weapon,
        }

    (out_dir / "rounds.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in round_records))
    for s, spec in ((0, seat0), (1, seat1)):
        streams[s].write(out_dir / f"side_{spec.name}_streams.npz")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    _log(f"done: {total} rounds  wins={wins}  -> {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.eval.h2h", description=__doc__)
    ap.add_argument("--side-a", required=True, type=SideSpec.parse,
                    help="name=run_dir:decode_config.json")
    ap.add_argument("--side-b", required=True, type=SideSpec.parse)
    ap.add_argument("--rounds", type=int, default=160,
                    help="total rounds (split evenly across matches)")
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--map", default="qnn_arena8")
    ap.add_argument("--swap", action="store_true",
                    help="side B takes seat 0 (run both arms for a fair pairing)")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--base-port", type=int, default=29000)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reward-json", type=Path, default=None)
    a = ap.parse_args(argv)
    run_h2h(a.side_a, a.side_b, out_dir=a.out, rounds=a.rounds,
            matches=a.matches, map_id=a.map, swap=a.swap, seed=a.seed,
            base_port=a.base_port, reward_json=a.reward_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
