"""Aim-skill characterization: tracking coherence (F4) + acquisition speed (F3).

Measures how well human players track enemies in their FOV (120° cone + LOS
traceline, recency==0 only). Lead-corrected aim point via canonical physics
tables — no reimplementation.

  F4 – Tracking coherence: fraction of engaged LOS frames where the
       lead-corrected angle to the most-aligned visible actor is < X°.
       Thresholds: 3°, 5°, 10°, 15°.

  F3 – Acquisition speed: frames from first LOS contact to first frame below
       ACQUIRE_THR_DEG (5°). Capped at 3 s (ACQUIRE_MAX_FRAMES). Failed
       acquisitions counted but excluded from the mean.

Output: ranked player list with demo filenames + skilled subset (top quartile).

Performance: episode-level vectorised numpy (no torch), multiprocessing per shard.

Usage:
  PYTHONPATH=src python -m qnn.eval.aim_kernel \\
      --collect-dir artifacts/collect/qwd \\
      [--split val|train|both] [--workers N] \\
      [--out runs/head_probe/_aim_skill.json]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import Any

COHERENCE_DEG     = [3.0, 5.0, 10.0, 15.0]
ACQUIRE_THR_DEG   = 5.0
ACQUIRE_MAX_FRAMES = 60        # 3 s at 20 Hz
MIN_ENGAGED_FRAMES = 30
TOKEN_ACTOR        = 1
MAX_ENTITY_SLOTS   = 16        # entity_count dtype is uint8, max value seen

_DIST_SCALE = 1000.0
_VEL_SCALE  = 2000.0
_WT_V_HORIZ = 2                # column index in weapon physics table (9, 7)

# One 20 Hz tick expressed in MODULE distance-units: rel is /_DIST_SCALE and vel is
# /_VEL_SCALE, so a frame of lead (vel·dt) lands in rel-units after ×(_VEL_SCALE/_DIST_SCALE).
# The single source of truth for the lead-cap unit conversion — the a25 lead module
# (qnn.model.lead_aim) and the human intercept baseline both import it from here.
TICK_DT_MODULE: float = 0.05 * _VEL_SCALE / _DIST_SCALE

# Actor hitbox half-width in game units — the fixed Quake player bbox (±16u;
# confirmed CONSTANT for actor tokens across the QWD corpus). The single source
# of truth for the hbw ruler's angular hitbox radius atan(halfw/range): the
# eval's _intercept_hbw (qnn.eval.run) and the a25 crest-gate decode
# (qnn.model.decode_actions) both import it from here.
ACTOR_HALFW_U: float = 16.0


def _lead_aim_angle_deg_live(
    rel: np.ndarray,     # (M, N, 3) view-frame rel pos, /DIST_SCALE (forward=+x)
    vel: np.ndarray,     # (M, N, 3) ACCEPTED FOR SIGNATURE COMPAT — see below
    imp: np.ndarray,     # (M,)      action-weapon context, impulse 0..8
    weapon_physics: np.ndarray, # (9, 7) build_model_weapon_scalars
    lead_hold_cap: "float | None" = None,         # INERT under the current-position anchor
    lead_hold_cap_radial: "float | None" = None,  # INERT under the current-position anchor
) -> np.ndarray:
    """Crosshair→discharge-anchor angle (degrees), (M, N).

    DISCHARGE LAW ANCHOR = the target's CURRENT POSITION with the per-weapon
    ballistic z-drop applied over the flight time TO that position (rev E,
    2026-08-06). Velocity lead is deliberately ZEROED: measured on the
    corpus, humans do not aim at the moving intercept on ANY projectile
    family at ANY range — the current-position anchor beats linear AND
    hazard-capped lead in median angular error at every range band
    (runs/head_probe/_human_lead_distribution.json, _human_lead_hazard_match
    .json, _human_lead_by_range.json). The ``vel`` param is accepted so the
    28 corpus/eval consumers keep their signatures, but only ``rel`` shapes
    the aim point; the hold caps are consequently inert here (they cap a
    velocity term that is now zero — the deployed LOOK decode's aim prior
    keeps them, see qnn.model.lead_aim.aim_prior_tangent_ffwd).

    Physics still owned by lead_aim.py (``weapon_trajectory`` keeps the
    hitscan ×100 boost — SG/SSG/LG are bit-identical under either anchor;
    ``compute_lead_aim`` supplies flight time + z-anchor). Runs torch-CPU.
    """
    import torch
    from qnn.model.lead_aim import compute_lead_aim, weapon_trajectory

    del vel  # current-position anchor: the target's motion never enters.
    rel_t = torch.from_numpy(np.ascontiguousarray(rel, dtype=np.float32))   # (M,N,3)
    ws_t  = torch.from_numpy(np.ascontiguousarray(weapon_physics, dtype=np.float32))
    imp_t = torch.from_numpy(np.ascontiguousarray(imp, dtype=np.int64))     # (M,)

    v_horiz, drop_const, drop_rate = weapon_trajectory(ws_t, imp_t)         # (M,) each
    aim = compute_lead_aim(rel_t, torch.zeros_like(rel_t), v_horiz,
                           drop_const, drop_rate,
                           lead_hold_cap, lead_hold_cap_radial)             # (M,N,3)

    norms = aim.norm(dim=-1).clamp_min(1e-9)
    cos_a = (aim[..., 0] / norms).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_a)).numpy()                       # (M,N)


def _build_physics_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return (weapon physics, aim_z_drop) as impulse-keyed numpy arrays.

    weapon physics (9, 7) normalized scalars — same as live aim uses.
    aim_z_drop    (9, 2) z-anchor (drop_const, drop_rate) per impulse.
    """
    from qnn.bc.weapon_physics import build_model_weapon_scalars
    from qnn.model.lead_aim import AIM_Z_DROP

    ws  = build_model_weapon_scalars().astype(np.float32)       # (9, 7)
    zdrop = np.array(AIM_Z_DROP, dtype=np.float32)              # (9, 2)
    return ws, zdrop


def action_attack_context(action: np.ndarray) -> np.ndarray:
    """Nearest-discharge weapon context for corpus-only human measurements.

    A27 labels ``action.attack`` only when a discharge actually occurs. Human aim
    references still need a ballistic ruler on the frames leading into and out of
    that discharge, so this analysis helper assigns each frame the nearest nonzero
    attack label within its episode. It is never an observation, model input,
    decoder state, or exported runtime value.
    """
    attack = np.asarray(action, dtype=np.int64).reshape(-1)
    out = np.where((attack >= 1) & (attack <= 8), attack, 0)
    nonzero = np.flatnonzero(out)
    if not len(nonzero):
        return out.astype(np.int8)
    ticks = np.arange(len(out))
    right_pos = np.searchsorted(nonzero, ticks, side="left")
    left_pos = np.clip(right_pos - 1, 0, len(nonzero) - 1)
    right_pos = np.clip(right_pos, 0, len(nonzero) - 1)
    left = nonzero[left_pos]
    right = nonzero[right_pos]
    nearest = np.where(ticks - left < right - ticks, left, right)
    return out[nearest].astype(np.int8)


# ── Core per-episode computation ────────────────────────────────────────────

def densify_entities(ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec, aux=None):
    """Flat per-frame entity streams → RAW padded ``(T, MAX_ENTITY_SLOTS, …)`` arrays —
    the shared densify the corpus-walk creators (intercept / acquisition / aim_range)
    used to each copy.

    Returns ``(all_rel, all_vel, all_los, *aux_padded)``. Arrays are RAW (int→float32,
    UNSCALED): the caller applies ``1/_DIST_SCALE`` / ``1/_VEL_SCALE`` itself, so raw-unit
    derivations (e.g. the hitbox distance in intercept) stay bit-exact. ``all_los`` =
    ``type == TOKEN_ACTOR & recency == 0``. ``aux`` is an optional list of flat
    per-entity arrays densified into ``(T, MAX_ENTITY_SLOTS)`` float32 alongside (e.g.
    ``entity_half_extents``); one padded array is appended per ``aux`` entry."""
    T = len(ep_cnt)
    ent_off = np.concatenate([[0], ep_cnt.cumsum()])
    all_rel = np.zeros((T, MAX_ENTITY_SLOTS, 3), dtype=np.float32)
    all_vel = np.zeros((T, MAX_ENTITY_SLOTS, 3), dtype=np.float32)
    all_los = np.zeros((T, MAX_ENTITY_SLOTS),    dtype=bool)
    aux_out = [np.zeros((T, MAX_ENTITY_SLOTS), dtype=np.float32) for _ in (aux or ())]
    for t in range(T):
        es, ee = int(ent_off[t]), int(ent_off[t + 1])
        n = min(ee - es, MAX_ENTITY_SLOTS)
        if n == 0:
            continue
        all_los[t, :n] = (ep_typ[es:es + n] == TOKEN_ACTOR) & (ep_rec[es:es + n] == 0.0)
        all_rel[t, :n] = ep_rel[es:es + n].astype(np.float32)
        all_vel[t, :n] = ep_vel[es:es + n].astype(np.float32)
        for j, a in enumerate(aux or ()):
            aux_out[j][t, :n] = a[es:es + n]
    return (all_rel, all_vel, all_los, *aux_out)


def iter_shard_episodes(sh, data_dir, obs=(), acts=()):
    """Walk one manifest shard's episodes for a corpus creator. Loads ``entity_count``
    plus the requested ``obs``/``acts`` arrays (mmap), computes frame + entity offsets,
    and yields per episode ``(ei, demo_idx, fsl, esl, arr)``:

      * ``fsl`` — the frame slice (index frame-major arrays: ``arr[k][fsl]``),
      * ``esl`` — the entity slice (index entity-major arrays: ``arr[k][esl]``),
      * ``arr`` — ``{key: mmap array}`` for ``entity_count`` + every requested key.

    Applies the shared ``fe > sh['rows']`` truncation guard; the engaged/threshold
    filter stays the caller's (it varies per creator). Empty shard → yields nothing."""
    ep_lens = [int(x) for x in sh["episode_lengths"]]
    if not ep_lens:
        return
    demo_idxs = sh.get("demo_idxs", [0] * len(ep_lens))
    dd = Path(data_dir)
    arr = {"entity_count": np.load(dd / sh["obs"]["entity_count"], mmap_mode="r")}
    for k in obs:
        path = sh["obs"].get(k)
        if path is not None:
            arr[k] = np.load(dd / path, mmap_mode="r")
            continue
        if k != "entity_recency":
            raise KeyError(k)
        # A27 has no recency feature: every combat row is current-frame.
        # Older human-analysis kernels use recency==0 only as their LOS mask,
        # so derive that analysis-only compatibility array from the A27
        # modality without putting recency back into the collect or model.
        modality_path = sh["obs"].get("entity_modality_id")
        if modality_path is None:
            raise KeyError(
                "entity_recency is absent and entity_modality_id is unavailable"
            )
        modality = np.load(dd / modality_path, mmap_mode="r")
        arr[k] = np.where(
            modality == 0, 0.0, np.finfo(np.float16).max
        ).astype(np.float16)
    for k in acts:
        arr[k] = np.load(dd / sh["actions"][k], mmap_mode="r")
    rows = sh["rows"]
    ent_off   = np.concatenate([[0], arr["entity_count"].cumsum()]).astype(np.int64)
    frame_off = np.concatenate([[0], np.cumsum(ep_lens)]).astype(np.int64)
    for ei, demo_idx in enumerate(demo_idxs):
        fs, fe = int(frame_off[ei]), int(frame_off[ei + 1])
        if fe > rows:
            break
        es, ee = int(ent_off[fs]), int(ent_off[fe])
        yield ei, int(demo_idx), slice(fs, fe), slice(es, ee), arr


def _process_episode(
    ep_cnt:     np.ndarray,   # (T,)    entity count per frame
    ep_rel:     np.ndarray,   # (E, 3)  int16 raw relative positions (u)
    ep_vel:     np.ndarray,   # (E, 3)  int16 raw velocities (u/s)
    ep_typ:     np.ndarray,   # (E,)    int8  entity types
    ep_rec:     np.ndarray,   # (E,)    f16   recency (0 = in LOS this tick)
    ep_attack:  np.ndarray,   # (T,)    action.attack (nonzero on discharge)
    ep_engaged: np.ndarray,   # (T,)    bool  target_probs argmax > 0
    weapon_np:  np.ndarray,   # (9, 7)  normalized weapon scalars
    zdrop_np:   np.ndarray,   # (9, 2)  z-anchor (drop_const, drop_rate)
) -> dict[str, Any]:
    T = len(ep_cnt)
    ent_off = np.concatenate([[0], ep_cnt.cumsum()])

    # Build padded (T, MAX_N) entity arrays — one pass over the flat arrays.
    all_rel = np.zeros((T, MAX_ENTITY_SLOTS, 3), dtype=np.float32)
    all_vel = np.zeros((T, MAX_ENTITY_SLOTS, 3), dtype=np.float32)
    all_los = np.zeros((T, MAX_ENTITY_SLOTS),    dtype=bool)

    for t in range(T):
        es, ee = int(ent_off[t]), int(ent_off[t + 1])
        n = min(ee - es, MAX_ENTITY_SLOTS)
        if n == 0:
            continue
        is_los = (ep_typ[es:es + n] == TOKEN_ACTOR) & (ep_rec[es:es + n] == 0.0)
        all_los[t, :n] = is_los
        all_rel[t, :n] = ep_rel[es:es + n].astype(np.float32)
        all_vel[t, :n] = ep_vel[es:es + n].astype(np.float32)

    # Normalize to model units.
    all_rel *= (1.0 / _DIST_SCALE)
    all_vel *= (1.0 / _VEL_SCALE)

    # Restrict to engaged frames that have at least one LOS actor.
    has_los = all_los.any(axis=1)
    eng = ep_engaged & has_los                    # (T,) — frames we can measure
    eng_idx = np.where(eng)[0]

    aim_angle = np.full(T, np.nan, dtype=np.float32)

    if len(eng_idx):
        rel = all_rel[eng_idx]                    # (M, 16, 3)
        vel = all_vel[eng_idx]
        los = all_los[eng_idx]                    # (M, 16) bool

        # Analysis-only action-weapon context per engaged frame. The v_horiz /
        # z-anchor (including the hitscan ×100 boost) still come from the live
        # trajectory implementation; no equipped-weapon observation is involved.
        imp = action_attack_context(ep_attack)[eng_idx]

        # Vectorised lead-corrected angle — (M, 16) frames × entity slots — via
        # the LIVE aim-prior geometry (compute_lead_aim/weapon_trajectory),
        # the same physics the deployed prior and the model-side coherence metric
        # use. Per-frame weapon impulse broadcasts across the 16 slots.
        angles = _lead_aim_angle_deg_live(rel, vel, imp, weapon_np)   # (M, 16)
        angles = np.where(los, angles, np.inf)    # mask non-LOS slots
        min_a  = angles.min(axis=1)               # (M,)
        aim_angle[eng_idx] = np.where(np.isfinite(min_a), min_a, np.nan)

    # F4 – tracking coherence.
    result: dict[str, Any] = {"n_frames": T, "n_engaged": int(eng.sum())}
    eng_ang = aim_angle[eng & np.isfinite(aim_angle)]
    for thr in COHERENCE_DEG:
        result[f"coh_{int(thr)}deg"] = (
            float((eng_ang < thr).mean()) if len(eng_ang) else float("nan"))

    # F3 – acquisition speed (sequential — event detection needs order).
    appeared = (np.where(
        (~ep_engaged[:-1]) & ep_engaged[1:])[0] + 1 if T > 1
        else np.array([], dtype=np.int32))

    acquire_times: list[int] = []
    n_failed = 0
    for af in appeared:
        window = aim_angle[int(af): min(int(af) + ACQUIRE_MAX_FRAMES, T)]
        hits = np.where(np.isfinite(window) & (window < ACQUIRE_THR_DEG))[0]
        if len(hits):
            acquire_times.append(int(hits[0]))
        else:
            n_failed += 1

    result["n_acquire_events"]  = len(appeared)
    result["n_acquire_failed"]  = n_failed
    result["acquire_mean"] = float(np.mean(acquire_times))   if acquire_times else float("nan")
    result["acquire_p50"]  = float(np.median(acquire_times)) if acquire_times else float("nan")
    result["acquire_p90"]  = float(np.percentile(acquire_times, 90)) if acquire_times else float("nan")
    return result


# ── Shard worker (runs in a subprocess) ─────────────────────────────────────

def _worker(args: tuple) -> list[dict]:
    si, sh, data_dir_str, weapon_np, zdrop_np, min_eng = args
    data_dir = Path(data_dir_str)
    ep_lens   = [int(x) for x in sh["episode_lengths"]]
    demo_idxs = sh.get("demo_idxs", [0] * len(ep_lens))
    if not ep_lens:
        return []

    cnt = np.load(data_dir / sh["obs"]["entity_count"],   mmap_mode="r")
    rel = np.load(data_dir / sh["obs"]["entity_rel"],     mmap_mode="r")
    vel = np.load(data_dir / sh["obs"]["entity_vel"],     mmap_mode="r")
    typ = np.load(data_dir / sh["obs"]["entity_types"],   mmap_mode="r")
    rec_path = sh["obs"].get("entity_recency")
    if rec_path is not None:
        rec = np.load(data_dir / rec_path, mmap_mode="r")
    else:
        modality = np.load(
            data_dir / sh["obs"]["entity_modality_id"], mmap_mode="r"
        )
        rec = np.where(
            modality == 0, 0.0, np.finfo(np.float16).max
        ).astype(np.float16)
    attack = np.load(data_dir / sh["actions"]["attack"], mmap_mode="r")
    tp  = np.load(data_dir / sh["actions"]["target_probs"], mmap_mode="r")

    rows          = sh["rows"]
    engaged_all   = tp.argmax(axis=1) > 0
    ent_offs      = np.concatenate([[0], cnt.cumsum()]).astype(np.int64)
    frame_offs    = np.concatenate([[0], np.cumsum(ep_lens)]).astype(np.int64)

    results = []
    for ei, (ep_len, demo_idx) in enumerate(zip(ep_lens, demo_idxs)):
        fs, fe = int(frame_offs[ei]), int(frame_offs[ei + 1])
        if fe > rows:
            break
        ep_engaged = engaged_all[fs:fe]
        if int(ep_engaged.sum()) < min_eng:
            continue
        es, ee = int(ent_offs[fs]), int(ent_offs[fe])
        stats = _process_episode(
            ep_cnt=np.asarray(cnt[fs:fe], dtype=np.int32),
            ep_rel=np.asarray(rel[es:ee]),
            ep_vel=np.asarray(vel[es:ee]),
            ep_typ=np.asarray(typ[es:ee]),
            ep_rec=np.asarray(rec[es:ee]),
            ep_attack=np.asarray(attack[fs:fe]),
            ep_engaged=ep_engaged,
            weapon_np=weapon_np,
            zdrop_np=zdrop_np,
        )
        stats["shard_idx"]   = si
        stats["episode_idx"] = ei
        stats["demo_idx"]    = int(demo_idx)
        results.append(stats)
    return results


# ── Top-level runner ─────────────────────────────────────────────────────────

def _run_split(
    collect_dir: Path,
    split: str,
    weapon_np: np.ndarray,
    zdrop_np: np.ndarray,
    n_workers: int,
) -> list[dict]:
    data_dir = collect_dir / f"precomputed_{split}"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    shards   = manifest["shards"]

    tasks = [
        (si, sh, str(data_dir), weapon_np, zdrop_np, MIN_ENGAGED_FRAMES)
        for si, sh in enumerate(shards)
    ]

    episodes: list[dict] = []
    if n_workers > 1:
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, batch in enumerate(pool.imap_unordered(_worker, tasks)):
                episodes.extend(batch)
                print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)
    else:
        for i, task in enumerate(tasks):
            episodes.extend(_worker(task))
            print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)

    print(f"  [{split}] {len(episodes)} episodes (>={MIN_ENGAGED_FRAMES} engaged frames)")
    return episodes


def run(
    collect_dir: Path,
    splits: list[str],
    out_path: Path,
    n_workers: int,
) -> None:
    # demo_idx is a position in the global alphabetically-sorted NAS QWD corpus.
    # done.log records which demos were processed in this collect run; indices
    # beyond its length are other QWD demos from the NAS not covered by this log.
    demo_index: dict[int, str] = {
        i: name
        for i, name in enumerate(
            (collect_dir / "done.log").read_text().strip().splitlines()
        )
    }

    weapon_np, zdrop_np = _build_physics_tables()

    all_episodes: list[dict] = []
    for split in splits:
        all_episodes.extend(
            _run_split(collect_dir, split, weapon_np, zdrop_np, n_workers))

    print(f"\nTotal: {len(all_episodes)} episodes across {splits}")

    # Per-player aggregation (frame-weighted coherence).
    from collections import defaultdict
    demo_buckets: dict[int, list[dict]] = defaultdict(list)
    for ep in all_episodes:
        demo_buckets[ep["demo_idx"]].append(ep)

    def _wt_mean(eps: list[dict], key: str) -> float:
        pairs = [(e[key], e["n_engaged"]) for e in eps if not math.isnan(e[key])]
        if not pairs:
            return float("nan")
        return float(sum(c * w for c, w in pairs) / sum(w for _, w in pairs))

    def _nanmean(vals: list[float]) -> float:
        v = [x for x in vals if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    players = []
    for did, eps in demo_buckets.items():
        players.append({
            "demo_idx":          did,
            "demo":              demo_index.get(did, f"unknown_{did}"),
            "n_episodes":        len(eps),
            "n_engaged_frames":  sum(e["n_engaged"] for e in eps),
            "coh_3deg":          round(_wt_mean(eps, "coh_3deg"),  4),
            "coh_5deg":          round(_wt_mean(eps, "coh_5deg"),  4),
            "coh_10deg":         round(_wt_mean(eps, "coh_10deg"), 4),
            "coh_15deg":         round(_wt_mean(eps, "coh_15deg"), 4),
            "acquire_mean":      round(_nanmean([e["acquire_mean"] for e in eps]), 2),
        })

    players.sort(
        key=lambda p: p["coh_5deg"] if not math.isnan(p["coh_5deg"]) else -1,
        reverse=True)

    coh5_vals = np.array(
        [p["coh_5deg"] for p in players if not math.isnan(p["coh_5deg"])],
        dtype=np.float32)
    pcts = {f"p{pc}": round(float(np.percentile(coh5_vals, pc)), 4)
            for pc in [25, 50, 75, 90, 95, 99]}
    p75 = float(np.percentile(coh5_vals, 75))
    skilled = [p for p in players if p["coh_5deg"] >= p75]

    out = {
        "splits":           splits,
        "collect_dir":      str(collect_dir),
        "measurement":      "lead_corrected_most_aligned_los_actor",
        "n_players":        len(players),
        "n_episodes_total": len(all_episodes),
        "player_coh5deg_percentiles": pcts,
        "p75_threshold":    round(p75, 4),
        "n_skilled_players": len(skilled),
        "players":          players,
        "skilled_players":  skilled,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\nPlayer coh@5° percentiles: {pcts}")
    print(f"Skilled players (P75+ coh@5°≥{p75:.3f}): {len(skilled)} / {len(players)}")
    print(f"\nTop 15 players:")
    for p in players[:15]:
        print(f"  {p['demo']:52s}  "
              f"coh@5°={p['coh_5deg']:.3f}  coh@10°={p['coh_10deg']:.3f}  "
              f"eps={p['n_episodes']:3d}  eng={p['n_engaged_frames']:6d}  "
              f"acq={p['acquire_mean']:.1f}f")
    print(f"\nWritten → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="both",
                    choices=["val", "train", "both"],
                    help="Which split(s) to process (default: both)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                    help="Parallel shard workers (default: nCPU-1)")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/head_probe/_aim_skill.json"))
    args = ap.parse_args()
    splits = ["train", "val"] if args.split == "both" else [args.split]
    print(f"Collect: {args.collect_dir}  splits={splits}  workers={args.workers}")
    run(args.collect_dir, splits, args.out, args.workers)


if __name__ == "__main__":
    main()
