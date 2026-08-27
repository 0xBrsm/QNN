"""Backfill the alignment-to-intercept (hbw) sidecar cache beside an existing
BC collect's shards (fire-at-alignment objective, rung 1;
agents/plans/fire-at-alignment-objective.md).

Writes ``shard_NNNNNN_act_align_hbw.npy`` (float16, ``(T_shard,)``) next to
each shard's existing arrays and registers ``actions.align_hbw`` in the
split's ``manifest.json`` so the standard train loader picks it up like any
other action key (``qnn.bc.streaming_source`` / ``qnn.bc.supervised_loop``
load every ``shard["actions"]`` entry generically — no loader change
needed). Structurally modeled on ``qnn.bc.cache_look_tan`` (batch backfill,
idempotent/skippable, multiprocess-per-shard, ``__main__`` CLI), but the
manifest update is NOT gated behind a separate ``--register`` step: unlike
``look_tan`` (a lossless re-encoding of an existing action array),
``align_hbw`` is a brand-new derived quantity nothing currently depends on,
so registering it as soon as it's computed cannot invalidate a run pinned
to the current collection fingerprint.

THE LABEL — per frame, the lead-corrected angular error to the most-aligned
in-LOS actor's INTERCEPT point, normalized by the hitbox-half-width (hbw)
angular radius at that actor's current range:

    hbw = lead_angle_deg / degrees(atan2(ACTOR_HALFW_U, range_u))

identical to :func:`qnn.eval.run._intercept_hbw` — ONE law shared by the
training label here, the closed-loop ruler (``_intercept_hbw`` /
``interception_dist``), and the human baseline
(``qnn.human.intercept``/``scripts/analysis/aim_intercept_skill.py``), so
the training signal is judged on the exact metric the sweep already
reports (the coh->intercept lesson: build every consumer off one kernel,
never a parallel reimplementation). ``range_u`` is the RAW (unlead)
distance to the actor's current position (matching ``_intercept_hbw``'s
caller, ``_lead_aim_cos_batched`` — the hbw normalizer is NOT re-measured
at the lead point). The lead geometry itself
(``qnn.model.lead_aim.compute_lead_aim`` / ``weapon_trajectory`` via
``qnn.eval.aim_kernel._lead_aim_angle_deg_live``) runs the deployed
LINEAR lead (no hazard-discount caps) — the same defaults
``_lead_aim_cos_batched`` uses for the model-side ruler.

WEAPON for the lead law: the FIRED weapon by attack context
(``qnn.eval.aim_kernel.action_attack_context`` over ``act.attack`` — the
op-attack convention: a discharge class, script-attributed to the nearest
shot in the episode, NEVER a held-weapon observation lane).

SENTINEL (-1.0, float16 — same storage convention as ``act_look_tan``):
a frame gets the sentinel when EITHER

  * no engaged target this frame — ``target_probs`` argmax is class 0
    (no engagement) OR no in-LOS actor token is present at all (mirrors
    ``qnn.eval.aim_kernel``'s ``eng = ep_engaged & has_los``); OR
  * no attributable weapon — ``action_attack_context`` returns 0 for this
    tick, i.e. the WHOLE episode has zero discharges (a27's
    nearest-discharge lookup only returns 0 when there is nothing nonzero
    anywhere in the episode to attribute).

numpy-only module (no top-level ``import torch`` — cache passes must stay
importable from the torch-free devcontainer python, the ``cache_look_tan``
precedent). The lead geometry itself needs torch at CALL time
(``qnn.eval.aim_kernel._lead_aim_angle_deg_live`` does its own deferred
``import torch``), so actually RUNNING this pass needs a container with
torch installed (the CPU container), same as ``qnn.human.intercept`` /
``qnn.eval.aim_kernel``.

Usage (CPU container — devcontainer python has no torch):
  PYTHONPATH=src python -m qnn.bc.cache_align_hbw \\
      --collect-dir artifacts/collect/qwd_v4d_v3vis [--workers N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from qnn.eval import aim_kernel as A

# Sentinel for "no engaged target OR no attributable weapon this frame" —
# imported by qnn.model.attack_with_head so the training-side weight law and
# this cache pass share one constant (never a duplicated magic number).
SENTINEL: float = -1.0


def _episode_align_hbw(
    ep_cnt: np.ndarray,      # (T,)   entity count per frame
    ep_rel: np.ndarray,      # (E, 3) int16 raw relative positions (u)
    ep_vel: np.ndarray,      # (E, 3) int16 raw velocities (u/s)
    ep_typ: np.ndarray,      # (E,)   int8  entity types
    ep_rec: np.ndarray,      # (E,)   f16   recency (0 = in LOS this tick)
    ep_attack: np.ndarray,   # (T,)   action.attack (nonzero on discharge)
    ep_target_probs: np.ndarray,  # (T, C) target-engagement distribution
    weapon_np: np.ndarray,   # (9, 7) normalized weapon scalars (build_model_weapon_scalars)
) -> np.ndarray:
    """Per-frame ``align_hbw`` (float32) for one episode's flat arrays.

    Mirrors ``qnn.eval.aim_kernel._process_episode``'s engaged/has_los gate
    and most-aligned-actor selection, plus ``qnn.human.intercept``'s
    per-discharge range/slot tracking — but normalizes with the FIXED
    ``ACTOR_HALFW_U`` constant (the ``_intercept_hbw`` law), not the
    per-entity ``entity_half_extents`` obs field ``qnn.human.intercept``
    uses (numerically identical for actor tokens — the docstring on
    ``ACTOR_HALFW_U`` confirms it's constant across the corpus — but this
    pass mirrors the model-side ruler's formula verbatim, per the
    fire-at-alignment plan's "one law everywhere").
    """
    T = len(ep_cnt)
    out = np.full(T, SENTINEL, dtype=np.float32)
    if T == 0:
        return out

    all_rel, all_vel, all_los = A.densify_entities(ep_cnt, ep_rel, ep_vel, ep_typ, ep_rec)
    dist_u = np.linalg.norm(all_rel, axis=2)              # raw units, (T, 16)
    rel_m = all_rel * (1.0 / A._DIST_SCALE)
    vel_m = all_vel * (1.0 / A._VEL_SCALE)

    has_los = all_los.any(axis=1)
    tp = np.asarray(ep_target_probs)
    engaged = has_los & (tp.argmax(axis=1) > 0)
    eng_idx = np.where(engaged)[0]
    if not len(eng_idx):
        return out

    # Episode-scoped nearest-discharge weapon attribution (op_attack
    # convention) — 0 only when the WHOLE episode has no discharge at all.
    imp_all = A.action_attack_context(ep_attack)
    imp_eng = imp_all[eng_idx].astype(np.int64)
    has_weapon = (imp_eng >= 1) & (imp_eng <= 8)
    valid_idx = eng_idx[has_weapon]
    if not len(valid_idx):
        return out

    rel_v = rel_m[valid_idx]
    vel_v = vel_m[valid_idx]
    los_v = all_los[valid_idx]
    dist_v = dist_u[valid_idx]
    imp_v = imp_eng[has_weapon]

    angles = A._lead_aim_angle_deg_live(rel_v, vel_v, imp_v, weapon_np)   # (M, 16) deg
    angles = np.where(los_v, angles, np.inf)
    slot = np.argmin(angles, axis=1)
    mrow = np.arange(len(valid_idx))
    best_ang = angles[mrow, slot]
    best_range = dist_v[mrow, slot]

    finite = np.isfinite(best_ang)
    if not finite.any():
        return out
    ang_radius_deg = np.degrees(
        np.arctan2(A.ACTOR_HALFW_U, np.maximum(best_range, 1e-3)))
    hbw = best_ang / np.maximum(ang_radius_deg, 1e-6)
    out[valid_idx[finite]] = hbw[finite].astype(np.float32)
    return out


def _process_one_shard(args: tuple) -> tuple[int, str, int, int]:
    """Worker: compute ``align_hbw`` for one shard, write to disk.

    Returns ``(shard_idx, fname, rows, n_scored)``. Skips recompute (just
    re-reports) if the sidecar file already exists on disk — the
    resume-after-partial-run idempotency the manifest-registration pass
    below relies on.
    """
    split_dir_str, shard_idx, shard = args
    split_dir = Path(split_dir_str)
    out_fname = f"shard{shard_idx:06d}_act_align_hbw.npy"
    out_path = split_dir / out_fname
    rows = int(shard["rows"])

    if out_path.exists():
        existing = np.asarray(np.load(out_path, mmap_mode="r"), dtype=np.float32)
        n_scored = int((existing > SENTINEL).sum())
        return shard_idx, out_fname, rows, n_scored

    weapon_np, _zdrop = A._build_physics_tables()
    shard_out = np.full(rows, SENTINEL, dtype=np.float32)
    for _ei, _demo_idx, fsl, esl, arr in A.iter_shard_episodes(
        shard, split_dir,
        obs=("entity_rel", "entity_vel", "entity_types", "entity_recency"),
        acts=("attack", "target_probs"),
    ):
        cnt = np.asarray(arr["entity_count"][fsl], dtype=np.int32)
        rel = np.asarray(arr["entity_rel"][esl])
        vel = np.asarray(arr["entity_vel"][esl])
        typ = np.asarray(arr["entity_types"][esl])
        rec = np.asarray(arr["entity_recency"][esl])
        attack = np.asarray(arr["attack"][fsl])
        tp = np.asarray(arr["target_probs"][fsl])
        shard_out[fsl] = _episode_align_hbw(
            cnt, rel, vel, typ, rec, attack, tp, weapon_np)

    n_scored = int((shard_out > SENTINEL).sum())
    np.save(out_path, shard_out.astype(np.float16))
    return shard_idx, out_fname, rows, n_scored


def _process_split(split_dir: Path, n_workers: int) -> None:
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = manifest["shards"]

    def _out_fname(i: int) -> str:
        return f"shard{i:06d}_act_align_hbw.npy"

    pending = [
        (str(split_dir), i, s) for i, s in enumerate(shards)
        if not (split_dir / _out_fname(i)).exists()
    ]
    print(f"  {split_dir.name}: {len(shards)} shards, {len(pending)} to compute")

    total_rows = 0
    total_scored = 0
    if pending:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = [ex.submit(_process_one_shard, a) for a in pending]
            done = 0
            for fut in as_completed(futures):
                _, _, rows, n_scored = fut.result()
                total_rows += rows
                total_scored += n_scored
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"    {done}/{len(pending)} shards "
                          f"({total_rows:,} rows, {total_scored:,} scored)")

    changed = False
    for i, s in enumerate(shards):
        fname = _out_fname(i)
        if (split_dir / fname).exists() and "align_hbw" not in s.get("actions", {}):
            s.setdefault("actions", {})["align_hbw"] = fname
            changed = True
    if changed:
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"    manifest updated: {manifest_path}")
    else:
        print(f"    manifest already up to date: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collect-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=min(30, os.cpu_count() or 4))
    args = ap.parse_args()
    for split in ("precomputed_train", "precomputed_val"):
        d = args.collect_dir / split
        if d.exists():
            _process_split(d, args.workers)
        else:
            print(f"  {split}: missing, skipping")


if __name__ == "__main__":
    main()
