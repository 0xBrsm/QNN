"""Human BLIND-FIRE rate — the corpus-side counterpart to
``qnn.diag.crest_frontier.blind_fire_mask``, under the IDENTICAL LOS rule.

MOTIVATION (rung-3 decode fire-bias sweep, 8/7): a counterfactual sweep on the
bot found 22.7%-29.1% of its SG discharges land on ticks with NO in-LOS actor
at all. Those "blind" discharges burn the refire cooldown but never reach the
crest instrument (``intercept_windows.npz`` gates on ``lead_valid``), and they
inflate the bot's TOTAL discharge rate relative to its AIMED (LOS-present)
rate -- which matters because the PPO cadence target
(``qnn.ppo.pfire_target.FireOccupancyTarget``, 1.543278/s for SG+SSG) is a
human rate CONDITIONAL ON ENGAGED-LOS (``qnn.human.op_attack``'s
``rate_per_s = op_attack_ticks / engaged_los_ticks * hz``, and by that
module's own construction a discharge with no LOS actor can never enter
either its numerator or its denominator -- ``eng`` requires ``has``).
That target is therefore SILENT on how often humans fire blind. This module
fills the gap: same corpus, same weapon attribution, same LOS predicate,
counted the other way around.

THE LOS RULE, verified identical on both sides of the human/bot comparison
before this module existed (see agents/plans/blind-fire-cadence.md):

  * bot   -- ``qnn.diag.crest_frontier.blind_fire_mask``: a discharge (engine
    ``shots_fired > 0``) on a tick where ``_lead_ruler_batched``'s
    ``lead_valid`` is false, i.e. the engine's ``QNN_PrimaryObservationIsCurrent``
    found no actor whose primary-observation timestamp equals the current
    tick (``src/engine/common/qnn_store.c``). Equivalently: no ``QNN_ENT_ACTOR``
    is being SIGHTed (in FOV, traced) THIS tick.
  * PPO reward path -- ``qnn.ppo.align_hbw.AlignHbw.los``: ``entity_types ==
    TOKEN_ACTOR & entity_modality_id == 0`` on the model's own packed obs.
    ``entity_modality_id`` is stamped by the SAME engine timestamp
    (``QNN_PrimaryObservationModalityId`` / ``QNN_PrimaryObservationIsCurrent``),
    so this is the identical fact read a second way, not a second law.
  * corpus (here) -- ``entity_types == TOKEN_ACTOR & entity_recency == 0``,
    the human-analysis convention every ``qnn.human`` / ``qnn.eval.aim_kernel``
    creator uses. A27-era collects (this repo's only collects) carry no
    ``entity_recency`` field; ``qnn.eval.aim_kernel.iter_shard_episodes``
    transparently derives it from ``entity_modality_id`` (0 -> 0.0, else a
    large sentinel) -- the SAME field ``AlignHbw.los`` reads, so "recency==0"
    and "modality==0" are one predicate under two names, not two rules.

Because the human demo corpus was recorded through the same C-worker percept
pipeline the bot's own obs comes from, "no in-LOS actor" means the same thing
on both sides: no actor entity is being actively sighted (traced, in FOV)
this exact tick, as opposed to remembered (MEMORY), heard (SOUND), or merely
in-PVS (PROXIMITY) -- see ``src/docs/vocab.md``'s modality table.

DIFFERS FROM ``qnn.human.op_attack`` DELIBERATELY: that module's ``eng`` gate
requires BOTH an LOS actor AND ``target_probs`` argmax > 0 (a hindsight
engagement label), so a human shot fired with an LOS actor present but no
labeled "engagement" falls outside its counted population entirely (neither
numerator nor denominator). This module's blind predicate has no such second
gate -- it mirrors the bot's ``blind_fire_mask`` exactly, which also carries
no engagement-belief condition, only LOS-actor presence. The two therefore
disagree on the ENGAGED-tick denominator (this module's is a strict superset
of op_attack's); both are reported so a reader can see the gap.

Output (``<collect>/human_baseline/_blind_fire_byweapon.json``):
  per weapon: total effective discharges, blind discharges (no LOS actor at
  all), blind rate, plus the pure-LOS engaged-tick count (no target_probs
  condition) for the LOS-only cadence comparison
  (``qnn.diag.crest_frontier``'s ``cadence_per_engaged_tick``).

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.blind_fire \\
      --collect-dir artifacts/collect/qwd_v4d_v3vis [--split val] [--workers 8] \\
      [--out <collect>/human_baseline/_blind_fire_byweapon.json]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from qnn.eval import aim_kernel as A

IMPULSE_NAME = {1: "Axe", 2: "SG", 3: "SSG", 4: "NG", 5: "SNG",
                6: "GL", 7: "RL", 8: "LG"}
NW = 8
TOKEN_ACTOR = 1                       # entity_types actor token (src/qnn/eval/aim_kernel.py)
MIN_ENGAGED_FRAMES_EP = 30            # matches qnn.human.op_attack's episode admission
HZ = 20.0

# SG/SSG (discrete hitscan) and NG/SNG (area fill) share physics per weapon
# and are pooled by the decode-fit calibration families
# (qnn.decode_fit.human_refs.CALIBRATION_FAMILIES) -- kept here only as
# documentation; this module reports per-impulse and lets the caller pool.


def _episode(cnt: np.ndarray, typ: np.ndarray, rec: np.ndarray,
             attack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-weapon ``(n_discharges, n_blind)`` [8,2] plus a 1-elt
    ``(engaged_los_ticks,)`` array for one episode's flat arrays.

    ``has_los[t]`` -- an in-LOS actor participates this tick, the identical
    predicate ``qnn.human.op_attack._episode`` computes for its ``has``.
    ``blind`` -- an effective discharge (``action.attack`` in 1..8,
    weapon-attributable) on a tick where ``has_los`` is false. UNLIKE
    ``qnn.human.op_attack``, this does NOT additionally require
    ``target_probs`` argmax > 0: the bot's ``blind_fire_mask`` carries no
    engagement-belief gate either, only LOS-actor presence, so matching it
    means dropping that second condition here too.

    Episode admission (``target_probs`` argmax > 0 for >= 30 frames) is
    still applied by the caller, purely to keep the episode POPULATION
    identical to ``qnn.human.op_attack``'s -- it does not gate individual
    ticks within an admitted episode.
    """
    T = len(cnt)
    off = np.concatenate([[0], np.asarray(cnt).cumsum()]).astype(np.int64)
    has_los = np.zeros(T, bool)
    for t in range(T):
        es, ee = int(off[t]), int(off[t + 1])
        if ee > es:
            has_los[t] = bool(((typ[es:ee] == TOKEN_ACTOR) & (rec[es:ee] == 0.0)).any())

    attack = np.asarray(attack, dtype=np.int64).reshape(-1)
    imp = A.action_attack_context(attack)
    fire = (attack >= 1) & (attack <= 8)

    counts = np.zeros((NW, 2), np.int64)   # [:, 0] = n_discharges, [:, 1] = n_blind
    ok = fire & (imp >= 1) & (imp <= 8)
    idx = np.where(ok)[0]
    eng_ticks = np.zeros(NW, np.int64)
    # pure-LOS engaged ticks per weapon context (no target_probs condition) --
    # the AlignHbw.all_los-equivalent denominator, for the LOS-only cadence
    # comparison. Weapon context still comes from the nearest-discharge
    # attribution (qnn.human.op_attack's convention), applied to EVERY
    # in-LOS tick, not just discharge ticks.
    los_idx = np.where(has_los & (imp >= 1) & (imp <= 8))[0]
    if len(los_idx):
        np.add.at(eng_ticks, imp[los_idx] - 1, 1)
    if not len(idx):
        return counts, eng_ticks
    np.add.at(counts, (imp[idx] - 1, 0), 1)
    blind_idx = idx[~has_los[idx]]
    if len(blind_idx):
        np.add.at(counts, (imp[blind_idx] - 1, 1), 1)
    return counts, eng_ticks


def _worker(args: tuple) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    sh, dd = args
    res: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _ei, dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, dd,
            obs=("entity_types", "entity_recency"),
            acts=("target_probs", "attack")):
        tp = np.asarray(arr["target_probs"][fsl])
        if int((tp.argmax(1) > 0).sum()) < MIN_ENGAGED_FRAMES_EP:
            continue
        counts, eng_ticks = _episode(
            np.asarray(arr["entity_count"][fsl], np.int32),
            np.asarray(arr["entity_types"][esl]),
            np.asarray(arr["entity_recency"][esl]),
            np.asarray(arr["attack"][fsl]))
        if int(dmi) not in res:
            res[int(dmi)] = (np.zeros((NW, 2), np.int64), np.zeros(NW, np.int64))
        rc, re = res[int(dmi)]
        rc += counts
        re += eng_ticks
    return res


def run(collect_dir: Path, splits: list[str], out_path: Path,
        n_workers: int) -> dict[str, Any]:
    """Compute the human per-weapon blind-fire reference for a collect and
    write it to ``out_path``. Signature matches ``qnn.human.op_attack.run``."""
    collect_dir = Path(collect_dir)
    per_counts: dict[int, np.ndarray] = {}
    per_eng: dict[int, np.ndarray] = {}
    split = splits[0] if splits else "val"
    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            continue
        tasks = [(sh, str(dd)) for sh in json.loads(man.read_text())["shards"]]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                for dmi, (counts, eng_ticks) in r.items():
                    if dmi not in per_counts:
                        per_counts[dmi] = np.zeros((NW, 2), np.int64)
                        per_eng[dmi] = np.zeros(NW, np.int64)
                    per_counts[dmi] += counts
                    per_eng[dmi] += eng_ticks
                print(f"  [{split}] {i + 1}/{len(tasks)} shards done", flush=True)

    weapons: dict[str, Any] = {}
    for impw in range(1, NW + 1):
        name = IMPULSE_NAME[impw]
        disc = np.array([per_counts[d][impw - 1, 0] for d in per_counts], np.int64)
        blind = np.array([per_counts[d][impw - 1, 1] for d in per_counts], np.int64)
        eng = np.array([per_eng[d][impw - 1] for d in per_eng], np.int64)
        tot_disc, tot_blind, tot_eng = int(disc.sum()), int(blind.sum()), int(eng.sum())
        entry: dict[str, Any] = {
            "impulse": impw,
            "n_discharges": tot_disc,
            "n_blind": tot_blind,
            "blind_rate": round(tot_blind / tot_disc, 4) if tot_disc else None,
            "engaged_los_ticks": tot_eng,
            "aimed_rate_per_s": (
                round((tot_disc - tot_blind) / tot_eng * HZ, 4) if tot_eng else None),
        }
        m = disc >= 50
        if m.sum() >= 6:
            rates = blind[m] / disc[m]
            entry["demo_blind_rate_q"] = {
                q: round(float(np.percentile(rates, p)), 4)
                for q, p in (("p10", 10), ("p25", 25), ("p50", 50),
                             ("p75", 75), ("p90", 90))}
            entry["n_demos"] = int(m.sum())
        weapons[name] = entry

    out = {
        "_meta": {
            "contract": ("blind discharge = action.attack in 1..8, "
                         "weapon-attributable (nearest-discharge context), on "
                         "a tick with NO in-LOS actor at all (entity_types == "
                         "ACTOR & entity_recency == 0, none present) -- the "
                         "IDENTICAL predicate qnn.diag.crest_frontier."
                         "blind_fire_mask applies to the bot's intercept "
                         "trace. Unlike qnn.human.op_attack, no target_probs "
                         "engagement condition is applied. engaged_los_ticks "
                         "counts every in-LOS tick attributed to this weapon, "
                         "discharge or not (no target_probs condition either) "
                         "-- the AlignHbw.all_los-equivalent denominator."),
            "collect_dir": str(collect_dir),
            "split": split,
            "hz": HZ,
            "n_demos": len(per_counts),
        },
        "weapons": weapons,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    pooled_disc = sum(int(per_counts[d][:, 0].sum()) for d in per_counts)
    pooled_blind = sum(int(per_counts[d][:, 1].sum()) for d in per_counts)
    print(f"\npooled blind-fire rate (all weapons): "
          f"{pooled_blind / max(pooled_disc, 1):.4f} over {pooled_disc} discharges, "
          f"{len(per_counts)} demos")
    print(f"Written -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/_blind_fire_byweapon.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_blind_fire_byweapon.json"
    run(args.collect_dir, [args.split], out, args.workers)


if __name__ == "__main__":
    main()
