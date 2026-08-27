"""Per-action-weapon OP-ATTACK RATE + inter-shot intervals (HUMAN).

The world-results attack reference for decode-fit stage 6: what a human's
attack DELIVERS per engaged-LOS second for each attack-with weapon, counted at
decision granularity (op-attack frames), not button hygiene (press runs).

  effective attack = action.attack in 1..8
  engaged-LOS      = target_probs argmax>0 AND >=1 LOS actor (recency==0)
  attack context   = nearest nonzero ``action.attack`` category in the episode

Same conventions as attack_discipline_byweapon.py; no lead kernel needed
(rates are alignment-unconditional -- alignment discipline stays that
script's job). Weapon-chain bolts (LG/NG 0.1s think shots between W_Attack
re-entries) are engine-owned on BOTH human and bot sides and excluded on
both, so the pair stays commensurate with the eval's per-LOS-tick
engine_los_attack tables.

Intervals are gaps between consecutive effective-attack frames within an
unbroken (engaged-LOS AND same attack context) run -- the within-engagement refire
texture ("histogram the results"), right-censoring dropped with the run.

Output (<collect>/human_baseline/_op_attack_rate_byweapon.json):
  per weapon: engaged-LOS ticks, op-attack ticks, rate/s, per-demo rate
  quantiles, interval quantiles + fixed-bin histogram (0..3 s, 0.05 s bins).

Usage:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=src \\
    python -m qnn.human.op_attack \\
      --collect-dir artifacts/collect/qwd [--split val] [--workers 8] \\
      [--out <collect>/human_baseline/_op_attack_rate_byweapon.json]
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
MIN_ENGAGED_FRAMES_EP = 30            # episode admit (matches qnn.eval.aim_kernel)
HZ = 20.0
INTERVAL_BINS = np.arange(0.0, 3.0 + 1e-9, 0.05)   # right-censor bucket appended


def _episode(cnt, typ, rec, attack, tp):
    """Per-weapon (eng_ticks, op_ticks) [8,2] + interval samples {imp: [sec]}."""
    T = len(cnt)
    off = np.concatenate([[0], np.asarray(cnt).cumsum()]).astype(np.int64)
    has = np.zeros(T, bool)
    for t in range(T):
        es, ee = int(off[t]), int(off[t + 1])
        if ee > es:
            has[t] = bool(((typ[es:ee] == TOKEN_ACTOR) & (rec[es:ee] == 0.0)).any())
    eng = (np.asarray(tp).argmax(1) > 0) & has

    attack = np.asarray(attack, dtype=np.int64).reshape(-1)
    imp = A.action_attack_context(attack)
    fire = (attack >= 1) & (attack <= 8)

    counts = np.zeros((NW, 2), np.int64)
    ivals: dict[int, list[float]] = {}
    ok = eng & (imp >= 1) & (imp <= 8)
    idx = np.where(ok)[0]
    if not len(idx):
        return counts, ivals
    np.add.at(counts, (imp[idx] - 1, 0), 1)
    np.add.at(counts, (imp[idx[fire[idx]]] - 1, 1), 1)

    # unbroken (engaged-LOS & same-weapon) runs -> intervals between op-attacks
    run_break = np.where(np.diff(idx) != 1)[0] + 1
    wpn_break = np.where(np.diff(imp[idx]) != 0)[0] + 1
    starts = np.unique(np.concatenate([[0], run_break, wpn_break]))
    bounds = np.append(starts, len(idx))
    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = idx[s:e]
        ft = seg[fire[seg]]
        if len(ft) >= 2:
            ivals.setdefault(int(imp[seg[0]]), []).extend(
                (np.diff(ft) / HZ).tolist())
    return counts, ivals


def _worker(args):
    sh, dd = args
    res: dict[int, tuple] = {}
    for _ei, dmi, fsl, esl, arr in A.iter_shard_episodes(
            sh, dd,
            obs=("entity_types", "entity_recency"),
            acts=("target_probs", "attack")):
        tp = np.asarray(arr["target_probs"][fsl])
        if int((tp.argmax(1) > 0).sum()) < MIN_ENGAGED_FRAMES_EP:
            continue
        counts, ivals = _episode(
            np.asarray(arr["entity_count"][fsl], np.int32), np.asarray(arr["entity_types"][esl]),
            np.asarray(arr["entity_recency"][esl]), np.asarray(arr["attack"][fsl]), tp)
        if int(dmi) not in res:
            res[int(dmi)] = (np.zeros((NW, 2), np.int64), {})
        rc, rv = res[int(dmi)]
        rc += counts
        for w, v in ivals.items():
            rv.setdefault(w, []).extend(v)
    return res


def run(collect_dir: Path, splits: list[str], out_path: Path, n_workers: int) -> dict[str, Any]:
    """Compute the human per-weapon op-attack rate/interval reference for a collect
    and write it to ``out_path``. Corpus-derived + model-agnostic — cached once per
    collect (qnn.human). Returns the written document. Signature matches the other
    corpus creators: (collect_dir, splits, out_path, n_workers)."""
    collect_dir = Path(collect_dir)
    per_counts: dict[int, np.ndarray] = {}
    per_ivals: dict[int, dict[int, list[float]]] = {}
    for split in splits:
        dd = collect_dir / f"precomputed_{split}"
        man = dd / "manifest.json"
        if not man.exists():
            continue
        tasks = [(sh, str(dd)) for sh in json.loads(man.read_text())["shards"]]
        with mp.Pool(min(n_workers, len(tasks))) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, tasks)):
                for dmi, (counts, ivals) in r.items():
                    if dmi not in per_counts:
                        per_counts[dmi] = np.zeros((NW, 2), np.int64)
                        per_ivals[dmi] = {}
                    per_counts[dmi] += counts
                    for w, v in ivals.items():
                        per_ivals[dmi].setdefault(w, []).extend(v)
                print(f"  [{split}] {i+1}/{len(tasks)} shards done", flush=True)

    weapons: dict[str, Any] = {}
    for impw in range(1, NW + 1):
        name = IMPULSE_NAME[impw]
        eng = np.array([per_counts[d][impw - 1, 0] for d in per_counts], np.int64)
        opa = np.array([per_counts[d][impw - 1, 1] for d in per_counts], np.int64)
        tot_eng, tot_op = int(eng.sum()), int(opa.sum())
        entry: dict[str, Any] = {
            "impulse": impw,
            "engaged_los_ticks": tot_eng,
            "op_attack_ticks": tot_op,
            "rate_per_s": round(tot_op / tot_eng * HZ, 4) if tot_eng else None,
        }
        # per-demo rate spread (demos with enough exposure on this weapon)
        m = eng >= 200
        if m.sum() >= 6:
            rates = opa[m] / eng[m] * HZ
            entry["demo_rate_q"] = {q: round(float(np.percentile(rates, p)), 4)
                                    for q, p in (("p10", 10), ("p25", 25), ("p50", 50),
                                                 ("p75", 75), ("p90", 90))}
            entry["n_demos"] = int(m.sum())
        iv = np.array([x for d in per_ivals for x in per_ivals[d].get(impw, [])],
                      np.float64)
        if len(iv) >= 50:
            hist, _ = np.histogram(np.clip(iv, 0, INTERVAL_BINS[-1] - 1e-6),
                                   bins=INTERVAL_BINS)
            entry["interval_q_s"] = {q: round(float(np.percentile(iv, p)), 4)
                                     for q, p in (("p25", 25), ("p50", 50),
                                                  ("p75", 75), ("p90", 90))}
            entry["interval_hist"] = {"bin_s": 0.05, "max_s": 3.0,
                                      "counts": hist.astype(int).tolist()}
            entry["n_intervals"] = int(len(iv))
        weapons[name] = entry

    out = {
        "_meta": {
            "contract": ("op-attack = (move_bit0 & input_mask_bit0) on engaged-LOS "
                         "frames (target argmax>0 & >=1 LOS actor), per attack-with weapon "
                         "(raw->impulse). Rates are decision-granularity world "
                         "results; commensurate with eval "
                         "engine_los_attack_by_lead_angle tables. Intervals within "
                         "unbroken engaged same-weapon runs."),
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
    pooled_eng = sum(int(per_counts[d][:, 0].sum()) for d in per_counts)
    pooled_op = sum(int(per_counts[d][:, 1].sum()) for d in per_counts)
    print(f"\npooled op-attack rate (all weapons): {pooled_op/pooled_eng*HZ:.3f}/s "
          f"over {pooled_eng} engaged-LOS ticks, {len(per_counts)} demos")
    print(f"Written -> {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-dir", type=Path, default=Path("artifacts/collect/qwd"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON (default: <collect>/human_baseline/_op_attack_rate_byweapon.json)")
    args = ap.parse_args()
    from qnn.human import baseline_dir
    out = args.out or baseline_dir(args.collect_dir) / "_op_attack_rate_byweapon.json"
    run(args.collect_dir, [args.split], out, args.workers)


if __name__ == "__main__":
    main()
