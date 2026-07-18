"""Fit a model's decode operating point after training — ONE module, ONE corpus
load on the model's training subset (segment_mask act.target!=0).

a25-ONLY: the run MUST carry the a25 shape (the 9-way ``attack_with`` weapon head
AND the seg-commitment ``move_seg`` head) — :func:`fit` raises otherwise. The
retired a24 fits (move sticky-tau, 8-way weapon sticky gate, scalar attack
threshold) were removed with the a24 arch; their decode laws no longer exist.

What this FITS (printed + returned; the pipeline emits — see
qnn.eval.decode_fit_pipeline, the only path that writes decode configs):
  * attack.bias_vec  — (8,) per-weapon attack operating point, fit to the human
                       per-class attack-with-k marginal over operative frames
                       (research/attack-head.md §11).
  * attack.stick_bias — selection hysteresis toward the held weapon, fit to the
                       human held-weapon switch rate (the joint head is stateless;
                       without it committed selection over-switches ~2.4x live).

The move channel is the a25 COMMITMENT decode — sticky-tau/hazard are retired
wholesale; move.commit_dur_tilt is fit by the pipeline's dedicated stage.

Forwarded WITHOUT the bench side-channel (deployment-faithful: the live policy.act
path teacher-forces nothing). CPU, no engine.

CLI:
  python -m qnn.bc.decode_fit --run-dir <run> --decode-config <cfg>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qnn.bc.supervised_loop import make_resident_source_from_cache
from qnn.diag.loader import load_policy as _diag_load_policy
from qnn.model import decode_config as _dc
from qnn.model.network import WEAPON_HEAD
from qnn.vocab import self_weapon_id_to_impulse

SEGMENT_MASK = {"act.target": {"$ne": 0}}   # the training subset the heads were fit on
WEAPON_NAMES = ["axe", "shotgun", "super_shotgun", "nailgun",
                "super_nailgun", "grenade", "rocket", "lightning"]


# ── policy load ──────────────────────────────────────────────────────────────
def _load_policy(run: Path):
    """Load a policy from a bench run directory.

    Delegates to :func:`qnn.diag.loader.load_policy` — the canonical
    single implementation shared with all analysis scripts.
    """
    return _diag_load_policy(run)


# ── shared helpers (op-filter / dwell math / committed stream) ───────────────
def _sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid. Clips before exp so the guard's −1e9
    suppression sentinel maps to 0.0 without an overflow warning (sigmoid is
    saturated well within ±60 in float32)."""
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))


def _to_np(x) -> np.ndarray:
    """Tensor (possibly on GPU) or array-like → host numpy. The resident source
    loads obs/actions onto policy.device, so cuda tensors must be copied first."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _op_mask(source) -> np.ndarray:
    """Per-frame OPERATIVE flag (input_mask bit 0): the frames on which the engine
    HONORS the attack input. Training rewrites the attack label to
    ``feasibility * demo_press`` and scores BCE/precision/recall/F1 only where op=1
    (``qnn.model.policy._compute_head_losses_and_metrics``); op=0 frames are no-op
    holds — a trigger held through cooldown or an auto-attack weapon's continuous hold,
    which the engine ignores — and the model's predictions there are uncalibrated.

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  EVERY comparison of MODEL attack behaviour to the HUMAN/DEMO attack stream    ║
    ║  MUST be restricted to op=1 frames FIRST. Raw ``mean(attack)`` over all frames ║
    ║  over-counts the human attack rate by every held-trigger cooldown frame, so a  ║
    ║  calibration fit against it is garbage. The segment_mask (act.target!=0) is    ║
    ║  the ENGAGED mask, NOT this one — engaged-but-in-cooldown frames exist.        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    im = source.actions.get("input_mask")
    if im is None:
        raise ValueError("operative filter requires actions['input_mask'] — recollect "
                         "the corpus on a post-input_mask branch")
    return (_to_np(im).reshape(-1).astype(np.uint8) & 0x1).astype(bool)


def _dwell_lengths(stream: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Run-lengths of the constant-weapon stretches within each episode (frames per
    stretch). Mirrors scripts/analysis/_weapon_temporal_fidelity.dwell_lengths so the
    dwell-EMD here is comparable to the §5 grader's published numbers."""
    offs = np.asarray(offsets, dtype=np.int64)
    out: list[int] = []
    for i in range(len(offs) - 1):
        s, e = int(offs[i]), int(offs[i + 1])
        if e <= s:
            continue
        w = stream[s:e]
        chg = np.flatnonzero(np.diff(w) != 0) + 1
        bounds = np.concatenate([[0], chg, [len(w)]])
        out.extend(np.diff(bounds).tolist())
    return np.asarray(out, dtype=np.float64)


def _dwell_emd(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein-1 (EMD) between two empirical dwell-length samples (sort-based)."""
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    a = np.sort(a); b = np.sort(b)
    grid = np.union1d(a, b)
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    widths = np.diff(np.concatenate([grid, grid[-1:]]))
    return float(np.sum(np.abs(ca - cb)[:-1] * widths[:-1]))


def _committed_stream(probs: np.ndarray, seed: np.ndarray, offsets: np.ndarray,
                      C: float, M: float) -> tuple[np.ndarray, np.ndarray]:
    """Carry-forward committed-weapon stream + per-frame 'committed' mask, under the
    sticky gate (commit argmax iff conf>=C and margin>=M, else hold the last commit).
    Each episode start is force-seeded from ``seed`` (impulses 1..8) so the carry never
    crosses episodes. This is the closed-loop proxy for the deployed gate: the engine's
    held weapon converges on the bot's own last commit, so the committed stream — not
    the human's demo-equipped weapon — is what the dwell distribution should be fit on.
    Same construction as _weapon_switch_threshold_eval._committed_stream."""
    pred_imp = probs.argmax(1) + 1
    top2 = np.sort(probs, axis=1)[:, -2:]
    conf = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    n = len(seed)
    arange = np.arange(n)
    offs = np.asarray(offsets, dtype=np.int64)
    ep_start = np.zeros(n, dtype=bool)
    ep_start[offs[:-1][offs[:-1] < n]] = True
    above = (conf >= C) & (margin >= M)
    take = above | ep_start                                  # episode start seeds
    take_value = np.where(above, pred_imp, seed)             # start & !above -> seed init
    com = take_value[np.maximum.accumulate(np.where(take, arange, -1))]
    return com, above


def _standing_intent(stream: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per-episode forward-fill of a weapon stream through 0 (no-select) frames →
    the standing HELD intent (1..8) at every frame. Leading 0s (before the first
    select in an episode) stay 0. This puts the human's *actual* intent on the SAME
    always-held basis as the gated committed stream (which holds a weapon between
    commits) — so dwell-EMD compares like with like, not a 0-inflated select stream."""
    s = np.asarray(stream).copy()
    offs = np.asarray(offsets, dtype=np.int64)
    for i in range(len(offs) - 1):
        a, b = int(offs[i]), int(offs[i + 1])
        seg = s[a:b]
        nz = seg != 0
        if not nz.any():
            continue
        idx = np.maximum.accumulate(np.where(nz, np.arange(b - a), 0))
        filled = seg[idx]
        filled[: int(np.argmax(nz))] = 0          # keep leading pre-first-select 0s
        s[a:b] = filled
    return s


def _fine_grid(center: float, lo: float, hi: float, half: float = 0.05, step: float = 0.01):
    """0.01-step grid spanning center +/- half, clamped to [lo, hi] (mirrors the
    coarse→fine refinement in _weapon_switch_threshold_eval)."""
    a = max(lo, round(center - half, 2))
    b = min(hi, round(center + half, 2))
    return list(np.round(np.arange(a, b + step / 2, step), 3))


# ── a25 9-way ATTACK-WITH per-weapon operating point (research/attack-head.md §11)
@torch.inference_mode()
def _forward_attack_with_logits(policy, source) -> np.ndarray:
    """Per-frame 9-way attack-with logits (N, 9). Same per-episode forward as
    _forward_weapon_logits, but reads the full (T, ATTACK_WITH_SIZE) head (the
    8-way reshape in _forward_weapon_logits would raise on this head)."""
    from qnn.model.bench.a25.attack_with_head import ATTACK_WITH_SIZE
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    out = np.empty((int(offsets[-1]), ATTACK_WITH_SIZE), np.float32)
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        T = e - s
        idx = torch.arange(s, e, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:]))
                   for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = policy.model(obs_seq, hidden=None, reset_mask=None)
        out[s:e] = logits[WEAPON_HEAD].reshape(T, ATTACK_WITH_SIZE).float().cpu().numpy()
    return out


def fit_attack_with(policy, source, l9: np.ndarray | None = None) -> dict[str, Any]:
    """Fit the a25 9-way attack-with per-weapon operating point ``attack.bias_vec``
    (8,) to the HUMAN per-class attack-with-k marginal over operative frames — the
    fired-weapon analog of fit_attack's scalar op-attack-rate match, one target per
    class (research/attack-head.md §11).

    Decode law: choice = argmax(l1..l8); attack iff (l_choice + bias_vec[choice]) > l0.
    Since bias_vec is applied POST-argmax it moves ONLY P(attack | choice==k), never
    selection — so for each class k, on the frames the model already selects k, solve
    the operating point to the per-class human rate:

        target_class[k]  = P(human fires k | op)                 # marginal, sums to agg
        cond_target[k]   = target_class[k] / P(argmax==k | op)   # rate WITHIN selected-k
        bias_vec[k]      = quantile(l0 - l_k over {op & argmax==k}, cond_target[k])

    cond_target > 1 means the class is SELECTION-limited (argmax picks it less often
    than the human fires it) — bias saturates (always attack when selected) and the
    class is flagged; on this model no non-axe class is selection-limited. Human and
    model rates are both computed on ``source`` (the engaged, segment-masked val
    subset), so the fit is in-distribution and self-consistent."""
    from qnn.model.bench.a25.attack_with_head import ATTACK_WITH_SIZE

    op = _op_mask(source)
    attack = _to_np(source.actions["attack"]).reshape(-1).astype(bool)
    weapon = _to_np(source.actions["weapon"]).reshape(-1).astype(np.int64)   # impulse 1..8
    if l9 is None:
        l9 = _forward_attack_with_logits(policy, source)
    l0 = l9[:, 0]
    weap = l9[:, 1:]
    choice = weap.argmax(1) + 1                                              # impulse 1..8
    nop = int(op.sum())

    NW = ATTACK_WITH_SIZE - 1
    bias_vec = [0.0] * NW
    rows = []
    eff = attack & op                                                       # op-attack presses
    for k in range(1, NW + 1):
        tgt = float((eff & (weapon == k)).sum()) / max(nop, 1)              # human P(fire k | op)
        sel = op & (choice == k)                                           # frames model selects k
        n_sel = int(sel.sum())
        sel_share = n_sel / max(nop, 1)
        cur = float((sel & (weap[:, k - 1] > l0)).sum()) / max(nop, 1)      # current attack-with-k rate
        if n_sel < 200 or sel_share <= 0.0:
            rows.append({"impulse": k, "target": round(tgt, 4), "sel_share": round(sel_share, 4),
                         "current": round(cur, 4), "bias": 0.0, "realized": round(cur, 4),
                         "selection_limited": bool(tgt > sel_share)})
            continue
        cond = tgt / sel_share                                              # rate WITHIN selected-k
        sel_lim = cond >= 1.0
        cond_c = min(cond, 1.0)
        d = l0[sel] - weap[sel, k - 1]                                      # attack iff bias > d
        b = float(np.quantile(d, cond_c)) if cond_c > 0.0 else float(-np.inf)
        bias_vec[k - 1] = 0.0 if not np.isfinite(b) else round(b, 4)
        realized = float((sel & ((weap[:, k - 1] + bias_vec[k - 1]) > l0)).sum()) / max(nop, 1)
        rows.append({"impulse": k, "target": round(tgt, 4), "sel_share": round(sel_share, 4),
                     "current": round(cur, 4), "cond_target": round(cond, 4),
                     "bias": bias_vec[k - 1], "realized": round(realized, 4),
                     "selection_limited": bool(sel_lim)})
    agg_h = float(eff.sum()) / max(nop, 1)
    agg_m_cur = float(((weap.max(1) > l0) & op).sum()) / max(nop, 1)
    return {"op_frames": nop, "human_agg": round(agg_h, 4), "model_agg_argmax": round(agg_m_cur, 4),
            "per_class": rows, "fit": {"attack.bias_vec": bias_vec}}


# ── a25 weapon-selection hysteresis fit (attack.stick_bias) ──────────────────
STICK_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)


def _stick_switch_sim(weap: np.ndarray, offsets: np.ndarray, held0: np.ndarray,
                      stick: float) -> float:
    """Committed-selection switch rate under held-weapon hysteresis.

    Selection law (a25 attack_with_decode_step): sel = argmax(weap + stick·
    onehot(held)); the emitted impulse re-points the server every tick, so
    held_{t+1} = sel_t. Per frame that reduces to: keep held iff
    weap[held] + stick >= max_k weap[k], else take the plain argmax. Returns
    switches / transitions — the same per-transition basis as the human
    self-weapon stream, so the parity target is rate-unit free."""
    top = weap.argmax(1)
    topv = weap.max(1)
    sw = tr = 0
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        held = int(held0[i])
        for t in range(s, e):
            sel = held if weap[t, held - 1] + stick >= topv[t] else int(top[t]) + 1
            if t > s:
                tr += 1
                if sel != held:
                    sw += 1
            held = sel
    return sw / max(tr, 1)


def fit_attack_stick_bias(source, l9: np.ndarray) -> dict[str, Any]:
    """Fit ``attack.stick_bias`` (selection hysteresis toward the held weapon) to
    the HUMAN held-weapon switch rate — the a25 attack_with analog of
    fit_weapon's dwell-EMD sticky gate. The joint 9-way head is stateless;
    without hysteresis its committed selection over-switches ~2.4x live
    (research/human-band.md, wave 6: stick 3.0 == human-parity switch rate,
    weapon-channel band distance 0.373→0.095). Objective: simulated committed-
    selection switch rate (selection law with held fed back, per episode) ==
    human self-weapon per-transition switch rate on the same engaged basis.
    Monotone decreasing in stick → coarse grid + nearest-parity pick."""
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    wid_raw = torch.as_tensor(_to_np(source.obs["self_weapon_id"]).reshape(-1)).long()
    self_imp = _to_np(self_weapon_id_to_impulse(wid_raw)).reshape(-1)
    sw = tr = 0
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s < 2:
            continue
        seg = self_imp[s:e]
        sw += int((seg[1:] != seg[:-1]).sum())
        tr += e - s - 1
    human_rate = sw / max(tr, 1)
    weap = l9[:, 1:]
    held0 = np.array([max(int(self_imp[int(offsets[i])]), 1)
                      for i in range(len(offsets) - 1)], dtype=np.int64)
    rows = []
    rates: list[tuple[float, float]] = []
    for stick in STICK_GRID:
        r = _stick_switch_sim(weap, offsets, held0, float(stick))
        rows.append({"stick": stick, "switch_rate": round(r, 5),
                     "delta": round(r - human_rate, 5)})
        rates.append((float(stick), r))
    # Rate is monotone decreasing in stick: bisect the bracketing grid interval
    # to exact offline parity instead of snapping to the nearest grid point
    # (grid-snap picked 4.0 where closed-loop parity sits ~3.0; the finer the
    # operating point here, the smaller the stage-6 closed-loop residual).
    best = min(rates, key=lambda sr: abs(sr[1] - human_rate))
    bracket = [sr for sr in rates if (sr[1] - human_rate) > 0]
    upper = [sr for sr in rates if (sr[1] - human_rate) <= 0]
    if bracket and upper:
        lo = max(bracket)[0]
        hi = min(upper)[0]
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            r = _stick_switch_sim(weap, offsets, held0, mid)
            if abs(r - human_rate) < abs(best[1] - human_rate):
                best = (mid, r)
            if r > human_rate:
                lo = mid
            else:
                hi = mid
    return {"human_switch_rate": round(human_rate, 5), "grid": rows,
            "best_rate": round(best[1], 5),
            "fit": {"attack.stick_bias": round(best[0], 3)}}


# ── orchestration: ONE policy, ONE source (training subset), both fits ────────
def _is_attack_with_run(run_dir: Path) -> bool:
    """True when the run's weapon head is the a25 9-way ``attack_with`` (probe
    override ``heads.weapon.type``). Mirrors decode_fit_pipeline._is_attack_with."""
    probe = run_dir / "config" / "probe.json"
    if not probe.exists():
        return False
    try:
        heads = (json.loads(probe.read_text()).get("overrides") or {}).get("heads") or {}
        return (heads.get("weapon") or {}).get("type") == "attack_with"
    except Exception:
        return False


def _is_move_seg_run(run_dir: Path) -> bool:
    """True when the run pins the a25 segment-commitment ``move_seg`` head.
    Mirrors decode_fit_pipeline._has_move_seg."""
    probe = run_dir / "config" / "probe.json"
    if not probe.exists():
        return False
    try:
        heads = (json.loads(probe.read_text()).get("overrides") or {}).get("heads") or {}
        return bool(heads.get("move_seg"))
    except Exception:
        return False


def fit(run_dir: Path, decode_config: Path) -> dict[str, Any]:
    """a25-ONLY orchestration: REQUIRES the a25 shape (attack_with + move_seg).

    Raises for any other head set — the a24 fits (move sticky-tau/hazard, 8-way
    weapon sticky gate, scalar attack threshold) are retired with the a24 arch;
    there is no fallback decode law to fit."""
    if not (_is_attack_with_run(run_dir) and _is_move_seg_run(run_dir)):
        raise RuntimeError(
            f"{run_dir}: decode_fit requires the a25 shape — a 9-way attack_with "
            "weapon head AND a seg-commitment move_seg head (config/probe.json "
            "overrides.heads). The a24 fits (move sticky-tau, weapon sticky gate, "
            "scalar attack threshold) are retired with the a24 arch."
        )
    resolved = _dc.resolve_decode_config(decode_config)
    del resolved  # validated (module import + guard contract); fits are config-free
    policy, probe = _load_policy(run_dir)
    corpus = Path(json.loads((run_dir / "config" / "machine.json").read_text())["bc_data_dir"])
    source = make_resident_source_from_cache(corpus / "precomputed_val", policy.device,
                                             segment_mask=SEGMENT_MASK)
    n_ep = len(source.episode_offsets) - 1
    print(f"[decode_fit] {n_ep} engaged episodes, {int(source.episode_offsets[-1])} frames "
          f"(probe={probe.get('head') or probe.get('base', '?')})", flush=True)
    # move channel = the a25 commitment decode; sticky/hazard retired wholesale
    # (move-head.md §8; move.commit_dur_tilt is fit by decode_fit_pipeline).
    hazard = {"_source": "retired (move_seg commitment decode)", "_retired": True}
    move = {"fit": {}, "skipped": "move_seg commitment decode — sticky/hazard retired"}
    print("[decode_fit] move: commitment decode (dur_tilt is fit by "
          "decode_fit_pipeline; no sticky/hazard fit)", flush=True)
    # a25 9-way attack_with head: the attack operating point is the per-weapon
    # attack.bias_vec (research/attack-head.md §11) and the selection hysteresis
    # is attack.stick_bias, fit to human switch-rate parity (the joint head is
    # stateless — without it committed selection over-switches ~2.4x live).
    l9 = _forward_attack_with_logits(policy, source)
    attack_with = fit_attack_with(policy, source, l9=l9)
    stick = fit_attack_stick_bias(source, l9)
    print(f"[decode_fit] attack_with: model_agg@argmax={attack_with['model_agg_argmax']} "
          f"human_agg={attack_with['human_agg']} bias_vec={attack_with['fit']['attack.bias_vec']}",
          flush=True)
    print(f"[decode_fit] stick_bias: human_switch={stick['human_switch_rate']} "
          f"fit={stick['fit']['attack.stick_bias']} (rate {stick['best_rate']})",
          flush=True)
    return {"move": move, "attack_with": attack_with, "stick": stick, "hazard": hazard,
            "fit": {**move["fit"], **attack_with["fit"], **stick["fit"]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--decode-config", required=True, type=Path)
    args = ap.parse_args()
    out = fit(args.run_dir, args.decode_config)

    m = out["move"]
    print(f"\nMOVE  SKIPPED: {m['skipped']}")
    aw = out["attack_with"]
    print(f"\nATTACK-WITH (a25 9-way per-weapon operating point; research/attack-head.md §11)")
    print(f"  engaged op-attack: model@argmax {aw['model_agg_argmax']:.4f} vs human {aw['human_agg']:.4f}")
    print(f"  {'wpn':>4} | {'target':>7} {'sel_share':>9} {'current':>7} {'bias':>7} {'realized':>8} {'sel_lim':>7}")
    for r in aw["per_class"]:
        print(f"  {r['impulse']:>4} | {r['target']:>7.4f} {r['sel_share']:>9.4f} {r['current']:>7.4f} "
              f"{r['bias']:>7.3f} {r['realized']:>8.4f} {str(r['selection_limited']):>7}")
    print(f"  → attack.bias_vec = {aw['fit']['attack.bias_vec']}")
    st = out["stick"]
    print(f"\nSTICK (attack.stick_bias — human switch-rate parity)")
    print(f"  human switch rate = {st['human_switch_rate']}  fit = "
          f"{st['fit']['attack.stick_bias']} (rate {st['best_rate']})")
    print(f"\n→ LOCKED ({' + '.join(sorted(out['fit']))}): {json.dumps(out['fit'])}")


if __name__ == "__main__":
    main()
