"""Fit a model's decode operating point after training — ONE module, ONE corpus
load on the model's training subset (segment_mask act.target!=0), both heads.

What this LOCKS (principled, auto-written with --write):
  * move sticky   — temporal forward; sticky-tau swept to the human fb/lr switch
                    rate under the real move decode.
  * weapon sticky — (conf, margin) swept to minimize the EMD between the committed
                    stream's dwell-time distribution and the human act.weapon dwell
                    distribution over the same combat stream.
  * attack thresh — attack threshold swept to match the human OPERATIVE attack rate
                    (the marginal-matching / calibration point). Single decode
                    param → single scalar target, like move-tau. This is the OFFLINE
                    calibration check (answers "is the baked 0.5 already calibrated"):
                    the void-firing operating point is closed-loop/OOD and is left to
                    the live sweep — offline, on the human trajectory, the look is the
                    human's, so the in-distribution fit cannot see void-firing.

Weapon gate objective (weapon-head.md §4/§5): match the switch *rhythm*, NOT
per-frame macro-F1 (structurally over-holds — rewards "hold when unsure", so it
always prefers an under-switching gate, label-independent) nor the scalar switch
rate (a single mean is degenerate: an over-holding gate matches the mean by
accident, so it can't tell a good gate from a frozen one; the dwell histogram is
not foolable that way). This objective was only "unresolved" while the SG-skewed
weapon label confound was live: rc1 papered over the over-occupancy with a
decode-side weapon_ban=[2] AND a high-confidence (macro-F1-ish) gate, so macro-F1
looked vindicated live. Once the labels were fixed (ban dropped in rc2/rc3) the
marginal is correct in training and the gate's only remaining job is switch
dynamics — for which dwell-EMD is the unambiguous fit. See project memory
`project_weapon_gate_principle_open`.

Forwarded WITHOUT the bench side-channel (deployment-faithful: the live policy.act
path teacher-forces nothing). CPU, no engine.

CLI:
  python -m qnn.bc.decode_fit --run-dir <run> --decode-config <cfg> [--write]
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
from qnn.model.bench.a24 import decode as _dec
from qnn.model.network import ATTACK_HEAD, WEAPON_HEAD
from qnn.schema import WEAPON_HEAD_SIZE
from qnn.vocab import self_weapon_id_to_impulse

SEGMENT_MASK = {"act.target": {"$ne": 0}}   # the training subset both heads were fit on
WEAPON_NAMES = ["axe", "shotgun", "super_shotgun", "nailgun",
                "super_nailgun", "grenade", "rocket", "lightning"]
MOVE_TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
WEAPON_CONFS = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
WEAPON_MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2]
ATTACK_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.20, 0.81, 0.05)]


# ── policy load ──────────────────────────────────────────────────────────────
def _load_policy(run: Path):
    """Load a policy from a bench run directory.

    Delegates to :func:`qnn.diag.loader.load_policy` — the canonical
    single implementation shared with all analysis scripts.
    """
    return _diag_load_policy(run)


# ── WEAPON gate fit (dwell-time EMD vs human) ────────────────────────────────
@torch.inference_mode()
def _forward_weapon_logits(policy, source) -> torch.Tensor:
    """Per-frame weapon logits (N, 8). Forward each episode as a (1, T) sequence
    through the plain network — NO bench side-channel, matching the live act()
    path which teacher-forces nothing. Per-episode (episode-batched forwards
    change the logits)."""
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    out = torch.empty((int(offsets[-1]), WEAPON_HEAD_SIZE), dtype=torch.float32)
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        T = e - s
        idx = torch.arange(s, e, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:]))
                   for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = policy.model(obs_seq, hidden=None, reset_mask=None)
        out[s:e] = logits[WEAPON_HEAD].reshape(T, WEAPON_HEAD_SIZE).float().cpu()
    return out


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


def _weapon_gate_sweep(probs, intent, offsets, fr, self_imp, human_dwell, human_switch,
                       confs, margins) -> tuple[list[dict], dict]:
    """Sweep (conf, margin) → committed-stream dwell-EMD vs the human standing-intent
    dwell. Returns all rows + the min-EMD cell."""
    rows = []
    best = {"conf": None, "margin": None, "dwell_emd": float("inf"), "switch_rate": 0.0}
    for c in confs:
        for mg in margins:
            com, _above = _committed_stream(probs, intent, offsets, float(c), float(mg))
            emd = _dwell_emd(_dwell_lengths(com, offsets), human_dwell)
            sw = float(np.mean(com[fr] != self_imp[fr])) if fr.any() else 0.0
            rows.append({"conf": round(float(c), 3), "margin": round(float(mg), 3),
                         "dwell_emd": emd, "switch_rate": sw, "delta_human": sw - human_switch})
            if emd < best["dwell_emd"]:
                best = {"conf": round(float(c), 3), "margin": round(float(mg), 3),
                        "dwell_emd": emd, "switch_rate": sw}
    return rows, best


def fit_weapon(policy, source, confs=WEAPON_CONFS, margins=WEAPON_MARGINS) -> dict[str, Any]:
    """COARSE→FINE sweep of (conf, margin); pick the operating point whose committed-
    stream DWELL-TIME distribution is closest (1-D EMD) to the human STANDING-INTENT
    dwell — the settled objective (module docstring; weapon-head.md §4/§5). Stage 1 is
    the 0.05 coarse grid; stage 2 refines at 0.01 around the coarse optimum (same
    coarse→fine as _weapon_switch_threshold_eval), so the locked gate isn't a
    grid-quantization artifact.

    Human target and gated stream are computed on the SAME val/segment/offsets AND
    the SAME always-held basis: the human side is act.weapon carried forward through
    no-select frames (``_standing_intent``), matching the committed stream which
    always holds a weapon between commits. (Don't compare a 0-inflated select stream
    to an always-held one — that's the stale-metric trap.) The leak-free attack-frame
    switch rate is carried alongside as a diagnostic, NOT the selector."""
    logits = _forward_weapon_logits(policy, source)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    wid_raw = torch.as_tensor(_to_np(source.obs["self_weapon_id"]).reshape(-1)).long()
    self_imp = _to_np(self_weapon_id_to_impulse(wid_raw)).reshape(-1)      # held impulse 1..8
    wlabel = _to_np(source.actions["weapon"]).reshape(-1)                  # raw act.weapon select
    zero_frac = float((wlabel == 0).mean())                               # how 0-inflated act.weapon is
    intent = _standing_intent(wlabel, offsets)                            # human standing held intent 1..8
    attack = _to_np(source.actions["attack"]).reshape(-1).astype(bool)
    op = _op_mask(source)                                                  # OPERATIVE filter (see _op_mask)
    fr = attack & op & (self_imp > 0)                                      # op-filtered leak-free diag frames
    human_switch = float(np.mean(intent[fr] != self_imp[fr])) if fr.any() else 0.0
    human_dwell = _dwell_lengths(intent, offsets)                          # the fit target (same basis)
    # Stage 1: coarse 0.05 grid.
    rows_c, best_c = _weapon_gate_sweep(probs, intent, offsets, fr, self_imp,
                                        human_dwell, human_switch, confs, margins)
    # Stage 2: fine 0.01 grid around the coarse optimum (conf capped at 0.99).
    cc = _fine_grid(best_c["conf"], 0.0, 0.99)
    mm = _fine_grid(best_c["margin"], 0.0, 0.6)
    rows_f, best_f = _weapon_gate_sweep(probs, intent, offsets, fr, self_imp,
                                        human_dwell, human_switch, cc, mm)
    best = best_f if best_f["dwell_emd"] <= best_c["dwell_emd"] else best_c
    hd = (dict(median=float(np.median(human_dwell)), mean=float(human_dwell.mean()),
               n=int(len(human_dwell)))
          if len(human_dwell) else dict(median=float("nan"), mean=float("nan"), n=0))
    return {"human_switch_rate": human_switch, "attack_frames": int(fr.sum()),
            "human_dwell": hd, "act_weapon_zero_frac": zero_frac,
            "grid": rows_c, "fine_grid": rows_f, "coarse_best": best_c, "best": best,
            "fit": {"weapon.sticky_confidence": best["conf"],
                    "weapon.sticky_margin": best["margin"]}}


# ── ATTACK threshold fit (operative-rate / calibration match) ────────────────
@torch.inference_mode()
def _forward_attack_logits(policy, source, guard=None):
    """Per-frame attack logit (N,). Per-episode (1, T) forward through the plain
    network — NO bench side-channel, matching the live act() path (same as
    _forward_weapon_logits).

    ``guard`` (a resolved decode config's ``guard_module`` adapter): when given,
    the RAW attack logit is passed through ``guard_attack_logit_for_export`` — the
    SAME guarded logit the live/exported model decodes (LG align/range, rocket
    self-splash, splash-proximity all drive the guarded logit to −1e9 ⇒ p≈0). The
    guard consumes the dequanted obs (``entity_scalars_raw`` etc.) + the SAMPLED
    move logits (jump row), reshaped to per-frame B=T exactly as ExportWrapper.
    Returns ``(raw, guarded)`` when a guard is supplied, else raw ``(N,)`` alone
    (the burst-sim caller relies on the bare-array form)."""
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    raw = torch.empty(int(offsets[-1]), dtype=torch.float32)
    guarded = torch.empty(int(offsets[-1]), dtype=torch.float32) if guard is not None else None
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e <= s:
            continue
        T = e - s
        idx = torch.arange(s, e, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:]))
                   for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = policy.model(obs_seq, hidden=None, reset_mask=None)
        attack_logit = logits[ATTACK_HEAD].reshape(T)
        raw[s:e] = attack_logit.float().cpu()
        if guard is not None:
            # Match ExportWrapper.forward's per-frame contract (B=T rows): flatten
            # the batch dim so the guard's mask (B,) broadcasts over attack (B,1).
            obs_flat = {k: v.reshape((T,) + tuple(v.shape[2:])) for k, v in obs_seq.items()}
            move_flat = logits["move"].reshape(T, 3, 3)      # (B, MOVE_AXES, classes)
            g = guard.guard_attack_logit_for_export(obs_flat, move_flat, attack_logit.reshape(T, 1))
            guarded[s:e] = g.reshape(T).float().cpu()
    if guard is None:
        return raw.numpy()
    return raw.numpy(), guarded.numpy()


def _attack_sweep(p_attack: np.ndarray, human: np.ndarray, thresholds):
    """Sweep attack threshold → model op-attack rate vs the human rate. Selector is
    the rate match (|Δrate|); precision/recall/F1 are diagnostics only."""
    r_h = float(human.mean())
    rows, best = [], {"threshold": 0.5, "delta_abs": float("inf")}
    for th in thresholds:
        atk = p_attack > th
        r_m = float(atk.mean())
        tp = int((atk & human).sum()); fp = int((atk & ~human).sum()); fn = int((~atk & human).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"threshold": round(float(th), 3), "model_rate": round(r_m, 4),
                     "rate_delta": round(r_m - r_h, 4), "precision": round(prec, 3),
                     "recall": round(rec, 3), "f1": round(f1, 3)})
        if abs(r_m - r_h) < best["delta_abs"]:
            best = {"threshold": round(float(th), 3), "delta_abs": abs(r_m - r_h),
                    "model_rate": round(r_m, 4), "f1": round(f1, 3)}
    return rows, best, r_h


def fit_attack(policy, source, thresholds=ATTACK_THRESHOLDS, guard=None) -> dict[str, Any]:
    """COARSE→FINE sweep of the attack threshold to the human OPERATIVE attack rate
    (the marginal-matching / calibration point). A single decode param → a single
    scalar target (human op-attack rate), mirroring fit_move's switch-rate match;
    precision/recall/F1 are carried as diagnostics, NOT the selector (attack is
    cooldown-dominated — geometry can't cross f1@0.5, attack-head.md). The
    void-attack operating point is closed-loop/OOD (the bot's own bad aim makes the
    off-target attack frames an in-distribution offline eval never contains) and is
    left to the live sweep; offline this answers whether the baked 0.5 is already
    calibrated.

    >>> OPERATIVE-FRAME FILTER (see _op_mask): the sweep runs on op=1 frames ONLY —
    >>> the frames the engine honours and the ONLY frames training scored the attack
    >>> head on. Both the human rate and the model rate are computed on that subset.
    >>> Comparing against the raw, all-frames attack rate (the original bug here)
    >>> inflates the human target with held-trigger cooldown frames and contaminates
    >>> the model rate with uncalibrated op=0 predictions — i.e. garbage calibration."""
    op = _op_mask(source)
    fwd = _forward_attack_logits(policy, source, guard=guard)
    raw_logit, guarded_logit = fwd if guard is not None else (fwd, None)
    attack = _to_np(source.actions["attack"]).reshape(-1).astype(bool)

    def _calibrate(logit):
        """Sweep threshold on sigmoid(logit) over op frames to the human op-attack
        rate; returns (best, r05, grid_coarse, grid_fine, r_h)."""
        p = _sigmoid(logit)[op]
        rows_c, best_c, r_h = _attack_sweep(p, human, thresholds)
        rows_f, best_f, _ = _attack_sweep(p, human, _fine_grid(best_c["threshold"], 0.05, 0.95))
        best = best_f if best_f["delta_abs"] <= best_c["delta_abs"] else best_c
        return best, float((p > 0.5).mean()), rows_c, rows_f, r_h

    human = attack[op]                                     # OPERATIVE frames only
    best, r05, rows_c, rows_f, r_h = _calibrate(raw_logit)
    raw_human = round(float(attack.mean()), 4)             # diagnostic: the un-filtered (wrong) rate
    out = {"human_rate": round(r_h, 4), "model_rate_at_0.5": round(r05, 4),
           "op_frames": int(op.sum()), "total_frames": int(op.size),
           "human_rate_unfiltered": raw_human,
           "calibrated_at_0.5": bool(abs(r05 - r_h) <= 0.01),
           "grid": rows_c, "fine_grid": rows_f, "coarse_best": None, "best": best,
           # The sweep runs on RAW sigmoid(logit) — no bias applied — so the
           # fitted threshold is the WHOLE operating point. Emit bias=0.0 with it:
           # decode composes attack iff sigmoid(logit + bias) >= tau, so a leftover
           # template bias (e.g. the rc1-era -1.1) would double-count and shift
           # the cut ~1.1 logits off the calibration (rc1v's tau=0.25 was chosen
           # to CANCEL that bias back to raw P>=0.5).
           "fit": {"attack.threshold": best["threshold"], "attack.bias": 0.0}}
    if guard is None:
        return out
    # GUARD-AWARE calibration: the live/exported model decodes attack from the
    # GUARDED logit (LG align/range, rocket self-splash, splash-proximity → −1e9 ⇒
    # p≈0), so the raw-logit τ above is mis-anchored — it compensates for
    # off-target attacks the
    # guard already kills. Re-calibrate on the guarded logit; that τ is the one to
    # bake. `guard_suppressed_op` = op frames the guard drove sub-0.5 that the raw
    # logit had ≥0.5 (the offline-vs-live spurious gap, quantified).
    g_best, g_r05, g_rows_c, g_rows_f, _ = _calibrate(guarded_logit)
    p_raw_op = _sigmoid(raw_logit)[op]
    p_grd_op = _sigmoid(guarded_logit)[op]
    suppressed = int(((p_raw_op >= 0.5) & (p_grd_op < 0.5)).sum())
    out["guarded"] = {
        "model_rate_at_0.5": round(g_r05, 4),
        "calibrated_at_0.5": bool(abs(g_r05 - r_h) <= 0.01),
        "grid": g_rows_c, "fine_grid": g_rows_f, "best": g_best,
        "guard_suppressed_op": suppressed,
        "guard_suppressed_op_frac": round(suppressed / max(int(op.sum()), 1), 4),
        "fit": {"attack.threshold": g_best["threshold"], "attack.bias": 0.0}}
    # The guard-aware τ is the deploy-faithful operating point → promote it to `fit`.
    out["fit"] = dict(out["guarded"]["fit"])
    return out


# ── MOVE sticky-tau fit (switch-rate match) ──────────────────────────────────
def _switch_rate(eps):
    sw = pr = 0
    for ep in eps:
        if len(ep) < 2:
            continue
        sw += int((ep[1:] != ep[:-1]).sum()); pr += len(ep) - 1
    return sw / max(pr, 1)


@torch.inference_mode()
def fit_move(policy, source, decode_cfg, hazard_table, taus=MOVE_TAUS) -> dict[str, Any]:
    net = policy.model
    hz = hazard_table   # the CORPUS hazard (config/move_hazard.json or computed) — the
    if not isinstance(hz, dict) or "lognorm" not in hz:   # equation tau is fit against
        raise ValueError("hazard_table must be a dict with a 'lognorm' {fb,lr} (mu,sigma) block")
    ln = hz["lognorm"]
    hazard_lognorm = torch.tensor([ln["fb"], ln["lr"]], dtype=torch.float32)  # (2,3,2)
    swb_eps = float(decode_cfg.params.get("move.switchback_eps", 0.1))
    stop_onset = bool(decode_cfg.params.get("move.stop_onset", True))
    params = {t: _dec.move_decode_params(tau_fb=t, tau_lr=t, swb_eps=swb_eps,
                                         stop_onset=stop_onset, hazard_lognorm=hazard_lognorm)
              for t in taus}
    offs = np.asarray(source.episode_offsets, np.int64)
    move_act = source.actions["move"]; self_wid = source.obs.get("self_weapon_id")
    # subsample episodes (MAX_EPISODES=250) — the switch-rate is stable on a
    # sample and the temporal forward can't be batched.
    n_ep = len(offs) - 1
    ep_ids = (range(n_ep) if n_ep <= 250 else np.linspace(0, n_ep - 1, 250).astype(int))
    human = [[], []]; decided = {t: [[], []] for t in taus}
    for ei in ep_ids:
        ei = int(ei)
        lo, hi = int(offs[ei]), int(offs[ei + 1]); T = hi - lo
        if T < 2:
            continue
        cls = np.asarray(move_act[lo:hi].cpu() if torch.is_tensor(move_act) else move_act[lo:hi]).astype(np.int8)
        human[0].append(cls[:, 0]); human[1].append(cls[:, 1])
        idx = torch.arange(lo, hi, dtype=torch.int64, device=policy.device)
        obs_seq = {k: v.index_select(0, idx).reshape((1, T) + tuple(v.shape[1:])) for k, v in source.obs.items()}
        _f, logits, _v, _nh, _tl = net(obs_seq, hidden=None, reset_mask=None)
        mlog = logits["move"].reshape(T, 9).cpu()
        wid = (np.asarray(self_wid[lo:hi].cpu() if torch.is_tensor(self_wid) else self_wid[lo:hi]).reshape(-1).astype(int)
               if self_wid is not None else np.zeros(T, int))
        for t in taus:
            st = _dec.move_decode_reset(params[t], rng_state=(0x9E3779B9 ^ (ei * 2654435761)) & 0xFFFFFFFF)
            fb = np.empty(T, np.int8); lr = np.empty(T, np.int8)
            for ti in range(T):
                _b, c, _l, st = _dec.move_decode_step(
                    mlog[ti], True, None, st, params[t])
                fb[ti] = c[0]; lr[ti] = c[1]
            decided[t][0].append(fb); decided[t][1].append(lr)
    hfb, hlr = _switch_rate(human[0]), _switch_rate(human[1])
    rows, best = [], {"fb": (None, 9.0), "lr": (None, 9.0)}
    for t in taus:
        fr, lrr = _switch_rate(decided[t][0]), _switch_rate(decided[t][1])
        rows.append((t, fr, fr - hfb, lrr, lrr - hlr))
        if abs(fr - hfb) < best["fb"][1]: best["fb"] = (t, abs(fr - hfb))
        if abs(lrr - hlr) < best["lr"][1]: best["lr"] = (t, abs(lrr - hlr))
    return {"human": {"fb": hfb, "lr": hlr}, "grid": rows,
            "fit": {"move.sticky_tau_fb": best["fb"][0], "move.sticky_tau_lr": best["lr"][0]}}


# ── orchestration: ONE policy, ONE source (training subset), both fits ────────
def fit(run_dir: Path, decode_config: Path) -> dict[str, Any]:
    resolved = _dc.resolve_decode_config(decode_config)
    policy, probe = _load_policy(run_dir)
    corpus = Path(json.loads((run_dir / "config" / "machine.json").read_text())["bc_data_dir"])
    source = make_resident_source_from_cache(corpus / "precomputed_val", policy.device,
                                             segment_mask=SEGMENT_MASK)
    n_ep = len(source.episode_offsets) - 1
    print(f"[decode_fit] {n_ep} engaged episodes, {int(source.episode_offsets[-1])} frames "
          f"(probe={probe.get('head') or probe.get('base', '?')})", flush=True)
    hazard = _resolve_corpus_hazard(run_dir, corpus)
    print(f"[decode_fit] hazard: {hazard['_source']} tick_hz={hazard.get('tick_hz')} "
          f"(log-normal equation, fb/lr mu,sigma)", flush=True)
    weapon = fit_weapon(policy, source)
    weapon["baked_gate"] = [policy.weapon_switch_confidence, policy.weapon_switch_margin]
    attack = fit_attack(policy, source, guard=resolved.guard_module)
    attack["baked_threshold"] = (
        None if policy.attack_threshold is None else float(policy.attack_threshold))
    move = fit_move(policy, source, resolved, hazard)
    # All three auto-lock: move-tau to the human fb/lr switch rate, weapon (conf,
    # margin) to the human dwell-time distribution (dwell-EMD), attack threshold to
    # the human operative attack rate (calibration; see module docstring).
    return {"move": move, "weapon": weapon, "attack": attack, "hazard": hazard,
            "fit": {**move["fit"], **weapon["fit"], **attack["fit"]}}


def _resolve_corpus_hazard(run_dir: Path, corpus: Path) -> dict[str, Any]:
    """The move-hazard table to fit against AND bake: the run's pinned
    config/move_hazard.json if present (run.init writes it from the corpus), else
    computed fresh from the corpus. Either way it's the corpus's own table — never
    a borrowed one (the rc3-inherited-rc1-hazard trap)."""
    pinned = run_dir / "config" / "move_hazard.json"
    if pinned.exists():
        h = json.loads(pinned.read_text())
        if "lognorm" not in h:
            raise ValueError(f"{pinned} has no 'lognorm' block — re-pin with the "
                             "equation hazard (move_hazard.lognorm_hazard_from_collect)")
        return {"lognorm": h["lognorm"], "tick_hz": h.get("tick_hz"),
                "_source": f"pinned {pinned}"}
    from qnn.model import move_hazard as _mh
    block = _mh.lognorm_hazard_from_collect(corpus, noncombat=True)
    return {"lognorm": block["lognorm"], "tick_hz": block.get("tick_hz"),
            "_source": f"computed from {corpus}"}


def _write_back(decode_config: Path, fit_params: dict[str, Any], hazard: dict[str, Any]) -> None:
    cfg = json.loads(decode_config.read_text())
    cfg["params"].update(fit_params)
    cfg["move_hazard"] = {"method": "lognorm", "lognorm": hazard["lognorm"],
                          "tick_hz": hazard.get("tick_hz")}
    decode_config.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"[decode_fit] wrote fitted params {fit_params} + corpus move_hazard "
          f"(tick_hz {hazard.get('tick_hz')}) -> {decode_config}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--decode-config", required=True, type=Path)
    ap.add_argument("--write", action="store_true", help="lock fitted params into the config")
    args = ap.parse_args()
    out = fit(args.run_dir, args.decode_config)

    m = out["move"]
    print(f"\nMOVE  human switch fb={m['human']['fb']:.4f} lr={m['human']['lr']:.4f}")
    print(f"{'tau':>5} | {'fb_sw':>7} {'fb_Δ':>9} | {'lr_sw':>7} {'lr_Δ':>9}")
    for t, fr, dfb, lrr, dlr in m["grid"]:
        print(f"{t:>5.2f} | {fr:>7.4f} {dfb:>+9.4f} | {lrr:>7.4f} {dlr:>+9.4f}")
    w = out["weapon"]
    print(f"\nWEAPON (dwell-EMD fit — match the human switch rhythm; weapon-head.md §4/§5)")
    print(f"  baked model gate = (conf {w['baked_gate'][0]}, margin {w['baked_gate'][1]})")
    print(f"  act.weapon 0-frac = {w['act_weapon_zero_frac']*100:.2f}%  (carried fwd to standing intent; "
          f">~0 means raw select stream would have been a stale, 0-inflated target)")
    print(f"  human switch rate={w['human_switch_rate']:.4f} ({w['attack_frames']} attack frames)  "
          f"human dwell median={w['human_dwell']['median']:.1f} mean={w['human_dwell']['mean']:.1f}")
    print(f"  full conf curve (best margin per conf):")
    by_conf = {}
    for r in w["grid"]:
        if r["conf"] not in by_conf or r["dwell_emd"] < by_conf[r["conf"]]["dwell_emd"]:
            by_conf[r["conf"]] = r
    for c in sorted(by_conf):
        r = by_conf[c]
        print(f"    conf {c:>4.2f} (m={r['margin']:.2f}): dwellEMD {r['dwell_emd']:>8.3f}  "
              f"sw_rate {r['switch_rate']:.4f}  Δhuman {r['delta_human']:>+8.4f}")
    bc = w["coarse_best"]; bf = w["best"]
    print(f"  coarse best : conf {bc['conf']:.2f} margin {bc['margin']:.2f}  dwellEMD {bc['dwell_emd']:.3f}")
    print(f"  REFINED best: conf {bf['conf']:.2f} margin {bf['margin']:.2f}  dwellEMD {bf['dwell_emd']:.3f}  (0.01 grid)")
    if w.get("fine_grid"):
        cs = sorted(w["fine_grid"], key=lambda r: r["dwell_emd"])[:6]
        print(f"  fine neighborhood @0.01 (top 6 by dwellEMD):")
        for r in cs:
            print(f"    conf {r['conf']:>4.2f} margin {r['margin']:.2f}: dwellEMD {r['dwell_emd']:>8.3f}")
    a = out["attack"]
    cal = "CALIBRATED" if a["calibrated_at_0.5"] else "miscalibrated — 0.5 is off"
    print(f"\nATTACK (threshold fit — match human OPERATIVE attack rate; offline calibration)")
    print(f"  baked threshold = {a['baked_threshold']}")
    print(f"  OP-FILTERED on {a['op_frames']}/{a['total_frames']} op=1 frames "
          f"({a['op_frames']/max(a['total_frames'],1)*100:.1f}%); "
          f"raw all-frame human rate would have been {a['human_rate_unfiltered']:.4f} (WRONG — held-trigger inflated)")
    print(f"  human op-attack rate = {a['human_rate']:.4f}  |  model rate @0.5 = {a['model_rate_at_0.5']:.4f}  → {cal}")
    print(f"  {'thr':>5} | {'rate':>7} {'Δhuman':>9} | {'prec':>6} {'rec':>6} {'f1':>6}")
    for r in a["grid"]:
        print(f"  {r['threshold']:>5.2f} | {r['model_rate']:>7.4f} {r['rate_delta']:>+9.4f} | "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
    print(f"  REFINED best: threshold {a['best']['threshold']:.2f}  (rate match, 0.01 grid; "
          f"P/R/F1 diagnostic only — void-firing is the closed-loop sweep)")

    print(f"\n→ LOCKED (move-tau + weapon gate + attack threshold + corpus move_hazard): "
          f"{json.dumps(out['fit'])} + hazard tick_hz={out['hazard'].get('tick_hz')}")
    if args.write:
        _write_back(args.decode_config, out["fit"], out["hazard"])


if __name__ == "__main__":
    main()
