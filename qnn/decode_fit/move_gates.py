"""The two per-checkpoint move-decode gate calibrations: jump τ and idle β.

    python -m qnn.decode_fit.move_gates --run-dir runs/head_probe/<run> \
        --cache-dir artifacts/collect/<corpus> [--out <report.json>]

Both knobs were hand-derived for ``a26rc1b`` (commit 2df9222a) by scripts that
were never committed; only their outputs survived, as
``runs/head_probe/_jump_{calib,tau}_a26rc1a.json`` and
``runs/head_probe/_idle_bias_fit_awposw.json``. This module is the recovered
kernel, so the second model that needs these gates does not need a third
throwaway script. ``--validate-against`` replays a recorded artifact pair and
prints the deltas, which is how the recovery was checked against a26rc1a.

Both are PER-CHECKPOINT constants, for the same reason
``move.commit_dur_tilt`` is (see qnn.eval.move_commit_fit): they inverta
THIS model's posterior against a human rate. Never carry a fitted τ or β to
another checkpoint — the posterior differs even when the corpus doesn't.

``jump.threshold`` (τ)
    The movearch jump head emits an engine-OUTCOME posterior that the AS-IS
    decode samples Bernoulli. τ replaces that with a deterministic gate (jump
    iff ``p_jump > τ``), so only confident, context-motivated frames fire.
    τ is placed by CUT FACTOR, not by frequency-matching a human rate
    (feedback_jump_no_rate_calibration): the default asks for the τ whose
    deterministic fire rate is 1/4 of the sampled rate on jump-FEASIBLE frames,
    the a26rc1b rule. The human rate is reported beside it as the diagnostic it
    is — a 4x cut lands the model at a FRACTION of the human rate, and that
    placement is judged in closed loop, never tuned to parity here.

    Feasibility is the operative filter ``input_mask`` bit 7 (ground jump) OR
    bit 6 (swim up) — the same construction ``qnn.diag.move.jump_discrim``
    uses, because raw ``ud == 2`` counts air presses the engine ignored and
    inflates every rate in sight.

``move.idle_none_bias`` (β, the lr lane)
    An additive bias on the seg head's ``none`` class buckets, scaled by
    ``(1 - E)`` so it vanishes in combat. β is fit on NO-ENEMY frames
    (``decode.world_enemy_present`` false — the same external signal the decode
    reads, not the target label) by matching the model's stand-still fraction
    to the human one.

    The model side is the DECODE-REALISTIC class marginal: the held class is
    masked out of the joint before renormalising, because at an onset the
    maximal-run law forbids re-committing to the class already held
    (``qnn.eval.move_commit_fit.masked_bucket_marginal`` does the same for the
    duration marginal, and teacher-forces the held class from the human action
    at t-1). Skipping that mask understates stand-still by ~8x — the unmasked
    marginal buries `none` under whatever moving class is currently held — and
    would inflate β by an order of magnitude.

    Exact and cheap either way: per frame only the three per-class log-sum-exps
    and the held class are needed, and the whole β curve follows in closed form.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qnn.model.move_seg_head import JOINT, N_BUCKETS, N_CLASSES
from qnn.model.network import JUMP_HEAD, MOVE_SEG_HEAD

# Seg-head joint layout per axis: [class0 buckets | class1 (none) buckets |
# class2 buckets]. Class 1 is `none`, which is the block move_commit_step biases
# (decode.py: `logits[:, N_BUCKETS:2*N_BUCKETS] + none_bias`).
NONE_CLASS = 1
AXIS_LR = 1
# Human stand-still is the same `none` class on the recorded lr action.
LR_NONE_ACTION = 1
# β grid reported alongside the fit (matches the a26rc1a artifact's grid).
BETA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
# Rounding that keys the fit's evaluation memo (two decades below the default
# tol, so it cannot move a placed β but does collapse float-arithmetic twins).
BETA_DECIMALS = 6
# τ grid reported alongside the fit.
TAU_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_CUT = 4.0


@torch.inference_mode()
def collect_gate_rows(policy, source, *, device=None) -> dict[str, np.ndarray]:
    """One padded-lane forward pass over the source → the per-frame arrays.

    Batched over episodes via ``qnn.diag.move_metrics.plan_batches`` (the same
    padded-lane strategy jump_discrim uses); a per-episode batch=1 loop is
    ~30x slower for no benefit, since nothing here is teacher-forced.

    Two reductions of the lr axis's 30-way joint are kept per frame, not the raw
    logits: the per-class log-sum-exp (which gives the SAMPLED class marginal)
    and the per-class max (which gives the ARGMAX/greedy class). Both are needed
    because the β curve's shape differs qualitatively between them, and that
    shape is what identifies which statistic a recorded fit used.
    """
    from qnn.diag.move_metrics import plan_batches
    from qnn.model.decode_actions import (move_engagement_signals,
                                            move_threat_signal)

    if device is None:
        device = torch.device(getattr(policy, "device", "cpu"))
    offsets = np.asarray(source.episode_offsets, dtype=np.int64)
    lengths = offsets[1:] - offsets[:-1]
    n_rows = int(offsets[-1])
    p_jump = np.zeros(n_rows, dtype=np.float32)
    lse_cls = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    max_cls = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    # Raw lr joint, kept for the rollout ruler (fp16: the decode samples from
    # it, and 3.9M x 30 in fp32 is 470 MB for no added fidelity).
    lr_joint = np.zeros((n_rows, JOINT), dtype=np.float16)
    enemy = np.zeros(n_rows, dtype=bool)
    engaged_active = np.zeros(n_rows, dtype=bool)
    threat = np.zeros(n_rows, dtype=bool)
    ammo_pools = np.zeros((n_rows, 4), dtype=np.float32)
    held_impulse = np.zeros(n_rows, dtype=np.int64)

    batches = plan_batches(lengths)
    print(f"[move-gates] {n_rows:,} rows, {len(lengths)} episodes, "
          f"{len(batches)} padded-lane batches", flush=True)
    for bi, batch in enumerate(batches):
        B = len(batch)
        Tmax = int(max(lengths[ei] for ei in batch))
        # Dequant boundary: the resident cache may hold a packed spatial atlas
        # and _forward_tensors bypasses act()'s dequant (idempotent — the same
        # call qnn.eval.move_commit_fit relies on).
        lanes = [policy._obs_tensors_dequant(
                    {k: v[int(offsets[ei]):int(offsets[ei + 1])]
                     for k, v in source.obs.items()})
                 for ei in batch]
        obs_seq = {
            k: torch.zeros((Tmax, B, *lanes[0][k].shape[1:]),
                           dtype=lanes[0][k].dtype, device=device)
            for k in lanes[0]
        }
        for b, lane in enumerate(lanes):
            for k, v in lane.items():
                obs_seq[k][:v.shape[0], b] = v
        _, logits, _, _, _ = policy._forward_tensors(obs_seq, hidden=None, masks=None)
        if JUMP_HEAD not in logits or MOVE_SEG_HEAD not in logits:
            raise ValueError(
                f"checkpoint is not movearch (heads: {sorted(logits)}) — both "
                "gates are movearch decode knobs: jump.threshold reads the jump "
                "head's outcome posterior and idle_none_bias biases the seg "
                "head's none class")
        pj = torch.sigmoid(logits[JUMP_HEAD].reshape(Tmax, B).float()).cpu().numpy()
        # Unflatten by JOINT, never by an assumed axis count. A water-ud seg
        # head emits THREE axes (fb, lr, ud) — 90 flat — and reshaping that to
        # (N, 2, -1) yields (N, 2, 45), silently splicing half the lr axis onto
        # fb and shifting every class block. QNNPolicy.act warns about exactly
        # this at its own seg slice; JOINT is the only safe divisor.
        seg_flat = logits[MOVE_SEG_HEAD].reshape(Tmax * B, -1)
        width = int(seg_flat.shape[1])
        if width % JOINT:
            raise ValueError(
                f"move_seg width {width} is not a multiple of JOINT={JOINT} "
                f"({N_CLASSES} classes x {N_BUCKETS} buckets) — the head's "
                "bucket law does not match this code's seg_bins")
        seg = (seg_flat.reshape(Tmax, B, width // JOINT, JOINT)[:, :, AXIS_LR, :]
               .float().reshape(Tmax, B, N_CLASSES, N_BUCKETS))
        lse = torch.logsumexp(seg, dim=3).cpu().numpy()
        mx = seg.max(dim=3).values.cpu().numpy()
        joint = seg.reshape(Tmax, B, JOINT).cpu().numpy()
        # ONE derivation of the decode's external inputs, shared with
        # QNNPolicy.act and the export wrapper — it raises on an obs that
        # cannot supply them rather than silently disabling the gate.
        flat = {k: v.reshape(Tmax * B, *v.shape[2:]) for k, v in obs_seq.items()}
        ep, ea, ammo, held = move_engagement_signals(flat, Tmax * B)
        thr = move_threat_signal(flat, Tmax * B)
        ep = ep.reshape(Tmax, B).cpu().numpy()
        ea = ea.reshape(Tmax, B).cpu().numpy()
        thr = thr.reshape(Tmax, B).cpu().numpy()
        ammo = ammo.reshape(Tmax, B, 4).cpu().numpy()
        held = held.reshape(Tmax, B).cpu().numpy()
        for b, ei in enumerate(batch):
            lo, hi = int(offsets[ei]), int(offsets[ei + 1])
            n = hi - lo
            p_jump[lo:hi] = pj[:n, b]
            lse_cls[lo:hi] = lse[:n, b]
            max_cls[lo:hi] = mx[:n, b]
            lr_joint[lo:hi] = joint[:n, b]
            enemy[lo:hi] = ep[:n, b]
            engaged_active[lo:hi] = ea[:n, b]
            threat[lo:hi] = thr[:n, b]
            ammo_pools[lo:hi] = ammo[:n, b]
            held_impulse[lo:hi] = held[:n, b]
        if bi % 10 == 0 or bi == len(batches) - 1:
            print(f"[move-gates]   batch {bi + 1}/{len(batches)}", flush=True)

    human = source.actions["move"].detach().cpu().numpy().astype(np.int64)
    im = (source.actions["input_mask"].detach().cpu().numpy()
          .astype(np.uint8).reshape(-1))
    # OPERATIVE filter: the engine only honours a jump press when feasible.
    feasible = (((im >> 7) & 1) | ((im >> 6) & 1)).astype(bool)
    # The training segment scalar: act.target == 1 - P(NO_TARGET), and the
    # engaged subset the heads were fit on is `$ne 0` — ANY off-NO_TARGET mass,
    # not a 0.5 threshold (qnn.bc.decode_fit.SEGMENT_MASK / filter_dsl).
    act_target = 1.0 - (source.actions["target_probs"]
                        .detach().cpu().numpy()[:, 0].astype(np.float64))
    # Held lr class at t-1, per episode (-1 at an episode's first row): the
    # class the maximal-run law masks out of the next onset decision.
    prev_lr = np.full(n_rows, -1, dtype=np.int64)
    for i in range(len(lengths)):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e - s > 1:
            prev_lr[s + 1:e] = human[s:e - 1, AXIS_LR]
    return {
        "p_jump": p_jump,
        "lse_cls": lse_cls,
        "max_cls": max_cls,
        "lr_joint": lr_joint,
        "engaged_active": engaged_active,
        "threat": threat,
        "ammo_pools": ammo_pools,
        "held_impulse": held_impulse,
        "episode_offsets": offsets,
        "prev_lr": prev_lr,
        "feasible": feasible,
        "human_jump": (human[:, 2] == 2) & feasible,
        "human_lr_none": human[:, AXIS_LR] == LR_NONE_ACTION,
        "enemy_present": enemy,
        "engaged": act_target != 0.0,
    }


def _det_rate(p_jump: np.ndarray, tau: float) -> float:
    return float((p_jump > tau).mean())


def fit_jump_tau(p_jump: np.ndarray, feasible: np.ndarray,
                 human_jump: np.ndarray, *, cut: float = DEFAULT_CUT,
                 label: str = "") -> dict[str, Any]:
    """τ whose deterministic fire rate is ``1/cut`` of the sampled rate.

    Solved on the empirical survival function of ``p_jump`` over feasible
    frames — the sampled rate IS ``mean(p_jump)`` there, so the target rate is
    a quantile lookup, refined against the realized rate because ties in
    ``p_jump`` make the survival function a step function.
    """
    pf = p_jump[feasible]
    if pf.size == 0:
        raise ValueError(f"{label or 'segment'}: no feasible frames")
    sampled = float(pf.mean())
    target = sampled / float(cut)
    tau = float(np.quantile(pf, 1.0 - target))
    realized = _det_rate(pf, tau)
    return {
        "label": label,
        "n_feasible": int(feasible.sum()),
        "n_total": int(p_jump.size),
        "human_jump_rate_feasible": float(human_jump[feasible].mean()),
        "sampled_rate_feasible": sampled,
        "target_rate": target,
        "cut_requested": float(cut),
        f"tau_for_{int(cut)}x_cut": tau,
        "det_gate_rate_at_tau": realized,
        "cut_factor_at_tau": (sampled / realized) if realized > 0 else float("inf"),
        "human_multiple_at_tau": (
            realized / float(human_jump[feasible].mean())
            if human_jump[feasible].any() else float("nan")),
        "mean_pjump_given_human_jump": (
            float(p_jump[human_jump].mean()) if human_jump.any() else float("nan")),
        "mean_pjump_given_no_human_jump": float(p_jump[~human_jump].mean()),
        "pjump_percentiles": {
            str(q): float(np.percentile(pf, q)) for q in (50, 90, 95, 99)},
        "grid": {
            f"{t:g}": {"det_gate_rate_feasible": _det_rate(pf, t),
                       "cut_factor_vs_sampled": (
                           sampled / _det_rate(pf, t)
                           if _det_rate(pf, t) > 0 else float("inf"))}
            for t in TAU_GRID},
    }


def _standstill_at(cls_logits: np.ndarray, prev_lr: np.ndarray, beta: float,
                   *, mask_held: bool = True, hard: bool = False) -> float:
    """Model lr stand-still fraction with ``beta`` on the none class.

    ``hard=False`` (pass per-class log-sum-exps): the SAMPLED class marginal —
    mean P(none), what an in-graph Gumbel-max move decode actually realises.
    ``hard=True`` (pass per-class maxes): the ARGMAX/greedy class fraction.

    ``mask_held`` drops the currently held class (``prev_lr``, teacher-forced
    from the human action at t-1) before renormalising, as the decode's onset
    mask does under the maximal-run law; a frame already holding `none`
    contributes 0, since the law cannot re-commit to it.

    The two statistics are NOT interchangeable: a sampled mean cannot move
    faster than 0.25 per unit β (the logistic bound), while an argmax fraction
    is unbounded, so their β curves have different shapes.
    """
    z = cls_logits.astype(np.float64).copy()
    z[:, NONE_CLASS] += beta
    if mask_held:
        held = prev_lr >= 0
        rows = np.nonzero(held)[0]
        z[rows, prev_lr[held]] = -np.inf
    if hard:
        return float((np.argmax(z, axis=1) == NONE_CLASS).mean())
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return float(p[:, NONE_CLASS].mean())


def rollout_standstill(rows: dict[str, np.ndarray], no_enemy: np.ndarray,
                       beta: float, *, dur_tilt=(0.0, 0.0), recommit: bool = False,
                       threat_break_hazard: float = 0.0,
                       idle_engagement_base: float = 0.5,
                       idle_cooldown_ticks: int = 20, seed: int = 0,
                       sample_episodes: float | None = None) -> float:
    """Realized per-frame ``lr == none`` OCCUPANCY under the real decode.

    This is the ruler the deployed bot actually expresses: the commitment decode
    samples a class AND a duration, so stand-still is the fraction of TICKS spent
    in `none`, not the per-onset probability of choosing it. β couples into both
    terms — it makes `none` likelier to be chosen and every chosen `none` then
    occupies several ticks — which is why this curve can rise faster per unit β
    than the 0.25 logistic bound that binds any per-frame marginal.

    Driven by ``move_commit_step`` itself rather than a re-derivation of its law:
    every episode is a lane, ticks advance together, and the commit state carries
    exactly as it does in play. Counted over ``no_enemy`` frames only (where the
    bias is active), but rolled out over ALL frames, since the held commitment
    crosses in and out of engagement.

    LANE RETIREMENT. Episodes are sorted by length DESCENDING, so the lanes still
    running at tick t are always the PREFIX ``[:n]`` of the batch: the per-tick
    slice is contiguous (a plain view, so ``move_commit_step``'s in-place state
    update still lands in the parent buffer, and a retired lane keeps the state it
    died with), and no expired row is ever stepped. The useful work is
    ``sum(lengths)`` lane-ticks; the padded ``Tmax x B`` rectangle this replaces
    did 34x that on qwd_v3 (135.4M vs 3.92M — 5859 episodes against a 23103-tick
    longest one), every bit of the excess on dead lanes. Wall time went 559s ->
    38s per evaluation, 15x rather than 34x: what is left is the per-TICK cost,
    paid 23103 times whatever the lane count, and only a shorter longest episode
    or a batched-over-time formulation would touch it.

    The COUNTED frames are exactly the padded version's — same rows, same
    selection, pinned by a test that swaps the decode for a deterministic
    function of the row and demands digit-for-digit equality. What retirement
    does change is the RNG: B shrinks per tick, so the shared torch generator is
    consumed in a different order and every draw lands on a different lane. The
    draws are still i.i.d. uniforms, so this is the same estimator of the same
    quantity — but it is a MONTE-CARLO estimator, and its value moves. Measured
    on qwd_v3 at the rc1 decode point (β=0.5): 0.249139 padded vs 0.250711
    retiring, against a seed-to-seed spread of 2.5e-4 over 3 seeds. Read this
    ruler at ~1e-3, and never expect a recorded β curve to reproduce digit-for-
    digit across the change.

    ``sample_episodes`` (fraction in (0, 1], deterministic given ``seed``) rolls
    out a random subset of EPISODES — whole episodes, never a slice of one,
    because a lane's occupancy is a PATH and a truncated lane would cut the
    commitments that make it. It buys much less than the fraction suggests, and
    both halves of that are worth knowing before reaching for it:

    * Cost. Once the dead lanes are gone the remaining cost is per TICK, and Tmax
      is set by the LONGEST surviving episode (23103 on qwd_v3), which random
      episode sampling barely touches. Measured: 38.7s at 1.0, 29.3s at 0.5,
      25.9s at 0.25 — 1.5x for a 4x smaller corpus.
    * Precision. The frame-level standard error is sqrt(p(1-p)/N), but ticks
      inside one commitment are perfectly correlated, so the effective N is
      roughly N / mean-commitment-duration. Measured seed-to-seed sd at the rc1
      point: 2.5e-4 at 1.0, ~1e-3 at 0.5 and 0.25, with the 0.25 mean sitting
      ~2e-3 high — i.e. it costs the resolution that separates two β candidates
      to save a third of the time.

    So: fine for a smoke test, not worth it for a fit. Leave it None (the
    default: every episode) for anything that gets reported.
    """
    from qnn.model.decode_actions import COMMIT_STATE_DIM, commit_reset_lanes, move_commit_step

    offsets = np.asarray(rows["episode_offsets"], dtype=np.int64)
    lengths = (offsets[1:] - offsets[:-1]).astype(np.int64)
    keep = lengths > 0
    lengths, starts = lengths[keep], offsets[:-1][keep]
    if sample_episodes is not None:
        frac = float(sample_episodes)
        if not 0.0 < frac <= 1.0:
            raise ValueError(
                f"sample_episodes must be a fraction in (0, 1]; got {frac}")
        n_ep = max(1, int(round(frac * len(lengths))))
        pick = np.random.default_rng(seed).choice(len(lengths), size=n_ep,
                                                 replace=False)
        lengths, starts = lengths[pick], starts[pick]
    # Sorted DESCENDING so "still running at tick t" is the prefix [:n_live].
    order = np.argsort(-lengths, kind="stable")
    lengths, starts = lengths[order], starts[order]
    B, Tmax = int(len(lengths)), int(lengths.max())

    # Row index of lane b at tick t is just starts[b] + t once the lanes are
    # sorted — no (Tmax, B) index/live rectangle to build or carry.
    starts_t = torch.as_tensor(starts)
    # fp16 in, converted per tick: the cache holds the joint in fp16 (the decode
    # samples from it), and a whole-corpus .astype(np.float32) is a 470 MB copy
    # for values that are exact either way.
    lr_joint = torch.as_tensor(rows["lr_joint"])
    enemy_t = torch.as_tensor(rows["enemy_present"])
    active_t = torch.as_tensor(rows["engaged_active"])
    threat_t = torch.as_tensor(rows["threat"])
    ammo_t = torch.as_tensor(rows["ammo_pools"])
    held_t = torch.as_tensor(rows["held_impulse"])
    sel_t = torch.as_tensor(no_enemy)

    commit = torch.as_tensor(commit_reset_lanes(), dtype=torch.float32).repeat(B, 1)
    if commit.shape[1] != COMMIT_STATE_DIM:
        raise ValueError(f"commit state width {commit.shape[1]} != {COMMIT_STATE_DIM}")
    # One (B, 2, JOINT) buffer for the whole rollout: fb stays all-zero (this
    # ruler reads the lr axis) and only the live prefix is rewritten per tick.
    # move_commit_step clones the per-axis logits it works on, so the buffer is
    # never mutated under us.
    seg = torch.zeros(B, 2, JOINT, dtype=torch.float32)
    torch.manual_seed(seed)
    # Accumulate in tensors: an int() per tick is a sync + two Python objects
    # per lane-tick, and at Tmax in the thousands that shows up in the profile.
    n_none = torch.zeros((), dtype=torch.int64)
    n_frames = torch.zeros((), dtype=torch.int64)
    n_live = B
    for t in range(Tmax):
        # Retire every lane whose episode ended at or before t. Descending sort
        # ⇒ they are always the tail, so the live set stays a contiguous prefix
        # and the commit state needs no compaction to stay aligned with it.
        while lengths[n_live - 1] <= t:
            n_live -= 1
        r = starts_t[:n_live] + t
        seg[:n_live, AXIS_LR, :] = lr_joint.index_select(0, r).float()
        cls = move_commit_step(
            seg[:n_live], commit[:n_live], greedy=False, dur_tilt=tuple(dur_tilt),
            threat=threat_t.index_select(0, r),
            threat_break_hazard=float(threat_break_hazard),
            recommit=bool(recommit),
            enemy_present=enemy_t.index_select(0, r),
            engaged_active=active_t.index_select(0, r),
            idle_none_bias=(0.0, float(beta)),
            idle_engagement_base=float(idle_engagement_base),
            idle_cooldown_ticks=int(idle_cooldown_ticks),
            ammo_pools=ammo_t.index_select(0, r),
            held_impulse=held_t.index_select(0, r))
        # No `live` mask: every stepped lane is live by construction.
        counted = sel_t.index_select(0, r)
        n_none += ((cls[:, AXIS_LR] == NONE_CLASS) & counted).sum()
        n_frames += counted.sum()
    if int(n_frames) == 0:
        raise ValueError("no counted frames in the rollout")
    return int(n_none) / int(n_frames)


def fit_idle_beta(rows: dict[str, np.ndarray], no_enemy: np.ndarray,
                  *, statistic: str = "rollout", mask_held: bool = True,
                  decode_kw: dict[str, Any] | None = None,
                  hi: float = 8.0, tol: float = 1e-3) -> dict[str, Any]:
    """β matching the model's no-enemy lr stand-still to the human fraction.

    ``statistic`` selects the model-side ruler: ``"sampled"`` (the class
    marginal, what the deployed Gumbel-max move decode realises) or ``"argmax"``
    (the greedy class). Every candidate's base value and β curve are reported
    under ``candidates`` so a recorded artifact can identify which ruler produced
    it — the curves differ in SHAPE, not just in level, so the base alone cannot
    tell them apart.

    Every β the fit touches goes through ONE memo table keyed on the rounded β,
    so no β is ever evaluated twice: the base, the reported grid and the
    refinement share it, and the table ships back as ``beta_evaluations`` so a
    caller that wants to print its own curve reads it instead of paying again.
    Bookkeeping for the closed-form rulers; for ``"rollout"`` one evaluation is a
    full decode rollout over the corpus (minutes), which is the whole reason the
    table exists.

    The root is refined by FALSE POSITION on the bracket the coarse grid already
    paid for, not by bisection from scratch: the occupancy curve is smooth and
    near-linear inside a grid interval, so tol=1e-3 lands in 3-6 probes where
    bisection needed ~13. Together with the memo that puts the whole rc1 rollout
    fit at 10 evaluations / 6.2 min (7 of them the grid the report has to show
    anyway), against ~23 / 3.6 h for grid-then-bisect. The bracket is preserved on
    every step (the interpolant is clamped into its interior), so a locally flat
    or noisy patch of the Monte-Carlo rollout ruler cannot walk the search off the
    root — rc1 landed β=1.5151 at 0.396636 against a human 0.396632.
    """
    decode_kw = dict(decode_kw or {})
    lc, mc = rows["lse_cls"][no_enemy], rows["max_cls"][no_enemy]
    pv = rows["prev_lr"][no_enemy]
    human_lr_none = rows["human_lr_none"]
    if lc.shape[0] == 0:
        raise ValueError("no no-enemy frames — cannot fit the idle bias")
    target = float(human_lr_none[no_enemy].mean())

    def curve_of(logits, hard, masked):
        return {f"{b:.1f}": _standstill_at(logits, pv, b, mask_held=masked,
                                           hard=hard) for b in BETA_GRID}

    candidates = {
        f"{name}{'' if masked else '_unmasked'}": {
            "base": _standstill_at(src, pv, 0.0, mask_held=masked, hard=hard),
            "grid_beta_to_standstill": curve_of(src, hard, masked),
        }
        for name, src, hard in (("sampled", lc, False), ("argmax", mc, True))
        for masked in (True, False)
    }

    def measure(b: float) -> float:
        if statistic == "rollout":
            return rollout_standstill(rows, no_enemy, b, **decode_kw)
        return _standstill_at(mc if statistic == "argmax" else lc, pv, b,
                              mask_held=mask_held, hard=statistic == "argmax")

    evals: dict[float, float] = {}

    def at(b: float) -> float:
        # Key on the ROUNDED β and evaluate that rounded value, so the table's
        # entry really is f(key) rather than f(something nearby). BETA_DECIMALS
        # is two orders below the default tol, i.e. below the ruler's own
        # resolution, so the rounding cannot move the placed β.
        key = round(float(b), BETA_DECIMALS)
        if key not in evals:
            evals[key] = measure(key)
        return evals[key]

    if statistic == "rollout":
        candidates["rollout"] = {
            "base": at(0.0),
            "grid_beta_to_standstill": {f"{b:.1f}": at(b) for b in BETA_GRID},
            "decode_kw": decode_kw,
        }

    base = at(0.0)
    if base >= target:
        # Already stiller than the human when disengaged: the bias can only add
        # stillness, so it has nothing to correct. Report it, do not fudge it.
        beta, at_beta = 0.0, base
    else:
        # Seed the bracket from the coarse grid — free, since the grid is
        # reported anyway and every point is already in the memo table.
        lo, f_lo, f_hi = 0.0, base, None
        for b in BETA_GRID:
            if b <= lo:
                continue
            v = at(b)
            if v < target:
                lo, f_lo = float(b), v
            else:
                hi, f_hi = float(b), v
                break
        if f_hi is None:
            # The whole reported grid is still below the human fraction: walk out
            # by doubling, as before.
            hi = max(float(hi), 2.0 * lo)
            while at(hi) < target:
                hi *= 2.0
                if hi > 1e4:
                    raise ValueError("idle β diverged — check the target fraction")
            f_hi = at(hi)
        beta, at_beta = lo, f_lo
        for step in range(1, 41):
            span = hi - lo
            if f_hi > f_lo:
                nxt = lo + (target - f_lo) * span / (f_hi - f_lo)
                # Keep the probe strictly inside the bracket: an interpolant that
                # lands on an endpoint (a flat or noisy patch) would not shrink
                # it, and the bracket is the only thing keeping a Monte-Carlo
                # ruler's wobble from walking the search off the root.
                nxt = min(max(nxt, lo + 0.02 * span), hi - 0.02 * span)
            else:
                # Non-increasing patch: no usable slope, so bisect.
                nxt = lo + 0.5 * span
            # Probe the memo's key itself, so the reported β is exactly the β
            # whose stand-still is reported beside it.
            nxt = round(nxt, BETA_DECIMALS)
            v = at(nxt)
            moved = span if step == 1 else abs(nxt - beta)
            beta, at_beta = nxt, v
            if v < target:
                lo, f_lo = nxt, v
            else:
                hi, f_hi = nxt, v
            # `tol` is a tolerance on β, and the error here is the STEP SIZE, not
            # the bracket width: false position converges with one endpoint left
            # stationary, so the bracket can stay wide around a root the probes
            # have already pinned. Stop when the estimate stops moving (or the
            # bracket does close, whichever comes first).
            if moved <= tol or hi - lo <= tol:
                break
        # β is the last PROBE, and its stand-still is the value measured there —
        # not an interpolation, and not one more rollout for a midpoint nobody
        # evaluated.
    return {
        "statistic": statistic,
        "mask_held": bool(mask_held),
        "n_no_enemy": int(no_enemy.sum()),
        "human_lr_standstill_no_enemy": target,
        "model_lr_standstill_no_enemy_base": base,
        "beta_lr_fit": beta,
        "model_lr_standstill_at_beta": at_beta,
        "already_stiller_than_human": bool(base >= target),
        # Every β this fit ever evaluated, so a caller can plot/print the curve
        # without re-running the ruler. str(), not %g: a %g key rounds to 6
        # SIGNIFICANT digits, which for a refined β is no longer the β that was
        # measured, and this table is meant to be read back as one.
        "beta_evaluations": {str(k): v for k, v in sorted(evals.items())},
        "n_beta_evaluations": len(evals),
        "grid_beta_to_standstill": candidates[
            "rollout" if statistic == "rollout"
            else f"{statistic}{'' if mask_held else '_unmasked'}"
        ]["grid_beta_to_standstill"],
        "candidates": candidates,
    }


def _load_rows_cache(path: Path, run_dir: Path, cache_dir: Path) -> dict | None:
    """Reload a saved forward pass, but only for the same (run, corpus) pair."""
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if (str(z["_run_dir"]) != str(run_dir)
            or str(z["_cache_dir"]) != str(cache_dir)):
        print(f"[move-gates] ignoring {path.name}: keyed to "
              f"{z['_run_dir']} / {z['_cache_dir']}", flush=True)
        return None
    print(f"[move-gates] reusing cached forward from {path}", flush=True)
    return {k: z[k] for k in z.files if not k.startswith("_")}


def fit_move_gates(run_dir: Path, cache_dir: Path | str, *,
                   cut: float = DEFAULT_CUT, statistic: str = "rollout",
                   mask_held: bool = True,
                   decode_kw: dict[str, Any] | None = None,
                   rows_cache: Path | None = None,
                   out_path: Path | None = None) -> dict[str, Any]:
    """Both gate fits for one checkpoint over its OWN pinned corpus.

    ``cache_dir`` must be the run's own corpus (``machine.json`` bc_data_dir):
    another collect is out of distribution and may not even share the obs
    layout, exactly as in qnn.eval.move_commit_fit.
    """
    from qnn.bc.supervised_loop import make_resident_source_from_cache
    from qnn.diag.loader import load_policy

    run_dir = Path(run_dir)
    rows = (_load_rows_cache(Path(rows_cache), run_dir, Path(cache_dir))
            if rows_cache else None)
    if rows is not None:
        return _fit_from_rows(rows, run_dir, cache_dir, cut=cut,
                              statistic=statistic, mask_held=mask_held,
                              decode_kw=decode_kw, out_path=out_path)
    policy, _probe = load_policy(run_dir, device="cpu")
    # segment_mask=None: the no-jump and no-enemy frames ARE the population
    # both gates are placed on — an engaged-only cache cannot see either.
    source = make_resident_source_from_cache(
        Path(cache_dir) / "precomputed_val", torch.device("cpu"),
        segment_mask=None)
    rows = collect_gate_rows(policy, source)
    if rows_cache:
        Path(rows_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(rows_cache, _run_dir=str(run_dir), _cache_dir=str(cache_dir),
                 **rows)
        print(f"[move-gates] cached forward -> {rows_cache}", flush=True)

    return _fit_from_rows(rows, run_dir, cache_dir, cut=cut,
                          statistic=statistic, mask_held=mask_held,
                          decode_kw=decode_kw, out_path=out_path)


def _fit_from_rows(rows: dict[str, np.ndarray], run_dir: Path,
                   cache_dir: Path | str, *, cut: float, statistic: str,
                   mask_held: bool, decode_kw: dict[str, Any] | None,
                   out_path: Path | None) -> dict[str, Any]:
    """Both fits from an already-collected forward pass."""
    run_dir = Path(run_dir)
    cks = (sorted((run_dir / "checkpoints").glob("best_*.pth"))
           or sorted((run_dir / "checkpoints").glob("bc_best_model.pth")))
    engaged = rows["engaged"]
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(cks[0]) if cks else "",
        "cache_dir": str(cache_dir),
        "n_rows": int(rows["p_jump"].size),
        "jump": {
            "engaged": fit_jump_tau(
                rows["p_jump"][engaged], rows["feasible"][engaged],
                rows["human_jump"][engaged], cut=cut, label="engaged"),
            "all": fit_jump_tau(
                rows["p_jump"], rows["feasible"], rows["human_jump"],
                cut=cut, label="all"),
        },
        "idle": fit_idle_beta(
            rows, ~rows["enemy_present"], statistic=statistic,
            mask_held=mask_held, decode_kw=decode_kw),
    }
    # The placed values: τ from the ENGAGED segment (the a26rc1b rule), rounded
    # the way a config carries it; β to 4 decimals.
    tau_key = f"tau_for_{int(cut)}x_cut"
    report["placed"] = {
        "jump.threshold": round(report["jump"]["engaged"][tau_key], 2),
        "move.idle_none_bias": [0.0, round(report["idle"]["beta_lr_fit"], 4)],
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1) + "\n")
    return report




def _validate(report: dict[str, Any], jump_tau_path: Path,
              idle_path: Path) -> None:
    """Print deltas against a recorded artifact pair (the a26rc1a recovery check)."""
    rec_tau = json.loads(Path(jump_tau_path).read_text())
    rec_idle = json.loads(Path(idle_path).read_text())
    print("\n=== validation vs recorded artifacts ===")
    for seg in ("engaged", "all"):
        if seg not in rec_tau:
            continue
        got, want = report["jump"][seg], rec_tau[seg]
        for k in ("n_feasible", "sampled_rate_feasible", "human_rate_feasible",
                  "tau_for_4x_cut", "det_gate_rate_at_tau"):
            w = want.get(k)
            g = got.get(k, got.get("human_jump_rate_feasible")
                        if k == "human_rate_feasible" else None)
            if w is None or g is None:
                continue
            d = (g - w) / w if isinstance(w, float) and w else (g - w)
            print(f"  jump.{seg}.{k:26} got {g!r:>22}  want {w!r:>22}  Δ {d:+.4%}"
                  if isinstance(w, float) else
                  f"  jump.{seg}.{k:26} got {g!r:>22}  want {w!r:>22}")
    for k in ("n_no_enemy", "human_lr_standstill_no_enemy",
              "model_lr_standstill_no_enemy_base", "beta_lr_fit"):
        if k not in rec_idle:
            continue
        g, w = report["idle"][k], rec_idle[k]
        print(f"  idle.{k:32} got {g!r:>22}  want {w!r:>22}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qnn.decode_fit.move_gates",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="the run's OWN pinned corpus (machine.json bc_data_dir)")
    ap.add_argument("--cut", type=float, default=DEFAULT_CUT,
                    help="sampled-rate cut factor τ is placed for (a26rc1b used 4)")
    ap.add_argument("--statistic", choices=("rollout", "sampled", "argmax"),
                    default="rollout",
                    help="model-side stand-still ruler for the idle fit; "
                         "'rollout' drives the real commitment decode and is "
                         "what the deployed bot expresses")
    ap.add_argument("--dur-tilt", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("FB", "LR"),
                    help="the base config's move.commit_dur_tilt (rollout ruler)")
    ap.add_argument("--recommit", action="store_true",
                    help="roll out with move.commit_recommit enabled")
    ap.add_argument("--threat-break-hazard", type=float, default=0.0,
                    help="the base config's move.threat_break_hazard")
    ap.add_argument("--sample-episodes", type=float, default=None,
                    help="roll out this FRACTION of episodes (deterministic). "
                         "A smoke-test knob: it costs the ~1e-3 resolution that "
                         "separates two β candidates and only saves ~1.5x at 0.25, "
                         "because the surviving cost is per tick and Tmax is set "
                         "by the longest episode")
    ap.add_argument("--no-mask-held", action="store_true",
                    help="do not mask the held class in the idle statistic")
    ap.add_argument("--rows-cache", type=Path, default=None,
                    help="save/reuse the forward pass here (keyed on run+corpus)")
    ap.add_argument("--out", type=Path, default=None, help="write the report here")
    ap.add_argument("--validate-against", nargs=2, type=Path, default=None,
                    metavar=("JUMP_TAU_JSON", "IDLE_JSON"),
                    help="print deltas vs a recorded artifact pair")
    a = ap.parse_args(argv)

    report = fit_move_gates(
        a.run_dir, a.cache_dir, cut=a.cut, statistic=a.statistic,
        mask_held=not a.no_mask_held, out_path=a.out,
        rows_cache=a.rows_cache,
        decode_kw={"dur_tilt": tuple(a.dur_tilt), "recommit": a.recommit,
                   "threat_break_hazard": a.threat_break_hazard,
                   "sample_episodes": a.sample_episodes})
    j = report["jump"]["engaged"]
    tau_key = f"tau_for_{int(a.cut)}x_cut"
    print(f"\n[move-gates] {report['n_rows']:,} rows")
    print(f"  jump engaged: feasible {j['n_feasible']:,}  sampled "
          f"{j['sampled_rate_feasible']:.6f}  human {j['human_jump_rate_feasible']:.6f}")
    print(f"    τ({a.cut:g}x cut) = {j[tau_key]:.4f} → det rate "
          f"{j['det_gate_rate_at_tau']:.6f} (cut {j['cut_factor_at_tau']:.3f}x, "
          f"{j['human_multiple_at_tau']:.3f}x human)")
    i = report["idle"]
    print(f"  idle: no-enemy {i['n_no_enemy']:,}  human standstill "
          f"{i['human_lr_standstill_no_enemy']:.4f}  model base "
          f"{i['model_lr_standstill_no_enemy_base']:.4f}")
    print(f"    β_lr = {i['beta_lr_fit']:.4f} → {i['model_lr_standstill_at_beta']:.4f}"
          + ("  (ALREADY STILLER THAN HUMAN — bias has nothing to correct)"
             if i["already_stiller_than_human"] else ""))
    print(f"  placed: {json.dumps(report['placed'])}")
    if a.validate_against:
        _validate(report, *a.validate_against)
    if a.out:
        print(f"\nwritten -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
