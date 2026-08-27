"""Look-segment commit decode — render look_seg onset commitments into a per-tick
turn-delta stream (Phase 2 of agents/plans/look-seg-head.md).

The look_seg head emits, at each segment onset, a JOINT (onset-class x duration-bucket)
+ a direction categorical. This module turns a committed stroke (amplitude center, dir,
duration bucket) into the per-tick theta magnitudes it plays out, using the corpus-fit
within-stroke velocity profile — fit PER Hz (qnn.model.look_seg_bins.bins_for_hz),
same as the duration/amplitude grid: the profile is a shape over TICKS, so it does not
transfer across tick rates. A hold commitment renders theta==0 for its duration.

This is the EXECUTOR half. The full look_commit_step (state lanes [cls, rem, elapsed,
amp, dir] + onset resampling + aim-prior steering + interrupt) mirrors move_commit_step
and consumes render_stroke_theta per tick. Built kernel-first so the renderer is unit-
verifiable in isolation before the stateful step is wired into the eval path.
"""
from __future__ import annotations

import numpy as np
import torch

from qnn.model.look_seg_bins import (
    N_DUR_BUCKETS, N_STROKE_AMP, bins_for_hz,
)

# Within-stroke normalized velocity profiles, per duration bucket, PER HZ FIT.
# Bucket 0 = 1-tick strokes (single impulse, no profile). Buckets 1..9 (dur>=2)
# carry the resampled 10-point shape (theta_t / total). Source:
# _look_seg_audit{,_10hz}.json velocity_profiles.by_bucket, transcribed verbatim
# (the audit already resamples to _N_PROFILE_PTS points; raw shapes are
# renormalized at render time to sum to 1 over the actual duration).
_VEL_PROFILE_BY_HZ: dict[int, dict[int, list[float]]] = {
    10: {  # runs/head_probe/_look_seg_audit_10hz.json
        1: [0.496, 0.496, 0.496, 0.4973, 0.4991, 0.5009, 0.5027, 0.504, 0.504, 0.504],
        2: [0.248, 0.248, 0.2859, 0.3618, 0.4377, 0.4425, 0.376, 0.3096, 0.2763, 0.2763],
        3: [0.1711, 0.1711, 0.221, 0.2782, 0.3071, 0.3192, 0.3061, 0.2509, 0.2026, 0.2026],
        4: [0.1373, 0.1423, 0.1924, 0.2256, 0.2197, 0.227, 0.2475, 0.2192, 0.1698, 0.1649],
        5: [0.1137, 0.1269, 0.1794, 0.1819, 0.1719, 0.1749, 0.1946, 0.2002, 0.1467, 0.1333],
        6: [0.0967, 0.1164, 0.1665, 0.1517, 0.1432, 0.1435, 0.1548, 0.1807, 0.1342, 0.1157],
        7: [0.0841, 0.1079, 0.1419, 0.1311, 0.1228, 0.1192, 0.1273, 0.155, 0.1263, 0.1016],
        8: [0.0584, 0.0907, 0.0935, 0.0878, 0.0851, 0.0829, 0.0839, 0.0942, 0.1005, 0.0691],
        9: [0.0274, 0.0427, 0.0378, 0.0359, 0.0347, 0.0352, 0.0354, 0.0371, 0.0419, 0.0338],
    },
    20: {  # runs/head_probe/_look_seg_audit.json (tick_hz 20, qwd_v4)
        1: [0.522, 0.522, 0.522, 0.5147, 0.5049, 0.4951, 0.4853, 0.478, 0.478, 0.478],
        2: [0.2722, 0.2722, 0.3035, 0.3661, 0.4288, 0.428, 0.3639, 0.2998, 0.2677, 0.2677],
        3: [0.1728, 0.1728, 0.2271, 0.2892, 0.3192, 0.33, 0.3103, 0.2397, 0.1779, 0.1779],
        4: [0.1184, 0.1245, 0.1857, 0.2347, 0.2553, 0.2637, 0.2598, 0.2082, 0.136, 0.1288],
        5: [0.0908, 0.1042, 0.1576, 0.1893, 0.2095, 0.2171, 0.2131, 0.1902, 0.1215, 0.1043],
        6: [0.0739, 0.0923, 0.1411, 0.158, 0.171, 0.179, 0.1813, 0.1771, 0.113, 0.0885],
        7: [0.0549, 0.0805, 0.1112, 0.1173, 0.118, 0.1234, 0.1333, 0.1368, 0.1018, 0.0664],
        8: [0.0284, 0.0598, 0.0662, 0.0634, 0.0615, 0.0616, 0.0648, 0.073, 0.0744, 0.0353],
        9: [0.0121, 0.0288, 0.0273, 0.0263, 0.0266, 0.0258, 0.0265, 0.027, 0.0313, 0.0154],
    },
}
_N_PROFILE_PTS = 10


def stroke_theta_schedule(bucket: int, dur: int, hz: int) -> np.ndarray:
    """Per-tick fractions of the stroke's total amplitude for a `dur`-tick stroke
    in duration `bucket`, at corpus tick rate `hz`. Returns an array of length
    `dur` summing to 1.0.

    dur==1 (or bucket 0): single impulse -> [1.0]. Otherwise the bucket's 10-pt
    profile (``hz``'s fit) is resampled to `dur` points and renormalized to sum 1.
    """
    dur = int(max(1, dur))
    profiles = _VEL_PROFILE_BY_HZ[hz]  # raises KeyError -> caller passed an unfit hz
    if dur == 1 or bucket <= 0 or bucket not in profiles:
        return np.ones(dur, dtype=np.float64) / dur
    prof = np.asarray(profiles[bucket], dtype=np.float64)
    # resample the 10-pt shape onto the stroke's `dur` tick centers
    src_x = np.linspace(0.0, 1.0, _N_PROFILE_PTS)
    dst_x = (np.arange(dur) + 0.5) / dur
    resamp = np.interp(dst_x, src_x, prof)
    s = resamp.sum()
    return resamp / s if s > 0 else np.ones(dur) / dur


def render_stroke_theta(amp_rad: float, bucket: int, dur: int, hz: int) -> np.ndarray:
    """Per-tick signed magnitude (radians) for a stroke: amp * velocity schedule.
    Sums to `amp_rad` over the stroke. Hold (amp==0) -> zeros."""
    sched = stroke_theta_schedule(bucket, dur, hz)
    return float(amp_rad) * sched


def amp_center_for_class(onset_class: int, hz: int) -> float:
    """onset_class 0 = hold (amp 0); 1..N_STROKE_AMP -> hz's amp_centers_rad."""
    if onset_class <= 0:
        return 0.0
    centers = bins_for_hz(hz).amp_centers_rad
    return float(centers[min(onset_class - 1, N_STROKE_AMP - 1)])


if __name__ == "__main__":  # standalone sanity — no torch/model needed
    for _hz in (10, 20):
        # 1) hold renders zeros
        assert np.allclose(render_stroke_theta(0.0, 0, 4, _hz), 0.0), "hold not zero"
        # 2) a stroke's per-tick magnitudes sum to its amplitude (mass conserved)
        for cls in range(1, N_STROKE_AMP + 1):
            amp = amp_center_for_class(cls, _hz)
            for bucket, dur in [(0, 1), (2, 3), (5, 6), (9, 30)]:
                sched = render_stroke_theta(amp, bucket, dur, _hz)
                assert len(sched) == max(1, dur), (_hz, cls, bucket, dur, len(sched))
                assert abs(sched.sum() - amp) < 1e-9, (_hz, cls, bucket, dur, sched.sum(), amp)
                assert (sched >= -1e-12).all() or amp >= 0, "monotone sign"
        # 3) bucket profile is non-uniform (a real velocity shape, not flat) for dur>1
        s = stroke_theta_schedule(2, 6, _hz)
        assert s.std() > 1e-3, "profile should be shaped, not flat"
        # 4) amp centers are monotincreasing in class
        centers = [amp_center_for_class(c, _hz) for c in range(1, N_STROKE_AMP + 1)]
        assert centers == sorted(centers), centers
        print(f"look_seg_decode renderer OK (hz={_hz}):",
              f"amp centers(deg)={[round(np.degrees(c),1) for c in centers]}",
              f"| stroke(bkt2,dur6) sched={np.round(s,3).tolist()}")


# ── Stateful commit step (Phase 2) ───────────────────────────────────────────
from qnn.model.look_seg_bins import (  # noqa: E402
    JOINT, N_DUR_BUCKETS, N_LOOK_DIR, N_ONSET_CLASSES,
)
from qnn.model.look_seg_segment import DUR_CAP  # noqa: E402

_HZ_TABLE = (10, 20)


def _build_sched_lut(hz: int) -> np.ndarray:
    """Dense schedule LUT [dur_bucket, dur, elapsed] -> amp-fraction for `hz`."""
    lut = np.zeros((N_DUR_BUCKETS, DUR_CAP + 1, DUR_CAP + 1), dtype=np.float32)
    for b in range(N_DUR_BUCKETS):
        for d in range(1, DUR_CAP + 1):
            lut[b, d, :d] = stroke_theta_schedule(b, d, hz)
    return lut


def _lo_hi(hz: int) -> "tuple[list[int], list[int]]":
    """Per-bucket actual-duration range (mirror of move's _BUCKET_LO/_HI)."""
    edges = bins_for_hz(hz).dur_edges
    lo = list(edges)
    hi = [e - 1 for e in edges[1:]] + [DUR_CAP]
    return lo, hi


# Per-hz dense LUT / amp-class table, built once at import (both fits are a few
# hundred KB total — trivial memory, no need for lazy per-call construction).
LOOK_SCHED_LUT_BY_HZ: dict[int, torch.Tensor] = {
    hz: torch.from_numpy(_build_sched_lut(hz)) for hz in _HZ_TABLE
}
_LO_BY_HZ: dict[int, list[int]] = {}
_HI_BY_HZ: dict[int, list[int]] = {}
for _hz in _HZ_TABLE:
    _LO_BY_HZ[_hz], _HI_BY_HZ[_hz] = _lo_hi(_hz)
_AMP_BY_CLASS_BY_HZ: dict[int, np.ndarray] = {
    hz: np.array([amp_center_for_class(c, hz) for c in range(N_ONSET_CLASSES)],
                 dtype=np.float32)  # class 0 (hold) -> 0.0
    for hz in _HZ_TABLE
}
# Direction categorical is hz-independent (uniform bins over [0, 2pi)).
_DIR_CENTERS = np.array([2 * np.pi * k / N_LOOK_DIR for k in range(N_LOOK_DIR)],
                        dtype=np.float32)


def look_commit_step(seg_logits, commit_state, *, hz: int, greedy=True, generator=None,
                     row_generators=None):
    """One tick of the look commitment. seg_logits: (B, JOINT+N_LOOK_DIR).
    commit_state: (B,5) lanes [cls, rem, elapsed, dur_bucket, dir_bin] (init any
    row with rem<=0 to force onset). ``hz`` selects the corpus-fit grid/profile
    (qnn.model.look_seg_bins.bins_for_hz) — pass the checkpoint's OWN
    resolved rate (``LookSegHead.hz``), never an ambient config. Returns per-tick
    (theta, phi) in radians and updates commit_state IN PLACE. theta feeds
    decode_look_from_polar's z-assembly.

    Sampled RNG: ``row_generators`` (a ``BatchedRNG`` for batched eval, or a list
    of per-row ``torch.Generator`` for per-env eval) gives per-row-independent
    streams — the eval-path contract, mirroring ``move_commit_step``. Falls back
    to the single ``generator`` (or default RNG) when ``row_generators`` is None.
    Greedy is deterministic and bit-parity with ``look_commit_step_graph``."""
    dev = seg_logits.device
    B = seg_logits.shape[0]
    lut = LOOK_SCHED_LUT_BY_HZ[hz].to(dev)
    amp_by_cls = torch.from_numpy(_AMP_BY_CLASS_BY_HZ[hz]).to(dev)
    dir_ctr = torch.from_numpy(_DIR_CENTERS).to(dev)
    lo = torch.as_tensor(_LO_BY_HZ[hz], dtype=torch.long, device=dev)
    hi = torch.as_tensor(_HI_BY_HZ[hz], dtype=torch.long, device=dev)

    cls = commit_state[:, 0].long()
    rem = commit_state[:, 1].long()
    need = rem <= 0

    joint = seg_logits[:, :JOINT]                                # (B, 90)
    dir_l = seg_logits[:, JOINT:JOINT + N_LOOK_DIR]              # (B, 16)
    if greedy:
        jidx = joint.argmax(dim=-1)
        didx = dir_l.argmax(dim=-1)
    else:
        from qnn.model.decode import BatchedRNG, inverse_cdf_sample, row_uniforms
        jp = torch.softmax(joint, dim=-1)
        dp = torch.softmax(dir_l, dim=-1)
        if isinstance(row_generators, BatchedRNG):
            # vectorized per-row-independent (joint, dir, in-bucket-u) draws — one
            # dispatch (mirrors move_commit_step's batched fast path).
            _uu = row_uniforms(row_generators, 3, dev)              # (B, 3)
            jidx = inverse_cdf_sample(jp, _uu[:, 0])
            didx = inverse_cdf_sample(dp, _uu[:, 1])
            u = _uu[:, 2]
        elif row_generators is not None:
            jidx = torch.empty(B, dtype=torch.long, device=dev)
            didx = torch.empty(B, dtype=torch.long, device=dev)
            u = torch.empty(B, device=dev)
            for i, g in enumerate(row_generators):
                jidx[i] = torch.multinomial(jp[i:i + 1], 1, generator=g).squeeze()
                didx[i] = torch.multinomial(dp[i:i + 1], 1, generator=g).squeeze()
                u[i] = torch.rand(1, generator=g, device=dev)
        elif generator is None:
            jidx = torch.multinomial(jp, 1).squeeze(-1)
            didx = torch.multinomial(dp, 1).squeeze(-1)
            u = torch.rand(B, device=dev)
        else:
            jidx = torch.multinomial(jp, 1, generator=generator).squeeze(-1)
            didx = torch.multinomial(dp, 1, generator=generator).squeeze(-1)
            u = torch.rand(B, device=dev, generator=generator)
    new_cls = jidx // N_DUR_BUCKETS                              # 0=hold, 1..8 stroke
    new_bkt = jidx % N_DUR_BUCKETS
    if greedy:
        new_dur = (lo[new_bkt] + hi[new_bkt]) // 2
    else:
        span = (hi[new_bkt] - lo[new_bkt] + 1).to(torch.float32)
        new_dur = lo[new_bkt] + (u * span).long().clamp(max=(hi[new_bkt] - lo[new_bkt]))

    # commit onset on `need` rows; else keep the active commitment
    cls = torch.where(need, new_cls, cls)
    bkt = torch.where(need, new_bkt, commit_state[:, 3].long())
    dir_bin = torch.where(need, didx, commit_state[:, 4].long())
    rem = torch.where(need, new_dur, rem)
    elapsed = torch.where(need, torch.zeros_like(rem), commit_state[:, 2].long())

    dur = (elapsed + rem).clamp(1, DUR_CAP)                     # dur = elapsed+rem
    frac = lut[bkt.clamp(0, N_DUR_BUCKETS - 1),
               dur, elapsed.clamp(0, DUR_CAP)]                  # (B,)
    theta = amp_by_cls[cls.clamp(0, N_ONSET_CLASSES - 1)] * frac  # hold cls0 -> 0
    phi = dir_ctr[dir_bin.clamp(0, N_LOOK_DIR - 1)]

    commit_state[:, 0] = cls.to(commit_state.dtype)
    commit_state[:, 1] = (rem - 1).to(commit_state.dtype)
    commit_state[:, 2] = (elapsed + 1).to(commit_state.dtype)
    commit_state[:, 3] = bkt.to(commit_state.dtype)
    commit_state[:, 4] = dir_bin.to(commit_state.dtype)
    return theta, phi


# ── Export state layout + graph twin (Phase 3) ────────────────────────────────
LOOK_COMMIT_STATE_DIM = 5
_LOOK_COMMIT_RESET_LANES = (0.0, 0.0, 0.0, 0.0, 0.0)


def look_commit_reset_lanes() -> "list[float]":
    """Per-episode reset init for the look commit state (the ONNX
    ``look_commit_state`` loopback slot): lanes [cls, rem, elapsed, dur_bucket,
    dir_bin]. ``rem == 0`` forces an onset on the first tick of every episode
    (need = rem <= 0), exactly like a freshly reset eager ``commit_state``.
    Stamped into ``state.loopback`` so the engine memsets it at episode start —
    the tremor/move_state precedent (zero engine change, no wire bump)."""
    return list(_LOOK_COMMIT_RESET_LANES)


def look_commit_step_graph(
    seg_logits: torch.Tensor,        # (B, JOINT + N_LOOK_DIR)
    commit_state: torch.Tensor,      # (B, LOOK_COMMIT_STATE_DIM) float32 — flat carried state
    *,
    hz: int,
    greedy: bool,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """TRACE-SAFE in-graph twin of :func:`look_commit_step` — the LOOK COMMIT
    DECODE LAW baked into the ONNX so deploy == the QNNPolicy.act look decode.
    ``hz`` must be the exporting checkpoint's OWN resolved rate (``LookSegHead.hz``)
    — it selects which fitted grid/profile LUT gets baked as constants.

    Returns ``(theta, phi, commit_state_out)``:
      theta            (B,) signed per-tick turn magnitude (rad); hold class -> 0.
      phi              (B,) per-tick direction (rad) — the committed stroke heading.
      commit_state_out (B, LOOK_COMMIT_STATE_DIM) float32 — updated flat state
                       (rebuilt by torch.stack; NO in-place slice assignment,
                       which the legacy ONNX exporter traces poorly — mirrors
                       move_commit_step_graph).

    ``greedy`` is a PYTHON bool (graph-build-time), baked as a taken branch:
      greedy  → argmax joint/dir + median-of-bucket duration (deterministic; the
                bit-exact parity/gate path against the eager greedy law).
      sampled → gumbel-max joint/dir + uniform-in-bucket duration (the deployed
                stochastic stream; ORT ``RandomUniformLike``, like the a24 jump
                sample), NOT torch.multinomial — so it differs from the eager
                sampled path by construction (validated distributionally, not
                bit-exact — the move-commit convention).

    Bit-for-bit with the eager greedy law: same onset-on-expiry rule, same
    velocity-profile LUT gather, same //2 median, same emit/rem/elapsed update."""
    from qnn.model.decode import gumbel_argmax  # ORT RandomUniformLike (no circular import)

    dev = seg_logits.device
    B = seg_logits.shape[0]
    _D1 = DUR_CAP + 1
    lut_flat = LOOK_SCHED_LUT_BY_HZ[hz].to(dev).reshape(N_DUR_BUCKETS * _D1 * _D1)
    amp_by_cls = torch.from_numpy(_AMP_BY_CLASS_BY_HZ[hz]).to(dev)
    dir_ctr = torch.from_numpy(_DIR_CENTERS).to(dev)
    lo = torch.as_tensor(_LO_BY_HZ[hz], dtype=torch.long, device=dev)
    hi = torch.as_tensor(_HI_BY_HZ[hz], dtype=torch.long, device=dev)

    cls = commit_state[:, 0].round().to(torch.int64)
    rem = commit_state[:, 1].round().to(torch.int64)
    prev_elapsed = commit_state[:, 2].round().to(torch.int64)
    prev_bkt = commit_state[:, 3].round().to(torch.int64)
    prev_dir = commit_state[:, 4].round().to(torch.int64)
    need = rem <= 0                                              # (B,) onset this tick

    joint = seg_logits[:, :JOINT]                               # (B, JOINT)
    dir_l = seg_logits[:, JOINT:JOINT + N_LOOK_DIR]             # (B, N_LOOK_DIR)
    if greedy:
        jidx = joint.argmax(dim=-1)
        didx = dir_l.argmax(dim=-1)
    else:
        jidx = gumbel_argmax(joint)
        didx = gumbel_argmax(dir_l)
    new_cls = torch.div(jidx, N_DUR_BUCKETS, rounding_mode="floor")   # 0=hold, 1..8 stroke
    new_bkt = jidx - new_cls * N_DUR_BUCKETS
    lo_b = lo.index_select(0, new_bkt)
    hi_b = hi.index_select(0, new_bkt)
    if greedy:
        new_dur = torch.div(lo_b + hi_b, 2, rounding_mode="floor")
    else:
        u = torch.rand(B, device=dev).clamp_(0.0, 1.0 - 1e-9)  # ORT RandomUniformLike
        span = (hi_b - lo_b + 1).to(torch.float32)
        new_dur = lo_b + torch.minimum((u * span).to(torch.long), hi_b - lo_b)

    # onset on `need` rows; else keep the active commitment (matches eager where-law)
    cls_o = torch.where(need, new_cls, cls)
    bkt_o = torch.where(need, new_bkt, prev_bkt)
    dir_o = torch.where(need, didx, prev_dir)
    rem_o = torch.where(need, new_dur, rem)
    elapsed_o = torch.where(need, torch.zeros_like(rem_o), prev_elapsed)

    dur = (elapsed_o + rem_o).clamp(1, DUR_CAP)
    bkt_c = bkt_o.clamp(0, N_DUR_BUCKETS - 1)
    el_c = elapsed_o.clamp(0, DUR_CAP)
    flat_idx = (bkt_c * _D1 + dur) * _D1 + el_c                 # 3D LUT gather -> flat index_select
    frac = lut_flat.index_select(0, flat_idx)                   # (B,)
    theta = amp_by_cls.index_select(0, cls_o.clamp(0, N_ONSET_CLASSES - 1)) * frac
    phi = dir_ctr.index_select(0, dir_o.clamp(0, N_LOOK_DIR - 1))

    commit_state_out = torch.stack([
        cls_o.to(torch.float32),
        (rem_o - 1).to(torch.float32),
        (elapsed_o + 1).to(torch.float32),
        bkt_o.to(torch.float32),
        dir_o.to(torch.float32),
    ], dim=-1)                                                  # (B, LOOK_COMMIT_STATE_DIM)
    return theta, phi, commit_state_out
