"""a25 segment-level look head — joint (onset-class × duration) at look onsets.

The look analogue of ``move_seg_head`` (agents/plans/look-seg-head.md §1): at
each look-segment onset predict the JOINT (onset class, duration bucket) of the
segment that starts there, plus a separate direction categorical at stroke
onsets. Onset class ∈ {hold} ∪ K stroke-amplitude bins (the Phase-0 K=8 grid);
segmentation = hold (θ==0) / stroke (θ>0 turning run, split on direction
reversal). Trains as a PASSENGER on the full graph: the per-frame polar look
head is untouched; this head's commit-decode deployment (Phase 2) comes only
after the offline win vs the A0 references (marginal + log-normal duration law,
runs/head_probe/_look_seg_baseline.json).

Labels are derived ON THE FLY in the loss from the batch's (T, B, 3) look
turn-delta vectors (``actions['look']``) via the same tangent log-map the polar
look head uses — no recollect, no loader hook, no dependence on the unregistered
``look_tan`` sidecars. The segmentation rule is the torch twin of the offline
numpy kernel in ``look_seg_segment.py`` (which the audit/baseline scorer use);
the shared grid constants live in ``look_seg_bins.py``. Segments right-censored
by the valid window are ignored; the first (left-censored) segment of a window
is not an onset.
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.look_bins import tangent_logmap
from qnn.model.look_seg_bins import (  # torch-free home
    JOINT, N_DUR_BUCKETS, N_LOOK_DIR, N_ONSET_CLASSES, LookSegBins, bins_for_hz,
    resolve_hz,
)

_IGNORE = -100
_TWO_PI = 2.0 * torch.pi

_EDGES_CACHE: dict[tuple, torch.Tensor] = {}
_AMP_CACHE: dict[tuple, torch.Tensor] = {}


def bucketize_duration(dur: torch.Tensor, dur_edges: "tuple[int, ...]") -> torch.Tensor:
    """duration frames (>=1) -> bucket 0..N_DUR_BUCKETS-1 (searchsorted over
    ``dur_edges``, the head's resolved ``LookSegBins.dur_edges``). Cached per
    device/dtype/edges (the host→device edge copy would otherwise synchronize
    on every train step)."""
    key = (dur.device.type, dur.device.index, dur.dtype, dur_edges)
    edges = _EDGES_CACHE.get(key)
    if edges is None:
        edges = torch.as_tensor(dur_edges, device=dur.device, dtype=dur.dtype)
        _EDGES_CACHE[key] = edges
    return (torch.searchsorted(edges, dur, right=True) - 1).clamp(0, N_DUR_BUCKETS - 1)


def _amp_centers(device: torch.device, dtype: torch.dtype,
                  amp_centers_rad: "tuple[float, ...]") -> torch.Tensor:
    key = (device.type, device.index, dtype, amp_centers_rad)
    c = _AMP_CACHE.get(key)
    if c is None:
        c = torch.as_tensor(amp_centers_rad, device=device, dtype=dtype)
        _AMP_CACHE[key] = c
    return c


# compiler.disable: the flipped-cummin scan trips an Inductor codegen bug on
# ROCm (the exact quarantine move_seg's derive_segment_targets documents).
# Grad-free integer/label prep — excluding just this keeps the loss compiled.
@torch.compiler.disable
def derive_look_seg_targets(
    look_vec: torch.Tensor,       # (T, B, 3) float — per-frame turn-delta unit vec
    valid: torch.Tensor,          # (T, B) bool
    bins: LookSegBins,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (joint, dir) long targets, both (T, B):

    * ``joint`` = onset_class * N_DUR_BUCKETS + dur_bucket at onset frames,
      -100 elsewhere (non-onsets, lane starts, right-censored segments).
    * ``dir``   = uniform direction bin at STROKE onsets, -100 elsewhere
      (holds carry no direction).

    Segmentation twin of ``look_seg_segment.segment_engaged`` — hold (θ==0) vs
    stroke (θ>0), a turning run split where |Δφ| between consecutive tangent
    vectors exceeds ``bins.reversal_rad``. Right-censoring is identical to
    move_seg's (a segment is complete only if the next onset arrives before
    any invalid frame or the window edge)."""
    T, B, _ = look_vec.shape
    z = tangent_logmap(look_vec.float())               # (T, B, 2)
    theta = torch.linalg.vector_norm(z, dim=-1)        # (T, B)
    turn = (theta > 0) & valid                         # (T, B) bool

    # Change points: hold<->turn transitions OR a within-turn reversal.
    chg = torch.zeros_like(valid)
    if T >= 2:
        z0, z1 = z[:-1], z[1:]
        dot = (z0 * z1).sum(dim=-1)                     # (T-1, B)
        crs = z0[..., 0] * z1[..., 1] - z0[..., 1] * z1[..., 0]
        dphi = torch.atan2(crs.abs(), dot)             # (T-1, B) in [0, pi]
        vv = valid[1:] & valid[:-1]
        state_chg = turn[1:] != turn[:-1]
        rev_chg = turn[1:] & turn[:-1] & (dphi > bins.reversal_rad)
        chg[1:] = vv & (state_chg | rev_chg)

    idx = torch.arange(T, device=z.device, dtype=torch.long).unsqueeze(1).expand(T, B)
    sentinel = torch.full((1, B), T, dtype=torch.long, device=z.device)

    def _next_after(mask: torch.Tensor) -> torch.Tensor:
        pos = torch.where(mask, idx, torch.full_like(idx, T))
        rev = torch.flip(torch.cat([pos[1:], sentinel], dim=0), [0])
        return torch.flip(torch.cummin(rev, dim=0).values, [0])

    next_chg = _next_after(chg)                         # (T, B)
    next_blk = _next_after(~valid)
    dur = torch.where(
        (next_chg < T) & (next_chg < next_blk),
        next_chg - idx,
        torch.zeros_like(next_chg),
    )
    onset = chg & (dur > 0)                             # onset with complete segment

    # Stroke amplitude = Σ θ over [t, t+dur) via exclusive prefix sum.
    pexcl = torch.zeros((T + 1, B), device=z.device, dtype=theta.dtype)
    pexcl[1:] = torch.cumsum(theta, dim=0)
    end_idx = (idx + dur).clamp(max=T)
    amp = pexcl.gather(0, end_idx) - pexcl.gather(0, idx)   # (T, B)

    centers = _amp_centers(z.device, amp.dtype, bins.amp_centers_rad)    # (K,)
    amp_bin = (amp.unsqueeze(-1) - centers).abs().argmin(dim=-1)   # (T, B) in 0..K-1
    is_hold = ~turn                                        # state at t (valid onsets only)
    cls = torch.where(is_hold, torch.zeros_like(amp_bin), amp_bin + 1)
    joint = cls * N_DUR_BUCKETS + bucketize_duration(dur.clamp(min=1), bins.dur_edges)
    joint_t = torch.where(onset, joint, torch.full_like(joint, _IGNORE))

    # Direction (stroke onsets only): uniform bin of the onset-frame tangent.
    phi = torch.atan2(z[..., 1], z[..., 0])               # (T, B) in (-pi, pi]
    phi = torch.remainder(phi, _TWO_PI)
    dir_bin = (phi / (_TWO_PI / N_LOOK_DIR)).long().clamp(0, N_LOOK_DIR - 1)
    dir_t = torch.where(onset & turn, dir_bin, torch.full_like(dir_bin, _IGNORE))
    return joint_t, dir_t


class LookSegHead(nn.Module):
    """MLP: motor features (readout [+ target.feat]) -> (B, JOINT + N_LOOK_DIR).

    First JOINT logits = the (onset-class × duration-bucket) joint; the tail
    N_LOOK_DIR logits = the direction categorical scored at stroke onsets.

    ``hz`` is the RESOLVED corpus tick rate (see ``look_seg_bins.resolve_hz``
    — the builder resolves HeadNodeSpec.hz==0 to LEGACY_HZ before this ctor
    runs, so ``hz`` here is always a concrete, table-backed value). Fixed for
    the head's lifetime: labels and decode both read ``self.bins``, so
    training and inference can never disagree about which grid a checkpoint
    was fit against."""

    def __init__(self, *, in_dim: int, d_hidden: int, activation: str, hz: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hz = int(hz)
        self.bins = bins_for_hz(self.hz)
        self.mlp = make_head_mlp(in_dim, JOINT + N_LOOK_DIR, d_hidden, activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features[..., : self.in_dim])

    # -- owned loss (dispatched from QNNPolicy._compute_head_losses_and_metrics)

    def look_seg_loss(
        self,
        logits,                      # full logits dict; ours under "look_seg"
        actions,
        valid_mask: torch.Tensor | None,
        compute_metrics: bool,
        dir_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        seg_logits = logits["look_seg"]                   # (B*, JOINT + N_LOOK_DIR)
        look = actions["look"]
        look_t = look if isinstance(look, torch.Tensor) else torch.as_tensor(look)
        if look_t.ndim != 3:
            # frame-shuffled batch: no time axis -> no segments to supervise
            z = seg_logits.sum() * 0.0
            return z, ({"loss_look_seg": z.detach()} if compute_metrics else {})
        look_t = look_t.to(device=seg_logits.device, dtype=torch.float32)
        T, B = look_t.shape[0], look_t.shape[1]
        v = (valid_mask.to(device=seg_logits.device).bool() if valid_mask is not None
             else torch.ones(T, B, dtype=torch.bool, device=seg_logits.device))
        joint_t, dir_t = derive_look_seg_targets(look_t, v, self.bins)

        flat = seg_logits.reshape(-1, JOINT + N_LOOK_DIR)
        joint_l = flat[:, :JOINT]
        dir_l = flat[:, JOINT:]
        flat_joint_t = joint_t.reshape(-1)
        flat_dir_t = dir_t.reshape(-1)

        ce_joint = torch.nn.functional.cross_entropy(
            joint_l, flat_joint_t, ignore_index=_IGNORE, reduction="sum")
        n_joint = (flat_joint_t != _IGNORE).sum()
        joint_loss = ce_joint / n_joint.clamp(min=1)

        ce_dir = torch.nn.functional.cross_entropy(
            dir_l, flat_dir_t, ignore_index=_IGNORE, reduction="sum")
        n_dir = (flat_dir_t != _IGNORE).sum()
        dir_loss = ce_dir / n_dir.clamp(min=1)

        # Direction is an auxiliary term (mirrors move_seg's ud axis): additive,
        # weight-scaled, and it never dilutes the joint skill that gates.
        loss = joint_loss + dir_loss * float(dir_weight)

        metrics: dict = {}
        if compute_metrics:
            metrics["loss_look_seg"] = joint_loss.detach()
            metrics["look_seg_nll"] = joint_loss.detach()
            metrics["look_seg_nll_dir"] = dir_loss.detach()
            metrics["look_seg_n"] = n_joint.detach().to(seg_logits.dtype)
            # look_seg_skill sufficient stats (research/head-metrics.md): clean
            # CE sum + true joint-class histogram over onset frames. The
            # supervised loop sums these across batches and derives
            # H_marg / dll / look_seg_skill on the common head ruler.
            metrics["looksegdist_ce_sum"] = ce_joint.detach()
            metrics["looksegdist_n"] = n_joint.detach().to(seg_logits.dtype)
            valid_joint = flat_joint_t[flat_joint_t != _IGNORE]
            hist = torch.bincount(valid_joint, minlength=JOINT).to(seg_logits.dtype)
            for j in range(JOINT):
                metrics[f"looksegdist_h_{j}"] = hist[j]
            # Direction skill (stroke onsets only) — reported, non-gating.
            metrics["looksegdirdist_ce_sum"] = ce_dir.detach()
            metrics["looksegdirdist_n"] = n_dir.detach().to(seg_logits.dtype)
            valid_dir = flat_dir_t[flat_dir_t != _IGNORE]
            dh = torch.bincount(valid_dir, minlength=N_LOOK_DIR).to(seg_logits.dtype)
            for d in range(N_LOOK_DIR):
                metrics[f"looksegdirdist_h_{d}"] = dh[d]
        return loss, metrics


# -- graph node registration --------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("look_seg", "canonical")
def _build_look_seg(head, dims, d_model):
    # readout-first feature layout lets a prefix slice drop target_feat when the
    # probe declares inputs=["readout"] (mirrors move_seg's builder).
    # coord_features_dim == base_features_dim unless a shared attack-intent
    # block is spliced in (network.slot_dims); this head is a CONSUMER of it.
    in_dim = dims["coord_features_dim"]
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    # head.hz == 0 means the graph doesn't stamp it (every pre-fix probe.json,
    # never written by this line before hz-parameterization) -> resolve_hz
    # maps that to LEGACY_HZ (10), the grid every such checkpoint actually
    # trained against.
    return LookSegHead(in_dim=in_dim, d_hidden=head.d_hidden,
                       activation=head.activation, hz=resolve_hz(head.hz))
