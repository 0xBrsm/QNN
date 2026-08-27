"""a25 segment-level move head — joint (class x duration) at segment onsets.

The successor to the retired move_hazard WHEN/WHAT split (research/move-head.md
§8): at each per-axis move segment onset, predict the JOINT (new class,
duration bucket) of the segment that starts there — the attack-with pattern
lifted one level. fb and lr only (ud/jump stays parked). Trains as a
PASSENGER head on the full graph: the per-frame move head is untouched; this
head's deployment integration (semi-Markov commitment decode) comes only
after the offline win vs the A0 references (marginal + log-normal law,
runs/bc/bench/_move_seg_baseline.json).

Labels are derived ON THE FLY in the loss from the batch's (T, B, 3) move
class tensor — no recollect, no loader hook (the failure mode that stalled
move_hazard phase 4). Segments right-censored by the valid window are
ignored; lane starts are not onsets.
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.seg_bins import FIB_EDGES, N_BUCKETS  # torch-free home
N_CLASSES = 3                        # {neg, none, pos}
N_AXES = 2                           # fb, lr
JOINT = N_CLASSES * N_BUCKETS        # 30-way per axis
_IGNORE = -100
_UD_BATCH_FLOOR = 16                 # denominator floor for the rare ud axis


_FIB_EDGES_CACHE: dict[tuple, torch.Tensor] = {}


def bucketize_duration(dur: torch.Tensor) -> torch.Tensor:
    """duration frames (>=1) -> bucket 0..9 (searchsorted over FIB_EDGES)."""
    # Cached per device/dtype: building the edge tensor from the Python list
    # is a synchronizing host→device copy, and this runs on every train step.
    key = (dur.device.type, dur.device.index, dur.dtype)
    edges = _FIB_EDGES_CACHE.get(key)
    if edges is None:
        edges = torch.as_tensor(FIB_EDGES, device=dur.device, dtype=dur.dtype)
        _FIB_EDGES_CACHE[key] = edges
    return (torch.searchsorted(edges, dur, right=True) - 1).clamp(0, N_BUCKETS - 1)


# compiler.disable: the flipped-cummin scan trips an Inductor codegen bug
# (MemoryDep assertion, torch 2.13 ROCm) that would otherwise quarantine the
# ENTIRE compiled loss method to eager. Label derivation is grad-free integer
# prep — excluding just this function keeps the rest of the loss compiled.
@torch.compiler.disable
def derive_segment_targets(
    move_classes: torch.Tensor,   # (T, B, 3) long — per-frame axis classes
    valid: torch.Tensor,          # (T, B) bool
    water: torch.Tensor | None = None,   # (T, B) bool — enables the ud axis
) -> torch.Tensor:
    """(T, B, 2) long joint targets: class*N_BUCKETS + bucket at onset frames,
    -100 everywhere else (non-onsets, lane starts, right-censored segments).

    With ``water`` supplied the output gains a third (ud) column derived the
    same way but valid ONLY on water frames — swim segments censor at water
    exit exactly like window edges (the move-arch consolidation plan). On
    land the ud column is always -100 (the jump head owns vertical there).
    """
    T, B, _ = move_classes.shape
    n_axes = N_AXES + (1 if water is not None else 0)
    out = torch.full((T, B, n_axes), _IGNORE, dtype=torch.long,
                     device=move_classes.device)
    axis_valid = [valid, valid] + ([valid & water.to(valid.dtype).bool()]
                                   if water is not None else [])
    for ai, axis in enumerate((0, 1, 2)[:n_axes]):        # fb, lr[, ud]
        v = axis_valid[ai]
        c = move_classes[..., axis]                       # (T, B)
        chg = torch.zeros_like(v)
        chg[1:] = (c[1:] != c[:-1]) & v[1:] & v[:-1]
        # frames-until-next-change AFTER t; 0 = censored (no change before an
        # invalid frame or the window edge). Closed form of the old per-t
        # reverse scan — a change is only reachable if no invalid frame sits
        # at or before it — via two flipped cummins instead of T sequential
        # steps (the Python loop was ~300 kernel launches per batch).
        idx = torch.arange(T, device=c.device, dtype=torch.long).unsqueeze(1)
        sentinel = torch.full((1, B), T, dtype=torch.long, device=c.device)

        def _next_after(mask: torch.Tensor) -> torch.Tensor:
            pos = torch.where(mask, idx, torch.full_like(idx.expand(T, B), T))
            # min position >= t+1 with mask set; row of sentinels closes the end
            rev = torch.flip(torch.cat([pos[1:], sentinel], dim=0), [0])
            return torch.flip(torch.cummin(rev, dim=0).values, [0])

        next_chg = _next_after(chg)                       # (T, B)
        next_blk = _next_after(~v)
        dur = torch.where(
            (next_chg < T) & (next_chg < next_blk),
            next_chg - idx,
            torch.zeros_like(next_chg),
        )
        ok = chg & (dur > 0)                              # onset with complete segment
        if ai == 2:
            # Swim-DOWN onsets are ignored, not trained: 756 segments in 35M
            # frames is unlearnable, each val down-event is an unbounded-NLL
            # tail that flips ud_skill's sign across seeds (130 of 1,647 val
            # onsets), and the decode story is already natural — in QW you
            # sink by not swimming up. The bot trades intentional diving for
            # a clean up-vs-release axis.
            ok = ok & (c != 0)
        joint = c * N_BUCKETS + bucketize_duration(dur.clamp(min=1))
        out[..., ai] = torch.where(ok, joint, torch.full_like(joint, _IGNORE))
    return out


class MoveSegHead(nn.Module):
    """MLP: motor features (readout [+ target.feat]) -> (B, n_axes, 30) joint
    logits. n_axes = 2 (fb, lr), or 3 with ``water_ud`` — the opt-in swim axis
    supervised on water frames only (default off = byte-identical)."""

    def __init__(self, *, in_dim: int, d_hidden: int, activation: str,
                 water_ud: bool = False) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.water_ud = bool(water_ud)
        self.n_axes = N_AXES + (1 if self.water_ud else 0)
        self.mlp = make_head_mlp(in_dim, self.n_axes * JOINT, d_hidden, activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features[..., : self.in_dim]).reshape(-1, self.n_axes, JOINT)

    # -- owned loss (dispatched from QNNPolicy._compute_head_losses_and_metrics)

    def move_seg_loss(
        self,
        logits,                      # full logits dict; ours under "move_seg"
        actions,
        valid_mask: torch.Tensor | None,
        compute_metrics: bool,
        ud_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        seg_logits = logits["move_seg"]                   # (B*, 2, 30)
        move = actions["move"]
        move_t = move if isinstance(move, torch.Tensor) else torch.as_tensor(move)
        if move_t.ndim != 3:
            # frame-shuffled batch: no time axis -> no segments to supervise
            z = seg_logits.sum() * 0.0
            return z, ({"loss_move_seg": z.detach()} if compute_metrics else {})
        move_t = move_t.to(device=seg_logits.device, dtype=torch.long)
        T, B = move_t.shape[0], move_t.shape[1]
        v = (valid_mask.to(device=seg_logits.device).bool() if valid_mask is not None
             else torch.ones(T, B, dtype=torch.bool, device=seg_logits.device))
        water = None
        if self.water_ud and "input_mask" in actions:
            im = actions["input_mask"]
            im_t = im if isinstance(im, torch.Tensor) else torch.as_tensor(im)
            im_t = im_t.to(device=seg_logits.device, dtype=torch.long).reshape(T, B)
            water = (((im_t >> 5) & 1) | ((im_t >> 6) & 1)) != 0
        targets = derive_segment_targets(move_t, v, water)  # (T, B, n_axes)
        n_axes = targets.shape[-1]
        flat_t = targets.reshape(-1, n_axes)
        flat_l = seg_logits.reshape(-1, self.n_axes, JOINT)
        losses = []
        metrics: dict = {}
        axis_names = ("fb", "lr", "ud")[:n_axes]
        for ai, name in enumerate(axis_names):
            ce = torch.nn.functional.cross_entropy(
                flat_l[:, ai], flat_t[:, ai], ignore_index=_IGNORE, reduction="sum")
            n = (flat_t[:, ai] != _IGNORE).sum()
            if name == "ud":
                # Rare-signal axis (~1 swim onset / 37k rows): a per-batch
                # mean over 0-2 onsets is a huge-variance gradient into the
                # shared trunk — the p3b NaN divergence (run.md post-mortem;
                # last healthy step had n_ud=34 over 1.28M rows). Floor the
                # denominator so a near-empty batch contributes a small,
                # proportional term instead of a full-scale single-sample
                # mean. Metrics/suffstats below stay true-mean (clamp min=1).
                losses.append(ce / n.clamp(min=_UD_BATCH_FLOOR))
            else:
                losses.append(ce / n.clamp(min=1))
            if compute_metrics:
                metrics[f"move_seg_nll_{name}"] = (ce / n.clamp(min=1)).detach()
                metrics[f"move_seg_n_{name}"] = n.detach().to(seg_logits.dtype)
                # move_seg_skill sufficient stats (research/head-metrics.md):
                # clean CE sum + true joint-class histogram over onset frames.
                # supervised_loop sums these across batches and derives
                # H_marg / dll / move_seg_skill on the common head ruler.
                metrics[f"movesegdist_ce_sum_{name}"] = ce.detach()
                metrics[f"movesegdist_n_{name}"] = n.detach().to(seg_logits.dtype)
                valid_targets = flat_t[:, ai][flat_t[:, ai] != _IGNORE]
                hist = torch.bincount(valid_targets, minlength=JOINT).to(seg_logits.dtype)
                for j in range(JOINT):
                    metrics[f"movesegdist_h_{name}_{j}"] = hist[j]
        # loss_move_seg keeps its historical meaning (fb/lr mean) so runs stay
        # comparable across the water_ud transition; the thin swim axis (2.9%
        # of frames; zero-onset batches are common) is a separate ADDITIVE
        # term that never dilutes fb/lr and never gates selection.
        loss = torch.stack(losses[:N_AXES]).mean()
        if compute_metrics:
            metrics["loss_move_seg"] = loss.detach()
        if len(losses) > N_AXES:
            # head_loss_weights key "move_seg_ud" scales the swim term
            # independently (trunk-share control for the rare axis).
            ud_loss = losses[N_AXES] * float(ud_weight)
            loss = loss + ud_loss
            if compute_metrics:
                metrics["loss_move_seg_ud"] = ud_loss.detach()
        return loss, metrics


# -- graph node registration --------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("move_seg", "canonical")
def _build_move_seg(head, dims, d_model):
    # readout-first feature layout lets a prefix slice drop target_feat when
    # the probe declares inputs=["readout"] (pointer present in these graphs).
    # coord_features_dim == base_features_dim unless a shared attack-intent
    # block is spliced in (network.slot_dims); this head is a CONSUMER of it.
    in_dim = dims["coord_features_dim"]
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    return MoveSegHead(in_dim=in_dim, d_hidden=head.d_hidden,
                       activation=head.activation,
                       water_ud=getattr(head, "water_ud", False))
