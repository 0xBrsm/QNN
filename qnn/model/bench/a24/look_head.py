"""Polar (magnitude × direction) look head — the human-likeness parameterization.

Instead of regressing a 3D direction or binning the two tangent axes independently,
this head decomposes the view-relative turn-delta into:

    mag ∈ {hold} ∪ {N_MAG foveated turn magnitudes}   (categorical)
    dir ∈ {N_DIR uniform directions in [0, 2π)}        (categorical)

  mlp_in   = features              (for look_cls: cat(gru_flat, target_feat=0) = MLP(GRU(CLS)))
  h        = mlp(mlp_in)           → split into (N_MAG+1) mag logits + N_DIR dir logits

Why polar (see qnn.model.look_bins): "hold" (don't turn) is a SINGLE protected bin
rather than the coincidence of two center bins, and a flick's yaw/pitch are one
direction (their correlation is represented, not factorized away). This is what lets
SAMPLING reproduce the human turn distribution — the dominant hold mode and directional
flicks both — which the decoded mean / per-axis bins cannot.

The head emits the two logit tensors (consumed by the look loss as hierarchical
cross-entropy) plus a deterministic ``look_predict`` (argmax reconstruction) for
diagnostics. ACTING is done by ``qnn.model.look_bins.polar_sample`` (sample → 3D
unit vector for the engine); PPO log-probs by ``polar_log_prob``.
"""
from __future__ import annotations

import torch
from torch import nn

from qnn.model._mlp import make_head_mlp
import torch.nn.functional as F

from qnn.model.look_bins import (
    N_DIR, N_MAG, polar_targets, polar_to_tangent,
    tangent_expmap, tangent_logmap,
)
from qnn.model.look_head import LookHeadInput, LookHeadOutput


class PurePolarLookHead(nn.Module):
    """Look head: MLP(features) → magnitude + direction categoricals (no prior, no pointer)."""

    def __init__(
        self, in_dim: int, d_hidden: int, activation: str,
    ) -> None:
        super().__init__()
        self.n_mag1 = N_MAG + 1
        self.mlp = make_head_mlp(in_dim, self.n_mag1 + N_DIR, d_hidden, activation)

    def forward(self, inp: LookHeadInput) -> LookHeadOutput:
        feats = inp.features
        h = self.mlp(feats)
        mag_logits = h[..., : self.n_mag1]              # (B*, N_MAG+1)  [0]=hold
        dir_logits = h[..., self.n_mag1:]               # (B*, N_DIR)

        # Deterministic readout for diagnostics only — ACTING samples (polar_sample).
        mag_bin = mag_logits.argmax(dim=-1)
        dir_bin = dir_logits.argmax(dim=-1)
        look_predict = tangent_expmap(polar_to_tangent(mag_bin, dir_bin))
        zero = torch.zeros_like(look_predict)
        return LookHeadOutput(
            look_predict=look_predict, look_prior=zero, look_delta=zero,
            look_mag_logits=mag_logits, look_dir_logits=dir_logits,
        )

    def look_loss(self, logits, look_label, valid, compute_metrics):
        """Hierarchical (magnitude × direction) cross-entropy on the valid look
        frames (the look-loss hook contract).

        `logits` carries this head's forwarded outputs; `look_label` is the
        normalized unit turn-delta for EVERY row (invalid rows are filled with
        the no-turn vector by the caller) and `valid` selects the scored rows.
        The loss folds `valid` into per-row weights instead of subset-indexing:
        boolean indexing calls nonzero(), whose device→host sync was the
        profiled training bottleneck. The label is mapped to the tangent
        log-map and discretized into (mag_bin, dir_bin); the loss is CE over
        the (N_MAG+1) magnitude bins plus — only on the non-hold rows — CE
        over the N_DIR direction bins (direction is undefined for hold). See
        qnn.model.look_bins.
        """
        mag = logits["_look_mag_logits"].reshape(-1, N_MAG + 1)
        dirl = logits["_look_dir_logits"].reshape(-1, N_DIR)
        z = tangent_logmap(look_label)
        mb, db = polar_targets(z)
        valid_f = valid.to(mag.dtype)
        n_valid = valid_f.sum().clamp(min=1.0)
        ce_mag = F.cross_entropy(mag, mb, reduction="none")
        loss = (ce_mag * valid_f).sum() / n_valid
        turn_f = (mb > 0).to(mag.dtype) * valid_f
        # Weighted-sum CE over zero rows is exactly zero. This keeps hold-only
        # batches on device instead of synchronizing `turn.any()` to Python.
        ce_dir = F.cross_entropy(dirl, db, reduction="none")
        loss = loss + (ce_dir * turn_f).sum() / turn_f.sum().clamp(min=1.0)
        metrics = {}
        if compute_metrics:
            # Metrics run on the sampled reporting step only — the subset
            # indexing (and its sync) is off the hot path, and the sums must
            # cover exactly the valid rows as before.
            mag = mag[valid]
            dirl = dirl[valid]
            z = z[valid]
            mb = mb[valid]
            db = db[valid]
            turn = mb > 0
            metrics["loss_look"] = loss.detach()
            # Distributional sufficient stats (additive raw sums, prefix lookdist_).
            # Polar emits model tangent log-density + the same per-axis binned tangent
            # histogram that the binned head uses, so look_dll is on the same
            # tangent-density scale and the two heads are directly comparable.
            with torch.no_grad():
                from qnn.model.look_bins import (
                    BIN_LOG_WIDTH, POLAR_LOG_CELL_AREA, N_BINS, bin_targets,
                )
                V = int(mag.shape[0])
                # Model tangent log-density: log p_mag + log p_dir - log(cell_area).
                # Hold frames (mb==0) carry no direction term.
                lpm = F.log_softmax(mag, dim=-1)
                lpd = F.log_softmax(dirl, dim=-1)
                log_area = POLAR_LOG_CELL_AREA.to(mag.device)[mb]   # (V,)
                logdens = lpm[torch.arange(V, device=mag.device), mb] - log_area
                logdens[turn] = logdens[turn] + lpd[turn, db[turn]]
                metrics["lookdist_polar_logdens_sum"] = logdens.sum().detach()
                metrics["lookdist_n"] = mag.new_tensor(float(V))
                # Human binned tangent hist — same format as binned head.
                bz = bin_targets(z)                                  # (V, 2)
                for a in (0, 1):
                    h = torch.bincount(bz[:, a], minlength=N_BINS).to(mag.dtype)
                    for b in range(N_BINS):
                        metrics[f"lookdist_h_{a}_{b}"] = h[b].detach()
        return loss, metrics


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("look", "polar")
def _build_look_polar(head, dims, d_model):
    return PurePolarLookHead(dims["motor_in"], head.d_hidden, head.activation)
