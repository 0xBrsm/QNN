"""XM (explorative-modeling) CONTINUOUS look head — bench ablation arm.

Best-of-K winner-take-all regression of the view-relative turn-delta, as the
counter-hypothesis to the canonical polar head's binning
(:mod:`qnn.model.look_head`). The decomposition is the same one polar uses —

    hold ∈ {0, 1}            (Bernoulli; "don't turn")
    turn ∈ R²                (continuous tangent log-map, ||z|| ≤ π)

— so the two arms score the same event on the same labels: the hold label is
``polar_targets(z)`` mag-bin 0, bit-for-bit the protected hold bin polar
classifies. What differs is the turn: instead of (magnitude × direction)
categoricals, a noise latent is drawn and an MLP REGRESSES the tangent, trained
best-of-K so the head can commit to one mode of a multi-modal turn distribution
instead of averaging across them (a plain L2/L1 regressor lands on the mean of
the modes — the failure the binned heads exist to avoid).

The K candidates are only meaningful as a SET, so there is no closed-form
likelihood here and no ``look_dll``: the reported score is the energy score
(``lookdist_xm_energy_sum``), a proper scoring rule for likelihood-free sample
distributions — lower is better. ``xm_spread`` is the collapse alarm: a head
whose K candidates coincide has degenerated to the point regressor this method
exists to beat, and its energy score is then meaningless.

Input is the shared feature cat, prefix-sliced to the head's declared graph
inputs — same a28 rule as every other head. ACTING samples the hold bit and one
turn candidate (``qnn.model.policy.act``, which routes θ/φ through the shared
``decode_look_from_polar`` override path so the aim-prior blend, feet-aim pitch
and hold semantics are polar's, not a second implementation).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from qnn.model._mlp import make_head_mlp
from qnn.model.look_bins import polar_targets, tangent_expmap, tangent_logmap

# Huber transition, in radians. 0.05 rad ≈ 2.9° — inside it the loss is
# quadratic (sub-degree tracking is fit on its own scale), outside it grows
# linearly so a wrong-mode candidate cannot dominate the batch gradient.
_HUBER_BETA = 0.05
# RELAXED winner-take-all: the losing candidates keep this fraction of the
# gradient (spread over K). Pure WTA (0.0) collapses — candidates that never
# win get no gradient, stop moving, and the head degenerates to one mode.
_WTA_EPSILON = 0.05


@dataclass(frozen=True, slots=True)
class XMLookHeadOutput:
    look_predict: torch.Tensor      # (B*, 3) one-draw reconstruction (diagnostics)
    look_hold_logit: torch.Tensor   # (B*, 1) hold logit (noise-FREE)
    look_features: torch.Tensor     # (B*, in_dim) sliced features, grad-carrying


class XMTangentLookHead(nn.Module):
    """Look head: Bernoulli hold + best-of-K continuous tangent regression."""

    def __init__(
        self, in_dim: int, d_hidden: int, activation: str,
        k_explore: int = 16, d_noise: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.k_explore = int(k_explore)
        self.d_noise = int(d_noise)
        if self.k_explore < 1:
            raise ValueError(f"k_explore must be >= 1, got {self.k_explore}")
        if self.d_noise < 1:
            raise ValueError(f"d_noise must be >= 1, got {self.d_noise}")
        # Hold is NOISE-FREE by construction: "am I turning at all" is not a
        # mode-selection problem, and a noise-driven hold would resample the
        # dominant behaviour of the head every tick.
        self.hold_mlp = make_head_mlp(in_dim, 1, d_hidden, activation)
        # RAW linear output — no tanh/normalize. The label is a tangent log-map
        # vector (||z|| = turn angle ≤ π); squashing would bake a scale the
        # foveated turn distribution does not have.
        self.turn_mlp = make_head_mlp(in_dim + self.d_noise, 2, d_hidden, activation)

    def draw_noise(self, rows: int, k: int, *, ref: torch.Tensor) -> torch.Tensor:
        """(k, rows, d_noise) latent draws on ``ref``'s device/dtype.

        Default global RNG: training seeds are fixed process-wide, and the
        acting path draws its own seeded noise (see ``qnn.model.policy.act``).
        """
        return torch.randn(
            (k, rows, self.d_noise), device=ref.device, dtype=ref.dtype)

    def turn_from_noise(
        self, features: torch.Tensor, noise: torch.Tensor,
    ) -> torch.Tensor:
        """(rows, in_dim) features × (k, rows, d_noise) noise → (k, rows, 2).

        The K candidates are ONE batched MLP pass ((K·rows) rows), not K passes.
        ``noise`` may also be (rows, d_noise) — a single draw, returned (rows, 2).
        """
        single = noise.dim() == 2
        if single:
            noise = noise.unsqueeze(0)
        k, rows = noise.shape[0], noise.shape[1]
        feat = features.reshape(rows, self.in_dim).unsqueeze(0).expand(k, rows, self.in_dim)
        z = self.turn_mlp(
            torch.cat([feat, noise], dim=-1).reshape(k * rows, self.in_dim + self.d_noise)
        ).reshape(k, rows, 2)
        return z[0] if single else z

    def forward(self, features: torch.Tensor) -> XMLookHeadOutput:
        feats = features[..., : self.in_dim]
        hold_logit = self.hold_mlp(feats)                          # (..., 1)
        flat = feats.reshape(-1, self.in_dim)
        z = self.turn_from_noise(flat, self.draw_noise(flat.shape[0], 1, ref=flat)[0])
        # One-draw readout for DIAGNOSTICS only — ACTING samples (policy.act).
        # Hold gating mirrors polar's mag_bin==0 rows: tangent 0 → (1,0,0).
        z = z.reshape(*feats.shape[:-1], 2)
        z = torch.where(hold_logit > 0.0, torch.zeros_like(z), z)
        return XMLookHeadOutput(
            look_predict=tangent_expmap(z),
            look_hold_logit=hold_logit,
            # The loss hook re-forwards the turn MLP with K draws, so the
            # SLICED features are pre-stashed here (forward-scoped, grad
            # intact). Repo policy forbids cross-component forward hooks.
            look_features=feats,
        )

    def look_loss(self, logits, look_label, valid, compute_metrics):
        """Bernoulli hold + best-of-K tangent regression on the valid look
        frames (the look-loss hook contract).

        `logits` carries this head's forwarded outputs; `look_label` is the
        normalized unit turn-delta for EVERY row (invalid rows are filled with
        the no-turn vector by the caller) and `valid` selects the scored rows.
        Like polar, `valid` is folded into per-row weights instead of
        subset-indexing: boolean indexing calls nonzero(), whose device→host
        sync was the profiled training bottleneck.

        The hold target is ``polar_targets(z)[0] == 0`` — the SAME protected
        hold bin polar classifies, so the two arms agree on what a hold is. The
        turn term scores only the non-hold rows (a hold has no direction), K
        candidates per row, relaxed winner-take-all: ``min_k + (ε/K)·Σ_k``.
        """
        feats = logits["_look_features"].reshape(-1, self.in_dim)
        hold_logit = logits["_look_hold_logit"].reshape(-1)
        z_label = tangent_logmap(look_label).reshape(-1, 2)
        mb, _ = polar_targets(z_label)
        hold_label = (mb == 0)
        dt = hold_logit.dtype
        valid_f = valid.reshape(-1).to(dt)
        n_valid = valid_f.sum().clamp(min=1.0)
        bce = F.binary_cross_entropy_with_logits(
            hold_logit, hold_label.to(dt), reduction="none")
        loss = (bce * valid_f).sum() / n_valid

        turn_f = (~hold_label).to(dt) * valid_f
        cand = self.turn_from_noise(
            feats, self.draw_noise(feats.shape[0], self.k_explore, ref=feats))
        per_k = F.smooth_l1_loss(
            cand, z_label.to(cand.dtype).unsqueeze(0).expand_as(cand),
            beta=_HUBER_BETA, reduction="none",
        ).sum(-1)                                                  # (K, R)
        # Weighted-sum over zero rows is exactly zero, so a hold-only (or
        # all-invalid) batch stays on device instead of synchronizing to Python.
        wta = per_k.min(dim=0).values + (_WTA_EPSILON / self.k_explore) * per_k.sum(dim=0)
        loss = loss + (wta * turn_f).sum() / turn_f.sum().clamp(min=1.0)

        metrics = {}
        if compute_metrics:
            # Metrics run on the sampled reporting step only — the subset
            # indexing (and its sync) is off the hot path, and the sums must
            # cover exactly the valid rows.
            metrics["loss_look"] = loss.detach()
            with torch.no_grad():
                from qnn.model.look_bins import N_BINS, bin_targets
                # float32 sums, NOT the head's autocast dtype: these are raw
                # sums the trainer accumulates over the report window
                # (_RAW_SUM_METRIC_PREFIXES), and a bf16 accumulator stops
                # counting once the running total outruns its 8-bit mantissa.
                f32 = torch.float32
                v = valid.reshape(-1)
                p_hold = torch.sigmoid(hold_logit[v]).float()       # (V,)
                zl = z_label[v]
                hl = hold_label[v]
                V = int(zl.shape[0])
                metrics["lookdist_n"] = torch.tensor(
                    float(V), dtype=f32, device=cand.device)
                metrics["lookdist_xm_hold_pred_sum"] = p_hold.sum()
                metrics["lookdist_xm_hold_label_sum"] = hl.to(f32).sum()
                # Turn rows only: the candidate SET is undefined on holds.
                t = v & (~hold_label)
                X = cand[:, t].float()                              # (K, T, 2)
                T = int(X.shape[1])
                metrics["lookdist_xm_turn_n"] = torch.tensor(
                    float(T), dtype=f32, device=cand.device)
                # Collapse alarm: per-row std across the K candidates. →0 means
                # the head degenerated to a point regressor (and the energy
                # score below stops meaning anything).
                # (K=1 has no spread to measure — the degenerate control arm.)
                metrics["lookdist_xm_spread_sum"] = (
                    X.std(dim=0).mean(dim=-1).sum() if (T > 0 and self.k_explore > 1)
                    else torch.zeros((), dtype=f32, device=cand.device))
                # ENERGY SCORE (proper scoring rule, lower=better) — the
                # likelihood-free stand-in for look_dll:
                #   E_k||X_k − z|| − ½·E_{k,k'}||X_k − X_k'||
                # Second term is the standard 1/K² estimator (k=k' included).
                if T > 0:
                    Xr = X.permute(1, 0, 2).contiguous()            # (T, K, 2)
                    zt = z_label[t].float().unsqueeze(1)            # (T, 1, 2)
                    d_lab = torch.linalg.vector_norm(Xr - zt, dim=-1).mean(dim=1)
                    d_self = torch.cdist(Xr, Xr).mean(dim=(1, 2))
                    metrics["lookdist_xm_energy_sum"] = (d_lab - 0.5 * d_self).sum()
                else:
                    metrics["lookdist_xm_energy_sum"] = torch.zeros(
                        (), dtype=f32, device=cand.device)
                # Human binned tangent hist — the SAME lookdist_h_* keys polar
                # emits (label side), plus the model-side marginal under the
                # binned head's existing lookdist_p_* convention. XM has no
                # closed-form bin distribution, so the model mass comes from ONE
                # sampled prediction per valid row (hold ~ Bernoulli(sigmoid),
                # turn = one random candidate) instead of a softmax; both sets
                # still sum to V per axis.
                bz = bin_targets(zl)                                # (V, 2)
                pick = torch.randint(0, self.k_explore, (V,), device=cand.device)
                z_samp = cand[:, v].float()[pick, torch.arange(V, device=cand.device)]
                hold_draw = (torch.rand_like(p_hold) < p_hold).unsqueeze(-1)
                z_samp = torch.where(hold_draw, torch.zeros_like(z_samp), z_samp)
                bp = bin_targets(z_samp)                            # (V, 2)
                for a in (0, 1):
                    h = torch.bincount(bz[:, a], minlength=N_BINS).to(f32)
                    p = torch.bincount(bp[:, a], minlength=N_BINS).to(f32)
                    for b in range(N_BINS):
                        metrics[f"lookdist_h_{a}_{b}"] = h[b].detach()
                        metrics[f"lookdist_p_{a}_{b}"] = p[b].detach()
        return loss, metrics


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("look", "xm_tangent")
def _build_look_xm_tangent(head, dims, d_model):
    # Shared-cat prefix slice: drop target.feat only when the head's declared
    # inputs omit it (the readout edge is validated first/required by spec).
    in_dim = (dims["coord_features_dim"] if "intent" in head.inputs
              else dims["base_features_dim"])
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    # 0 = unspecified in the spec (the `hz` precedent) → the head's own default.
    kwargs = {}
    if head.k_explore:
        kwargs["k_explore"] = head.k_explore
    if head.d_noise:
        kwargs["d_noise"] = head.d_noise
    return XMTangentLookHead(in_dim, head.d_hidden, head.activation, **kwargs)
