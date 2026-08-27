"""Move-axis labeler — bidirectional inverse dynamics for QWD/MVD.

Bidirectional dilated TCN that recovers move_fb / move_lr from observed
velocity trajectories. Trained on QWD ground truth (real usercmds),
applied to MVD-forced collects to replace velocity-sign labels with
context-aware predictions.

Scope: fb/lr only. ud (jump) and fire stay on deterministic C-side
rules — bakeoff measured ud at 97.5% accuracy via ground->air + vel_z
delta, and the fire gap (precision=0.903, recall=0.656) is structural
button-vs-shot semantics, not a modeling problem.

See:
  - project_seq_labeler_axes        labeler vs policy distinction
  - project_qwd_rate_distribution   corpus design (bc_included + trick at >=70 Hz)
  - scripts/move_inference_bakeoff.md   measured baselines
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The numpy feature/spec/decode layer was factored into qnn.labeler.data so
# the lightgbm GBT path can reuse it without importing torch.  Re-export the
# names here so existing call sites (this module's MoveLabeler, qnn.labeler.
# train, and the gbt_*_stack scripts) are unchanged.
#
# Feature layout (per-frame npy cache, ~14 B/frame core):
#   self_velocity     fp16 (T, 3)   body-frame, pre-normalized by C worker
#   self_movement_id  uint8 (T,)    0=ground, 1=air, 2+=water
#   look              fp16 (T, 3)   per-emit view delta (forward dot anchor_basis)
#   act_move (target) uint8 (T,)    packed press byte (fb|lr<<2|ud<<4 + bits)
#
# Why look as 3-vec rather than yaw_rate scalar: at native-rate emit,
# QNN_FillLook produces look[i] = cur_forward dot anchor_basis[i],
# where anchor advances after every emit (qnn_collect_main.c:972). So at
# resample_hz=0, look is the per-native-frame view delta as a unit vector
# with no atan2 / wrap-seam math required.
from .data import (  # noqa: F401  (re-exported)
    BASELINE_DIM,
    BASELINE_EPS,
    CORE_FEAT_DIM,
    GBT_STACK_DIM,
    N_CLASSES,
    VEL_CLIP,
    VELOCITY_SCALE,
    FeatureSpec,
    _baseline_classes_from_vel,
    build_features,
    decode_move_fb_lr,
    decode_move_fb_lr_ud,
)


# ── model hyperparameters ─────────────────────────────────────────────────────

CHANNELS    = 64
TCN_LAYERS  = 6           # dilations 1,2,4,8,16,32 -> +-126-frame RF (~+-1.8s @ 70Hz)
KERNEL_SIZE = 5
DROPOUT     = 0.1

PREDICTED_AXES = ("fb", "lr")     # ud handled by C-rule, not predicted (legacy)
PREDICTED_AXES_WITH_UD = ("fb", "lr", "ud")


# ── model ─────────────────────────────────────────────────────────────────────

class FastDilatedConv1d(nn.Module):
    """Drop-in for ``nn.Conv1d(dilation=d, padding=(k-1)*d//2)`` (odd kernel),
    computed via space-to-batch so ROCm/MIOpen uses its fast DENSE-conv kernel
    instead of the slow dilated-conv GEMM fallback (~13x faster: dilation>1 hits
    a no-fast-solver path, dilation=1 does not — measured on the 8060S).

    Splits the sequence into ``d`` interleaved phases (elements d apart become
    consecutive), runs a dilation-1 conv on ``(B*d, C, T/d)``, and reshapes
    back.  Numerically identical to the equivalent nn.Conv1d (verified to
    <1e-6).  ``weight``/``bias`` are named to match nn.Conv1d, so checkpoints
    trained with the plain Conv1d encoder load into this module unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self._pad = (kernel_size - 1) // 2   # 'same' pad for the per-phase dense conv
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mirror nn.Conv1d's default initialization so behaviour matches when
        # this replaces a plain Conv1d in fresh runs.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, C, T)
        d = self.dilation
        if d == 1:
            return F.conv1d(x, self.weight, self.bias, padding=self._pad)
        B, C, T = x.shape
        pad = (d - T % d) % d
        if pad:
            x = F.pad(x, (0, pad))          # right-pad to a multiple of d (zeros, as in the conv pad)
        Tp = x.shape[2]
        # (B,C,Tp) -> (B,C,Tp/d,d) -> (B,d,C,Tp/d) -> (B*d, C, Tp/d): phase r = elements r, d+r, 2d+r…
        x = x.reshape(B, C, Tp // d, d).permute(0, 3, 1, 2).reshape(B * d, C, Tp // d)
        y = F.conv1d(x, self.weight, self.bias, padding=self._pad)
        # reshape back so output position i*d+r comes from phase r at index i
        y = y.reshape(B, d, self.out_channels, Tp // d).permute(0, 2, 3, 1).reshape(B, self.out_channels, Tp)
        return y[:, :, :T]


class MoveLabeler(nn.Module):
    """Bidirectional dilated TCN; logits for fb and lr, 3 classes each.

    If ``use_baseline_skip`` is set, the model expects a baseline kwarg to
    forward (per-frame one-hot of the deterministic sign-of-velocity rule
    for fb and lr).  Encoder logits are added to a learned linear projection
    of the baseline, so the model starts by trusting the rule and the
    encoder only has to learn the residual.  Baseline projections are
    initialized to the identity (3x3) so logits at init equal the
    baseline's one-hot — a strong prior the encoder can pull away from.
    """

    def __init__(
        self,
        feat_dim: int = CORE_FEAT_DIM,
        channels: int = CHANNELS,
        n_layers: int = TCN_LAYERS,
        kernel_size: int = KERNEL_SIZE,
        p_drop: float = DROPOUT,
        use_baseline_skip: bool = False,
        baseline_skip_init_scale: float = 3.0,
        skip_fb: bool = True,
        skip_lr: bool = True,
        predict_ud: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric (non-causal) padding")

        layers: list[nn.Module] = []
        in_ch = feat_dim
        for i in range(n_layers):
            dilation = 2 ** i
            layers += [
                # space-to-batch dilated conv — same math as
                # nn.Conv1d(dilation=dilation, padding=padding) but avoids the
                # slow MIOpen dilated-conv fallback (padding is implicit here).
                FastDilatedConv1d(in_ch, channels, kernel_size, dilation),
                nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
            ]
            in_ch = channels
        self.encoder = nn.Sequential(*layers)
        self.head_fb = nn.Linear(channels, N_CLASSES)
        self.head_lr = nn.Linear(channels, N_CLASSES)
        self.predict_ud = predict_ud
        if predict_ud:
            # Jump head — shares the encoder with fb/lr.  ud truth comes from
            # bits 4-5 of move_packed (usercmd.upmove > 0 → ud=pos=jump press).
            self.head_ud = nn.Linear(channels, N_CLASSES)

        self.use_baseline_skip = use_baseline_skip
        self.skip_fb = use_baseline_skip and skip_fb
        self.skip_lr = use_baseline_skip and skip_lr
        if self.skip_fb:
            self.baseline_fb = nn.Linear(N_CLASSES, N_CLASSES, bias=False)
        if self.skip_lr:
            self.baseline_lr = nn.Linear(N_CLASSES, N_CLASSES, bias=False)
        # Identity init: baseline one-hot maps to a logit boost on its
        # own class.  Scale controls the prior strength:
        #   1.0 ≈ softmax probability ~0.58 on the baseline class
        #   3.0 ≈ ~0.85 (consistent winner across the skip-on experiments)
        if self.skip_fb or self.skip_lr:
            with torch.no_grad():
                eye = torch.eye(N_CLASSES) * baseline_skip_init_scale
                if self.skip_fb:
                    self.baseline_fb.weight.copy_(eye)
                if self.skip_lr:
                    self.baseline_lr.weight.copy_(eye.clone())

        self._kernel_size = kernel_size
        self._n_layers    = n_layers

    def forward(self, x: torch.Tensor,
                baseline: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """x: (B, T, F) float -> {"fb": (B, T, 3), "lr": (B, T, 3)} logits.

        If skip_fb or skip_lr is active, ``baseline`` must be (B, T, 6) =
        fb one-hot (3) concatenated with lr one-hot (3).
        """
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        fb_logits = self.head_fb(h)
        lr_logits = self.head_lr(h)
        if self.skip_fb or self.skip_lr:
            if baseline is None:
                raise ValueError("baseline skip enabled but no baseline passed to forward")
            if self.skip_fb:
                fb_oh = baseline[..., :N_CLASSES]
                fb_logits = fb_logits + self.baseline_fb(fb_oh)
            if self.skip_lr:
                lr_oh = baseline[..., N_CLASSES:2 * N_CLASSES]
                lr_logits = lr_logits + self.baseline_lr(lr_oh)
        out = {"fb": fb_logits, "lr": lr_logits}
        if self.predict_ud:
            out["ud"] = self.head_ud(h)
        return out

    @property
    def receptive_field(self) -> int:
        """Half-width of the symmetric receptive field, in frames."""
        return (self._kernel_size - 1) // 2 * (2 ** self._n_layers - 1)


# ── module-level smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    T = 256
    rng = np.random.default_rng(0)

    vel = rng.normal(0, 0.4, (T, 3)).astype(np.float32)
    mid = rng.integers(0, 3, size=T, dtype=np.int32)
    look = rng.normal(0, 0.3, (T, 3)).astype(np.float32)
    look[:, 0] = 1.0 - 0.05 * rng.random(T)  # near-identity
    for spec in [FeatureSpec()]:
        feats = build_features(
            vel, mid, look,
            spec=spec,
        )
        model = MoveLabeler(feat_dim=spec.dim)
        x = torch.from_numpy(feats).unsqueeze(0)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"spec={spec}  feat_dim={spec.dim}  "
              f"out=fb{tuple(out['fb'].shape)}, lr{tuple(out['lr'].shape)}  "
              f"params={params}  rf=+-{model.receptive_field}")
