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

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ── feature layout ────────────────────────────────────────────────────────────
#
# Storage (per-frame npy cache, ~14 B/frame core):
#   self_velocity     fp16 (T, 3)   body-frame, pre-normalized by C worker
#   self_movement_id  uint8 (T,)    0=ground, 1=air, 2+=water
#   look              fp16 (T, 3)   per-emit view delta (forward dot anchor_basis)
#   c_rule_fire (opt) uint8 (T,)
#   c_rule_jump (opt) uint8 (T,)
#   act_move (target) uint8 (T,)    packed (fb | lr<<2 | ud<<4)
#
# Why look as 3-vec rather than yaw_rate scalar: at native-rate emit,
# QNN_FillLookAndSwitch produces look[i] = cur_forward dot anchor_basis[i],
# where anchor advances after every emit (qnn_collect_main.c:972). So at
# resample_hz=0, look is the per-native-frame view delta as a unit
# vector with no atan2 / wrap-seam math required.

VELOCITY_SCALE = 2000.0    # informational; matches QNN_VELOCITY_SCALE
CORE_FEAT_DIM  = 9         # vel(3) + mid_oh(3) + look(3)
BASELINE_DIM   = 6         # per-axis one-hot {neg, none, pos} for fb + lr
BASELINE_EPS   = 0.01      # normalized 20 u/s — same threshold as the bakeoff B variant
GBT_STACK_DIM  = 6         # per-axis softmax probs from GBT (fb 3 + lr 3) as TCN inputs
WEAPON_OH_DIM  = 9         # one-hot of weapon_id ∈ {0..8} (0=none, 1=axe..8=LG)
VEL_CLIP       = 1.0       # clip body-frame normalized vel to [-1, 1] (matches QW max ~700 u/s)


@dataclass(frozen=True)
class FeatureSpec:
    is_firing:         bool = False
    is_jumping:        bool = False
    use_weapon_id:     bool = False   # one-hot of server-held weapon (9 dims)
    use_baseline:      bool = False   # per-frame sign(velocity) baseline concatenated into input
    use_baseline_skip: bool = False   # baseline bypasses encoder and adds directly to output logits
    # When use_baseline_skip is set, restrict the skip to specific axes.
    # Empty means both fb and lr.  Values: "fb", "lr", "fb_lr".
    baseline_skip_axes: str = "fb_lr"
    clip_velocity:     bool = False   # clip body-frame normalized vel to ±VEL_CLIP
    use_gbt_stack:     bool = False   # add 6 GBT softmax probs as input features

    @property
    def dim(self) -> int:
        return (CORE_FEAT_DIM
            + int(self.is_firing)
            + int(self.is_jumping)
            + (WEAPON_OH_DIM if self.use_weapon_id else 0)
            + (BASELINE_DIM if self.use_baseline else 0)
            + (GBT_STACK_DIM if self.use_gbt_stack else 0))

    @property
    def skip_fb(self) -> bool:
        return self.use_baseline_skip and "fb" in self.baseline_skip_axes

    @property
    def skip_lr(self) -> bool:
        return self.use_baseline_skip and "lr" in self.baseline_skip_axes


# ── model hyperparameters ─────────────────────────────────────────────────────

CHANNELS    = 64
TCN_LAYERS  = 6           # dilations 1,2,4,8,16,32 -> +-126-frame RF (~+-1.8s @ 70Hz)
KERNEL_SIZE = 5
DROPOUT     = 0.1

N_CLASSES      = 3                # {0: neg, 1: none, 2: pos}
PREDICTED_AXES = ("fb", "lr")     # ud handled by C-rule, not predicted (legacy)
PREDICTED_AXES_WITH_UD = ("fb", "lr", "ud")


# ── feature build ─────────────────────────────────────────────────────────────

def build_features(
    self_velocity: np.ndarray,                      # (T, 3)
    self_movement_id: np.ndarray,                   # (T,)
    look: np.ndarray,                               # (T, 3)
    *,
    c_rule_fire: np.ndarray | None = None,          # (T,) {0,1}
    c_rule_jump: np.ndarray | None = None,          # (T,) {0,1}
    weapon_id: np.ndarray | None = None,            # (T,) uint8 ∈ {0..8}
    gbt_probs: np.ndarray | None = None,            # (T, 6) softmax probs (fb 3 + lr 3)
    spec: FeatureSpec | None = None,
) -> np.ndarray:
    """(T, spec.dim) float32 input matrix.

    All inputs are already in the right scale: self_velocity is body-frame
    pre-normalized by the C worker; look is unit-norm by construction.
    """
    if spec is None:
        spec = FeatureSpec(
            is_firing=c_rule_fire is not None,
            is_jumping=c_rule_jump is not None,
            use_weapon_id=weapon_id is not None,
            use_gbt_stack=gbt_probs is not None,
        )

    vel = np.asarray(self_velocity, dtype=np.float32)
    if spec.clip_velocity:
        vel = np.clip(vel, -VEL_CLIP, VEL_CLIP)
    mid = np.asarray(self_movement_id, dtype=np.int32).reshape(-1)
    mid_oh = np.stack([
        (mid == 0).astype(np.float32),
        (mid == 1).astype(np.float32),
        (mid >= 2).astype(np.float32),
    ], axis=1)
    lk = np.asarray(look, dtype=np.float32)

    parts: list[np.ndarray] = [vel, mid_oh, lk]

    if spec.is_firing:
        if c_rule_fire is None:
            raise ValueError("spec.is_firing set but c_rule_fire is None")
        parts.append(np.asarray(c_rule_fire, dtype=np.float32).reshape(-1, 1))
    if spec.is_jumping:
        if c_rule_jump is None:
            raise ValueError("spec.is_jumping set but c_rule_jump is None")
        parts.append(np.asarray(c_rule_jump, dtype=np.float32).reshape(-1, 1))
    if spec.use_weapon_id:
        if weapon_id is None:
            raise ValueError("spec.use_weapon_id set but weapon_id is None")
        wid = np.asarray(weapon_id, dtype=np.int32).reshape(-1)
        wid = np.clip(wid, 0, WEAPON_OH_DIM - 1)
        weapon_oh = np.zeros((wid.shape[0], WEAPON_OH_DIM), dtype=np.float32)
        weapon_oh[np.arange(wid.shape[0]), wid] = 1.0
        parts.append(weapon_oh)
    if spec.use_baseline:
        # Per-axis sign-of-velocity baseline (fb, lr) as 2x one-hot {neg, none, pos}.
        # Same threshold the bakeoff B+shift variant used; lets the model start
        # from the deterministic-rule output and learn to refine it rather than
        # rediscover the sign rule from raw velocity.
        baseline = _baseline_classes_from_vel(vel, eps=BASELINE_EPS)  # (T, 2) {0,1,2}
        baseline_oh = np.zeros((vel.shape[0], BASELINE_DIM), dtype=np.float32)
        for axis in range(2):
            for cls in range(3):
                baseline_oh[:, axis * 3 + cls] = (baseline[:, axis] == cls).astype(np.float32)
        parts.append(baseline_oh)
    if spec.use_gbt_stack:
        if gbt_probs is None:
            raise ValueError("spec.use_gbt_stack set but gbt_probs is None")
        parts.append(np.asarray(gbt_probs, dtype=np.float32).reshape(-1, GBT_STACK_DIM))

    feats = np.concatenate(parts, axis=1).astype(np.float32)
    assert feats.shape[1] == spec.dim, f"feat dim {feats.shape[1]} != spec.dim {spec.dim}"
    return feats


def _baseline_classes_from_vel(vel: np.ndarray, eps: float = BASELINE_EPS) -> np.ndarray:
    """Per-frame sign-of-velocity baseline: (T, 3) -> (T, 2) {0=neg, 1=none, 2=pos}.

    Only fb (axis 0) and lr (axis 1) — same axes the labeler predicts.  ud is
    handled by the C-rule jump detector.
    """
    v = np.asarray(vel, dtype=np.float32)
    fb = np.where(v[:, 0] >  eps, 2, np.where(v[:, 0] < -eps, 0, 1))
    lr = np.where(v[:, 1] >  eps, 2, np.where(v[:, 1] < -eps, 0, 1))
    return np.stack([fb, lr], axis=1).astype(np.int64)


# ── target decode ─────────────────────────────────────────────────────────────

def decode_move_fb_lr(packed: np.ndarray) -> np.ndarray:
    """uint8 packed -> (N, 2) int64 [fb, lr] in {0=neg, 1=none, 2=pos}.

    ud bits (4-5) are dropped; this labeler doesn't predict that axis.
    """
    p = np.asarray(packed, dtype=np.uint8).ravel()
    return np.stack([p & 0x3, (p >> 2) & 0x3], axis=1).astype(np.int64)


def decode_move_fb_lr_ud(packed: np.ndarray) -> np.ndarray:
    """uint8 packed -> (N, 3) int64 [fb, lr, ud] in {0=neg, 1=none, 2=pos}.

    ud encodes the jump press: 0/1=no press (ud=neg never used in QW;
    most frames are none), 2=jump press at this native tick.
    """
    p = np.asarray(packed, dtype=np.uint8).ravel()
    return np.stack([p & 0x3, (p >> 2) & 0x3, (p >> 4) & 0x3], axis=1).astype(np.int64)


# ── model ─────────────────────────────────────────────────────────────────────

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
            padding  = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size, dilation=dilation, padding=padding),
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
    fire = rng.integers(0, 2, size=T, dtype=np.uint8)
    jump = rng.integers(0, 2, size=T, dtype=np.uint8)

    for spec in [
        FeatureSpec(),
        FeatureSpec(is_firing=True, is_jumping=True),
    ]:
        feats = build_features(
            vel, mid, look,
            c_rule_fire=fire if spec.is_firing else None,
            c_rule_jump=jump if spec.is_jumping else None,
            spec=spec,
        )
        model = MoveLabeler(feat_dim=spec.dim)
        x = torch.from_numpy(feats).unsqueeze(0)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"spec={spec}  feat_dim={spec.dim}  "
              f"out=fb{tuple(out['fb'].shape)}, lr{tuple(out['lr'].shape)}  "
              f"params={params}  rf=+-{model.receptive_field}")
