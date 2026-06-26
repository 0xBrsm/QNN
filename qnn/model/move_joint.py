"""Joint move head — models the full fb/lr/ud distribution instead of three
independent per-axis softmaxes.

STATUS: explored, NOT promoted. This was built to test whether cross-axis
coupling closes a move joint-combo gap. The offline ablation
(scripts/analysis/move_joint_ablation.py) found the gap is negligible: the
human fb/lr/ud axes are nearly independent (total correlation TC ≈ 0.008 nats),
the 0.698 CLS→GRU model's true move_kl_joint is ≈ 0.003 nats (BELOW the TC
ceiling), and pairwise/full coupling buys only ~0.005 nats over the independent
baseline — noise-level. (The earlier "3.3 nat gap" was a combo-ordering bug in
the diagnostic, since fixed — see qnn.model.policy.) So the independent
factorization is correct for this data; this module is kept only as the tested
primitive behind that negative result. Don't wire it into a probe without a
fresh reason.

This head emits logits over the 27 combos in the canonical ordering used
everywhere else in the codebase:

    combo = fb + 3*lr + 9*ud          fb,lr,ud ∈ {0=neg, 1=none, 2=pos}

Three coupling modes (a single knob), increasing in structure:

  * ``none``     — unary only. Joint energy E(i,j,k) = u_fb[i]+u_lr[j]+u_ud[k];
                   softmax over 27 factorizes EXACTLY into the product of three
                   per-axis softmaxes, so this is bit-equivalent to today's
                   independent head. The baseline.
  * ``pairwise`` — unary + three global 3×3 interaction tables (fb×lr, fb×ud,
                   lr×ud; 27 extra params). The minimal augmentation that can
                   represent cross-axis correlation; degrades to ``none`` when
                   the tables are zero. Captures the pairwise structure that
                   dominates Quake movement (diagonal strafe, bunny-hop).
  * ``full``     — an unrestricted 27-way head (feats→27). Can represent any
                   joint, including 3-way coupling; shares no strength across
                   combos with a common class.

The pairwise tables are global (data-independent) parameters by default — the
cleanest "is coupling worth it?" ablation. Feature-conditioned pairwise is a
later refinement.
"""

from __future__ import annotations

import torch
from torch import nn

from qnn.actions import MOVE_AXIS_CLASSES, MOVE_AXES

N_COMBO = MOVE_AXIS_CLASSES ** MOVE_AXES  # 27
COUPLING_MODES = ("none", "pairwise", "full")


def combo_index(move_axes: torch.Tensor) -> torch.Tensor:
    """(N,3) per-axis class indices [fb,lr,ud] → (N,) combo index fb+3*lr+9*ud."""
    return (
        move_axes[..., 0]
        + MOVE_AXIS_CLASSES * move_axes[..., 1]
        + (MOVE_AXIS_CLASSES ** 2) * move_axes[..., 2]
    ).long()


def _to_combo_order(e_natural: torch.Tensor) -> torch.Tensor:
    """(B,3,3,3) energy indexed [fb,lr,ud] → (B,27) flat in combo order.

    Flat combo index must be fb + 3*lr + 9*ud. A C-order reshape of a tensor
    indexed [ud,lr,fb] yields ud*9 + lr*3 + fb = combo, so permute
    [fb,lr,ud]→[ud,lr,fb] first.
    """
    b = e_natural.shape[0]
    return e_natural.permute(0, 3, 2, 1).reshape(b, N_COMBO)


class MoveJointHead(nn.Module):
    """Feature → 27-way joint move logits, with a coupling-mode knob."""

    def __init__(
        self,
        in_dim: int,
        *,
        mode: str = "pairwise",
        d_hidden: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if mode not in COUPLING_MODES:
            raise ValueError(f"mode must be one of {COUPLING_MODES}, got {mode!r}")
        self.mode = mode
        self.in_dim = int(in_dim)
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.trunk = nn.Sequential(nn.Linear(in_dim, d_hidden), act())
        out_dim = N_COMBO if mode == "full" else MOVE_AXES * MOVE_AXIS_CLASSES  # 27 or 9
        self.proj = nn.Linear(d_hidden, out_dim)
        if mode == "pairwise":
            # Global interaction tables. Zero-init → starts exactly at the
            # independent model and only departs if coupling reduces loss.
            self.w_fblr = nn.Parameter(torch.zeros(MOVE_AXIS_CLASSES, MOVE_AXIS_CLASSES))
            self.w_fbud = nn.Parameter(torch.zeros(MOVE_AXIS_CLASSES, MOVE_AXIS_CLASSES))
            self.w_lrud = nn.Parameter(torch.zeros(MOVE_AXIS_CLASSES, MOVE_AXIS_CLASSES))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, 27) unnormalized joint logits in combo order."""
        h = self.trunk(feats)
        z = self.proj(h)
        if self.mode == "full":
            return z  # already (B, 27), combo-ordered by construction of the target
        # Unary energy: E[fb,lr,ud] = u_fb[fb] + u_lr[lr] + u_ud[ud].
        u = z.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)  # (B,3,3) → axes fb,lr,ud
        u_fb, u_lr, u_ud = u[:, 0, :], u[:, 1, :], u[:, 2, :]
        e = (
            u_fb[:, :, None, None]
            + u_lr[:, None, :, None]
            + u_ud[:, None, None, :]
        )  # (B,3,3,3) indexed [fb,lr,ud]
        if self.mode == "pairwise":
            e = e + self.w_fblr[None, :, :, None]   # [fb,lr] over ud
            e = e + self.w_fbud[None, :, None, :]   # [fb,ud] over lr
            e = e + self.w_lrud[None, None, :, :]   # [lr,ud] over fb
        return _to_combo_order(e)
