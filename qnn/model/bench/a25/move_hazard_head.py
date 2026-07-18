"""a25 move-hazard head — the learned WHEN/termination law for move.

This is the a25-generation sixth head. It replaces the *tabulated* semi-Markov
hazard the a24 decode carries (``move_hazard_*`` stamps + the sticky gate) with
a learned, context-conditioned release law on the big trunk.

WHAT vs WHEN
-----------
The move head (3 axes × {neg, none, pos}) stays exactly as-is and stays
**calibrated** — it owns the WHAT (which direction). This head owns only the
**WHEN**: per axis it predicts

    P(release the currently-held class this tick | axis, held_class, dwell_age, CLS)

— a single release-hazard logit per axis. On a decoded release the new class is
sampled from the move head's softmax renormalized over the non-held classes; the
hazard never touches the WHAT distribution. Jump onset is just the ud none-row
of this same surface (held=none on ud, release → pos), so it folds in with no
special case — no separate jump head.

Architecture (see src/docs/move-head.md "the hazard gap")
---------------------------------------------------------
One shared MLP, run once per axis (fb/lr/ud) with an axis one-hot, so the three
axes share statistics while staying independent (no cross-axis term — the move
axes are ~independent, move-head.md §4; ``move_joint`` was the disproven
alternative). Per-axis input:

    [ cls_feat,  one-hot(held_class)=3,  one-hot(axis)=3,  log1p(dwell_age)=1 ]

``dwell_age`` is fed **explicitly** — it is the variable the human dwell hazard
is a function of, and the per-axis ``held_class`` selects the class-conditioned
release law (press rows vs the none row differ). Recurrence is deliberately NOT
added: the trunk's GRU already carries temporal context, and an explicit dwell
scalar leaves nothing for a per-head GRU to integrate — a per-head recurrence
would only re-open the prev-action copy path the move work fenced off
(move-head.md §3). All obs/combat context comes through ``cls_feat`` (the CLS
readout the move head already reads), so the hazard gradient shapes the trunk to
encode dodge-relevant context the static table never could.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model._mlp import make_head_mlp

# Per-axis features appended to the CLS readout: held-class one-hot + axis
# one-hot + the (log1p) dwell-age scalar.
_EXTRA_IN = MOVE_AXIS_CLASSES + MOVE_AXES + 1


@dataclass(frozen=True, slots=True)
class MoveHazardHeadInput:
    cls_feat: torch.Tensor    # (B*, d)        trunk readout (CLS / GRU-integrated CLS)
    held_class: torch.Tensor  # (B*, MOVE_AXES) int — currently-held class per axis
    dwell_age: torch.Tensor   # (B*, MOVE_AXES)     — ticks the held class has run


@dataclass(frozen=True, slots=True)
class MoveHazardHeadOutput:
    hazard_logits: torch.Tensor  # (B*, MOVE_AXES) — per-axis release logit


class MoveHazardHead(nn.Module):
    """Per-axis release-hazard head (the a25 WHEN-law). See module docstring.

    ``in_dim`` is the trunk readout width; the leading ``[..., :in_dim]`` slice
    mirrors the a24 CLS heads (drops a zeroed ``target_feat`` tail when the
    pointer is Off).
    """

    # Exposed so the Network can construct this head's input without importing
    # a bench.a25 type directly — it reads ``self.move_hazard_head.Input`` off the
    # built instance (the head owns its IO contract).
    Input = MoveHazardHeadInput

    def __init__(self, in_dim: int, d_hidden: int, activation: str) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(self.in_dim + _EXTRA_IN, 1, d_hidden, activation)

    def forward(self, inp: MoveHazardHeadInput) -> MoveHazardHeadOutput:
        cls = inp.cls_feat[..., : self.in_dim]                          # (B, d)
        batch = cls.shape[0]
        held = inp.held_class.long().clamp(0, MOVE_AXIS_CLASSES - 1)    # (B, A)
        dwell = inp.dwell_age.to(cls.dtype)                            # (B, A)

        cls_e = cls.unsqueeze(1).expand(batch, MOVE_AXES, cls.shape[-1])      # (B,A,d)
        held_oh = F.one_hot(held, MOVE_AXIS_CLASSES).to(cls.dtype)           # (B,A,3)
        axis_oh = torch.eye(MOVE_AXES, dtype=cls.dtype, device=cls.device) \
            .unsqueeze(0).expand(batch, MOVE_AXES, MOVE_AXES)               # (B,A,A)
        dwell_f = torch.log1p(dwell.clamp(min=0.0)).unsqueeze(-1)           # (B,A,1)

        x = torch.cat([cls_e, held_oh, axis_oh, dwell_f], dim=-1)           # (B,A,d+7)
        logits = self.mlp(x).squeeze(-1)                                    # (B,A)
        return MoveHazardHeadOutput(hazard_logits=logits)

    @staticmethod
    def hazard_loss(
        hazard_logits: torch.Tensor,   # (N, MOVE_AXES)
        release_target: torch.Tensor,  # (N, MOVE_AXES) in {0,1}
        valid: torch.Tensor,           # (N, MOVE_AXES) bool/float mask
        compute_metrics: bool = False,
    ):
        """Calibrated discrete-time hazard loss — masked BCE on the binary
        per-axis release event ``y = 1[class switches next tick]``.

        NO class reweighting: the objective is a *calibrated* release
        probability (the move head's WHAT marginal is calibrated and reweighting
        the hazard would inflate it into over-switching). Judge it on
        AUC/calibration, not F1. ``valid`` masks out-of-distribution frames and
        episode-end (right-censored) frames where no next-tick label exists.
        """
        tgt = release_target.to(hazard_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(hazard_logits, tgt, reduction="none")
        mask = valid.to(hazard_logits.dtype)
        denom = mask.sum().clamp(min=1.0)
        loss = (bce * mask).sum() / denom
        if not compute_metrics:
            return loss, {}
        with torch.no_grad():
            prob = torch.sigmoid(hazard_logits)
            metrics = {
                "loss_move_hazard": loss.detach(),
                # calibration sanity: mean observed vs predicted release rate
                "move_hazard_rate_obs": (tgt * mask).sum() / denom,
                "move_hazard_rate_pred": (prob * mask).sum() / denom,
            }
        return loss, metrics


# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("move_hazard", "canonical")
def _build_move_hazard_canonical(head, dims, d_model):
    return MoveHazardHead(in_dim=dims["motor_in"], d_hidden=head.d_hidden, activation=head.activation)
