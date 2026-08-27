"""a25 land-jump head — 2-class per-tick jump predictor (head type ``"jump"``).

The move-arch consolidation (agents/plans/a25-move-arch-consolidation.md)
splits the legacy ud axis's double duty: swim (a held, water-only segment)
goes to the segment head; the land jump — an EDGE event, press length p50 =
1 tick / p99 = 3 in the corpus (runs/head_probe/_ud_axis_audit.json) — gets
this dedicated binary head.

Label and scoring population (v2, engine-OUTCOME convention — the house
input_mask pattern; user call 2026-07-10):

    y      = op_input bit 2 AND NOT water   ("a land jump actually FIRED
             this tick" — computed IN THE ENGINE at collect by the debugged
             QNN_PackOpInput path: jump_press && pmove-eval)
    scored = NOT water AND valid_mask       (water = input_mask bits 5|6;
             the seg head's water-ud axis owns those frames — the scored
             population equals the head's decode domain)

Negatives on frames where a press fired nothing (held button swallowed by
the engine's anti-pogo debounce, airborne, etc.) are CORRECT signal —
"choosing jump now is wrong" — and where the reason is observable the model
learns the reason; where it isn't (the hold) it learns the rate, and
sampling that rate reproduces the human per-context jump frequency. No
feasibility gate exists at decode: the posterior prices the debounce in,
and the engine enforces it mechanically.

Loss is a plain BCE for the clean likelihood (skill suffstats want it, and
the eventual decode wants a calibrated posterior — judge on calibration and
sampled rate, never argmax F1: the movedist analysis showed jump "collapse"
is an argmax artifact). ``pos_weight`` (graph-spec param, default 1.0 = off)
multiplies the positive-frame BCE terms for the ~2.4% pos rate, applied to
the TRAINING term only — the ``jumpdist_*`` skill stats always use the clean
unweighted BCE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp

_IGNORE = -100


class JumpHead(nn.Module):
    def __init__(self, *, in_dim: int, d_hidden: int, activation: str,
                 pos_weight: float = 1.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, 1, d_hidden, activation)
        self.pos_weight = float(pos_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features[..., : self.in_dim]).squeeze(-1)  # (B*,)

    # -- owned loss (dispatched via the policy's ``jump_loss`` hook) ---------

    def jump_loss(
        self,
        logits,                      # full logits dict; ours under "jump"
        actions,
        valid_flat: torch.Tensor | None,
        compute_metrics: bool,
    ) -> tuple[torch.Tensor, dict]:
        jl = logits["jump"].reshape(-1)
        dev = jl.device
        if "op_input" not in actions or "input_mask" not in actions:
            z = jl.sum() * 0.0
            return z, ({"loss_jump": z.detach()} if compute_metrics else {})

        def _flat(v, dtype):
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            return t.to(device=dev, dtype=dtype).reshape(-1)

        im = _flat(actions["input_mask"], torch.long)
        water = (((im >> 5) & 1) | ((im >> 6) & 1)) != 0
        ud_op = (_flat(actions["op_input"], torch.long) >> 2) & 1
        y = (ud_op != 0) & ~water                 # land jump OUTCOME
        y = y.to(torch.float32)
        scored = ~water
        if valid_flat is not None:
            scored = scored & valid_flat.to(device=dev).bool().reshape(-1)
        n = scored.sum().clamp_min(1).to(jl.dtype)

        bce = F.binary_cross_entropy_with_logits(
            jl, y, reduction="none")                       # clean, per-frame
        term = bce
        if self.pos_weight != 1.0:
            w = torch.where(y > 0.5, bce.new_full((), self.pos_weight),
                            bce.new_ones(()))
            term = bce * w
        sm = scored.to(term.dtype)
        loss = (term * sm).sum() / n

        metrics: dict = {}
        if compute_metrics:
            metrics["loss_jump"] = loss.detach()
            # jump_skill sufficient stats: clean BCE sum + pos count → the
            # binary-marginal skill deriver (same ruler as attack_skill).
            metrics["jumpdist_ce_sum"] = (bce * sm).sum().detach()
            metrics["jumpdist_n"] = n.detach()
            metrics["jumpdist_pos"] = (y * sm).sum().detach()
            # DECISION-ONLY population → jump_op_skill. `scored = ~water`
            # excludes water frames but NOT airborne ones, where a ground jump
            # is impossible and `y` is therefore a deterministic 0. Measured on
            # qwd_v5main val: 316,282 of 942,841 scored frames (33.6%) are
            # ground-jump-infeasible and carry EXACTLY 0 positives, dragging the
            # base rate 1.604% -> 1.066% and h_marg 0.0822 -> 0.0590.
            # Emitted alongside, never replacing — the committed records were all
            # measured on the wider population.
            jump_feas = ((im >> 7) & 1) != 0          # input_mask bit 7
            sm_op = (scored & jump_feas).to(term.dtype)
            metrics["jumpdist_op_ce_sum"] = (bce * sm_op).sum().detach()
            metrics["jumpdist_op_n"] = sm_op.sum().clamp_min(1).detach()
            metrics["jumpdist_op_pos"] = (y * sm_op).sum().detach()
            with torch.no_grad():
                p = torch.sigmoid(jl)
                pred = (p > 0.5) & scored
                pos = (y > 0.5) & scored
                metrics["tp_jump"] = (pred & pos).sum().to(jl.dtype)
                metrics["fp_jump"] = (pred & ~pos).sum().to(jl.dtype)
                metrics["fn_jump"] = (~pred & pos).sum().to(jl.dtype)
                # Calibration pair: expected sampled rate vs the human rate on
                # the SAME scored population — the number the decode fit reads.
                # (pred_rate_/pos_rate_ prefixes are epoch-averaged by the loop.)
                metrics["pred_rate_jump"] = ((p * sm).sum() / n).detach()
                metrics["pos_rate_jump"] = ((y * sm).sum() / n).detach()
        return loss, metrics


# -- graph node registration --------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("jump", "canonical")
def _build_jump(head, dims, d_model):
    # Same readout-first prefix-slice contract as move_seg: dropping
    # target.feat from inputs shrinks the consumed prefix.
    # coord_features_dim == base_features_dim unless a shared attack-intent
    # block is spliced in (network.slot_dims); this head is a CONSUMER of it.
    in_dim = (dims["coord_features_dim"] if "intent" in head.inputs
              else dims["base_features_dim"])
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    # Spec default pos_weight=0.0 means "unset" → no reweighting (1.0).
    pw = float(getattr(head, "pos_weight", 0.0))
    return JumpHead(in_dim=in_dim, d_hidden=head.d_hidden,
                    activation=head.activation,
                    pos_weight=pw if pw > 0.0 else 1.0)
