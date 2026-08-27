"""a27 MTP future-attack aux head — censored time-to-next-op-discharge
(head type ``"attack_future"``).

Phase 1a of agents/plans/cross-head-coordination.md, specified in
agents/plans/mtp-attack-future-probe.md. ONE 5-way categorical per frame
predicting how far away the next operative discharge is (see
:mod:`qnn.model.attack_future_bins` for the bucket contract). The point is not
the head's own accuracy: forcing the trunk to encode "attack imminent" BEFORE
it happens gives the look/move heads that signal through the shared readout.

TRAINING-ONLY. The head reads ``features_base_flat`` (readout [+ target.feat]),
deliberately NOT the motor vector — the motor vector carries
``weapon_context = softmax(attack_logits) @ embed``, so feeding it would let
this head's gradient flow back into the attack head and move the very attack
marginal the probe is meant to hold fixed. It is dropped at export
(``Network.aux_training_heads = False``), never rides the wire, and never
participates in decode.

Metrics keep two rulers on one softmax:
  * ``attackfuturedist_*``      — clean 5-way CE + true-class histogram →
    ``attack_future_skill`` on the common head ruler; the ``_op_`` twin is the
    same block scored on operative frames only (input_mask bit 0), the
    jump_head.py decision-only pattern.
  * ``attackfuturemargdist_*``  — the binary marginal P(fire ≤ HORIZON) = 1−p₀
    (the attack_with_head.py dual-ruler trick) → ``attack_future_marg_skill``,
    a rate the coordination analysis can read directly.

Loss is a plain CE: the dominant class is ~88%, comparable to look_seg's hold
class, so the primary arm needs no reweighting. ``pos_weight`` (graph-spec
param, default 1.0 = off) multiplies the event-class (1..4) terms as an escape
hatch if epoch-1 recall collapses to class 0; it applies to the TRAINING term
only — the ``attackfuture*dist_*`` suffstats always use the clean CE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.model._mlp import make_head_mlp
from qnn.model.attack_future_bins import IGNORE as _IGNORE, N_CLASSES

# The action-stream column the BC source derives per episode (see
# qnn.bc.supervised_loop) and this head consumes.
ACTION_KEY = "attack_future_bucket"


class AttackFutureHead(nn.Module):
    """MLP: base features (readout [+ target.feat]) -> (B*, N_CLASSES)."""

    def __init__(self, *, in_dim: int, d_hidden: int, activation: str,
                 pos_weight: float = 1.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, N_CLASSES, d_hidden, activation)
        self.pos_weight = float(pos_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Same readout-first prefix-slice contract as move_seg / jump.
        return self.mlp(features[..., : self.in_dim])

    # -- owned loss (dispatched via the policy's ``attack_future_loss`` hook) --

    def attack_future_loss(
        self,
        logits,                      # full logits dict; ours under "attack_future"
        actions,
        valid_flat: torch.Tensor | None,
        compute_metrics: bool,
    ) -> tuple[torch.Tensor, dict]:
        af = logits["attack_future"].reshape(-1, N_CLASSES)
        dev = af.device
        if ACTION_KEY not in actions:
            # Corpus/loader without the derived column: contribute nothing
            # rather than fabricating a label (look_seg's no-op guard).
            z = af.sum() * 0.0
            return z, ({"loss_attack_future": z.detach()} if compute_metrics else {})

        def _flat(v, dtype):
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            return t.to(device=dev, dtype=dtype).reshape(-1)

        target = _flat(actions[ACTION_KEY], torch.long)
        if valid_flat is not None:
            target = target.masked_fill(
                ~valid_flat.to(device=dev).bool().reshape(-1), _IGNORE)
        scored = target != _IGNORE
        n = scored.sum().clamp_min(1).to(af.dtype)

        # Per-sample clean CE (suffstats want the unweighted likelihood);
        # ignored rows are clamped to a valid index and zeroed by `scored`.
        ce = F.cross_entropy(af, target.clamp_min(0), reduction="none")
        term = ce
        if self.pos_weight != 1.0:
            w = torch.where(target > 0, ce.new_full((), self.pos_weight),
                            ce.new_ones(()))
            term = ce * w
        sm = scored.to(term.dtype)
        loss = (term * sm).sum() / n

        metrics: dict = {}
        if compute_metrics:
            metrics["loss_attack_future"] = loss.detach()
            # Label-validity backstop (train.py's n_<head>_valid gate): a
            # weighted head that scored zero labels is mis-wired, not slow.
            metrics["n_attack_future_valid"] = scored.sum().to(af.dtype).detach()
            with torch.no_grad():
                safe_t = target.clamp_min(0)
                hist = torch.bincount(
                    safe_t[scored], minlength=N_CLASSES).to(af.dtype)
                metrics["attackfuturedist_ce_sum"] = (ce * sm).sum().detach()
                metrics["attackfuturedist_n"] = n.detach()
                for c in range(N_CLASSES):
                    metrics[f"attackfuturedist_h_{c}"] = hist[c]

                # DECISION-ONLY twin (jump_head.py's jumpdist_op_* pattern):
                # the scored population above is EVERY valid frame — future
                # prediction is not gated by the current tick's cooldown — but
                # the coordination question ("is a discharge coming?") is only
                # meaningful where the model is making fire decisions. Emitted
                # alongside, never replacing.
                if "input_mask" in actions:
                    op = (_flat(actions["input_mask"], torch.long) & 1) != 0
                    scored_op = scored & op
                    sm_op = scored_op.to(term.dtype)
                    hist_op = torch.bincount(
                        safe_t[scored_op], minlength=N_CLASSES).to(af.dtype)
                    metrics["attackfuturedist_op_ce_sum"] = (ce * sm_op).sum().detach()
                    metrics["attackfuturedist_op_n"] = sm_op.sum().clamp_min(1).detach()
                    for c in range(N_CLASSES):
                        metrics[f"attackfuturedist_op_h_{c}"] = hist_op[c]

                # Binary marginal P(fire within the horizon) = 1 - p0, scored
                # as a clean BCE on the same softmax — the free ruler the
                # attack-with head takes off its 9-way conditional.
                p0 = F.softmax(af, dim=-1)[:, 0].clamp(1e-7, 1.0 - 1e-7)
                y = (target > 0) & scored
                bce = torch.where(y, -torch.log(1.0 - p0), -torch.log(p0))
                metrics["attackfuturemargdist_ce_sum"] = (bce * sm).sum().detach()
                metrics["attackfuturemargdist_n"] = n.detach()
                metrics["attackfuturemargdist_pos"] = y.sum().to(af.dtype).detach()
                # Calibration pair on the same scored population.
                metrics["pred_rate_attack_future"] = (
                    ((1.0 - p0) * sm).sum() / n).detach()
                metrics["pos_rate_attack_future"] = (
                    y.sum().to(af.dtype) / n).detach()
        return loss, metrics


# -- graph node registration --------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("attack_future", "canonical")
def _build_attack_future(head, dims, d_model):
    # Same readout-first prefix-slice contract as move_seg / jump: dropping
    # target.feat from inputs shrinks the consumed prefix.
    in_dim = dims["base_features_dim"]
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    # Spec default pos_weight=0.0 means "unset" → no reweighting (1.0).
    pw = float(getattr(head, "pos_weight", 0.0))
    # ARM COMPARABILITY: restore the RNG stream across this head's
    # construction. Network assigns the aux module LAST so its xavier draw in
    # _init_weights lands after every other module's — but nn.Linear.__init__
    # ALSO draws (reset_parameters), and that draw happens BEFORE _init_weights
    # runs, which shifted every shared tensor in the whole network (measured:
    # 23 of 74 tensors differed from the control arm at the same seed). The aux
    # head's own weights come from the xavier pass, so discarding its
    # constructor draw costs nothing and makes the aux/control/zero-weight arms
    # bit-identical at init. Model construction is CPU-side (build_network runs
    # before .to(device)), so the CPU generator is the whole stream.
    # Pinned by tests/model/test_attack_future_head.py.
    state = torch.random.get_rng_state()
    try:
        return AttackFutureHead(in_dim=in_dim, d_hidden=head.d_hidden,
                                activation=head.activation,
                                pos_weight=pw if pw > 0.0 else 1.0)
    finally:
        torch.random.set_rng_state(state)
