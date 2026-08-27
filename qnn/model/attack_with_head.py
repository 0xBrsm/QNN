"""A27 attack head — 9-way joint attack decision (attack type
``"attack_with"``).

The reframe (research/weapon-head.md §10-§14 aftermath): holding a weapon has
no foundational value in QW — select-and-fire lands in the same usercmd (the
fire-script mechanic that pollutes ~84% of raw select edges proves it), so the
decision the model owns is "attack WITH weapon k or don't attack", one 9-way
categorical per frame:

    class 0      — no (effective) attack
    class 1..8   — attack, firing weapon k (engine impulse order axe..LG)

This head occupies the graph's ``attack`` slot and owns its loss via the
``attack_loss`` hook that
:meth:`QNNPolicy._compute_head_losses_and_metrics` dispatches before the
legacy binary path. The label is read directly from the A27 action stream:

    label      = act.attack                         (0..8 categorical truth)
    ignore     = padding rows (valid_mask) → -100

The collector records a nonzero attack class only on an effective discharge, so
release-dump selects (the ``-weapon`` script half) self-exclude without exposing
held weapon state to the model.

Loss is a PLAIN cross-entropy (no pos_weight / focal): the probe's decode is
greedy argmax (preserves coordinated commits like the rocket jump), which
wants a calibrated posterior, and the skill metrics below want a clean
likelihood.

Metrics keep both canonical rulers comparable:
  * ``attackdist_*``  — clean binary NLL of the MARGINAL P(attack) = 1 - p0
    → ``attack_skill``, directly comparable to the canonical attack head.
  * ``attackchoicedist_*`` — clean CE of P(impulse | attack), scored on
    true-attack frames.
  * ``tp/fp/fn_attack_<class>`` — conditional-argmax confusion counts on
    true-attack frames; feeds the standard macro-F1 epoch aggregation.
  * ``attack_rate_{human,argmax,mass}`` — the split-mass decode watch-item:
    argmax attacks only when one weapon class alone beats class 0; ``mass``
    is the rate under P(attack) > 0.5. A persistent argmax<<mass gap means
    the conditional is too flat for greedy decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from qnn.bc.cache_align_hbw import SENTINEL as _ALIGN_SENTINEL
from qnn.model import action_labels
from qnn.model._mlp import make_head_mlp
from qnn.model.action_labels import register_label_derive
from qnn.schema import WEAPON_HEAD_SIZE

ATTACK_WITH_SIZE = 1 + WEAPON_HEAD_SIZE  # 9: no-attack + axe..thunderbolt
_IGNORE = -100


def _align_weight(
    align_hbw: torch.Tensor, target: torch.Tensor, gamma: float,
) -> torch.Tensor:
    """Per-frame multiplicative CE weight on POSITIVE (fire) frames from the
    corpus's ``align_hbw`` sidecar (``qnn.bc.cache_align_hbw``):
    ``w = exp(-gamma * hbw)`` — aligned fires (small hbw) get MORE weight,
    wild fires (large hbw) get less. Sentinel rows (no engaged target / no
    attributable weapon at that frame) and every negative (non-attack) frame
    get weight 1.0 — untouched.

    MARGINAL-PRESERVING: rescaled so the positive class's total weight mass
    equals its frame COUNT (the pre-reweight uniform-1.0 mass) — the reweight
    only moves gradient WITHIN the positive class, never the aggregate
    fire-vs-no-fire balance the CE teaches (a28 already under-fires 45-52%;
    this objective must shift WHEN the model fires, never HOW MUCH).
    """
    pos = target > 0
    has_signal = pos & (align_hbw > _ALIGN_SENTINEL)
    raw = torch.where(
        has_signal,
        torch.exp(-gamma * align_hbw.clamp_min(0.0)),
        torch.ones_like(align_hbw),
    )
    n_pos = pos.sum().clamp_min(1).to(raw.dtype)
    pos_mass = (raw * pos.to(raw.dtype)).sum().clamp_min(1e-12)
    scale = n_pos / pos_mass
    return torch.where(pos, raw * scale, torch.ones_like(raw))


@dataclass(frozen=True, slots=True)
class AttackSelectorInput:
    """Input to the categorical 0..8 attack selector."""

    selector: torch.Tensor
    feasibility_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class AttackSelectorOutput:
    """Categorical attack logits. a28: logits ONLY — the selector no longer
    emits a motor-conditioning context (weapon_ctx removed)."""

    logits: torch.Tensor


def _flat_long(value, device: torch.device) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.to(device=device, dtype=torch.long).reshape(-1)


@register_label_derive("attack.v1")
def attack_with_target(
    actions,
    device: torch.device,
    valid_flat: torch.Tensor | None,
    **_unused,
) -> torch.Tensor:
    """(N,) long target in {-100, 0..8} from the action streams.

    Label contract ``attack.v1`` — the A27 discharge-only semantics: the
    action column IS the class, no carried intent, no self-heal.
    """
    target = _flat_long(actions["attack"], device)
    if valid_flat is not None:
        target = target.masked_fill(~valid_flat, _IGNORE)
    return target


# ── Selector metrics ────────────────────────────────────────────────
#
# The metric names are a downstream contract (supervised_loop scans
# tp_<infix>_<cls> prefixes; bc_summary carries every key into run
# records), so the naming lives in one table and the arithmetic exists
# once. Pinned by tests/model/test_selector_metric_keys.py.


@dataclass(frozen=True)
class _SelectorMetricNames:
    """Per-slot metric names for the shared selector metric block."""

    loss: str          # loss_attack        / loss_weapon
    cls_infix: str     # tp_attack_<cls>    / tp_weapon_<cls>
    cond_dist: str     # attackchoicedist_* / weapondist_*
    n_valid: str       # n_attack_choice_valid / n_weapon_valid
    cond_acc: str      # acc_attack_choice  / acc_weapon
    cond_f1: str       # f1_attack_choice   / f1_weapon
    confidence: str    # confidence_attack  / confidence_weapon


_ATTACK_SLOT_NAMES = _SelectorMetricNames(
    loss="loss_attack", cls_infix="attack", cond_dist="attackchoicedist",
    n_valid="n_attack_choice_valid", cond_acc="acc_attack_choice",
    cond_f1="f1_attack_choice", confidence="confidence_attack")

def _feasibility(actions, device) -> "torch.Tensor | None":
    """Per-frame feasibility bit (input_mask bit 0), or None if absent.

    Same derivation the label uses in attack_with_target, surfaced separately so
    the metric block can score the decision-only population.
    """
    if "input_mask" not in actions:
        return None
    return (_flat_long(actions["input_mask"], device) & 1) != 0


def _selector_metrics(
    loss: torch.Tensor,
    aw_logits: torch.Tensor,
    target: torch.Tensor,
    names: _SelectorMetricNames,
    feas: torch.Tensor | None = None,
) -> dict:
    """The 85-key metric block for one 9-class selector slot.

    ``feas`` is the per-frame feasibility bit (input_mask bit 0 — "would
    W_Attack fire if button0=1 this tick"). When supplied, the block adds the
    ``attackdist_op_*`` sufficient stats scored on feasible frames only, i.e.
    the frames where the head's decision can actually actuate. See the comment
    at the emission site.
    """
    from qnn.model.policy import ATTACK_WEAPON_CLASS_NAMES

    metrics: dict[str, torch.Tensor] = {names.loss: loss.detach()}
    with torch.no_grad():
        scored = target != _IGNORE
        n_scored = scored.sum().to(aw_logits.dtype)
        log_probs = F.log_softmax(aw_logits, dim=-1)
        probs = log_probs.exp()
        pred = probs.argmax(dim=-1)
        y_attack = (target > 0) & scored

        # ── marginal attack: binary metrics + attack_skill stats ──
        # Names are SHARED with the other slot by design.
        pred_attack = (pred > 0) & scored
        tp = (pred_attack & y_attack).sum()
        fp = (pred_attack & ~y_attack & scored).sum()
        fn = (~pred_attack & y_attack).sum()
        tn = (~pred_attack & ~y_attack & scored).sum()
        metrics["tp_attack"] = tp.detach()
        metrics["fp_attack"] = fp.detach()
        metrics["fn_attack"] = fn.detach()
        metrics["tn_attack"] = tn.detach()
        n_total = (tp + fp + fn + tn).clamp(min=1)
        metrics["acc_attack"] = ((tp + tn).float() / n_total).detach()
        prec = tp.float() / (tp + fp).clamp(min=1)
        rec = tp.float() / (tp + fn).clamp(min=1)
        metrics["precision_attack"] = prec.detach()
        metrics["recall_attack"] = rec.detach()
        metrics["f1_attack"] = (2.0 * prec * rec / (prec + rec).clamp(min=1e-6)).detach()

        # Clean binary NLL of P(attack) = 1 - p0 → attack_skill,
        # comparable to the canonical attack head's ruler.
        # -log(1-p0) on attack frames, -log(p0) on no-attack frames.
        p0 = probs[:, 0].clamp(1e-7, 1.0 - 1e-7)
        bce = torch.where(y_attack, -torch.log(1.0 - p0), -torch.log(p0))
        metrics["attackdist_ce_sum"] = (bce * scored).sum().detach()
        metrics["attackdist_n"] = n_scored.detach()
        metrics["attackdist_pos"] = y_attack.sum().to(aw_logits.dtype).detach()

        # DECISION-ONLY population → attack_op_skill. `scored` above keeps every
        # frame with a non-IGNORE label, but with input_mask on the label is the
        # engine OUTCOME (feasibility AND demo press), so on an infeasible frame
        # — weapon on cooldown, no ammo, not owned — the label is a
        # deterministic 0. Those are free correct answers: they were 59.8% of the
        # scored frames on qwd_v5main val, dragging the base rate 14.2% -> 5.7%
        # and h_marg 0.409 -> 0.219. That is why exposing attack_finished to the
        # model was historically worth +0.32 attack_skill while decode already
        # gates fires on attack_finished <= 1e-6: it predicted the frames that
        # cannot actuate.
        #
        # Emitted ALONGSIDE the diluted keys, not instead of them: every
        # committed record was measured on the old population and silently
        # redefining the key would invalidate those comparisons.
        if feas is not None:
            scored_op = scored & feas
            n_op = scored_op.sum().to(aw_logits.dtype)
            metrics["attackdist_op_ce_sum"] = (bce * scored_op).sum().detach()
            metrics["attackdist_op_n"] = n_op.detach()
            metrics["attackdist_op_pos"] = (
                (y_attack & scored_op).sum().to(aw_logits.dtype).detach()
            )

        # Decode watch-item: greedy-argmax vs mass vs human attack rate.
        denom = n_scored.clamp(min=1.0)
        metrics["attack_rate_human"] = (y_attack.sum().to(aw_logits.dtype) / denom).detach()
        metrics["attack_rate_argmax"] = (pred_attack.sum().to(aw_logits.dtype) / denom).detach()
        mass_attack = ((1.0 - p0) > 0.5) & scored
        metrics["attack_rate_mass"] = (mass_attack.sum().to(aw_logits.dtype) / denom).detach()

        # ── conditional weapon on true-attack frames ──
        K = WEAPON_HEAD_SIZE
        cond_logp = log_probs[:, 1:] - torch.logsumexp(log_probs[:, 1:], dim=-1, keepdim=True)
        cond_pred = cond_logp.argmax(dim=-1)                     # 0..7
        w_target = (target - 1).clamp(min=0)                     # 0..7 where y_attack
        cond_nll = -cond_logp.gather(1, w_target.unsqueeze(-1)).squeeze(-1)
        metrics[f"{names.cond_dist}_ce_sum"] = (cond_nll * y_attack).sum().detach()
        metrics[f"{names.cond_dist}_n"] = y_attack.sum().to(aw_logits.dtype).detach()

        # Confusion counts on attack frames (sentinel row K = non-attack).
        # Vectorized: 1 scatter_add instead of an 8-iteration Python loop
        # with ~10 tensor ops each — ~5-8s/epoch at bs=4096 on the probe loop.
        safe_t = torch.where(y_attack, w_target, torch.full_like(w_target, K))
        safe_p = torch.where(y_attack, cond_pred, torch.full_like(cond_pred, K))
        flat_idx = (safe_p * (K + 1) + safe_t).long()
        conf = torch.zeros((K + 1) * (K + 1), dtype=torch.float32, device=aw_logits.device)
        conf.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        conf = conf.view(K + 1, K + 1)[:K, :K]
        tp_all = conf.diagonal()
        fp_all = conf.sum(dim=1) - tp_all
        fn_all = conf.sum(dim=0) - tp_all
        valid_count = conf.sum()
        metrics[names.n_valid] = valid_count.detach().to(aw_logits.dtype)
        metrics[names.cond_acc] = (tp_all.sum() / valid_count.clamp(min=1.0)).detach()
        metrics[names.confidence] = probs.max(dim=-1).values.mean().detach()
        infix = names.cls_infix
        class_f1s = []
        for cls_idx, cls_name in ATTACK_WEAPON_CLASS_NAMES:
            tpc, fpc, fnc = tp_all[cls_idx], fp_all[cls_idx], fn_all[cls_idx]
            metrics[f"tp_{infix}_{cls_name}"] = tpc.detach()
            metrics[f"fp_{infix}_{cls_name}"] = fpc.detach()
            metrics[f"fn_{infix}_{cls_name}"] = fnc.detach()
            prec_c = tpc / (tpc + fpc).clamp(min=1.0)
            rec_c = tpc / (tpc + fnc).clamp(min=1.0)
            f1_c = 2.0 * prec_c * rec_c / (prec_c + rec_c).clamp(min=1e-6)
            metrics[f"precision_{infix}_{cls_name}"] = prec_c.detach()
            metrics[f"recall_{infix}_{cls_name}"] = rec_c.detach()
            metrics[f"f1_{infix}_{cls_name}"] = f1_c.detach()
            metrics[f"pos_rate_{infix}_{cls_name}"] = (
                (tpc + fnc) / valid_count.clamp(min=1.0)
            ).detach()
            metrics[f"{names.cond_dist}_h_{cls_idx}"] = (tpc + fnc).detach().to(aw_logits.dtype)
            class_f1s.append(f1_c)
        metrics[names.cond_f1] = torch.stack(class_f1s).mean().detach()
    return metrics


class AttackWithHead(nn.Module):
    """Single MLP: selector (GRU(CLS) readout) → 9 logits.

    Emits the selector output consumed by Network's categorical attack
    slot. a28: logits only — no soft-mix context embedding (weapon_ctx
    removed; the motor heads see exactly their declared graph inputs).

    Realized-alignment edge + DART (``aim_dim`` / ``dart_p``)
    ---------------------------------------------------------
    Alignment edge (fire-at-alignment rung 3 arm A′; supersedes probe B-i's
    3-column form). When the graph declares the ``aim`` edge, Network appends
    a 17-wide alignment block (per-weapon crest payouts, their realized
    one-tick deltas, has_target — see
    ``qnn.model.network.alignment_edge_block``) to the TAIL of this head's
    input cat, so the head's slice is ``(context | aim)`` with
    ``context = in_dim - aim_dim`` (the declared gru / target.feat edges).
    The ``aim2`` edge (A″; crest-ceiling-handoff.md "Candidate next steps"
    §3) extends this to 49 dims — the same 17 columns plus a forward-
    projected tail (``qnn.model.lead_aim.weapon_alignment_projected``) —
    still one tail block, still ``aim_dim`` wide, still gradient-isolated.

    INTENT KEYING — the alignment block's lead geometry is weapon-keyed, and
    ``QNNPolicy._aim_prior_geometry`` keys it at act time by THIS tick's
    attack-with intent. That keying is unavailable HERE: this head *is* the
    intent producer, so a same-tick key would make its own input a function of
    its own logits. The block is therefore keyed by the PREVIOUS tick's attack
    class (``obs['attack_intent_prev']``, the column the ``prev_attack`` intent
    node already plumbs — teacher-forced at train/val). The consequence is a
    one-tick-stale WEAPON key on the geometry; the geometry itself (target
    position, velocity, pointer softmax) is current.

    ``dart_p`` > 0 applies train-time channel dropout to the CONTEXT portion of
    that slice ONLY, never the alignment block and never any other head's
    inputs (the perturbation is on this head's local copy). Under teacher
    forcing the context predicts the human's trigger through the human's aim;
    degrading it is what forces reliance onto the truthful alignment channel.
    Eval/act are untouched (``self.training`` gate ⇒ nn.Dropout identity).
    """

    def __init__(
        self, *, in_dim: int, d_hidden: int, activation: str,
        feasibility_mask: bool = False, focal_gamma: float = 0.0,
        pos_weight: float = 0.0,
        aim_dim: int = 0, dart_p: float = 0.0,
        align_weight_gamma: float = 0.0,
        label: str,
    ) -> None:
        super().__init__()
        # Which action-label semantics this head was/is trained against
        # (qnn.model.action_labels).  REQUIRED and explicit: this module is
        # one architecture that has served three different label contracts
        # across generations, and until this field the binding was implied
        # by the graph slot name — which is how an a27 probe silently
        # trained the selector on a retired all-zero column.  Validated
        # here so a bad contract name fails at construction, not at the
        # first backward pass.
        contract = action_labels.contract(str(label))
        if not contract.selector:
            raise ValueError(
                f"AttackWithHead label must be a selector contract; "
                f"{contract.name!r} is not (selector contracts: "
                f"{list(action_labels.selector_contracts())})"
            )
        if contract.classes != ATTACK_WITH_SIZE:
            raise ValueError(
                f"AttackWithHead emits {ATTACK_WITH_SIZE} logits but label "
                f"contract {contract.name!r} is {contract.classes}-class "
                f"({contract.label})"
            )
        self.label_contract = contract.name
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, ATTACK_WITH_SIZE, d_hidden, activation)
        # Fire-only family calibration.  This intercept is deliberately NOT
        # added to the nine selector logits: those logits also own weapon
        # choice, so shifting them would silently change the selected weapon.
        # Network publishes the vector beside the raw logits and every fire
        # consumer adds only the selected entry to the class-vs-no-attack
        # margin.  Zero preserves older checkpoints exactly.
        self.fire_bias = nn.Parameter(torch.zeros(ATTACK_WITH_SIZE - 1))
        # P1 feasibility-masking (agents/plans/attack-finished-masking-refactor.md).
        # When set, Network builds a per-class additive mask from obs (owned+ammo
        # AND refire-cooldown) and passes it via AttackSelectorInput.feasibility_mask;
        # the head no longer has to LEARN feasibility, so the fragile OOD read
        # (the a1 arsenal-fold collapse) never gets learned. All default-off ⇒
        # byte-identical to the pre-P1 head.
        self.wants_feasibility_mask = bool(feasibility_mask)
        self.focal_gamma = float(focal_gamma)
        self.pos_weight = float(pos_weight)
        # Fire-at-alignment objective knob (agents/plans/
        # fire-at-alignment-objective.md, rung 1). 0.0 = off ⇒ this head's
        # loss never touches actions["align_hbw"] and is byte-identical to
        # a head without the knob.
        self.align_weight_gamma = float(align_weight_gamma)
        if self.align_weight_gamma < 0.0:
            raise ValueError(
                f"align_weight_gamma must be >= 0, got {self.align_weight_gamma}")
        # (context | aim) split of this head's own input slice.
        self.aim_dim = int(aim_dim)
        if not 0 <= self.aim_dim < self.in_dim:
            raise ValueError(
                f"aim_dim {self.aim_dim} must be in [0, in_dim={self.in_dim})")
        self.context_dim = self.in_dim - self.aim_dim
        self.dart_p = float(dart_p)
        if not 0.0 <= self.dart_p < 1.0:
            raise ValueError(f"dart_p must be in [0, 1), got {self.dart_p}")
        # Always constructed (p=0 is an exact identity, adds no parameters and
        # draws no RNG) so the module tree is the same across arms.
        self.context_dropout = nn.Dropout(self.dart_p)

    def dart_context(self, x: torch.Tensor) -> torch.Tensor:
        """DART the CONTEXT sub-slice of ``x`` (``(..., in_dim)``); return the
        re-joined ``(context | aim)`` tensor.

        Off (exact identity, no split, no allocation) when ``dart_p == 0`` or
        in eval mode. The branch is on Python constants / ``self.training``, not
        on tensor data, so torch.compile guards it rather than graph-breaking;
        the mask itself is nn.Dropout's inverted-dropout bernoulli (expectation
        preserved, no data-dependent control flow).
        """
        if self.dart_p == 0.0 or not self.training:
            return x
        if self.aim_dim == 0:
            return self.context_dropout(x)
        return torch.cat(
            [self.context_dropout(x[..., : self.context_dim]),
             x[..., self.context_dim:]],
            dim=-1,
        )

    def forward(self, inp: AttackSelectorInput) -> AttackSelectorOutput:
        logits = self.mlp(self.dart_context(inp.selector[..., : self.in_dim]))
        if inp.feasibility_mask is not None:
            logits = logits + inp.feasibility_mask.to(logits.dtype)
        return AttackSelectorOutput(logits=logits)

    def _focal_ce(
        self, logits: torch.Tensor, target: torch.Tensor,
        align_hbw: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """CE with optional focal down-weighting of easy frames (focal_gamma),
        a multiplicative up-weight on the rare attack positives (classes
        1..8; pos_weight), and an optional marginal-preserving alignment
        reweight of the positives (align_weight_gamma; see ``_align_weight``).
        Reduces to the plain mean CE when all three are 0/None."""
        scored = target != _IGNORE
        denom = scored.sum().clamp_min(1).to(logits.dtype)
        # Clamp ignored (-100) to a valid index for the per-sample CE, then zero
        # them out via `scored` so they never contribute.
        tgt = target.clamp_min(0)
        nll = F.cross_entropy(logits, tgt, reduction="none")  # (N,)
        term = nll
        if self.focal_gamma > 0.0:
            p_true = torch.exp(-nll)
            term = ((1.0 - p_true) ** self.focal_gamma) * nll
        if self.pos_weight > 0.0:
            w = torch.where(target > 0, term.new_full((), self.pos_weight),
                            term.new_ones(()))
            term = term * w
        if self.align_weight_gamma > 0.0 and align_hbw is not None:
            term = term * _align_weight(align_hbw, target, self.align_weight_gamma)
        term = term * scored.to(term.dtype)
        return term.sum() / denom

    # -- owned loss -----------------------------------------------------------

    def attack_loss(
        self,
        logits,
        actions,
        valid_flat: torch.Tensor | None,
        compute_metrics: bool,
        obs=None,
    ) -> tuple[torch.Tensor, dict]:
        from qnn.model.network import ATTACK_HEAD  # lazy: avoid import-order knots

        aw_logits = logits[ATTACK_HEAD].reshape(-1, ATTACK_WITH_SIZE)
        # Target comes from the DECLARED label contract, not from the slot
        # this head happens to occupy.  The slot picks the logits tensor;
        # the contract picks the label semantics.
        target = action_labels.derive_for(self.label_contract)(
            actions, aw_logits.device, valid_flat)
        if self.wants_feasibility_mask and obs is not None \
                and "self_weapon_readiness" in obs:
            # The label is the action-side firing intent. If the observation
            # contradicts it, exclude the frame instead of manufacturing a
            # target from sensed arsenal state. The exclusion predicate MUST
            # mirror Network._weapon_feasibility_mask's FULL predicate
            # (ownership/ammo AND cooldown): that mask bakes -1e9 into the
            # contradicted class's logit, so a fire label guarded on only the
            # readiness half scores CE against a -1e9 logit — a ~1e9 loss row
            # that dominates the head's gradient direction. Never triggered
            # by a native 20 Hz corpus, but a composed one pairs window-start
            # obs with any-fire-in-window labels: the human fired on the
            # second sub-tick, one native tick before cooldown expiry at the
            # decision tick (found 2026-08-03, 10 Hz decimation epoch-1 gate).
            from qnn.model.network import Network  # lazy: avoid import-order knots
            readiness = obs["self_weapon_readiness"].to(aw_logits.device)
            readiness = readiness.reshape(-1, readiness.shape[-1])
            tgt_ready = readiness.gather(1, target.clamp(1, 8).unsqueeze(1) - 1).squeeze(1)
            infeasible = tgt_ready <= Network._FEAS_OWNED_AMMO
            if "self_arsenal_scalars" in obs:
                af = obs["self_arsenal_scalars"][..., 0].to(aw_logits.device).reshape(-1)
                infeasible = infeasible | (af > Network._FEAS_AF_READY)
            target = target.masked_fill((target >= 1) & infeasible, _IGNORE)
        align_hbw = None
        if self.align_weight_gamma > 0.0:
            # Fail loud here too (belt-and-suspenders with container.py's
            # startup gate, qnn.bc.container._required_actions_for_config):
            # a probe that turns this knob on must never silently fall back
            # to an unweighted loss because the corpus lacks the sidecar.
            if "align_hbw" not in actions:
                raise RuntimeError(
                    "AttackWithHead.align_weight_gamma > 0 but actions has no "
                    "'align_hbw' key — run `python -m qnn.bc.cache_align_hbw "
                    "--collect-dir <corpus>` on this collect first."
                )
            align_hbw = torch.as_tensor(
                actions["align_hbw"], device=aw_logits.device, dtype=torch.float32,
            ).reshape(-1)
        # Class-0 frames are real labels, so a microbatch is all-ignore only
        # if valid_mask zeroes it entirely — same (weaker) caveat as the
        # categorical CE's mean reduction.
        if self.focal_gamma > 0.0 or self.pos_weight > 0.0 or self.align_weight_gamma > 0.0:
            loss = self._focal_ce(aw_logits, target, align_hbw=align_hbw)
        else:
            loss = F.cross_entropy(aw_logits, target, ignore_index=_IGNORE, reduction="mean")
        if not compute_metrics:
            return loss, {}

        feas = _feasibility(actions, aw_logits.device)
        return loss, _selector_metrics(
            loss, aw_logits, target, _ATTACK_SLOT_NAMES, feas=feas)



# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("attack", "attack_with")
def _build_attack(head, dims, d_model):
    # Declared-edge cat, plus the realized-alignment tail only when the head
    # declares it (the intent-edge builders' coord-vs-base pattern). "aim2"
    # (A″ forward-projected extension) reuses the same dims["aim_dim"] slot —
    # slot_dims already sized it to AIM2_DIM when the spec resolved aim2_edge.
    aim = dims["aim_dim"] if ("aim" in head.inputs or "aim2" in head.inputs) else 0
    return AttackWithHead(
        in_dim=dims["weapon_coord_in"] if aim else dims["weapon_in"],
        d_hidden=head.d_hidden,
        activation=head.activation,
        feasibility_mask=getattr(head, "feasibility_mask", False),
        focal_gamma=getattr(head, "focal_gamma", 0.0),
        pos_weight=getattr(head, "pos_weight", 0.0),
        aim_dim=aim,
        dart_p=getattr(head, "dart_p", 0.0),
        align_weight_gamma=getattr(head, "align_weight_gamma", 0.0),
        label=head.resolved_label)
