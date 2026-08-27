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

from qnn.model import action_labels
from qnn.model._mlp import make_head_mlp
from qnn.model.attack_head import AttackSelectorInput, AttackSelectorOutput
from qnn.model.action_labels import register_label_derive
from qnn.schema import WEAPON_HEAD_SIZE

ATTACK_WITH_SIZE = 1 + WEAPON_HEAD_SIZE  # 9: no-attack + axe..thunderbolt
_IGNORE = -100


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


# ── Shared selector metrics ─────────────────────────────────────────
#
# attack_loss and weapon_loss compute the SAME numbers; only the metric
# names differ, and not by a uniform prefix — the marginal-attack block is
# named identically in both (both heads report P(attack) on one ruler, so
# epoch aggregation treats it as a single series), while the conditional
# block diverges: acc_attack_choice vs acc_weapon, attackchoicedist_* vs
# weapondist_*, n_attack_choice_valid vs n_weapon_valid. Those names are a
# downstream contract (supervised_loop scans tp_<infix>_<cls> prefixes;
# bc_summary carries every key into run records), so the naming lives in
# one table and the arithmetic exists once.
# Pinned by tests/model/test_selector_metric_keys.py.


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

_WEAPON_SLOT_NAMES = _SelectorMetricNames(
    loss="loss_weapon", cls_infix="weapon", cond_dist="weapondist",
    n_valid="n_weapon_valid", cond_acc="acc_weapon",
    cond_f1="f1_weapon", confidence="confidence_weapon")


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

    Emits the selector output consumed by Network's categorical attack slot;
    motor-head context contract are untouched; ``context`` is the soft-mix
    embedding over the 9 classes (motor heads condition on attack-with
    intent; equipped-weapon state does not exist in the observation).
    """

    def __init__(
        self, *, in_dim: int, d_model: int, d_hidden: int, activation: str,
        feasibility_mask: bool = False, focal_gamma: float = 0.0,
        pos_weight: float = 0.0,
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
        self.embed = nn.Embedding(ATTACK_WITH_SIZE, int(d_model))
        # P1 feasibility-masking (agents/plans/attack-finished-masking-refactor.md).
        # When set, Network builds a per-class additive mask from obs (owned+ammo
        # AND refire-cooldown) and passes it via AttackSelectorInput.feasibility_mask;
        # the head no longer has to LEARN feasibility, so the fragile OOD read
        # (the a1 arsenal-fold collapse) never gets learned. All default-off ⇒
        # byte-identical to the pre-P1 head.
        self.wants_feasibility_mask = bool(feasibility_mask)
        self.focal_gamma = float(focal_gamma)
        self.pos_weight = float(pos_weight)

    def forward(self, inp: AttackSelectorInput) -> AttackSelectorOutput:
        logits = self.mlp(inp.selector[..., : self.in_dim])
        if inp.feasibility_mask is not None:
            # Gate BEFORE the context softmax so the intent context fed to the
            # motor heads reflects only feasible weapons too.
            logits = logits + inp.feasibility_mask.to(logits.dtype)
        context = F.softmax(logits, dim=-1) @ self.embed.weight
        return AttackSelectorOutput(logits=logits, context=context)

    def _focal_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE with optional focal down-weighting of easy frames (focal_gamma)
        and a multiplicative up-weight on the rare attack positives (classes
        1..8; pos_weight). Reduces to the plain mean CE when both are 0."""
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
            # The label is the action-side firing intent. If the observation's
            # ownership/ammo signal contradicts it, exclude that frame instead
            # of manufacturing a target from sensed arsenal state.
            readiness = obs["self_weapon_readiness"].to(aw_logits.device)
            readiness = readiness.reshape(-1, readiness.shape[-1])
            tgt_ready = readiness.gather(1, target.clamp(1, 8).unsqueeze(1) - 1).squeeze(1)
            target = target.masked_fill((target >= 1) & (tgt_ready <= 0.1 + 1e-4), _IGNORE)
        # Class-0 frames are real labels, so a microbatch is all-ignore only
        # if valid_mask zeroes it entirely — same (weaker) caveat as the
        # categorical CE's mean reduction.
        if self.focal_gamma > 0.0 or self.pos_weight > 0.0:
            loss = self._focal_ce(aw_logits, target)
        else:
            loss = F.cross_entropy(aw_logits, target, ignore_index=_IGNORE, reduction="mean")
        if not compute_metrics:
            return loss, {}

        feas = _feasibility(actions, aw_logits.device)
        return loss, _selector_metrics(
            loss, aw_logits, target, _ATTACK_SLOT_NAMES, feas=feas)



    def weapon_loss(
        self,
        logits,
        actions,
        valid_flat: torch.Tensor | None,
        compute_metrics: bool,
        obs=None,
    ) -> tuple[torch.Tensor, dict]:
        from qnn.model.network import WEAPON_HEAD  # lazy: avoid import-order knots

        aw_logits = logits[WEAPON_HEAD].reshape(-1, ATTACK_WITH_SIZE)
        # When masking is on, self-heal the label on frames where the intent
        # weapon can't fire → the held (actually-fired) weapon. Needs obs
        # (self_weapon_id + self_weapon_readiness, both dequant outputs the
        # arsenal token already consumes).
        held_impulse = intent_ready = None
        if self.wants_feasibility_mask and obs is not None \
                and "self_weapon_id" in obs and "self_weapon_readiness" in obs:
            dev = aw_logits.device
            swid = _flat_long(obs["self_weapon_id"], dev)                 # raw ENTITY_IDS 3..10
            held_impulse = torch.where((swid >= 3) & (swid <= 10),
                                       swid - 2, torch.zeros_like(swid))   # → impulse 1..8 (0=invalid)
            readiness = obs["self_weapon_readiness"]
            readiness = readiness.to(dev).reshape(-1, readiness.shape[-1])  # (N, 8)
            intent = _flat_long(actions["weapon"], dev).clamp(1, 8)
            intent_ready = readiness.gather(1, (intent - 1).unsqueeze(1)).squeeze(1)  # (N,)
        # Ditto: the declared contract owns the label, not the slot name.
        target = action_labels.derive_for(self.label_contract)(
            actions, aw_logits.device, valid_flat,
            held_impulse=held_impulse, intent_ready=intent_ready)
        if held_impulse is not None:
            # After self-heal, a tiny residual (~1% of fire frames) still labels
            # a class the mask calls infeasible — the held weapon's own readiness
            # reads empty though the demo fired it (obs ammo artifact). Those are
            # bad data: drop to IGNORE so the mask never masks a scored label.
            tgt_ready = readiness.gather(1, target.clamp(1, 8).unsqueeze(1) - 1).squeeze(1)
            target = target.masked_fill((target >= 1) & (tgt_ready <= 0.1 + 1e-4), _IGNORE)
        # Class-0 frames are real labels, so a microbatch is all-ignore only
        # if valid_mask zeroes it entirely — same (weaker) caveat as the
        # canonical weapon CE's mean reduction.
        if self.focal_gamma > 0.0 or self.pos_weight > 0.0:
            loss = self._focal_ce(aw_logits, target)
        else:
            loss = F.cross_entropy(aw_logits, target, ignore_index=_IGNORE, reduction="mean")
        if not compute_metrics:
            return loss, {}

        feas = _feasibility(actions, aw_logits.device)
        return loss, _selector_metrics(
            loss, aw_logits, target, _WEAPON_SLOT_NAMES, feas=feas)

# ── a26-line (weapon-slot) target + loss — self-heal semantics ─────────
# Ported verbatim from the a26 line so weapon-slot attack_with graphs
# train on this branch; the a27 attack-slot path above is untouched.
@register_label_derive("weapon.v2")
def attack_with_target_weapon_slot(
    actions,
    device: torch.device,
    valid_flat: torch.Tensor | None,
    *,
    held_impulse: torch.Tensor | None = None,
    intent_ready: torch.Tensor | None = None,
    **_unused,
) -> torch.Tensor:
    """(N,) long target in {-100, 0..8} from the batch's action streams.

    Label contract ``weapon.v2`` — the A25/A26 carried select-intent
    semantics.  A27 corpora do not satisfy it (``act_weapon`` is retired
    and always 0); it is kept registered so a25/a26 checkpoints keep
    loading for cross-line eval.

    ``act_weapon`` is de-scripted deliberate *select-intent* (carried forward),
    which on ~5% of fire frames names a weapon the player can't actually fire
    (unowned / no ammo) — the engine then deterministically fires the HELD
    weapon. When ``held_impulse`` (self_weapon_id→impulse) and ``intent_ready``
    (readiness of the intent weapon, per-frame) are supplied, self-heal those
    frames to the held weapon = what actually fired. This keeps the label a
    fireable class, so a readiness feasibility mask can never mask it (the P1
    divergence). Feasible-intent frames (incl. owned switch-lag) are untouched.
    """
    attack = _flat_long(actions["attack"], device)
    weapon = _flat_long(actions["weapon"], device)
    eff = attack > 0
    if "input_mask" in actions:
        feas = (_flat_long(actions["input_mask"], device) & 1) != 0
        eff = eff & feas
    if held_impulse is not None and intent_ready is not None:
        # "can't fire with intent" = intent weapon unowned/empty (readiness<=0.1
        # floor). Substitute the held weapon (what the engine actually fired).
        heal = eff & (intent_ready <= 0.1 + 1e-4) & (held_impulse >= 1) & (held_impulse <= 8)
        weapon = torch.where(heal, held_impulse, weapon)
    target = torch.where(eff, weapon, torch.zeros_like(weapon))
    # Effective attack with no weapon held (pre-spawn/dead edge frames) is
    # unattributable — drop rather than mislabel as class 0.
    target = target.masked_fill(eff & (weapon == 0), _IGNORE)
    if valid_flat is not None:
        target = target.masked_fill(~valid_flat, _IGNORE)
    return target

# -- graph node registration ------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("weapon", "attack_with")
def _build_weapon_attack_with(head, dims, d_model):
    """a26-line checkpoints place the 9-way selector on the WEAPON slot (the
    a27 refactor moved it to the attack slot); same module, second registry
    row, so those checkpoints load for cross-line eval.

    The slot no longer implies the label: ``head.label`` carries the
    contract, defaulted by :mod:`qnn.model.graph.spec` to this slot's
    historical semantics (``weapon.v2``) when an older spec omits it."""
    return AttackWithHead(
        in_dim=dims["weapon_in"], d_model=d_model, d_hidden=head.d_hidden,
        activation=head.activation,
        feasibility_mask=getattr(head, "feasibility_mask", False),
        focal_gamma=getattr(head, "focal_gamma", 0.0),
        pos_weight=getattr(head, "pos_weight", 0.0),
        label=head.resolved_label)


@register_head("attack", "attack_with")
def _build_attack(head, dims, d_model):
    return AttackWithHead(
        in_dim=dims["weapon_in"], d_model=d_model, d_hidden=head.d_hidden,
        activation=head.activation,
        feasibility_mask=getattr(head, "feasibility_mask", False),
        focal_gamma=getattr(head, "focal_gamma", 0.0),
        pos_weight=getattr(head, "pos_weight", 0.0),
        label=head.resolved_label)
