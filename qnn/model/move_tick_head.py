"""BENCH ARM — per-tick (per-frame) move head, revived for the seg-vs-frame decision.

**This is NOT a canonical head.** It exists to run cell C3 of
``agents/plans/seg-vs-frame-decision.md``: a controlled arm that swaps the
canonical ``move_seg`` commitment head for the pre-a28 per-tick move head, so
the seg-vs-frame question is decided by closed-loop rulers instead of by the
a24 priors alone. The canonical graph (``bases/core.json``) does not and must
not carry it; ``qnn.model.graph`` accepts it only as the bench head slot
``move_tick`` (type ``per_tick``).

Provenance
----------
* Module recovered from ``2e7f1793^:src/qnn/model/move_head.py`` — the
  canonical per-tick head as it stood at deletion time (a27-era): ONE
  ``Linear(features, 9)`` reshaped to (B*, 3 axes, 3 classes), three
  INDEPENDENT 3-class categorical axes (fb, lr, ud) over {neg, none, pos}.
  No priors, no approach-prior logits, no residual — that variant died with
  the a24 line long before the deletion commit.
* Loss + metric block recovered from ``2e7f1793^:src/qnn/model/policy.py``
  (``_compute_head_losses_and_metrics``, the ``MOVE_HEAD in logits`` branch),
  moved into the head under the a25+ head-owns-its-loss contract
  (``move_seg_loss`` / ``jump_loss`` / ``look_seg_loss`` pattern).
* The act-time decode (sticky-τ gate + semi-Markov log-normal dwell hazard)
  lives in :mod:`qnn.model.move_tick_decode`, recovered from
  ``2a4db619^:src/qnn/model/bench/a24/decode.py``. Per-frame INDEPENDENT
  sampling of this head destroys the human 88-99% hold autocorrelation
  (2.6-2.7x over-switch) — an arm without that decode is a strawman.

Deliberate deviations from the recovered code (all documented, none silent)
--------------------------------------------------------------------------
* Loss composition mirrors ``move_seg`` rather than the historical
  equal-weight mean over three axes: ``mean(CE_fb, CE_lr) + ud_weight *
  CE_ud``. C1's move_seg carries its rare ud axis as a separate ADDITIVE term
  at ``move_seg_ud = 0.25``; matching that composition is what makes the fb/lr
  trunk share comparable between the two arms. ``loss_move_tick`` reports the
  fb/lr mean (the ``loss_move_seg`` convention).
* The ud axis's rare-positive reweighting is a graph-spec ``pos_weight``
  (the ``jump`` head precedent) instead of the retired policy-level
  ``jump_pos_weight``/``jump_distance_sigma`` knobs. The distance-weighted
  jump shoulder (``qnn.bc.loss_shaping``) is NOT revived — the C1 controls
  run it at 0.0.
* Per-class precision/recall metric keys are dropped (argmax point metrics
  the head-metrics doc deprecates); acc/macro-F1 and the full ``movedist_*``
  sufficient statistics are kept verbatim, so
  ``supervised_loop._move_distribution_metrics_from_sums`` derives
  ``move_dll`` / ``move_skill`` / ``move_kl_joint`` on the SAME ruler as the
  historical a24/a25 records with no loop change.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qnn.actions import (
    MOVE_AXES,
    MOVE_AXIS_CLASSES,
    MOVE_AXIS_NAMES,
    MOVE_CLASS_NEG,
    MOVE_CLASS_NONE,
    MOVE_CLASS_POS,
)
from qnn.model._mlp import make_head_mlp

OUT_DIM = MOVE_AXES * MOVE_AXIS_CLASSES  # 9 logits
N_FBLR = 2                               # fb, lr — the axes the sticky decode owns


class MoveTickHead(nn.Module):
    """MLP: motor features (readout [+ target.feat]) -> (B*, 3, 3) axis logits.

    Same readout-first prefix-slice contract as ``move_seg`` / ``jump``:
    dropping ``target.feat`` from the declared inputs shrinks the consumed
    prefix, it never re-layouts the shared feature cat.
    """

    def __init__(self, *, in_dim: int, d_hidden: int, activation: str,
                 pos_weight: float = 1.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.mlp = make_head_mlp(in_dim, OUT_DIM, d_hidden, activation)
        # ud (jump/swim) POS-class reweighting for the ~4% positive rate.
        # Training term only — the movedist_* suffstats stay clean.
        self.pos_weight = float(pos_weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features[..., : self.in_dim]).reshape(
            -1, MOVE_AXES, MOVE_AXIS_CLASSES)

    # -- owned loss (dispatched from QNNPolicy._compute_head_losses_and_metrics)

    def move_tick_loss(
        self,
        logits,                       # full logits dict; ours under "move_tick"
        actions,
        valid_flat: torch.Tensor | None,
        compute_metrics: bool,
        ud_weight: float = 1.0,
        input_mask_on: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Per-axis CE against the engine-OUTCOME move label.

        Label rewrite (recovered verbatim from the pre-a28 policy block):
        with ``input_mask`` on, axis i's label is (demo intent) AND
        (per-direction feasibility from the input_mask bits).

          fb (axis 0) / lr (axis 1): feasibility bits 1-4 are always 1 while
              alive (pmove always processes fmove/smove) — the demo intent IS
              the engine outcome, no rewrite.
          ud (axis 2): direction-specific. POS feasibility = bit 7 (ground
              jump) OR bit 6 (swim up); NEG feasibility = bit 5 (swim down).
              Infeasible intent is rewritten to NONE — the engine could not
              have honoured that press.

        No frames are dropped; every valid frame trains against the outcome.
        """
        move_logits = logits["move_tick"]
        move_pred = move_logits.reshape(-1, MOVE_AXES, MOVE_AXIS_CLASSES)
        dev = move_pred.device
        if "move" not in actions:
            z = move_pred.sum() * 0.0
            return z, ({"loss_move_tick": z.detach()} if compute_metrics else {})

        move_t = actions["move"]
        move_t = move_t if isinstance(move_t, torch.Tensor) else torch.as_tensor(move_t)
        move_target = move_t.to(device=dev, dtype=torch.long).reshape(-1, MOVE_AXES)

        valid = (valid_flat.to(device=dev).bool().reshape(-1)
                 if valid_flat is not None
                 else torch.ones(move_target.shape[0], dtype=torch.bool, device=dev))

        if input_mask_on:
            if "input_mask" not in actions:
                raise RuntimeError(
                    "move_tick head: input_mask=True but actions['input_mask'] "
                    "is absent — the ud axis label is the engine OUTCOME and "
                    "cannot be derived without the feasibility bits.")
            im = actions["input_mask"]
            im = im if isinstance(im, torch.Tensor) else torch.as_tensor(im)
            im_flat = im.to(device=dev, dtype=torch.long).reshape(-1)
            up_neg_feas = ((im_flat >> 5) & 1) != 0        # swim down
            up_pos_feas = ((im_flat >> 6) & 1) != 0        # swim up
            jump_feas = ((im_flat >> 7) & 1) != 0          # ground jump
            ud_pos_feas = jump_feas | up_pos_feas
            ud_intent = move_target[:, 2]
            pos_mask = (ud_intent == MOVE_CLASS_POS) & ud_pos_feas
            neg_mask = (ud_intent == MOVE_CLASS_NEG) & up_neg_feas
            rewritten = move_target.clone()
            rewritten[:, 2] = torch.where(
                pos_mask, torch.full_like(ud_intent, MOVE_CLASS_POS),
                torch.where(neg_mask, torch.full_like(ud_intent, MOVE_CLASS_NEG),
                            torch.full_like(ud_intent, MOVE_CLASS_NONE)))
            move_target = rewritten

        is_real = valid.any()
        vf = valid.to(move_pred.dtype)
        n = vf.sum().clamp_min(1.0)

        ud_class_weight = None
        if self.pos_weight != 1.0:
            ud_class_weight = torch.tensor(
                [1.0, 1.0, self.pos_weight], dtype=torch.float32, device=dev)

        # Dense CE with the mask folded into the weight (never boolean
        # indexing): x[valid] calls nonzero(), whose device->host sync was the
        # profiled training bottleneck — the same fix the look/target blocks got.
        ce_per_axis: list[torch.Tensor] = []
        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
            w = ud_class_weight if axis_name == "ud" else None
            ce_pf = F.cross_entropy(
                move_pred[:, axis_i, :], move_target[:, axis_i],
                weight=w, reduction="none")
            ce_per_axis.append((ce_pf * vf).sum() / n)

        # move_seg composition: fb/lr mean is THE head loss; ud rides as a
        # separate additive term so the rare axis never dilutes fb/lr.
        loss = torch.stack(ce_per_axis[:N_FBLR]).mean()
        ud_loss = ce_per_axis[2] * float(ud_weight)

        metrics: dict = {}
        if compute_metrics:
            metrics["loss_move_tick"] = loss.detach()
            metrics["loss_move_tick_ud"] = ud_loss.detach()
            for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
                metrics[f"loss_move_tick_{axis_name}"] = ce_per_axis[axis_i].detach()
            metrics.update(self._distribution_metrics(move_pred, move_target, valid))
        return loss + ud_loss, metrics

    # -- movedist_* sufficient statistics (the historical move ruler) --------

    @torch.no_grad()
    def _distribution_metrics(
        self,
        move_pred: torch.Tensor,       # (V*, 3, 3) logits
        move_target: torch.Tensor,     # (V*, 3) engine-outcome classes
        valid: torch.Tensor,           # (V*,) bool
    ) -> dict:
        """Additive suffstats consumed by
        ``supervised_loop._move_distribution_metrics_from_sums`` -> move_dll /
        move_skill / move_kl_marg / move_kl_joint / jump calibration.

        FLOAT32 DELIBERATELY: these are COUNTS (tens of thousands per batch)
        summed across the epoch. bf16 carries 8 mantissa bits, so a histogram
        kept in the autocast dtype quantizes above 256 and the derived skill
        is garbage. ``_flush_tensor_dict``'s stack type-promotes, so a float32
        metric mixes with bf16 ones safely.
        """
        metrics: dict = {}
        jv = valid
        njv = int(jv.sum().item())
        if njv == 0:
            return metrics
        f32 = torch.float32
        pf = torch.softmax(move_pred[jv].float(), dim=-1)          # (V,3,3)
        logpf = torch.log_softmax(move_pred[jv].float(), dim=-1)
        tj = move_target[jv]                                        # (V,3)
        ar = torch.arange(njv, device=move_pred.device)
        metrics["movedist_n"] = torch.tensor(float(njv), dtype=f32,
                                             device=move_pred.device)
        per_axis_acc: list[torch.Tensor] = []
        per_axis_f1: list[torch.Tensor] = []
        argmax_all = pf.argmax(dim=-1)                              # (V,3)
        for axis_i, axis_name in enumerate(MOVE_AXIS_NAMES):
            metrics[f"movedist_ce_{axis_name}"] = (
                -logpf[ar, axis_i, tj[:, axis_i]].sum()).to(f32)
            hist_a = torch.bincount(tj[:, axis_i],
                                    minlength=MOVE_AXIS_CLASSES).to(f32)
            pred_a = pf[:, axis_i, :].sum(0).to(f32)
            for c in range(MOVE_AXIS_CLASSES):
                metrics[f"movedist_h_{axis_name}_{c}"] = hist_a[c]
                metrics[f"movedist_p_{axis_name}_{c}"] = pred_a[c]
            # argmax point metrics (kept: acc + macro-F1 only)
            pred_axis = argmax_all[:, axis_i]
            true_axis = tj[:, axis_i]
            per_axis_acc.append((pred_axis == true_axis).to(f32).mean())
            class_f1 = []
            for cls_idx in (MOVE_CLASS_NEG, MOVE_CLASS_NONE, MOVE_CLASS_POS):
                pred_cls = pred_axis == cls_idx
                true_cls = true_axis == cls_idx
                tp = (pred_cls & true_cls).sum().to(f32)
                fp = (pred_cls & ~true_cls).sum().to(f32)
                fn = (~pred_cls & true_cls).sum().to(f32)
                prec = tp / (tp + fp).clamp(min=1.0)
                rec = tp / (tp + fn).clamp(min=1.0)
                class_f1.append(2.0 * prec * rec / (prec + rec).clamp(min=1e-6))
            per_axis_f1.append(torch.stack(class_f1).mean())
            metrics[f"acc_move_{axis_name}"] = per_axis_acc[-1]
            metrics[f"f1_move_{axis_name}"] = per_axis_f1[-1]
        metrics["acc_move"] = torch.stack(per_axis_acc).mean()
        metrics["f1_move"] = torch.stack(per_axis_f1).mean()

        # Joint combo histogram, combo = fb + 3*lr + 9*ud (27 bins). The model
        # is per-frame axis-INDEPENDENT, so its implied joint is the summed
        # outer product. permute(2,1,0) so the C-order reshape lands on the
        # same combo index as the human histogram (else move_kl_joint compares
        # scrambled bins).
        combo_idx = tj[:, 0] + 3 * tj[:, 1] + 9 * tj[:, 2]
        jh = torch.bincount(combo_idx, minlength=27).to(f32)
        jp = torch.einsum("vi,vj,vk->ijk", pf[:, 0, :], pf[:, 1, :],
                          pf[:, 2, :]).permute(2, 1, 0).reshape(27).to(f32)
        for m in range(27):
            metrics[f"movedist_jh_{m}"] = jh[m]
            metrics[f"movedist_jp_{m}"] = jp[m]
        ud_i = MOVE_AXIS_NAMES.index("ud")
        metrics["movedist_ampos_ud"] = (
            argmax_all[:, ud_i] == MOVE_CLASS_POS).sum().to(f32)
        return metrics


# -- graph node registration --------------------------------------------------
from qnn.model.node_registry import register_head  # noqa: E402


@register_head("move_tick", "per_tick")
def _build_move_tick(head, dims, d_model):
    # Same readout-first prefix-slice contract as move_seg / jump.
    # coord_features_dim == base_features_dim unless a shared attack-intent
    # block is spliced in (network.slot_dims); this head is a CONSUMER of it.
    in_dim = (dims["coord_features_dim"] if "intent" in head.inputs
              else dims["base_features_dim"])
    if "target.feat" not in head.inputs:
        in_dim -= d_model
    pw = float(getattr(head, "pos_weight", 0.0))
    # ARM COMPARABILITY (the attack_future precedent, network.py +
    # attack_future_head._build_attack_future): restore the RNG stream across
    # this head's construction. nn.Linear.__init__ draws in reset_parameters
    # BEFORE Network._init_weights runs, so a head of a different width would
    # otherwise shift every shared tensor in the network away from the control
    # arm at the same seed. The head's own weights come from the xavier pass,
    # so discarding its constructor draw costs nothing. Model construction is
    # CPU-side (build_network runs before .to(device)), so the CPU generator is
    # the whole stream. Pinned by tests/model/test_move_tick_arm.py.
    state = torch.random.get_rng_state()
    try:
        return MoveTickHead(in_dim=in_dim, d_hidden=head.d_hidden,
                            activation=head.activation,
                            pos_weight=pw if pw > 0.0 else 1.0)
    finally:
        torch.random.set_rng_state(state)
