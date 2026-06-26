"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

import faulthandler as _faulthandler
import json
import os
import sys as _sys
import threading as _threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import time as _time

# Enable C-level traceback on signals (SEGV / FPE / ABRT). Chunked-prefetch
# BC path occasionally tickles platform-specific issues (WSL DXG pinned
# memory, dtype mismatches in index_copy_) that segfault below the Python
# layer; without this we get no traceback at all.
_faulthandler.enable(file=_sys.stderr, all_threads=True)

import numpy as np
import torch

from qnn import filter_dsl
from qnn.vocab import MAX_TOKEN_OBJECTS, TOKEN_ACTOR
from qnn.bc.class_weights import attack_class_weights
from qnn.schema import OBS_DIM
from qnn.model.network import ModelConfig
from qnn.model.policy import QNNPolicy
from qnn.bc.container import (
    BCSourceBundle,
    build_behavior_cloning_sources,
    effective_head_loss_weights,
    validate_cache_for_training,
    validate_source_bundle_compatible,
)
from qnn.utils.io import write_json
from qnn.utils.repro import set_global_seed, write_experiment_manifest


_MODEL_INIT_LOCK = _threading.Lock()


@dataclass(slots=True)
class BCConfig:
    """Behavior cloning configuration.

    Model arch lives in ``model`` (a ``ModelConfig`` instance) — the sole
    source of truth for architecture. ``build_run_bc_config()`` constructs
    it from the frozen run's ``config/model.json``. Every field of this
    dataclass is required; no Python-level defaults.
    """
    output_dir: str
    bc_data_dir: str
    seed: int
    batch_size: int
    sequence_length: int  # 0 = full episode (no chunking)
    epochs: int
    lr: float
    # Model architecture (source of truth: config/model.json)
    model: ModelConfig
    max_grad_norm: float  # gradient clipping for BPTT stability
    tbptt_limit: int  # max ticks before detaching gradient graph (0 = no limit)
    fixed_tick_hz: int
    device: str
    head_loss_weights: str  # JSON string, e.g. '{"move":1.5,"weapon":0.0}'
    regression_threshold: float
    regression_patience: int
    lr_min: float
    warmup_epochs: int
    prometheus_pushgateway_url: str
    train_eval_interval: int
    train_eval_gap_threshold: float
    train_eval_val_regression_threshold: float
    train_eval_train_improve_threshold: float
    # Performance tuning (sourced from machine.json).
    pin_memory: bool
    prefetch: int
    snapshot_interval: int
    # When true (and the model has no recurrence), bypass the
    # streaming=False: preload the whole corpus to device once (default;
    # the unified-memory APU win — no host duplicate of the GPU tensors).
    # streaming=True: lazy mmap reads per batch from shard files.
    streaming: bool
    dtype: str                 # "fp32" | "bf16" | "fp16"
    step_report_interval_seconds: int
    attack_pos_weight_override: float  # >0 overrides auto-computed neg/pos ratio for attack BCE
    attack_focal_gamma: float          # >0 swaps attack BCE for focal BCE with this gamma; 0 disables
    # Lin et al. per-class focal prefactor: alpha on positives,
    # (1 - alpha) on negatives. Active only when attack_focal_gamma > 0;
    # 0.5 is neutral. See QNNPolicy.__init__.
    attack_focal_alpha: float
    # >0 enables distance-weighted BCE on the attack head: per-frame loss
    # weight = 1 - gaussian-of-distance-to-nearest-true-fire so wrong-by-
    # one-frame FPs cost a small fraction of wrong-by-100-frames FPs.
    # Tune from the FP timing histogram (~3 at 20 Hz is a sensible
    # starting point; see scripts/analysis/fire_fp_timing.py). 0 disables.
    attack_distance_sigma: float
    jump_pos_weight: float          # >1.0 upweights POS class on move ud-axis CE
    jump_pos_weight_end: float      # >0 linearly decays jump_pos_weight epoch-wise; -1 disables
    # Same Gaussian shoulder as attack_distance_sigma, applied to the
    # ud-axis (jump) CE. Jumps are also a sparse press-or-not decision
    # with human timing noise, so the same shaping is appropriate;
    # expect a different sigma than fire because jump bursts are shorter
    # and rarer. 0 disables.
    jump_distance_sigma: float
    # Per-frame predicate (MongoDB DSL, qnn.filter_dsl) over the stored
    # action/obs arrays plus derived scalars (see _flatten_episode_arrays).
    # None / empty = no masking. Example:
    # {"act.target": {"$ne": 0}} = combat-only training (drops frames
    # with zero target mass; act.target = 1 - target_probs[:, 0]).
    segment_mask: "dict | None"
    # Per-idx predicate over entity token fields. Indices where the
    # predicate evaluates to False have their entity arrays zeroed
    # (positions preserved). Target mass on hidden indices is folded
    # into NO_TARGET so the target head is never trained toward a
    # token the model cannot see.
    token_mask: "dict | None"
    # When true, swap each head's label from the raw demo button
    # (`usercmd`) to the engine outcome
    # (`act = max(usercmd − infeasibility_mask, 0)`). Decodes the
    # per-axis bits of actions["input_mask"] (packed by
    # QNN_PackInputMask on the C side):
    #   bit 0    = attack            → fire label
    #   bits 1-2 = forward 3-class   → fb axis label
    #   bits 3-4 = side    3-class   → lr axis label
    #   bits 5-6 = up      3-class   ─┐ collapsed into the ud axis:
    #   bit 7    = jump              ─┘ jump | up-pos → POS, up-neg → NEG
    # No keep mask is needed downstream — the label is the engine
    # outcome by construction, so there's a single ``f1_attack`` (no
    # masked variant). Requires the recollected corpus that carries
    # act_input_mask.npy. See QNNPolicy._compute_head_losses_and_metrics.
    input_mask: bool
    # Expected collection identity (qnn.collection_fingerprint). Empty =
    # log-only mode.
    collection_fingerprint: str
    # Per-frame engagement EMA decay rate (see
    # qnn.bc.supervised_loop._compute_engagement_ema). α=0.5 was picked by
    # the MI analysis in scripts/attack_prior_methodical.py and is the
    # historical default; head-probe runs can override per-run via
    # probe.json["engagement_ema_alpha"].
    engagement_ema_alpha: float = 0.5
    # Opt-in: when true, the attack-head BCE LOSS is computed against
    # ``actions["attack_shifted"]`` (per-episode ``attack[t] OR attack[next
    # op-frame in same episode]``). Val precision/recall/F1 metrics still
    # use the original ``actions["attack"]`` label so we can compare to
    # the baseline apples-to-apples — only the loss-tensor target shifts.
    # Motivation: signed-offset analysis showed the current best model
    # has a systematic anticipatory bias, with ~31% of its predicted
    # positives leading the demo's actual fires by exactly one op-frame
    # (one cooldown cycle). The +1 op-frame target shift credits the
    # model for those otherwise-penalized leads; asymmetric on purpose
    # since lag is rare. Default False = bit-identical to current
    # behavior.
    attack_label_shift: bool = False
    # Opt-in: when true, the attack-head LOSS and val precision/recall/F1
    # metric only see op=1 frames (input_mask bit 0 == 1). op=0 frames
    # contribute no gradient and don't count in the confusion matrix;
    # pos_weight is computed from op=1 counts only. When false (default
    # = historical behavior), op=0 frames stay in the loss with label
    # forced to 0 by `feasibility * demo_press`, pos_weight is corpus-
    # wide, and the metric scores all frames. See QNNPolicy.attack_op_only.
    attack_op_only: bool = False
    # >0 swaps the binned look head's hard-argmin cross-entropy for a
    # distance-aware Gaussian soft-target CE with this σ (radians). Smooths
    # the foveated sub-degree center bins; 0 = standard one-hot CE. Only the
    # binned look_cls probe consumes it. See QNNPolicy.look_label_smoothing_sigma.
    look_label_smoothing_sigma: float = 0.0
    # --- Autostop: stop when the run is clearly neither learning nor
    # reorganizing. Replaces the inert regression_patience (which keyed on
    # mae_move/mae_look — never emitted for distributional heads). A run is
    # killed only when BOTH hold for `autostop_patience` consecutive epochs:
    #   (1) not learning  — val _selection_score did not beat the running best
    #                        by at least autostop_min_improve, AND
    #   (2) not reorganizing — weight_drift_l2 fell to ≤ autostop_drift_frac ×
    #                        the drift measured at the last improving epoch
    #                        (the learning-phase reference). High drift on a
    #                        val plateau = still reorganizing → keep going.
    # autostop_patience<=0 disables. Never fires before autostop_min_epoch.
    autostop_patience: int = 0
    autostop_min_improve: float = 0.001
    autostop_drift_frac: float = 0.5
    autostop_min_epoch: int = 6
    # Catastrophic hard-stop: if val_selection regresses > this margin ABOVE the
    # running best, stop immediately regardless of drift/patience/min_epoch (catches
    # divergence the reorganizing-veto would otherwise mask). 0 disables; always on
    # (independent of autostop_patience) since divergence should stop even runs that
    # left the normal autostop off. 0.5 is well above normal epoch-to-epoch noise.
    autostop_catastrophic_margin: float = 0.5


from qnn.bc.loop import (
    MidEpochState as _MidEpochState,
    make_resident_source_from_cache as _make_resident_source_from_cache,
    run_epoch as _run_epoch,
)


def _selection_score(metrics: Mapping[str, float]) -> float:
    """Composite selection metric for combat-objective BC.

    Lower is better. Every head contributes ``(1 − <head>_skill)`` — the
    fraction of that head's marginal entropy it did NOT capture. Skill is the
    common ruler (``<head>_dll / H_marg``, a proper scoring rule normalised to
    [.,1)); summing the normalised terms weights all five heads equally, unlike
    raw nats/KL/BCE whose scales differ ~20×. argmax F1/acc are diagnostics and
    are NOT used here. The single source of truth is src/docs/head-metrics.md.

    Heads absent from a run contribute a neutral 0 (skill defaults to 1). The
    fallbacks below only fire on the train-proxy path, where the distributional
    sufficient stats may not have been emitted; the val path that actually
    selects checkpoints always has ``<head>_skill``.
    """
    def head_error(head: str, *, loss_key: str | None = None) -> float:
        skill = metrics.get(f"{head}_skill")
        if skill is not None:
            return 1.0 - float(skill)
        # Train-proxy fallback only: a clean skill wasn't emitted, so fall back
        # to the raw NLL (lower=better) as a rough monotone proxy. Never used
        # for checkpoint selection.
        if loss_key is not None and loss_key in metrics:
            return float(metrics[loss_key])
        return 0.0  # neutral — head not present this run

    return (
        head_error("move", loss_key="loss_move")
        + head_error("look", loss_key="loss_look")
        + head_error("target", loss_key="loss_target")
        + head_error("attack", loss_key="loss_attack")
        + head_error("weapon", loss_key="loss_weapon")
    )


def _train_eval_schedule(
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    *,
    interval: int,
    gap_threshold: float,
    val_regression_threshold: float,
    train_improve_threshold: float,
) -> tuple[float, float, list[str]]:
    train_proxy_sum = _selection_score(train_metrics)
    val_sum = _selection_score(val_metrics)
    proxy_gap = val_sum - train_proxy_sum

    reasons: list[str] = []
    safe_interval = max(int(interval), 0)
    if safe_interval > 0 and (epoch + 1) % safe_interval == 0:
        reasons.append(f"interval/{safe_interval}")
    if proxy_gap > float(gap_threshold):
        reasons.append("proxy_gap")

    if history:
        prev = history[-1]
        prev_train_proxy_sum = float(
            prev.get(
                "train_proxy_sum",
                1.0 - float(prev.get("train_acc_target", 0.0)),
            )
        )
        prev_val_sum = 1.0 - float(prev.get("val_acc_target", 0.0))
        val_regression = val_sum - prev_val_sum
        train_delta = train_proxy_sum - prev_train_proxy_sum
        if (
            val_regression > float(val_regression_threshold)
            and train_delta < -float(train_improve_threshold)
        ):
            reasons.append("val_regressed_train_improved")

    return train_proxy_sum, proxy_gap, reasons


def _autostop_decision(
    *,
    selection_metric: float,
    prev_best: float,
    weight_drift_l2: float,
    drift_ref: "float | None",
    stall: int,
    epoch: int,
    patience: int,
    min_improve: float,
    drift_frac: float,
    min_epoch: int,
    catastrophic_margin: float = 0.0,
) -> "tuple[bool, int, float | None, str]":
    """Decide whether to stop a BC run that is neither learning nor reorganizing.

    Pure function of this epoch's signals + carried state. Returns
    ``(stop, new_stall, new_drift_ref, reason)``:

      * learning      — ``selection_metric`` beat ``prev_best`` (the running best
                        BEFORE this epoch's update) by ≥ ``min_improve``. On an
                        improving epoch the stall counter resets and the drift
                        reference is refreshed to this epoch's ``weight_drift_l2``
                        (the learning-phase drift at the current LR).
      * reorganizing  — ``weight_drift_l2`` is still > ``drift_frac`` × the
                        reference drift. High weight motion on a val plateau means
                        the model is still moving and may yet break out → keep going.

    Stop only when BOTH fail for ``patience`` consecutive epochs (and not before
    ``min_epoch``). ``patience<=0`` disables. The reference being the *last
    improving* epoch's drift makes the test scale-free and LR-decay aware: it
    compares plateau motion to how much the weights moved while this same model
    was still learning, not to the big epoch-0 init-settle.

    CATASTROPHIC hard-stop (``catastrophic_margin>0``): if val_selection regresses
    more than ``catastrophic_margin`` ABOVE the running best, stop IMMEDIATELY —
    bypassing the reorganizing-veto, patience, and min_epoch. This catches
    divergence/blow-ups, which the drift-veto otherwise masks: a gradient explosion
    produces *huge* weight drift that reads as "productive reorganization", so the
    normal stop never fires while val craters. (Real failure mode: a run diverged
    ~ep19, val_selection shot positive, drift spiked — and the normal stop held it.)
    """
    improved = selection_metric < (prev_best - min_improve)
    if improved:
        new_stall = 0
        new_drift_ref = weight_drift_l2
    else:
        new_stall = stall + 1
        new_drift_ref = drift_ref
    # Catastrophic regression: val collapsed far below the best (divergence). prev_best
    # is inf until a best exists, so `> inf + margin` is False at the first epoch.
    if catastrophic_margin > 0.0 and selection_metric > prev_best + catastrophic_margin:
        return (
            True, new_stall, new_drift_ref,
            f"CATASTROPHIC: val_selection {selection_metric:+.4f} regressed "
            f">{catastrophic_margin:g} above best {prev_best:+.4f} (divergence — "
            f"drift-veto/patience bypassed)",
        )
    reorganizing = (
        new_drift_ref is not None
        and new_drift_ref > 0.0
        and weight_drift_l2 > drift_frac * new_drift_ref
    )
    stop = (
        patience > 0
        and (epoch + 1) >= min_epoch
        and new_stall >= patience
        and not reorganizing
    )
    reason = ""
    if stop:
        _ref = new_drift_ref if new_drift_ref is not None else float("nan")
        reason = (
            f"{new_stall} epochs without ≥{min_improve:g} val-selection improvement; "
            f"drift {weight_drift_l2:.3f} ≤ {drift_frac:g}×{_ref:.3f} "
            f"(settled — not reorganizing)"
        )
    return stop, new_stall, new_drift_ref, reason



# --- Data loading ---

def _unpack_move_axes(packed: np.ndarray) -> np.ndarray:
    """Expand the on-disk packed move byte to (T, 3) uint8 axis class indices.

    The packed byte mirrors the input_mask layout (one bit per direction):

      bit 0 = attack press
      bit 1 = fb neg            bit 2 = fb pos
      bit 3 = lr neg            bit 4 = lr pos
      bit 5 = ud neg            bit 6 = ud pos (upmove > 0 — swim-up /
                                                jumppad / ladder)
      bit 7 = jump press (explicit jump button)

    Each axis returns a 3-class label in {0=neg, 1=none, 2=pos}:
      class = 1 + (pos_bit) - (neg_bit)

    For the ud axis the model treats jump and upmove>0 as the same
    "press +Z" intent (only one ud_pos class in the action head), so
    pos_bit = bit 6 OR bit 7.  Both bits cleared → class=1 (none);
    only neg → 0; either pos → 2.

    Attack and jump are extracted separately by ``_unpack_attack_bit`` and
    ``_unpack_jump_bit``. Materializes a fresh array (no longer mmap-backed)
    — fine because action labels are tiny relative to obs.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    one = np.uint8(1)
    fb_neg = (arr >> 1) & one
    fb_pos = (arr >> 2) & one
    lr_neg = (arr >> 3) & one
    lr_pos = (arr >> 4) & one
    ud_neg = (arr >> 5) & one
    # ud_pos folds in jump (bit 7) so both swim/jumppad upmove and the
    # explicit jump button supervise the same "press +Z" class.
    ud_pos = ((arr >> 6) & one) | ((arr >> 7) & one)
    fb = (one + fb_pos - fb_neg).astype(np.uint8)
    lr = (one + lr_pos - lr_neg).astype(np.uint8)
    ud = (one + ud_pos - ud_neg).astype(np.uint8)
    return np.ascontiguousarray(np.stack([fb, lr, ud], axis=-1))


def _unpack_attack_bit(packed: np.ndarray) -> np.ndarray:
    """Extract the attack bit (bit 0) from the packed move byte.

    Returns a (T,) uint8 in {0, 1}. The heads consume attack as a (T,)
    binary stream; this synthesizes it from the move byte that the
    collector packs in qnn.bc.collect._compact_action_arrays.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    return np.ascontiguousarray(arr & 0x1)


def _unpack_jump_bit(packed: np.ndarray) -> np.ndarray:
    """Extract the explicit jump press bit (bit 7) from the packed move byte.

    Diagnostic / debug signal. The model's ud-axis class label keeps the
    legacy unified behavior (jump-press OR swim-up → ud_pos); this is the
    standalone ground-jump indicator for analysis paths that need to tell
    swim-up from jump.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    return np.ascontiguousarray((arr >> 7) & 0x1)


# Hoisted to avoid per-episode import-statement lookup × 44k calls.
from qnn.bc.target_labeler import (
    label_enemy_target_probs as _LABEL_TARGETS,
    DEFAULT_LABELER_CONFIG as _LABELER_DEFAULT_CONFIG,
)

def _densify_obs_for_labeler(obs_padded: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert padded native obs arrays to the float layout the target
    labeler reads.

    Numpy-only equivalent of running ``SelfDequantizer + EntityDequantizer``
    on CPU, but producing *only the fields the labeler actually touches*:

      - ``self_scalars[:, 0]`` (health, normalized by MAX_HEALTH) — for
        the dead-frame mask
      - ``entity_scalars_raw[:, :, {HALFEXT, REL, VEL, TEAM, RECENCY}]``
        at the actor-layout offsets
      - ``entity_ids[:, :, {1=modality, 2=player_id}]``
      - ``entity_types``

    Non-actor entity indices are left as zero — the labeler masks to
    ``entity_types == TOKEN_ACTOR`` before reading any scalar offset, so
    the projectile/item/mover branches of the full dequantizer would be
    discarded anyway. Skipping them here saves ~600s of load time on
    the production corpus vs. running the full model-side dequantizers.
    """
    from qnn import engine_norm as en

    health  = np.asarray(obs_padded["health"])
    T = health.shape[0]
    et = np.asarray(obs_padded["entity_types"]).astype(np.int64, copy=False)
    N = et.shape[1]

    # self_scalars: labeler reads only idx 0 (_SELF_HEALTH_OFFSET).
    self_scalars = np.zeros((T, 17), dtype=np.float32)
    self_scalars[:, 0] = health.astype(np.float32) / en.MAX_HEALTH

    # entity_scalars_raw at actor offsets. Mirrors the actor branch of
    # EntityDequantizer (qnn.model.dequant) exactly:
    #   [0:3]   half_extents / DIST_SCALE
    #   [3:6]   rel           / DIST_SCALE
    #   [7:10]  vel           / MAX_VELOCITY
    #   [16]    team
    #   [18]    recency       / TIME_SCALE
    # Non-actor indices are zeroed; labeler masks them out anyway.
    entity_scalars = np.zeros((T, N, 19), dtype=np.float32)
    half = np.asarray(obs_padded["entity_half_extents"]).astype(np.float32) / en.DIST_SCALE
    rel  = np.asarray(obs_padded["entity_rel"]).astype(np.float32) / en.DIST_SCALE
    vel  = np.asarray(obs_padded["entity_vel"]).astype(np.float32) / en.MAX_VELOCITY
    team = np.asarray(obs_padded["entity_team"]).astype(np.float32)
    recency = np.asarray(obs_padded["entity_recency"]).astype(np.float32) / en.TIME_SCALE
    actor_mask = (et == TOKEN_ACTOR)
    if actor_mask.any():
        mask3 = actor_mask[..., None]
        entity_scalars[..., 0:3]  = np.where(mask3, half, 0.0)
        entity_scalars[..., 3:6]  = np.where(mask3, rel,  0.0)
        entity_scalars[..., 7:10] = np.where(mask3, vel,  0.0)
        entity_scalars[..., 16]   = np.where(actor_mask, team,    0.0)
        entity_scalars[..., 18]   = np.where(actor_mask, recency, 0.0)

    # entity_ids: labeler reads indices 1 (modality) and 2 (player_id).
    entity_ids = np.stack([
        np.asarray(obs_padded["entity_subject_id"]).astype(np.int64, copy=False),
        np.asarray(obs_padded["entity_modality_id"]).astype(np.int64, copy=False),
        np.asarray(obs_padded["entity_player_id"]).astype(np.int64, copy=False),
    ], axis=-1)

    return {
        "self_scalars":       self_scalars,
        "entity_types":       et,
        "entity_scalars_raw": entity_scalars,
        "entity_ids":         entity_ids,
    }


def _compute_target_probs(
    obs_padded: dict[str, np.ndarray],
    actions: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the target labeler on a padded-native episode.

    Returns ``(T, TARGET_PROBS_CLASSES) float32`` — same output the
    collector used to bake into the cache. Recomputing at training
    start (a) decouples labeler config from the wire format
    (fingerprint stays stable when you tune LabelerConfig), and (b)
    is lossless: the model trains on the exact f32 distribution the
    labeler emits, no sparse-encoding truncation on multi-hot rows.

    Cost: ~3 µs/frame on CPU. For an 8M-frame corpus that's ~25s of
    one-time startup overhead, amortized across the whole run.
    """
    from qnn.bc.target_labeler import (
        label_enemy_target_probs,
        DEFAULT_LABELER_CONFIG,
    )
    legacy_obs = _densify_obs_for_labeler(obs_padded)
    return label_enemy_target_probs(
        legacy_obs, actions, config=DEFAULT_LABELER_CONFIG,
    )


def _flatten_episode_arrays(obs: dict, actions: dict) -> dict[str, Any]:
    """Build a flat ``field_path -> np.ndarray`` view of an episode for
    qnn.filter_dsl predicate evaluation.

    Paths mirror the on-disk layout:
        act.<head>   →  action_arrays[head]
        obs.<chan>   →  obs_arrays[chan]
    """
    flat: dict[str, Any] = {}
    for head, arr in actions.items():
        flat[f"act.{head}"] = arr
        if head == "target_probs" and isinstance(arr, np.ndarray) and arr.ndim == 2:
            # Per-frame engagement scalar (= 1 - P(NO_TARGET)) so segment_mask
            # can express the no-engagement filter as `{"act.target":
            # {"$ne": 0}}` without column-indexing in filter_dsl.
            flat["act.target"] = 1.0 - arr[:, 0]
    for chan, arr in obs.items():
        flat[f"obs.{chan}"] = arr
    return flat


def _filter_referenced_keys(predicate: Any) -> set[str]:
    """Collect every leaf field path referenced by a filter predicate."""
    if not isinstance(predicate, dict):
        return set()
    keys: set[str] = set()
    for key, value in predicate.items():
        if key in ("$and", "$or"):
            if isinstance(value, list):
                for sub in value:
                    keys |= _filter_referenced_keys(sub)
        elif key == "$not":
            keys |= _filter_referenced_keys(value)
        elif key.startswith("$"):
            continue
        else:
            keys.add(key)
    return keys


def _flatten_token_arrays(obs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Build the per-token namespace used by train-time ``token_mask``."""
    return {
        "type": np.asarray(obs["entity_types"]),
        "modality": np.asarray(obs["entity_modality_id"]),
        "pid": np.asarray(obs["entity_player_id"]),
        "subject": np.asarray(obs["entity_subject_id"]),
        # Historical token masks used route_idx for idx-identity-like
        # filtering. Native v1 stores subject id in this position; keep
        # the alias so old configs fail less mysteriously.
        "route_idx": np.asarray(obs["entity_subject_id"]),
    }


def _token_keep_mask(
    obs: Mapping[str, np.ndarray],
    token_mask: Mapping[str, Any] | None,
) -> np.ndarray | None:
    if not token_mask:
        return None
    keep = np.asarray(
        filter_dsl.eval_filter(_flatten_token_arrays(obs), token_mask),
        dtype=bool,
    )
    expected = np.asarray(obs["entity_types"]).shape
    if keep.shape != expected:
        raise ValueError(
            f"token_mask predicate must produce a per-token bool array of "
            f"shape {expected}; got {keep.shape}"
        )
    return keep


def _mask_token_array(key: str, arr: np.ndarray, keep: np.ndarray) -> np.ndarray:
    out = np.asarray(arr).copy()
    fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
    out[~keep] = fill
    return out


def _mask_target_probs_for_tokens(
    target_probs: np.ndarray,
    indptr: np.ndarray,
    keep: np.ndarray | None,
) -> np.ndarray:
    if keep is None:
        return np.asarray(target_probs)
    td = np.asarray(target_probs, dtype=np.float32).copy()
    rows = td.shape[0]
    counts = (indptr[1:rows + 1] - indptr[:rows]).astype(np.int64, copy=False)
    if counts.sum() == 0:
        return td
    row_idx = np.repeat(np.arange(rows, dtype=np.int64), counts)
    idx_idx = np.arange(keep.shape[0], dtype=np.int64) - np.repeat(indptr[:rows], counts)
    drop = (~keep) & (idx_idx < (td.shape[1] - 1))
    if not np.any(drop):
        return td
    moved_rows = row_idx[drop]
    moved_cols = idx_idx[drop] + 1
    moved = np.zeros(rows, dtype=td.dtype)
    np.add.at(moved, moved_rows, td[moved_rows, moved_cols])
    td[moved_rows, moved_cols] = 0.0
    td[:, 0] += moved
    return td


def _slice_native_episode(
    obs_arrays: Mapping[str, np.ndarray],
    row_start: int,
    row_end: int,
    indptr: np.ndarray,
) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for key, arr in obs_arrays.items():
        if key in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            tok_lo = int(indptr[row_start])
            tok_hi = int(indptr[row_end])
            obs[key] = arr[tok_lo:tok_hi]
        else:
            obs[key] = arr[row_start:row_end]
    return obs


# Native-format entity field categories. Self/spatial fields are
# row-indexed (axis 0 = frame). Entity per-token fields are
# token-indexed (axis 0 = token, leading dim varies per row). The
# scalar entity_count is row-indexed.
_NATIVE_ROW_INDEXED_OBS_FIELDS = frozenset({
    "health", "effective_armor",
    "ammo_shells", "ammo_nails", "ammo_rockets", "ammo_cells",
    "vel", "attack_finished",
    "self_weapon_id", "self_movement_id", "self_items",
    "spatial_dir",
    "spatial_nearest_dist", "spatial_mean_dist",
    "spatial_openness", "spatial_clearance", "spatial_traversable",
    "spatial_dropoff", "spatial_solid_frac", "spatial_water_frac",
    "spatial_slime_frac", "spatial_lava_frac",
    "entity_count",
})

_NATIVE_TOKEN_INDEXED_OBS_FIELDS = frozenset({
    "entity_types", "entity_subject_id", "entity_modality_id",
    "entity_player_id", "entity_event_count",
    "entity_event_actions", "entity_event_sources",
    "entity_half_extents", "entity_rel", "entity_vel",
    "entity_path", "entity_path_dist", "entity_eta", "entity_recency",
    "entity_facing", "entity_team", "entity_score",
    "entity_amount", "entity_regen", "entity_state",
})

# GROUND sector index in spatial_dir's (T, 9, 3) layout — set in
# qnn.vocab.SPATIAL_SECTOR_IDS as "Ground_State". The C-side
# QNN_BuildGroundSpatial projects world-down (0,0,-1) into the view
# frame, which by AngleVectors math is exactly (sin(pitch), 0,
# -cos(pitch)) — see src/engine/common/qnn_spatial.c:175-181. So the
# i8 spatial_dir[:, 7, 0] column already encodes sin(view_pitch).
_GROUND_SECTOR_IDX = 7


def _inject_view_pitch_from_spatial_dir(
    obs_arrays: dict[str, "np.ndarray"],
) -> dict[str, "np.ndarray"]:
    """Backfill view_pitch from spatial_dir for legacy shards.

    Pre-self-split corpora don't carry a dedicated view_pitch field.
    Recover it from spatial_dir[GROUND, 0] = sin(pitch) (i8 quant
    against 127), reproject to the (deg / 90) × 127 i8 quantization
    that the new wire emits and the SelfDequantizer consumes. Cheap
    enough to do once at shard open — engine clamps pitch to ±70°, so
    sin → arcsin round-trip stays inside the i8 range with sub-degree
    error.

    Idempotent: if the shard already has view_pitch (fresh collect with
    the new emit), pass through unchanged.
    """
    if "view_pitch" in obs_arrays:
        return obs_arrays
    spatial_dir = obs_arrays.get("spatial_dir")
    if spatial_dir is None:
        return obs_arrays  # nothing to derive from; defer to caller's error
    sin_p = np.asarray(spatial_dir[:, _GROUND_SECTOR_IDX, 0], dtype=np.float32) / 127.0
    np.clip(sin_p, -1.0, 1.0, out=sin_p)
    # arcsin → normalize by π/2 (equivalent to deg / 90) → i8 quant.
    pitch_norm = np.arcsin(sin_p) * (2.0 / np.pi)
    view_pitch = np.round(pitch_norm * 127.0).clip(-127, 127).astype(np.int8)
    out = dict(obs_arrays)
    out["view_pitch"] = view_pitch
    return out

# Sentinel for empty entity indices in the padded (T, MAX_TOKEN_OBJECTS,
# ...) materialization. -1 in entity_types matches the ObsEmbedding's
# `entity_mask = (entity_types == TOKEN_ACTOR)` semantics; the
# ObsEmbedding key-padding mask flips on non-actor types so the
# transformer simply ignores empty indices.
_ENTITY_TYPES_EMPTY_SENTINEL = -1


def _materialize_padded_entity(
    obs: dict[str, np.ndarray], n_max: int,
) -> dict[str, np.ndarray]:
    """Pad an episode's token-indexed entity fields to ``(T, n_max, ...)``.

    Required by the trainer's GPU-resident / chunked prefetch paths
    which both index batches along axis 0 (frame) and need a constant
    second-dim for tensor concatenation. The unpadded
    ``(total_tokens, ...)`` layout is preserved on disk per the
    engine_norm phase 2 spec — this pad is a load-time materialization,
    not a re-write of the shard.

    ``entity_count`` (T,) drives the per-row valid-prefix; trailing
    indices are zeroed (``entity_types`` gets -1 sentinels so the
    ObsEmbedding's actor-only mask works unchanged).
    """
    counts = obs.get("entity_count")
    if counts is None:
        return obs  # legacy already-padded layout (test path)
    counts_np = np.asarray(counts, dtype=np.int64)
    T = counts_np.shape[0]
    indptr_local = np.concatenate([[0], np.cumsum(counts_np)])
    # Build a single (T, n_max) gather index that's reused across every
    # token-indexed field. valid[t, j] iff idx j is occupied for row t.
    # gather_idx points at flat[0] for invalid indices — the result is
    # masked back out via np.where before being returned, so the bogus
    # read is harmless.
    indices = np.arange(n_max, dtype=np.int64)
    counts_clamped = np.minimum(counts_np, n_max)
    valid = indices[None, :] < counts_clamped[:, None]
    gather_idx = np.where(valid, indptr_local[:T, None] + indices[None, :], 0)

    out = dict(obs)
    for key in list(obs.keys()):
        if key not in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            continue
        flat = np.asarray(obs[key])
        fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
        if flat.shape[0] == 0:
            # Empty episode (no tokens across any row). All indices
            # invalid; skip the gather, emit a fill-only tensor.
            out[key] = np.full((T, n_max) + flat.shape[1:], fill, dtype=flat.dtype)
            continue
        padded = flat[gather_idx]  # (T, n_max, *per_token_shape)
        # Broadcast the (T, n_max) mask up to padded's rank so np.where
        # zeros (or sentinels) the trailing invalid indices.
        if padded.ndim > 2:
            mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2))
        else:
            mask = valid
        out[key] = np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))
    return out


def _pad_entity_batch(
    unpadded_obs: dict[str, np.ndarray],
    indptr: np.ndarray,
    row_start: int,
    row_end: int,
    n_max: int,
) -> dict[str, np.ndarray]:
    """Pad token-indexed entity fields for a contiguous row range.

    Vectorized batch-side equivalent of :func:`_materialize_padded_entity`.
    Operates on already-sliced row arrays whose token data lives in
    ``unpadded_obs[key]`` indexed by the per-episode ``indptr`` of
    length ``n_samples + 1``. Returns a dict with only the token-indexed
    keys padded to ``(row_end - row_start, n_max, *per_token_shape)``.

    Caller is responsible for stitching with row-indexed fields, which
    are sliced upstream. This split keeps the per-batch pad work tight:
    one vectorized ``flat[gather_idx]`` per token-indexed key over just
    the rows in this batch.
    """
    n_rows = row_end - row_start
    if n_rows <= 0:
        return {}
    # Local indptr restricted to the requested row range, offset so its
    # values index into the unpadded[key] arrays' contiguous token range.
    indptr_slice = indptr[row_start:row_end + 1]
    counts = (indptr_slice[1:] - indptr_slice[:-1]).astype(np.int64, copy=False)
    indices = np.arange(n_max, dtype=np.int64)
    counts_clamped = np.minimum(counts, n_max)
    valid = indices[None, :] < counts_clamped[:, None]
    # Absolute per-row token start into the unpadded[key] arrays.
    # ``indptr`` is the per-episode cumulative entity_count, so
    # indptr_slice[:-1] points at the first valid token for each row.
    row_starts = indptr_slice[:-1].astype(np.int64, copy=False)
    gather_idx = np.where(valid, row_starts[:, None] + indices[None, :], 0)

    out: dict[str, np.ndarray] = {}
    for key, flat in unpadded_obs.items():
        if key not in _NATIVE_TOKEN_INDEXED_OBS_FIELDS:
            continue
        flat_arr = np.asarray(flat)
        fill = _ENTITY_TYPES_EMPTY_SENTINEL if key == "entity_types" else 0
        if flat_arr.shape[0] == 0:
            out[key] = np.full((n_rows, n_max) + flat_arr.shape[1:], fill, dtype=flat_arr.dtype)
            continue
        padded = flat_arr[gather_idx]
        if padded.ndim > 2:
            mask = valid.reshape(valid.shape + (1,) * (padded.ndim - 2))
        else:
            mask = valid
        out[key] = np.where(mask, padded, np.asarray(fill, dtype=padded.dtype))
    return out


@dataclass(slots=True)
class _ShardEpisodeMeta:
    row_start: int
    row_end: int
    tok_start: int
    tok_end: int
    ep_indptr: np.ndarray
    sort_key: tuple[int, int, int]


@dataclass(slots=True)
class _ShardSegment:
    src_row_start: int
    src_row_end: int
    meta: _ShardEpisodeMeta


def _episode_ids(
    shard: Mapping[str, Any],
    fallback_idx_start: int,
) -> tuple[list[int], list[int], list[int]]:
    lengths = [int(n) for n in shard.get("episode_lengths", [])]
    demo_idxs = shard.get("demo_idxs")
    if demo_idxs is None or len(demo_idxs) != len(lengths):
        demo_idxs = list(range(fallback_idx_start, fallback_idx_start + len(lengths)))
    episode_idxs = shard.get("episode_idxs")
    if episode_idxs is None or len(episode_idxs) != len(lengths):
        episode_idxs = [0] * len(lengths)
    return lengths, [int(v) for v in demo_idxs], [int(v) for v in episode_idxs]


def _build_indptr(entity_count: np.ndarray) -> np.ndarray:
    counts = np.asarray(entity_count, dtype=np.int64)
    indptr = np.empty(counts.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr


def _target_runs(
    obs: Mapping[str, np.ndarray],
    actions: Mapping[str, np.ndarray],
    segment_mask: dict | None,
) -> list[tuple[int, int]]:
    if not segment_mask:
        first = next(iter(actions.values()))
        return [(0, int(first.shape[0]))]
    flat = _flatten_episode_arrays(dict(obs), dict(actions))
    mask = np.asarray(filter_dsl.eval_filter(flat, segment_mask), dtype=bool)
    from qnn.bc.collect import _runs_from_mask
    return [(int(s), int(e)) for s, e in _runs_from_mask(mask)]


def _shard_segments(
    shard: Mapping[str, Any],
    fallback_idx_start: int,
    obs_arrays: Mapping[str, np.ndarray],
    action_arrays: Mapping[str, np.ndarray],
    shard_indptr: np.ndarray,
    segment_mask: dict | None,
) -> list[_ShardSegment]:
    lengths, demo_idxs, episode_idxs = _episode_ids(shard, fallback_idx_start)
    predicate_keys = _filter_referenced_keys(segment_mask)
    needs_token_obs = any(
        key.startswith("obs.") and key[4:] in _NATIVE_TOKEN_INDEXED_OBS_FIELDS
        for key in predicate_keys
    )
    segments: list[_ShardSegment] = []
    src_start = 0
    row_cursor = 0
    tok_cursor = 0
    for n_samples, demo_idx, episode_idx in zip(lengths, demo_idxs, episode_idxs):
        src_end = src_start + n_samples
        actions = {
            head: values[src_start:src_end]
            for head, values in action_arrays.items()
        }
        if segment_mask:
            if needs_token_obs:
                ep_indptr = (
                    shard_indptr[src_start:src_end + 1] - shard_indptr[src_start]
                ).astype(np.int64, copy=True)
                unpadded = _slice_native_episode(
                    obs_arrays, src_start, src_end, shard_indptr,
                )
                padded = _pad_entity_batch(
                    unpadded, ep_indptr, 0, int(n_samples), MAX_TOKEN_OBJECTS,
                )
                obs_for_filter = {
                    key: (padded[key] if key in padded else value)
                    for key, value in unpadded.items()
                }
            else:
                obs_for_filter = {
                    key[4:]: obs_arrays[key[4:]][src_start:src_end]
                    for key in predicate_keys
                    if key.startswith("obs.") and key[4:] in obs_arrays
                }
        else:
            obs_for_filter = {}
        runs = _target_runs(obs_for_filter, actions, segment_mask)
        for segment_idx, (local_start, local_end) in enumerate(runs):
            row_start = src_start + local_start
            row_end = src_start + local_end
            ep_indptr = (
                shard_indptr[row_start:row_end + 1] - shard_indptr[row_start]
            ).astype(np.int64, copy=True)
            n_rows = row_end - row_start
            n_toks = int(ep_indptr[-1])
            segments.append(_ShardSegment(
                src_row_start=row_start,
                src_row_end=row_end,
                meta=_ShardEpisodeMeta(
                    row_start=row_cursor,
                    row_end=row_cursor + n_rows,
                    tok_start=tok_cursor,
                    tok_end=tok_cursor + n_toks,
                    ep_indptr=ep_indptr,
                    sort_key=(demo_idx, episode_idx, segment_idx),
                ),
            ))
            row_cursor += n_rows
            tok_cursor += n_toks
        src_start = src_end
    return segments


# ---------------------------------------------------------------------------
# Prometheus pushgateway integration (optional).
# ---------------------------------------------------------------------------

_PROM_METRICS_TO_PUSH = (
    "val_acc_target",
    "train_acc_target",
    "train_loss", "val_loss",
)


def _push_metrics_to_prometheus(
    gateway_url: str,
    epoch_metrics: Dict[str, float],
    epoch: int,
    variant: str,
    config: BCConfig,
    *,
    _warned: list[bool] = [False],  # noqa: B006 — mutable default for singleton state
) -> None:
    """Push selected epoch metrics to a Prometheus pushgateway.

    No-ops silently when prometheus_client is not installed or the push fails.
    Only prints a warning on the first failure to avoid log spam.
    """
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    except ImportError:
        if not _warned[0]:
            print("  [bc] prometheus_client not installed — skipping metrics push")
            _warned[0] = True
        return

    try:
        registry = CollectorRegistry()
        epoch_gauge = Gauge(
            "bc_epoch", "Current training epoch",
            labelnames=["variant", "lr", "batch_size"],
            registry=registry,
        )
        epoch_gauge.labels(variant=variant, lr=str(config.lr), batch_size=str(config.batch_size)).set(epoch)

        for metric_name in _PROM_METRICS_TO_PUSH:
            if metric_name not in epoch_metrics:
                continue
            safe_name = f"bc_{metric_name}"
            g = Gauge(
                safe_name, metric_name,
                labelnames=["variant", "lr", "batch_size"],
                registry=registry,
            )
            g.labels(variant=variant, lr=str(config.lr), batch_size=str(config.batch_size)).set(
                epoch_metrics[metric_name]
            )

        push_to_gateway(gateway_url, job="bc_training", registry=registry)
    except Exception as exc:
        if not _warned[0]:
            print(f"  [bc] WARNING: Prometheus push failed ({exc}); suppressing further warnings")
            _warned[0] = True


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------

def run_behavior_cloning(
    config: BCConfig,
    seed_checkpoint: str = "",
    *,
    model_factory: Callable[[int, ModelConfig], "torch.nn.Module"] | None = None,
    graph: Any | None = None,
    side_channel_provider: Callable[..., Any] | None = None,
    source_bundle: BCSourceBundle | None = None,
    release_sources: bool = True,
    log_label: str = "",
    cancel_event: "_threading.Event | None" = None,
) -> Dict[str, float]:
    """Run BC training.

    ``side_channel_provider`` is forwarded to ``QNNPolicy``; bench probe runs
    pass ``qnn.model.bench.side_channels.bench_side_channel_scope`` so the
    label-derived bench contexts are entered around each forward pass.

    ``model_factory`` is the same hook ``QNNPolicy.__init__`` takes; pass
    one to swap in an ablation module (e.g. a per-head probe from
    ``qnn.model.bench``) without forking the trainer. When ``None`` the
    canonical ``Network`` is built from ``config.model``.

    ``graph`` (a ``qnn.model.graph.GraphSpec``) is the declarative
    assembly path — the model is built by ``build_network`` and the spec
    is persisted into every checkpoint (``meta.model_graph``). Mutually
    exclusive with ``model_factory``.

    Fine-tuning from a seed checkpoint ignores ``model_factory``/``graph``
    because ``QNNPolicy.load`` reconstructs the saved architecture; passing
    either is rejected to fail loud rather than silently dropping it.
    """
    log_prefix = f"  [bc {log_label}]" if log_label else "  [bc]"

    def _log(message: str) -> None:
        print(f"{log_prefix} {message}")

    set_global_seed(config.seed)
    # Episode shuffle uses a fixed seed (42) independent of the model init
    # seed, so all ablation runs see the same episode ordering per epoch.
    # This rng is saved/restored in checkpoints so resume produces the
    # same ordering as a continuous run.
    _SHUFFLE_SEED = 42
    rng = np.random.default_rng(_SHUFFLE_SEED)

    # Fall back to PUSHGATEWAY_URL env var if config doesn't specify one.
    if not config.prometheus_pushgateway_url:
        env_url = os.environ.get("PUSHGATEWAY_URL", "")
        if env_url:
            object.__setattr__(config, "prometheus_pushgateway_url", env_url)
            _log(f"Prometheus pushgateway: {env_url}")

    if not str(config.output_dir).strip():
        raise RuntimeError("Behavior cloning requires output_dir")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    head_loss_weights = effective_head_loss_weights(config.head_loss_weights)

    # Configure mixed-precision autocast via the env var that QNNPolicy reads.
    os.environ["QNN_AUTOCAST_DTYPE"] = config.dtype
    _log(f"dtype={config.dtype}")

    obs_dim = OBS_DIM
    # QNNPolicy construction intentionally seeds torch's global RNG for
    # deterministic init. Keep that short section serialized so
    # in-process parallel ablations do not race each other's initial
    # weights.
    with _MODEL_INIT_LOCK:
        if seed_checkpoint and Path(seed_checkpoint).exists():
            if model_factory is not None or graph is not None:
                raise RuntimeError(
                    "model_factory/graph are incompatible with seed_checkpoint — "
                    "QNNPolicy.load rebuilds the saved architecture itself."
                )
            _log(f"Fine-tuning from seed: {seed_checkpoint}")
            model = QNNPolicy.load(seed_checkpoint, device=config.device)
        else:
            model = QNNPolicy(
                obs_dim=obs_dim,
                model=config.model,
                jump_pos_weight=config.jump_pos_weight,
                attack_focal_gamma=config.attack_focal_gamma,
                attack_focal_alpha=config.attack_focal_alpha,
                attack_distance_sigma=config.attack_distance_sigma,
                jump_distance_sigma=config.jump_distance_sigma,
                look_label_smoothing_sigma=config.look_label_smoothing_sigma,
                seed=config.seed,
                device=config.device,
                model_factory=model_factory,
                graph=graph,
                side_channel_provider=side_channel_provider,
            )
    # input_mask is a training-time toggle, not a ModelConfig field —
    # set after construction so the same checkpoint can be retrained
    # either way (and so seed_checkpoint resumes pick up the run's
    # current config rather than the seed run's). When true, the head's
    # label is the engine outcome (act = max(usercmd − mask, 0)); when
    # false, the label is the raw demo button (usercmd).
    model.input_mask = bool(config.input_mask)
    model.attack_op_only = bool(config.attack_op_only)
    # attack_label_shift is also a training-time label-rewrite toggle
    # (no model arch impact, no checkpoint meta). Off by default; when on,
    # the attack-head LOSS reads actions["attack_shifted"] (built by the
    # source above) and val metrics keep using the original attack label.
    model.attack_label_shift = bool(config.attack_label_shift)

    if source_bundle is None:
        source_bundle = build_behavior_cloning_sources(config, head_loss_weights=head_loss_weights)
    else:
        validate_source_bundle_compatible(
            config, source_bundle, head_loss_weights=head_loss_weights,
        )

    train_source = source_bundle.train_source
    val_source = source_bundle.val_source
    actual_fp = source_bundle.actual_fingerprint

    sample_counts = source_bundle.sample_counts
    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available")
    weights = attack_class_weights(
        train_source,
        head_loss_weights=head_loss_weights,
        override=float(config.attack_pos_weight_override),
        device=model.device,
        op_only=bool(config.attack_op_only),
    )
    _train_eval_n_eps = len(val_source.episodes)

    # Parse per-head loss weights from JSON string if provided.
    hlw: Dict[str, float] | None = None
    if config.head_loss_weights:
        hlw = dict(head_loss_weights)

    # Best _selection_score seen so far — composite Σ_head (1 − head_skill).
    # NOT a loss; selection error, lower is better. See src/docs/head-metrics.md.
    best_selection_score = float("inf")
    best_epoch = -1
    history: list[Dict[str, float]] = []
    start_epoch = 0
    # Running peak weight drift — the reference for the headline ``reorg``
    # scalar (current drift ÷ peak; ~1 = reorganizing as hard as ever, →0 =
    # converged/stuck). Raw weight_drift_l2 stays in bc_history.
    _max_weight_drift = 0.0

    # Regression-based stopping state.
    _best_move = float("inf")
    _best_look = float("inf")
    _best_max_reg = float("inf")  # for checkpoint selection: min of max(move_reg, look_reg)
    _best_reg_epoch = -1
    _reg_violations = 0

    # Autostop state (not-learning + not-reorganizing). See _autostop_decision.
    _autostop_stall = 0
    _autostop_drift_ref: "float | None" = None

    # NAS archive: save every epoch checkpoint to SMB share for offsite backup.
    _NAS_CHECKPOINTS = r"\\pi.local\nqcorpus\bc_checkpoints"
    _smb_available = False
    try:
        import smbclient
        smbclient.ClientConfig(username="guest", password="", require_secure_negotiate=False)
        smbclient.register_session(
            "pi.local", username="guest", password="",
            auth_protocol="ntlm", require_signing=False,
        )
        _variant_name = output.parent.name or output.name
        _variant_dir = _NAS_CHECKPOINTS + "\\" + _variant_name
        smbclient.makedirs(_variant_dir, exist_ok=True)
        _smb_available = True
        _log(f"NAS archive available: {_variant_dir}")
    except Exception:
        _smb_available = False
        _log("NAS archive not available — skipping offsite backup")

    # Mid-epoch state: rolling file for deterministic resume within an epoch.
    mid_epoch_path = output / "snapshot.pt"
    _MID_EPOCH_SAVE_INTERVAL = config.snapshot_interval

    # Resume from checkpoint if available.
    checkpoint_path = output / "bc_training_checkpoint.pt"
    if checkpoint_path.exists():
        import torch as _torch_resume
        from qnn.utils.checkpoint_converter import (
            migrate_entity_embed,
            migrate_obs_embedding_self_token_builder,
            migrate_self_scalars,
        )
        ckpt = _torch_resume.load(checkpoint_path, map_location=model.device, weights_only=False)
        migrate_entity_embed(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        migrate_self_scalars(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        migrate_obs_embedding_self_token_builder(ckpt["model_state_dict"])
        model.model.load_state_dict(ckpt["model_state_dict"])
        # Resume compat: accept new key, fall back to the old "best_val_loss"
        # name (a misnomer that held the same selection score).
        best_selection_score = float(ckpt.get(
            "best_selection_score", ckpt.get("best_val_loss", float("inf")),
        ))
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        _best_move = ckpt.get("_best_move", float("inf"))
        _best_look = ckpt.get("_best_look", float("inf"))
        _best_max_reg = ckpt.get("_best_max_reg", float("inf"))
        _best_reg_epoch = ckpt.get("_best_reg_epoch", -1)
        _reg_violations = ckpt.get("_reg_violations", 0)
        _autostop_stall = ckpt.get("_autostop_stall", 0)
        _autostop_drift_ref = ckpt.get("_autostop_drift_ref", None)
        # Optimizer state restored after first supervised step creates it.
        _resume_optimizer_state = ckpt.get("optimizer_state_dict")
        # Restore rng state so resume produces the same episode ordering
        # as a continuous run.
        _saved_rng_state = ckpt.get("rng_state")
        if _saved_rng_state is not None:
            rng.bit_generator.state = _saved_rng_state
        _log(f"Resuming from epoch {start_epoch} (best_selection={best_selection_score:.4f} at epoch {best_epoch})")
    else:
        _resume_optimizer_state = None

    # Mid-epoch resume: if we have a mid-epoch state file, use it to
    # resume within the current epoch instead of restarting it.
    _mid_epoch_resume: _MidEpochState | None = None
    if mid_epoch_path.exists():
        import torch as _torch_mid
        try:
            _mid_ckpt = _torch_mid.load(mid_epoch_path, map_location=model.device, weights_only=False)
            if _mid_ckpt.get("epoch") == start_epoch:
                model.model.load_state_dict(_mid_ckpt["model_state_dict"])
                _resume_optimizer_state = _mid_ckpt.get("optimizer_state_dict")
                _mid_epoch_resume = _mid_ckpt["mid_epoch_state"]
                rng.bit_generator.state = _mid_ckpt["rng_state"]
                _log(
                    f"Mid-epoch resume: epoch {start_epoch}, "
                    f"step {_mid_epoch_resume.opt_steps}, "
                    f"chunk {_mid_epoch_resume.next_episode}"
                )
            else:
                mid_epoch_path.unlink()
        except Exception as exc:
            _log(f"Mid-epoch state load failed: {exc}")
            mid_epoch_path.unlink(missing_ok=True)

    # torch.compile: tested but net negative for this model size (189K params).
    # The fused kernels don't help when individual ops are already microseconds,
    # and the compile wrapper adds overhead (val: 100s → 120s per epoch).
    # Revisit if model size increases significantly.

    # Per-step reporting: aggregate every ~1024 samples, then wall-clock gate
    # actual logging/flushes so perf runs do not spend most of their time
    # printing and rewriting the step log.
    _report_every = max(1, 1024 // max(config.batch_size, 1)) if config.batch_size > 0 else 0
    _step_log: list[Dict[str, float]] = []
    _step_report_interval = max(int(config.step_report_interval_seconds), 0)
    _last_step_report_time = _time.monotonic() - _step_report_interval

    def _on_step(step_metrics: Dict[str, float]) -> None:
        nonlocal _last_step_report_time
        _now = _time.monotonic()
        if _step_report_interval > 0 and (_now - _last_step_report_time) < _step_report_interval:
            return
        _last_step_report_time = _now
        step_metrics["epoch"] = float(epoch)
        _step_log.append(step_metrics)
        mae_parts = [f"{k}={v:.4f}" for k, v in sorted(step_metrics.items()) if k.startswith("mae_")]
        _log(
            f"  step {int(step_metrics.get('opt_step', 0)):>5d}  "
            f"loss={step_metrics.get('loss', 0):.4f}  "
            f"{'  '.join(mae_parts)}"
        )
        # Flush step log to disk every report interval for live monitoring.
        write_json(output / "bc_step_log.json", {"steps": _step_log})

    def _save_mid_epoch(state: _MidEpochState) -> None:
        bc_opt = model._optimizers.get("bc")
        mid_data = {
            "epoch": epoch,
            "model_state_dict": {
                k.replace("_orig_mod.", ""): v
                for k, v in model.model.state_dict().items()
            },
            "optimizer_state_dict": bc_opt.state_dict() if bc_opt else None,
            "mid_epoch_state": state,
            "rng_state": rng.bit_generator.state,
        }
        torch.save(mid_data, mid_epoch_path)

    _active_lr = config.lr
    _lr_override_path = output / "lr_override.json"

    import math as _math
    from datetime import datetime as _datetime, timezone as _tz

    import gc as _gc

    _prev_epoch_weights: Dict[str, torch.Tensor] | None = None

    for epoch in range(start_epoch, config.epochs):
        # Reclaim Python + CUDA allocator pool at each epoch boundary.
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optional linear decay of the ud-axis pos_weight across epochs.
        # Lets us start with high pos_weight (push recall hard while the head
        # is randomly initialized) and end with low pos_weight (let precision
        # recover as the head calibrates).  -1.0 sentinel disables decay.
        if config.jump_pos_weight_end > 0 and config.epochs > 1:
            alpha = float(epoch) / float(config.epochs - 1)
            current_pw = (1.0 - alpha) * float(config.jump_pos_weight) + alpha * float(config.jump_pos_weight_end)
            model.jump_pos_weight = current_pw
            _log(f"jump_pos_weight (decay): epoch {epoch}/{config.epochs - 1}  alpha={alpha:.3f}  pw={current_pw:.3f}")
        # Snapshot weights at the start of this epoch so we can compute
        # L2 drift from the end-of-last-epoch state as a "is the model still
        # actively changing?" signal.
        _epoch_start_weights = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
        # Hot-reload LR: drop {"lr": 0.001, "lr_min": 0.0003} into lr_override.json.
        _lr = config.lr
        _lr_min = config.lr_min
        if _lr_override_path.exists():
            try:
                _ovr = _json.loads(_lr_override_path.read_text())
                _lr = float(_ovr.get("lr", _lr))
                _lr_min = float(_ovr.get("lr_min", _lr_min))
                _log(f"lr_override.json: lr={_lr}, lr_min={_lr_min}")
            except Exception as exc:
                _log(f"lr_override.json parse error: {exc}")

        # LR schedule: optional linear warmup then optional cosine decay.
        _warmup = config.warmup_epochs
        if _warmup > 0 and epoch < _warmup:
            # Linear warmup from lr_min (or near-zero) to lr.
            _base = _lr_min if _lr_min > 0 else _lr * 0.01
            _active_lr = _base + (_lr - _base) * (epoch / _warmup)
        elif _lr_min > 0:
            # Cosine decay from lr to lr_min over post-warmup epochs.
            _post_warmup = epoch - _warmup
            _post_total = max(config.epochs - 1 - _warmup, 1)
            progress = _post_warmup / _post_total
            _active_lr = _lr_min + 0.5 * (_lr - _lr_min) * (1 + _math.cos(_math.pi * progress))
        else:
            _active_lr = _lr

        if epoch == start_epoch or epoch > start_epoch:
            _log(f"LR={_active_lr:.6f}")

        _t_train_start = _time.monotonic()
        train_metrics = _run_epoch(
            model,
            train_source,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            tbptt_limit=config.tbptt_limit,
            class_weights=weights,
            lr=_active_lr,
            rng=rng,
            max_grad_norm=config.max_grad_norm,
            head_loss_weights=hlw,
            step_callback=_on_step,
            report_every=_report_every,
            report_interval_seconds=float(_step_report_interval),
            save_state_callback=_save_mid_epoch,
            snapshot_interval=_MID_EPOCH_SAVE_INTERVAL,
            resume_state=_mid_epoch_resume,
        )
        _mid_epoch_resume = None
        _t_train_end = _time.monotonic()
        if _resume_optimizer_state is not None:
            bc_opt = model._optimizers.get("bc")
            if bc_opt is not None:
                bc_opt.load_state_dict(_resume_optimizer_state)
                _resume_optimizer_state = None
        _t_val_start = _time.monotonic()
        val_metrics = _run_epoch(
            model,
            val_source,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            tbptt_limit=config.tbptt_limit,
            head_loss_weights=hlw,
        )
        _t_val_only_end = _time.monotonic()
        train_proxy_sum, train_proxy_gap, train_eval_reasons = _train_eval_schedule(
            epoch,
            history,
            train_metrics,
            val_metrics,
            interval=config.train_eval_interval,
            gap_threshold=config.train_eval_gap_threshold,
            val_regression_threshold=config.train_eval_val_regression_threshold,
            train_improve_threshold=config.train_eval_train_improve_threshold,
        )
        train_eval_metrics: Dict[str, float] = {}
        train_eval_sum: float | None = None
        _train_eval_secs = 0.0
        train_eval_ran = val_source.n_total_rows > 0 and bool(train_eval_reasons)
        if train_eval_ran:
            # Clean train eval (model.eval mode, no dropout) on a train subset
            # only when scheduled or when proxy metrics suggest a gap issue.
            _t_train_eval_start = _time.monotonic()
            # Reuse the train source by truncating to the validation episode
            # count — works for both resident and streaming sources.
            train_eval_source = train_source.head(_train_eval_n_eps)
            train_eval_metrics = _run_epoch(
                model,
                train_eval_source,
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                tbptt_limit=config.tbptt_limit,
                head_loss_weights=hlw,
                )
            _train_eval_secs = _time.monotonic() - _t_train_eval_start
            train_eval_sum = _selection_score(train_eval_metrics)
        _t_val_end = _time.monotonic()
        _train_secs = _t_train_end - _t_train_start
        _val_only_secs = _t_val_only_end - _t_val_start
        _val_secs = _t_val_end - _t_val_start
        train_rows = float(train_metrics.get("n_rows", sample_counts["train"]))
        val_rows = float(val_metrics.get("n_rows", sample_counts["val"]))
        train_eval_rows = float(train_eval_metrics.get("n_rows", 0.0)) if train_eval_ran else 0.0
        train_rows_per_sec = train_rows / _train_secs if _train_secs > 0 else 0.0
        val_rows_per_sec = val_rows / _val_only_secs if _val_only_secs > 0 else 0.0
        train_eval_rows_per_sec = train_eval_rows / _train_eval_secs if _train_eval_secs > 0 else 0.0
        _wall_clock = _datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if train_eval_ran:
            _log(
                f"timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  "
                f"train_eval={_train_eval_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]"
            )
        else:
            _log(f"timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]")
        # Headline per-head summary: one normalised skill per head — the
        # fraction of that head's marginal entropy it captures (0 = base rate,
        # →1 = fully determined). Comparable across heads and the terms of the
        # selection composite. All raw metrics (dll/kl/nll/loss/f1/per-class)
        # stay in bc_history.json for analysis. See src/docs/head-metrics.md.
        _headline_keys = (
            "move_skill", "look_skill", "target_skill", "attack_skill", "weapon_skill",
        )
        skill_str = "  ".join(
            f"{k}={float(val_metrics[k]):.4f}"
            for k in _headline_keys if k in val_metrics
        )

        # Composite selection: Σ_head (1 − head_skill), lower is better.
        val_selection_score = _selection_score(val_metrics)
        selection_metric = val_selection_score
        improved = selection_metric < best_selection_score

        # Weight drift: L2 of (weights now) - (weights at epoch start).
        # Non-zero drift in a plateau = model still reorganizing; zero = stuck.
        # Accumulate squared diffs on GPU, single host sync at the end.
        _cur_state = model.model.state_dict()
        _diffs = [((_cur_state[_k] - _start_v) ** 2).sum()
                  for _k, _start_v in _epoch_start_weights.items()
                  if _cur_state[_k].dtype.is_floating_point]
        _weight_drift_l2 = torch.stack(_diffs).sum().sqrt().item() if _diffs else 0.0

        _grad_mean = train_metrics.get("grad_norm_mean")
        _grad_max = train_metrics.get("grad_norm_max")

        # Two interpretable summaries replace the raw
        # train_proxy/proxy_gap/train_eval/gap/grad/drift clutter on the line
        # (all of which remain in bc_history.json):
        #   overfit = val selection error − the train reference (held-out
        #     train_eval when it ran, else the noisier train proxy). >0 = val
        #     worse than train (memorising); ~0 = generalising; <0 = val ahead.
        #   reorg   = this epoch's weight drift ÷ the running peak drift.
        #     ~1 = reorganising as hard as ever; →0 = converged / stuck.
        _train_ref = train_eval_sum if (train_eval_ran and train_eval_sum is not None) else train_proxy_sum
        _overfit = val_selection_score - _train_ref
        _max_weight_drift = max(_max_weight_drift, _weight_drift_l2)
        _reorg = _weight_drift_l2 / _max_weight_drift if _max_weight_drift > 0.0 else 1.0

        epoch_line = (
            f"{log_prefix} Epoch {epoch + 1}/{config.epochs}  "
            f"sel={val_selection_score:.4f}  "
            f"overfit={_overfit:+.4f}  "
            f"reorg={_reorg:.2f}  "
            f"{'*' if improved else ''}  "
            f"train_rps={train_rows_per_sec:.1f}  val_rps={val_rows_per_sec:.1f}  "
            f"{skill_str}"
        )
        print(epoch_line)

        # Assemble and record per-epoch metrics.
        epoch_metrics: Dict[str, Any] = {
            "epoch": float(epoch),
            "train_secs": _train_secs,
            "val_secs": _val_secs,
            "val_only_secs": _val_only_secs,
            "train_eval_secs": _train_eval_secs,
            "wall_clock": _wall_clock,
            "train_proxy_sum": train_proxy_sum,
            "train_proxy_gap": train_proxy_gap,
            "train_eval_ran": train_eval_ran,
            "train_eval_reason": ",".join(train_eval_reasons),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_eval_rows": train_eval_rows,
            "effective_train_rows_per_sec": train_rows_per_sec,
            "effective_val_rows_per_sec": val_rows_per_sec,
            "effective_train_eval_rows_per_sec": train_eval_rows_per_sec,
        }
        epoch_metrics["weight_drift_l2"] = _weight_drift_l2

        for key, value in train_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"train_{key}"] = float(value)
        for key, value in train_eval_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"train_eval_{key}"] = float(value)
        for key, value in val_metrics.items():
            if key == "_next_hidden":
                continue
            epoch_metrics[f"val_{key}"] = float(value)
        history.append(epoch_metrics)

        # Write history and step log incrementally so results survive crashes.
        write_json(output / "bc_history.json", {"history": history})
        if _step_log:
            write_json(output / "bc_step_log.json", {"steps": _step_log})

        # Epoch sentinel: external watchers can poll this file to detect
        # epoch completion across any training mode (BC, PPO, etc.).
        (output / "epoch_done").write_text(
            json.dumps({"epoch": epoch, "wall_clock": _wall_clock, "mode": "bc"}) + "\n"
        )

        # Push metrics to Prometheus pushgateway (no-op when URL is empty).
        if config.prometheus_pushgateway_url:
            _push_metrics_to_prometheus(
                config.prometheus_pushgateway_url,
                epoch_metrics,
                epoch,
                variant=Path(config.output_dir).name,
                config=config,
            )

        # Regression-based stopping: track per-head bests and regression.
        val_move = val_metrics.get("mae_move", float("inf"))
        val_look = val_metrics.get("mae_look", float("inf"))
        _best_move = min(_best_move, val_move)
        _best_look = min(_best_look, val_look)
        move_reg = val_move - _best_move
        look_reg = val_look - _best_look

        # Checkpoint selection: best val MAE sum.  Regression gate is purely
        # for stopping, not model selection. Capture the pre-update best so the
        # autostop "did we improve?" test sees this epoch's running best.
        _autostop_prev_best = best_selection_score
        if selection_metric < best_selection_score:
            best_selection_score = selection_metric
            best_epoch = epoch
            model.save(output / "bc_best_model.pth")

        if move_reg > config.regression_threshold or look_reg > config.regression_threshold:
            _reg_violations += 1
        else:
            _reg_violations = 0

        _log(
            f"  regression: move={move_reg:+.4f} look={look_reg:+.4f} "
            f"violations={_reg_violations}/{config.regression_patience}"
        )

        # Autostop: not-learning + not-reorganizing (the live replacement for
        # the inert regression_patience). Updates carried (_autostop_stall,
        # _autostop_drift_ref); break handled below, after the checkpoint save.
        _autostop_now, _autostop_stall, _autostop_drift_ref, _autostop_reason = _autostop_decision(
            selection_metric=val_selection_score,
            prev_best=_autostop_prev_best,
            weight_drift_l2=_weight_drift_l2,
            drift_ref=_autostop_drift_ref,
            stall=_autostop_stall,
            epoch=epoch,
            patience=config.autostop_patience,
            min_improve=config.autostop_min_improve,
            drift_frac=config.autostop_drift_frac,
            min_epoch=config.autostop_min_epoch,
            catastrophic_margin=config.autostop_catastrophic_margin,
        )
        if config.autostop_patience > 0:
            _ref_str = (f"{_autostop_drift_ref:.3f}" if _autostop_drift_ref is not None else "n/a")
            _log(
                f"  autostop: stall={_autostop_stall}/{config.autostop_patience} "
                f"drift={_weight_drift_l2:.3f} ref={_ref_str}"
            )

        # Save resumable checkpoint every epoch (latest + epoch-stamped).
        bc_opt = model._optimizers.get("bc")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": {
                k.replace("_orig_mod.", ""): v
                for k, v in model.model.state_dict().items()
            },
            "optimizer_state_dict": bc_opt.state_dict() if bc_opt else None,
            "best_selection_score": best_selection_score,
            "best_epoch": best_epoch,
            "history": history,
            "_best_move": _best_move,
            "_best_look": _best_look,
            "_best_max_reg": _best_max_reg,
            "_best_reg_epoch": _best_reg_epoch,
            "_reg_violations": _reg_violations,
            "_autostop_stall": _autostop_stall,
            "_autostop_drift_ref": _autostop_drift_ref,
            "rng_state": rng.bit_generator.state,
        }
        torch.save(ckpt_data, checkpoint_path)
        # Epoch completed cleanly — remove the rolling mid-epoch state.
        mid_epoch_path.unlink(missing_ok=True)
        # Epoch-stamped copy so we can resume from any epoch.
        epoch_ckpt_dir = output / "checkpoints"
        epoch_ckpt_dir.mkdir(exist_ok=True)
        torch.save(ckpt_data, epoch_ckpt_dir / f"bc_checkpoint_epoch{epoch:03d}.pt")

        # Archive checkpoint and best model to NAS.
        if _smb_available:
            try:
                import smbclient as _smb
                import shutil as _shutil
                for src in [checkpoint_path, output / "bc_best_model.pth"]:
                    if src.exists():
                        nas_dest = _variant_dir + "\\" + src.name
                        with open(src, "rb") as local_f:
                            with _smb.open_file(nas_dest, mode="wb") as remote_f:
                                _shutil.copyfileobj(local_f, remote_f)
            except Exception as exc:
                _log(f"NAS archive failed: {exc}")

        if cancel_event is not None and cancel_event.is_set():
            _log(f"Cancellation requested — stopping after epoch {epoch + 1}")
            break

        if _reg_violations >= config.regression_patience:
            _log(
                f"Regression stop: {config.regression_patience} consecutive epochs "
                f"above threshold {config.regression_threshold}. Best epoch: {best_epoch + 1}"
            )
            break

        if _autostop_now:
            _log(f"Autostop: {_autostop_reason}. Best epoch: {best_epoch + 1}")
            break

    if best_epoch < 0:
        model.save(output / "bc_best_model.pth")

    # Free the training model + optimizer state before loading the best
    # checkpoint for the final val pass. In normal single-run mode we also
    # release the device-resident TRAIN tensors to avoid briefly holding
    # training model + final_model + train data + val data. Shared-source
    # ablation runs keep those tensors alive on purpose.
    del model
    if release_sources:
        train_source.release_device_tensors()
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_model = QNNPolicy.load(
        output / "bc_best_model.pth", device=config.device, model_factory=model_factory,
    )
    # QNNPolicy.load builds a fresh policy without the side-channel provider, so
    # re-attach it — otherwise side-channel-dependent heads (GTTargetPointer,
    # engagement_ema, …) fail in the final eval pass. (The per-epoch `model` got it
    # at construction.)
    if side_channel_provider is not None:
        final_model._side_channel_provider = side_channel_provider
    # Defensive: QNNPolicy.load restores input_mask from the checkpoint
    # meta if present; for pre-fix checkpoints that lack the field the
    # default is False, which would silently swap the val label distribution.
    # Override from the BCConfig so the trainer's final-pass numbers always
    # match the training-time val pass.
    final_model.input_mask = bool(config.input_mask)
    final_model.attack_label_shift = bool(config.attack_label_shift)
    final_model.attack_op_only = bool(config.attack_op_only)

    if val_source.n_total_rows > 0:
        final_val_metrics = _run_epoch(
            final_model,
            val_source,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            tbptt_limit=config.tbptt_limit,
            head_loss_weights=hlw,
        )
    else:
        final_val_metrics = {"loss": 0.0}
    if release_sources:
        val_source.release_device_tensors()

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_selection_score": best_selection_score,
        "final_val_loss": float(final_val_metrics["loss"]),
        "num_train_samples": int(sample_counts["train"]),
        "num_val_samples": int(sample_counts["val"]),
        "epochs_ran": len(history),
    }
    if history:
        last = history[-1]
        for key in (
            "effective_train_rows_per_sec",
            "effective_val_rows_per_sec",
            "effective_train_eval_rows_per_sec",
            "train_rows",
            "val_rows",
            "train_eval_rows",
        ):
            if key in last:
                summary[f"final_{key}"] = float(last[key])
    if actual_fp is not None:
        summary["collection_fingerprint"] = actual_fp["fingerprint"]
    for key, value in final_val_metrics.items():
        if key == "_next_hidden":
            continue
        summary[f"final_val_{key}"] = float(value)

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}


# ── Runner entry point (called by run.router) ──────────────────────

def run(ctx: "RunnerContext") -> dict[str, object]:
    """Run BC pipeline from a frozen run directory."""
    import dataclasses as _dc
    import time as _time

    from qnn.run.config import build_run_bc_config
    from qnn.run.common import RunnerContext, base_results, finalize_results, prepare_bc_run_outputs

    results = base_results(ctx)
    stage_timings: dict[str, float] = {}

    bc_cfg = build_run_bc_config(ctx.run_cfg, ctx.device)
    prepare_bc_run_outputs(ctx.run_cfg, resume=ctx.resume)

    bc_data_dir = Path(bc_cfg.get("bc_data_dir", ""))
    train_cache = bc_data_dir / "precomputed_train"
    if not train_cache.exists():
        raise RuntimeError(
            f"BC training data not found at {train_cache}. "
            f"Run python -m qnn.bc.collect first."
        )

    seed_checkpoint = str(ctx.run_cfg.get("checkpoint_path", ""))
    started = _time.monotonic()
    valid_keys = {f.name for f in _dc.fields(BCConfig)}
    unknown = sorted(set(bc_cfg) - valid_keys)
    if unknown:
        raise RuntimeError(
            f"BC config has {len(unknown)} unknown key(s) (typo or removed feature): {unknown}. "
            "Either remove them from the run's train.json/model.json or add them to BCConfig."
        )
    results["bc"] = run_behavior_cloning(BCConfig(**bc_cfg), seed_checkpoint=seed_checkpoint)
    stage_timings["bc"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)


# ── Standalone eval entry point ────────────────────────────────────────────────
# python -m qnn.bc.train --eval-only --run-dir runs/bc/<name> [--data-dir ...]

def _eval_only(run_dir: Path, data_dir: Path | None, device: str, batch_size: int) -> None:
    import json as _json
    checkpoint = run_dir / "checkpoints" / "bc_best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No best-model checkpoint at {checkpoint}")

    if data_dir is None:
        machine_cfg = _json.loads((run_dir / "config" / "machine.json").read_text())
        data_dir = Path(machine_cfg["bc_data_dir"])

    val_cache = data_dir / "precomputed_val"
    if not val_cache.exists():
        raise FileNotFoundError(f"Val cache not found: {val_cache}")

    train_cfg = _json.loads((run_dir / "config" / "train.json").read_text())
    tbptt = int(train_cfg.get("tbptt_limit", 256))

    print(f"  checkpoint : {checkpoint}")
    print(f"  val data   : {val_cache}")
    print(f"  device     : {device}  batch_size: {batch_size}  tbptt: {tbptt}")

    model = QNNPolicy.load(str(checkpoint), device=device)
    model.model.eval()

    validate_cache_for_training(val_cache, required_actions=frozenset())
    source = _make_resident_source_from_cache(val_cache, model.device)
    print(f"  val episodes: {len(source.episodes)}")

    metrics = _run_epoch(
        model,
        source,
        batch_size=batch_size,
        sequence_length=0,
        tbptt_limit=tbptt,
    )

    print("\n--- val metrics ---")
    for k, v in sorted(metrics.items()):
        if k == "_next_hidden":
            continue
        print(f"  {k:<30s}  {v:.6f}")


if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Evaluate a BC best-model checkpoint on the val set.")
    _ap.add_argument("--eval-only", action="store_true", required=True)
    _ap.add_argument("--run-dir", type=Path, required=True, help="Run directory (contains config/ and checkpoints/)")
    _ap.add_argument("--data-dir", type=Path, default=None, help="Override bc_data_dir from machine.json")
    _ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    _ap.add_argument("--batch-size", type=int, default=256)
    _args = _ap.parse_args()
    _eval_only(_args.run_dir, _args.data_dir, _args.device, _args.batch_size)
