"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import time as _time

import numpy as np
import torch

from qnn import filter_dsl
from qnn.bc.class_weights import fire_class_weights
from qnn.schema import OBS_DIM
from qnn.model.policy import (
    HEAD_LOSS_WEIGHTS,
    QNNPolicy,
)
from qnn.utils.io import write_json
from qnn.utils.repro import set_global_seed, write_experiment_manifest


@dataclass(slots=True)
class BCConfig:
    """Behavior cloning configuration.

    Architecture defaults (trunk_hidden, gru_hidden, n_heads, n_layers,
    ffn_dim) must stay in sync with the frozen run's
    ``config/model.json`` — ``build_run_bc_config()`` injects those
    values from the run directory before training starts.
    """
    output_dir: str
    bc_data_dir: str
    seed: int
    batch_size: int
    sequence_length: int  # 0 = full episode (no chunking)
    epochs: int
    lr: float
    use_gru: bool
    # --- Architecture params (authoritative source: frozen run config/model.json) ---
    gru_hidden: int
    trunk_hidden: int
    # look smoothing is now applied during bc_collect.py, not at training time
    max_grad_norm: float  # gradient clipping for BPTT stability
    tbptt_limit: int  # max ticks before detaching gradient graph (0 = no limit)
    n_heads: int
    n_layers: int
    ffn_dim: int  # feedforward hidden dim in transformer layers (typically 4 * d_model)
    d_model: int
    attn_dropout: float
    # --- End architecture params ---
    fixed_tick_hz: int
    device: str
    # Per-head loss weights.  Heads not listed default to 1.0.
    head_loss_weights: str  # JSON string, e.g. '{"move":1.5,"weapon":0.0}'
    regression_threshold: float  # max acceptable regression from per-head best
    regression_patience: int  # consecutive epochs above threshold before stopping
    lr_min: float  # >0 enables cosine decay from lr to lr_min over all epochs
    warmup_epochs: int  # >0 enables linear warmup from ~0 to lr over this many epochs
    prometheus_pushgateway_url: str  # e.g. "http://pi:9091"; empty = disabled
    train_eval_interval: int  # run clean train eval every N epochs (0 = disabled)
    train_eval_gap_threshold: float  # early trigger when val - train proxy exceeds this
    train_eval_val_regression_threshold: float  # early trigger when val regresses beyond this
    train_eval_train_improve_threshold: float  # paired with val regression trigger
    # Performance tuning (sourced from machine.json by build_run_bc_config).
    pin_memory: bool           # pinned host buffers + non-blocking h2d
    prefetch: int              # 0 = off, N = keep N batches staged ahead
    microbatch_size: int       # forward-pass batch size for gradient accumulation (0 = batch_size)
    snapshot_interval: int     # seconds between rolling mid-epoch state snapshots (0 = disabled)
    dtype: str                 # "fp32" | "bf16" | "fp16" — autocast for forward+loss
    step_report_interval_seconds: int = 60  # wall-clock cadence for step logging (0 = every callback)
    use_weapon_head: bool = False  # v21 dense desired-weapon selector + action-head weapon context
    weapon_switch_confidence: float = 0.65  # live switch gate: top weapon probability threshold
    weapon_switch_margin: float = 0.15  # live switch gate: top1-top2 probability margin
    fire_pos_weight_override: float = 0.0  # >0 overrides the auto-computed pos_weight (neg/pos) for fire BCE loss
    jump_pos_weight: float = 1.0  # >1.0 upweights the POS class on the move ud-axis CE
    jump_pos_weight_end: float = -1.0  # >0 enables linear decay of jump_pos_weight from this start value to the end value across all epochs (epoch 0 = start, last epoch = end). -1.0 disables decay and holds at start.
    target_focal_gamma: float = 0.0  # >0 applies focal modulation (1-p_t)^gamma to the target-slot CE; concentrates gradient on hard (exception) frames where slot != 0

    weapon_use_gru: bool = True       # weapon_selector includes gru_flat (decisive: dropping cost f1_weapon -0.07)
    weapon_context_from_obs: bool = True  # motor heads condition on currently-held weapon from obs (self_weapon_id) instead of softmax(weapon_logits); decouples weapon intent from motor actuation so the 1-2 frame switch lag at inference doesn't corrupt fire/look/move
    gru_target_query: bool = False    # route GRU output (instead of self_readout) into the target attention query; planned follow-up for target commitment/hysteresis
    hard_target_feat: bool = False    # target_feat is the entity vector at a single chosen slot (BC GT during training, argmax at eval) instead of a soft attention pool. Decouples target-head loss tuning from motor-head training distribution
    weapon_in_target_query: bool = False  # additive currently-held weapon embed on the target attention query so target choice can condition on weapon (RL pulls distant, shotgun pulls close); independent of hard_target_feat
    linear_slot_prior: bool = False    # additive logit prior linear in slot index on the target pointer; encodes the slot ordering (slot 0 most likely) directly so the residual only has to override when a non-slot-0 target is better
    head_bottleneck_dim: "int | dict" = 192
    head_use_relu: bool = True         # ReLU is the operative ingredient (confirmed via disentangle); capacity reduction alone hurts
    head_activation: "str | None" = None  # "none" | "relu" | "gelu"; if None, derived from head_use_relu
    # Per-frame predicate (MongoDB DSL, qnn.filter_dsl) over the
    # stored action/obs arrays.  Each loaded episode is masked by the
    # predicate at load time; contiguous surviving runs become discrete
    # segments keyed by (demo_idx, episode_idx, segment_idx) so GRU
    # state resets at each boundary.  None / empty = no masking.
    # Example: {"act.target": {"$ne": -100}} = combat-only training.
    segment_mask: "dict | None" = None
    # Per-slot predicate (same MongoDB DSL) over entity token fields.
    # Slots where the predicate evaluates to False have their entity
    # arrays zeroed (entity_types set to -1, scalars/ids/event arrays to
    # 0); slot positions are preserved so target labels remain valid.
    # Target rows pointing into a zeroed slot are flipped to -100 so CE
    # skips them.  Field paths: ``type``, ``modality``, ``pid``,
    # ``route_idx``.  None / empty = no masking.  Equivalent to the
    # deprecated ``--entity-filter pvs_actors`` collect flag:
    #   {"type": 1, "pid": {"$gt": 0}, "modality": 0}
    # (Actors never get modality 1 = PROXIMITY; SIGHT is the only
    # PVS-visible modality for actors.)
    token_mask: "dict | None" = None
    # Expected collection identity (qnn.collection_fingerprint).  When
    # non-empty, the trainer verifies the data dir's fingerprint
    # matches before loading.  Empty = log-only mode.  See
    # ``qnn.collection_fingerprint`` for the override env var.
    collection_fingerprint: str = ""


from qnn.bc.loop import MidEpochState as _MidEpochState, PrecomputedEpisode as _PrecomputedEpisode, run_epoch as _run_precomputed_supervised


def _selection_score(metrics: Mapping[str, float]) -> float:
    """Composite selection metric for combat-objective BC.

    Lower is better. Each head contributes additively; missing metrics default
    to a neutral value so runs with subsets of heads still produce monotonic
    improvement signals.
    """
    target_error = 1.0 - float(metrics.get("acc_target", 1.0))
    # Move: 3-axis macro-F1 (each axis macro-averages the 3 classes
    # neg/none/pos).  Scaled by 3 to match the magnitude of the historical
    # axis-sum-of-error form so selection scores line up with prior runs.
    move_err = 3.0 * (1.0 - float(metrics.get("f1_move", 1.0)))
    # Look: cos_sim ranges in [-1, 1]; convert to a "1 - cos" error in [0, 2].
    look_err = 1.0 - float(metrics.get("cos_sim_look", 1.0))
    # Fire: F1 ranges in [0, 1]; convert to a "1 - f1" error in [0, 1].
    fire_f1 = float(metrics.get("f1_fire_global", metrics.get("f1_fire", 1.0)))
    fire_err = 1.0 - fire_f1
    # Weapon: macro-F1 across 8 classes — equal weight regardless of
    # frequency so rare-weapon failures don't disappear into the dominant
    # rocket-launcher class.
    weapon_f1 = float(metrics.get("f1_weapon_global", metrics.get("f1_weapon", 1.0)))
    weapon_err = 1.0 - weapon_f1
    return target_error + move_err + look_err + fire_err + weapon_err


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





# --- Data loading ---

def _unpack_move_axes(packed: np.ndarray) -> np.ndarray:
    """Expand the on-disk packed move byte to (T, 3) uint8 axis class indices.

    The collector packs three 3-class axis indices (each in {0=neg, 1=none,
    2=pos}) into bits 0-1 (fb), 2-3 (lr), 4-5 (ud) of a single uint8.
    Materializes a fresh array (no longer mmap-backed) — fine because action
    labels are tiny relative to obs.
    """
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError(f"expected (T,) packed move, got shape {arr.shape}")
    fb = (arr      ) & 0x3
    lr = (arr >> 2 ) & 0x3
    ud = (arr >> 4 ) & 0x3
    return np.ascontiguousarray(np.stack([fb, lr, ud], axis=-1))


def _madvise_sequential(arr: np.ndarray) -> None:
    """Hint the kernel to read-ahead and drop pages behind the cursor.

    Mmap'd training shards can be tens of GB.  Without this hint the
    page cache fills with every page ever touched, competing with WSL2
    VM memory.  MADV_SEQUENTIAL lets the kernel reclaim pages that the
    training loop has already consumed.
    """
    import mmap as mmap_mod
    mm = getattr(arr, '_mmap', None)
    if mm is not None and hasattr(mm, 'madvise'):
        mm.madvise(mmap_mod.MADV_SEQUENTIAL)


def _effective_head_loss_weights(raw: str) -> Dict[str, float]:
    weights = dict(HEAD_LOSS_WEIGHTS)
    if not raw:
        return weights
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"head_loss_weights must be a JSON object, got {type(parsed).__name__}")
    for head, value in parsed.items():
        weights[str(head)] = float(value)
    return weights


def _require_action_files(
    action_files: Mapping[str, str],
    required_actions: frozenset[str],
    *,
    cache_dir: Path,
) -> None:
    missing = sorted(required_actions.difference(action_files))
    if not missing:
        return
    raise RuntimeError(
        f"{cache_dir} is missing required action arrays {missing}. "
        "Recollect BC data on this branch before training."
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
    for chan, arr in obs.items():
        flat[f"obs.{chan}"] = arr
    return flat


def _split_episode_on_mask(obs: dict, actions: dict,
                            mask: np.ndarray) -> list[tuple[dict, dict, int]]:
    """Split an episode into one ``(obs, actions, n_samples)`` tuple per
    surviving run in ``mask``.  Runs are emitted in source-frame order
    so the i-th run gets segment_idx = i deterministically. """
    from qnn.bc.collect import _runs_from_mask  # local import: shared helper
    runs = _runs_from_mask(mask)
    out: list[tuple[dict, dict, int]] = []
    for s, e in runs:
        sub_obs = {k: v[s:e] for k, v in obs.items()}
        sub_act = {k: v[s:e] for k, v in actions.items()}
        out.append((sub_obs, sub_act, int(e - s)))
    return out


def _load_precomputed(
    cache_dir: Path,
    *,
    required_actions: frozenset[str] = frozenset(),
    segment_mask: dict | None = None,
    token_mask: dict | None = None,
) -> list[_PrecomputedEpisode]:
    """Load precomputed episodes with real memory-mapped .npy arrays.

    Episodes are returned sorted globally by ``(demo_idx, episode_idx,
    segment_idx)``.  ``demo_idx`` is the position of each demo in the
    collector's canonical sorted demo list; ``episode_idx`` is the
    0-based ordinal of each surviving run when the collector segmented
    the demo on the filter config's ``drop_tick_labels`` mask;
    ``segment_idx`` is the 0-based ordinal of each surviving run inside
    that episode after applying the train-time ``segment_mask``
    predicate (or 0 if no mask is set).  This makes training-time
    shuffle a pure function of the seed, the dataset, and the mask —
    independent of which worker finished first during collection.

    Shards without ``demo_idxs`` fall back to load order; shards
    without ``episode_idxs`` default to a single episode_idx=0 per
    episode.  No ``segment_mask`` keeps each episode as one
    ``segment_idx=0`` trajectory (today's behavior). """
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != "sharded_v1":
        raise RuntimeError(
            f"{cache_dir}/manifest.json: expected sharded_v1 format. "
            "Recollect BC data with the current collector."
        )

    indexed: list[tuple[tuple[int, int, int], _PrecomputedEpisode]] = []
    fallback_idx = 0
    for shard in manifest.get("shards", []):
        _require_action_files(shard["actions"], required_actions, cache_dir=cache_dir)
        obs_arrays = {
            key: np.load(cache_dir / fname, mmap_mode="r")
            for key, fname in shard["obs"].items()
        }
        action_arrays = {
            head: np.load(cache_dir / fname, mmap_mode="r")
            for head, fname in shard["actions"].items()
        }
        # Unpack the bit-packed move byte to (T, 6) once per shard so
        # per-episode slices below produce (n_samples, 6) views directly.
        if "move" in action_arrays:
            action_arrays["move"] = _unpack_move_axes(action_arrays["move"])
        for arr in obs_arrays.values():
            _madvise_sequential(arr)
        for arr in action_arrays.values():
            if isinstance(arr, np.memmap):
                _madvise_sequential(arr)
        episode_lengths = shard.get("episode_lengths", [])
        demo_idxs = shard.get("demo_idxs")
        if demo_idxs is None or len(demo_idxs) != len(episode_lengths):
            demo_idxs = list(range(fallback_idx, fallback_idx + len(episode_lengths)))
        fallback_idx += len(episode_lengths)
        episode_idxs = shard.get("episode_idxs")
        if episode_idxs is None or len(episode_idxs) != len(episode_lengths):
            episode_idxs = [0] * len(episode_lengths)
        start = 0
        for n_samples, demo_idx, episode_idx in zip(
                episode_lengths, demo_idxs, episode_idxs):
            end = start + int(n_samples)
            obs = {key: values[start:end] for key, values in obs_arrays.items()}
            actions = {head: values[start:end] for head, values in action_arrays.items()}
            if token_mask:
                from qnn.bc.token_filter import apply_token_mask, clear_targets_on_masked_slots
                obs = apply_token_mask(obs, token_mask)
                if "target" in actions:
                    actions["target"] = clear_targets_on_masked_slots(
                        actions["target"], obs,
                    )
            if segment_mask:
                flat = _flatten_episode_arrays(obs, actions)
                mask = np.asarray(filter_dsl.eval_filter(flat, segment_mask), dtype=bool)
                runs = _split_episode_on_mask(obs, actions, mask)
                for segment_idx, (sub_obs, sub_act, n) in enumerate(runs):
                    indexed.append((
                        (int(demo_idx), int(episode_idx), int(segment_idx)),
                        _PrecomputedEpisode(
                            obs=sub_obs,
                            actions=sub_act,
                            n_samples=n,
                            sort_key=(int(demo_idx), int(episode_idx), int(segment_idx)),
                        ),
                    ))
            else:
                indexed.append((
                    (int(demo_idx), int(episode_idx), 0),
                    _PrecomputedEpisode(
                        obs=obs,
                        actions=actions,
                        n_samples=int(n_samples),
                        sort_key=(int(demo_idx), int(episode_idx), 0),
                    ),
                ))
            start = end
    indexed.sort(key=lambda item: item[0])
    return [ep for _, ep in indexed]



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

def run_behavior_cloning(config: BCConfig, seed_checkpoint: str = "") -> Dict[str, float]:
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
            print(f"  [bc] Prometheus pushgateway: {env_url}")

    if not str(config.output_dir).strip():
        raise RuntimeError("Behavior cloning requires output_dir")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load precomputed .npy caches (produced by python -m qnn.bc.collect)
    bc_data_dir = Path(config.bc_data_dir) if hasattr(config, "bc_data_dir") else Path(config.output_dir).parent
    train_cache = bc_data_dir / "precomputed_train"
    val_cache = bc_data_dir / "precomputed_val"
    if not train_cache.exists():
        raise RuntimeError(f"BC training data not found at {train_cache}. Run python -m qnn.bc.collect first.")

    # Verify the dataset identity matches what the run expects.  When
    # config.collection_fingerprint is set the check is strict (raises
    # FingerprintMismatch); otherwise the current fingerprint is just
    # logged + stamped into bc_summary for audit.
    from qnn import collection_fingerprint
    expected_fp = config.collection_fingerprint or None
    actual_fp = collection_fingerprint.verify(
        expected_fingerprint=expected_fp,
        data_dir=bc_data_dir,
    )
    if actual_fp is not None:
        print(f"  [bc] collection fingerprint: {actual_fp['fingerprint']}")
    elif expected_fp is None:
        print(f"  [bc] collection fingerprint: (absent — pre-fingerprint collection)")

    head_loss_weights = _effective_head_loss_weights(config.head_loss_weights)
    required_actions_set: set[str] = set()
    if head_loss_weights.get("target", 1.0) > 0.0:
        required_actions_set.add("target")
    if config.use_weapon_head and head_loss_weights.get("weapon", 1.0) > 0.0:
        required_actions_set.add("weapon")
    required_actions = frozenset(required_actions_set)

    print(f"  [bc] Loading training data: {train_cache}")
    if config.segment_mask:
        print(f"  [bc] segment_mask: {config.segment_mask}")
    if config.token_mask:
        print(f"  [bc] token_mask: {config.token_mask}")
    train_episodes = _load_precomputed(
        train_cache, required_actions=required_actions,
        segment_mask=config.segment_mask,
        token_mask=config.token_mask,
    )
    val_episodes = _load_precomputed(
        val_cache, required_actions=required_actions,
        segment_mask=config.segment_mask,
        token_mask=config.token_mask,
    ) if val_cache.exists() else []

    sample_counts = {
        "train": sum(ep.n_samples for ep in train_episodes),
        "val": sum(ep.n_samples for ep in val_episodes),
    }

    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available")

    # Configure mixed-precision autocast via the env var that QNNPolicy reads.
    os.environ["QNN_AUTOCAST_DTYPE"] = config.dtype
    print(f"  [bc] dtype={config.dtype}")

    obs_dim = OBS_DIM
    if seed_checkpoint and Path(seed_checkpoint).exists():
        print(f"  [bc] Fine-tuning from seed: {seed_checkpoint}")
        model = QNNPolicy.load(seed_checkpoint, device=config.device)
    else:
        model = QNNPolicy(
            obs_dim=obs_dim,
            trunk_hidden=config.trunk_hidden,
            gru_hidden=config.gru_hidden,
            use_gru=config.use_gru,
            seed=config.seed,
            device=config.device,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            ffn_dim=config.ffn_dim,
            attn_dropout=config.attn_dropout,
            use_weapon_head=config.use_weapon_head,
            weapon_switch_confidence=config.weapon_switch_confidence,
            weapon_switch_margin=config.weapon_switch_margin,
            jump_pos_weight=config.jump_pos_weight,
            target_focal_gamma=config.target_focal_gamma,
            weapon_use_gru=config.weapon_use_gru,
            weapon_context_from_obs=config.weapon_context_from_obs,
            gru_target_query=config.gru_target_query,
            hard_target_feat=config.hard_target_feat,
            weapon_in_target_query=config.weapon_in_target_query,
            linear_slot_prior=config.linear_slot_prior,
            head_bottleneck_dim=config.head_bottleneck_dim,
            head_use_relu=config.head_use_relu,
            head_activation=config.head_activation,
        )

    weights = fire_class_weights(
        train_episodes,
        head_loss_weights=head_loss_weights,
        override=float(config.fire_pos_weight_override),
        device=model.device,
    )

    # Parse per-head loss weights from JSON string if provided.
    hlw: Dict[str, float] | None = None
    if config.head_loss_weights:
        hlw = dict(head_loss_weights)

    best_val_loss = float("inf")
    best_epoch = -1
    history: list[Dict[str, float]] = []
    start_epoch = 0

    # Regression-based stopping state.
    _best_move = float("inf")
    _best_look = float("inf")
    _best_max_reg = float("inf")  # for checkpoint selection: min of max(move_reg, look_reg)
    _best_reg_epoch = -1
    _reg_violations = 0

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
        print(f"  [bc] NAS archive available: {_variant_dir}")
    except Exception:
        _smb_available = False
        print("  [bc] NAS archive not available — skipping offsite backup")

    # Mid-epoch state: rolling file for deterministic resume within an epoch.
    mid_epoch_path = output / "snapshot.pt"
    _MID_EPOCH_SAVE_INTERVAL = config.snapshot_interval

    # Resume from checkpoint if available.
    checkpoint_path = output / "bc_training_checkpoint.pt"
    if checkpoint_path.exists():
        import torch as _torch_resume
        from qnn.utils.checkpoint_converter import migrate_entity_embed, migrate_self_scalars
        ckpt = _torch_resume.load(checkpoint_path, map_location=model.device, weights_only=False)
        migrate_entity_embed(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        migrate_self_scalars(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
        model.model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        _best_move = ckpt.get("_best_move", float("inf"))
        _best_look = ckpt.get("_best_look", float("inf"))
        _best_max_reg = ckpt.get("_best_max_reg", float("inf"))
        _best_reg_epoch = ckpt.get("_best_reg_epoch", -1)
        _reg_violations = ckpt.get("_reg_violations", 0)
        # Optimizer state restored after first supervised step creates it.
        _resume_optimizer_state = ckpt.get("optimizer_state_dict")
        # Restore rng state so resume produces the same episode ordering
        # as a continuous run.
        _saved_rng_state = ckpt.get("rng_state")
        if _saved_rng_state is not None:
            rng.bit_generator.state = _saved_rng_state
        print(f"  [bc] Resuming from epoch {start_epoch} (best_val={best_val_loss:.4f} at epoch {best_epoch})")
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
                print(f"  [bc] Mid-epoch resume: epoch {start_epoch}, "
                      f"step {_mid_epoch_resume.opt_steps}, "
                      f"chunk {_mid_epoch_resume.next_episode}")
            else:
                mid_epoch_path.unlink()
        except Exception as exc:
            print(f"  [bc] Mid-epoch state load failed: {exc}")
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
        print(f"  [bc]   step {int(step_metrics.get('opt_step', 0)):>5d}  "
              f"loss={step_metrics.get('loss', 0):.4f}  "
              f"{'  '.join(mae_parts)}")
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
            print(f"  [bc] jump_pos_weight (decay): epoch {epoch}/{config.epochs - 1}  alpha={alpha:.3f}  pw={current_pw:.3f}")
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
                print(f"  [bc] lr_override.json: lr={_lr}, lr_min={_lr_min}")
            except Exception as exc:
                print(f"  [bc] lr_override.json parse error: {exc}")

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
            print(f"  [bc] LR={_active_lr:.6f}")

        _t_train_start = _time.monotonic()
        train_metrics = _run_precomputed_supervised(
            model,
            train_episodes,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            class_weights=weights,
            lr=_active_lr,
            rng=rng,
            max_grad_norm=config.max_grad_norm,
            tbptt_limit=config.tbptt_limit,
            head_loss_weights=hlw,
            step_callback=_on_step,
            report_every=_report_every,
            report_interval_seconds=float(_step_report_interval),
            pin_memory=config.pin_memory,
            prefetch=config.prefetch,
            microbatch_size=config.microbatch_size,
            save_state_callback=_save_mid_epoch,
            snapshot_interval=_MID_EPOCH_SAVE_INTERVAL,
            resume_state=_mid_epoch_resume,
        )
        # Mid-epoch state consumed — don't reuse on next epoch.
        _mid_epoch_resume = None
        _t_train_end = _time.monotonic()
        # Restore optimizer state on first epoch after resume.
        if _resume_optimizer_state is not None:
            bc_opt = model._optimizers.get("bc")
            if bc_opt is not None:
                bc_opt.load_state_dict(_resume_optimizer_state)
                _resume_optimizer_state = None
        _t_val_start = _time.monotonic()
        val_metrics = _run_precomputed_supervised(
            model,
            val_episodes,
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            tbptt_limit=config.tbptt_limit,
            head_loss_weights=hlw,
            pin_memory=config.pin_memory,
            prefetch=config.prefetch,
            microbatch_size=config.microbatch_size,
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
        train_eval_ran = bool(val_episodes) and bool(train_eval_reasons)
        if train_eval_ran:
            # Clean train eval (model.eval mode, no dropout) on a train subset
            # only when scheduled or when proxy metrics suggest a gap issue.
            _t_train_eval_start = _time.monotonic()
            train_eval_metrics = _run_precomputed_supervised(
                model,
                train_episodes[:len(val_episodes)],
                batch_size=config.batch_size,
                sequence_length=config.sequence_length,
                tbptt_limit=config.tbptt_limit,
                head_loss_weights=hlw,
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
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
            print(
                f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  "
                f"train_eval={_train_eval_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]"
            )
        else:
            print(f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]")
        # Headline per-head summary: one number per head (F1 where the
        # class imbalance makes accuracy misleading), plus per-axis move F1
        # so axis-specific regressions surface in the log.
        _headline_keys = (
            "acc_target",
            "f1_move", "f1_move_fb", "f1_move_lr", "f1_move_ud",
            "cos_sim_look",
            "f1_fire",
            "f1_weapon",
        )
        mae_str = "  ".join(
            f"{k}={float(val_metrics[k]):.4f}"
            for k in _headline_keys if k in val_metrics
        )

        # Composite selection: target acc + move/fire/weapon macro-F1 + look cos.
        val_selection_score = _selection_score(val_metrics)
        selection_metric = val_selection_score
        improved = selection_metric < best_val_loss

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

        epoch_line = (
            f"  [bc] Epoch {epoch + 1}/{config.epochs}  "
            f"train_proxy={train_proxy_sum:.4f}  "
            f"val={val_selection_score:.4f}  "
            f"proxy_gap={train_proxy_gap:+.4f}  "
        )
        if train_eval_ran and train_eval_sum is not None:
            epoch_line += (
                f"train_eval={train_eval_sum:.4f}  "
                f"gap={val_selection_score - train_eval_sum:+.4f}  "
                f"[{','.join(train_eval_reasons)}]  "
            )
        else:
            epoch_line += "train_eval=skipped  "
        epoch_line += f"{'*' if improved else ''}  "
        if _grad_mean is not None:
            epoch_line += (
                f"grad_mean={_grad_mean:.3f}  "
                f"grad_max={_grad_max:.3f}  "
            )
        epoch_line += f"drift={_weight_drift_l2:.3f}  "
        epoch_line += f"train_rps={train_rows_per_sec:.1f}  val_rps={val_rows_per_sec:.1f}  "
        epoch_line += mae_str
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
        # for stopping, not model selection.
        if selection_metric < best_val_loss:
            best_val_loss = selection_metric
            best_epoch = epoch
            model.save(output / "bc_best_model.pth")

        if move_reg > config.regression_threshold or look_reg > config.regression_threshold:
            _reg_violations += 1
        else:
            _reg_violations = 0

        print(f"  [bc]   regression: move={move_reg:+.4f} look={look_reg:+.4f} "
              f"violations={_reg_violations}/{config.regression_patience}")

        # Save resumable checkpoint every epoch (latest + epoch-stamped).
        bc_opt = model._optimizers.get("bc")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": {
                k.replace("_orig_mod.", ""): v
                for k, v in model.model.state_dict().items()
            },
            "optimizer_state_dict": bc_opt.state_dict() if bc_opt else None,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "_best_move": _best_move,
            "_best_look": _best_look,
            "_best_max_reg": _best_max_reg,
            "_best_reg_epoch": _best_reg_epoch,
            "_reg_violations": _reg_violations,
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
                print(f"  [bc] NAS archive failed: {exc}")

        if _reg_violations >= config.regression_patience:
            print(f"  [bc] Regression stop: {config.regression_patience} consecutive epochs "
                  f"above threshold {config.regression_threshold}. Best epoch: {best_epoch + 1}")
            break

    if best_epoch < 0:
        model.save(output / "bc_best_model.pth")

    final_model = QNNPolicy.load(output / "bc_best_model.pth", device=config.device)

    final_val_metrics = _run_precomputed_supervised(
        final_model,
        val_episodes,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        tbptt_limit=config.tbptt_limit,
        pin_memory=config.pin_memory,
        prefetch=config.prefetch,
    ) if val_episodes else {"loss": 0.0}

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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

    val_episodes = _load_precomputed(val_cache)
    print(f"  val episodes: {len(val_episodes)}")

    metrics = _run_precomputed_supervised(
        model,
        val_episodes,
        batch_size=batch_size,
        sequence_length=0,
        tbptt_limit=tbptt,
        pin_memory=False,
        prefetch=0,
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
