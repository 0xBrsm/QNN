"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from quake_ai.actions import (
    ACTION_HEADS,
    CONTINUOUS_ACTION_HEADS,
    DISCRETE_ACTION_HEADS,
)
from quake_ai.model.observation import OBS_DIM
from quake_ai.model.policy import QNNPolicy
from quake_ai.utils.io import write_json
from quake_ai.utils.repro import set_global_seed, write_experiment_manifest

_ACTION_HEAD_NAMES = list(ACTION_HEADS.keys())
_CONTINUOUS_HEAD_NAMES = [head for head in _ACTION_HEAD_NAMES if head in CONTINUOUS_ACTION_HEADS]
_DISCRETE_HEAD_NAMES = [head for head in _ACTION_HEAD_NAMES if head in DISCRETE_ACTION_HEADS]
_RAW_SUM_METRIC_PREFIXES = (
    "n_",
    "correct_",
    "l1_sum_",
    "tp_",
    "fp_",
    "fn_",
    "target_pos_",
    "pred_pos_",
)
_AVERAGED_METRIC_PREFIXES = (
    "acc_",
    "mae_",
)


@dataclass(slots=True)
class BCConfig:
    """Behavior cloning configuration.

    Architecture defaults (trunk_hidden, gru_hidden, n_heads, n_layers,
    ffn_dim, readout, action_history_tokens) must stay in sync with
    the frozen run's ``config/model.json`` — ``build_run_bc_config()`` injects
    those values from the run directory before training starts.
    """
    output_dir: str
    bc_data_dir: str
    seed: int
    batch_size: int
    sequence_length: int  # 0 = full episode (no chunking)
    epochs: int
    lr: float
    patience: int
    use_gru: bool
    # --- Architecture params (authoritative source: frozen run config/model.json) ---
    gru_hidden: int
    trunk_hidden: int
    class_weight_power: float
    class_weight_min: float
    class_weight_max: float
    # look smoothing is now applied during bc_collect.py, not at training time
    max_grad_norm: float  # gradient clipping for BPTT stability
    tbptt_limit: int  # max ticks before detaching gradient graph (0 = no limit)
    n_heads: int
    n_layers: int
    ffn_dim: int  # feedforward hidden dim in transformer layers (typically 4 * d_model)
    d_model: int
    attn_dropout: float
    action_history_tokens: int  # number of recent action ticks as transformer tokens (0 = disabled)
    readout: str  # "cls" or "self"
    # --- End architecture params ---
    fixed_tick_hz: int
    device: str
    # Per-head loss weights.  Heads not listed default to 1.0.
    # Set recall_0..3 to 0.0 to exclude them from gradient budget.
    head_loss_weights: str  # JSON string, e.g. '{"move":1.5,"recall_0":0.0}'
    focal_gamma: float  # 0.0 = standard CE, >0 = focal loss for discrete heads
    sparse_discrete: bool  # true = sparse masking for fire/jump/switch, false = use focal/CE on all ticks
    look_deadzone: float  # >0 zeros look labels where turn magnitude < threshold; 0.0 = disabled
    regression_stop: bool  # use regression-based stopping instead of patience
    regression_threshold: float  # max acceptable regression from per-head best
    regression_patience: int  # consecutive epochs above threshold before stopping
    lr_min: float  # >0 enables cosine decay from lr to lr_min over all epochs
    warmup_epochs: int  # >0 enables linear warmup from ~0 to lr over this many epochs
    prometheus_pushgateway_url: str  # e.g. "http://pi:9091"; empty = disabled


@dataclass(slots=True)
class _PrecomputedEpisode:
    """One episode's observations and actions as contiguous arrays."""
    obs: dict[str, np.ndarray]      # key → (n_samples, ...) arrays
    actions: dict[str, np.ndarray]   # head → (n_samples, ...) arrays
    n_samples: int



# --- Action class counting ---

def _init_action_counts() -> dict[str, np.ndarray]:
    return {head: np.ones(ACTION_HEADS[head], dtype=np.float32) for head in _DISCRETE_HEAD_NAMES}


def _accumulate_action_counts(
    ep: _PrecomputedEpisode,
    action_counts: dict[str, np.ndarray],
) -> None:
    for head in _DISCRETE_HEAD_NAMES:
        np.add.at(action_counts[head], ep.actions[head], 1.0)


def _class_weights_from_counts(
    counts: Mapping[str, np.ndarray],
    *,
    power: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for head, values in counts.items():
        scaled = np.power(values.astype(np.float32, copy=False), -float(power)).astype(np.float32, copy=False)
        scaled = scaled / max(float(np.mean(scaled)), 1e-8)
        scaled = np.clip(scaled, float(min_weight), float(max_weight))
        weights[head] = scaled.astype(np.float32, copy=False)
    return weights



# --- Data loading ---

def _load_precomputed(cache_dir: Path) -> list[_PrecomputedEpisode]:
    """Load precomputed episodes with real memory-mapped .npy arrays."""
    import json
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    episodes: list[_PrecomputedEpisode] = []
    for entry in manifest:
        obs = {key: np.load(cache_dir / fname, mmap_mode="r")
               for key, fname in entry["obs"].items()}
        actions = {head: np.load(cache_dir / fname, mmap_mode="r")
                   for head, fname in entry["actions"].items()}
        episodes.append(_PrecomputedEpisode(obs=obs, actions=actions, n_samples=entry["n_samples"]))
    return episodes



# --- Training loop ---

def _generate_chunk_indices(
    episodes: Sequence[_PrecomputedEpisode],
    sequence_length: int,
) -> list[tuple[int, int, int]]:
    """Generate (episode_idx, start_offset, chunk_length) for all chunks."""
    chunk_len = max(int(sequence_length), 1)
    indices: list[tuple[int, int, int]] = []
    for ep_idx, ep in enumerate(episodes):
        for start in range(0, ep.n_samples, chunk_len):
            length = min(chunk_len, ep.n_samples - start)
            indices.append((ep_idx, start, length))
    return indices


def _run_precomputed_supervised(
    model: QNNPolicy,
    episodes: Sequence[_PrecomputedEpisode],
    batch_size: int,
    sequence_length: int,
    *,
    class_weights: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    lr: float | None = None,
    rng: np.random.Generator | None = None,
    max_grad_norm: float = 1.0,
    tbptt_limit: int = 256,
    head_loss_weights: Mapping[str, float] | None = None,
    focal_gamma: float = 0.0,
    sparse_discrete: bool = True,
    look_deadzone: float = 0.0,
    step_callback: Any | None = None,  # called every report_every optimizer steps with running metrics
    report_every: int = 0,  # 0 = disabled
) -> Dict[str, float]:
    _empty: Dict[str, float] = {"loss": 0.0, "accuracy": 0.0, "n_rows": 0.0}
    if not episodes:
        return _empty

    # Process each episode as one continuous sequence with GRU hidden state
    # carried forward.  Gradients are truncated every *tbptt_limit* ticks to
    # cap memory, but the hidden state itself is never reset within an episode.
    # Optimizer steps every _ACCUM_CHUNKS chunks (not episodes) for frequent
    # weight updates while maintaining gradient stability.
    training = class_weights is not None and lr is not None
    if training:
        model.model.train()
    else:
        model.model.eval()
    use_full_episode = int(sequence_length) <= 0
    _ACCUM_CHUNKS = max(1, int(batch_size))  # step optimizer every N chunks

    # Shuffle episode order (not tick order within episodes).
    ep_order: list[int] = list(range(len(episodes)))
    if rng is not None:
        ep_order = [int(i) for i in rng.permutation(len(episodes))]

    total_rows = 0
    total_loss = 0.0
    total_accuracy = 0.0
    raw_metric_totals: Dict[str, float] = {}
    averaged_metric_totals: Dict[str, float] = {}
    accum_count = 0

    opt_steps = 0
    _report_rows = 0
    _report_loss = 0.0
    _report_avg_totals: Dict[str, float] = {}
    _report_raw_totals: Dict[str, float] = {}

    if training:
        model.bc_zero_grad()

    for ep_idx in ep_order:
        ep = episodes[ep_idx]
        if ep.n_samples == 0:
            continue

        hidden = None  # Reset GRU at episode start.
        # Full episode: chunk by tbptt_limit for gradient truncation only.
        # The hidden state carries across chunks (detached from graph).
        if use_full_episode:
            chunk_size = max(int(tbptt_limit), 64) if tbptt_limit > 0 else ep.n_samples
        else:
            chunk_size = max(int(sequence_length), 1)

        for start in range(0, ep.n_samples, chunk_size):
            end = min(start + chunk_size, ep.n_samples)
            length = end - start

            # Slice this chunk: shape (length, 1, ...) for batch_size=1
            obs_chunk: dict[str, np.ndarray] = {}
            for key, arr in ep.obs.items():
                obs_chunk[key] = arr[start:end].reshape(length, 1, *arr.shape[1:])
            act_chunk: dict[str, np.ndarray] = {}
            for head in _ACTION_HEAD_NAMES:
                chunk = ep.actions[head][start:end]
                if head in _CONTINUOUS_HEAD_NAMES:
                    act_chunk[head] = chunk.reshape(length, 1, ACTION_HEADS[head])
                else:
                    act_chunk[head] = chunk.reshape(length, 1)

            if training:
                metrics = model.supervised_step(
                    obs_chunk, act_chunk, class_weights, lr=lr,
                    hidden=hidden,
                    accumulate_only=True,
                    head_loss_weights=head_loss_weights,
                    focal_gamma=focal_gamma,
                    sparse_discrete=sparse_discrete,
                    look_deadzone=look_deadzone,
                )
            else:
                metrics = model.evaluate_supervised(
                    obs_chunk, act_chunk,
                    hidden=hidden,
                    focal_gamma=focal_gamma,
                    sparse_discrete=sparse_discrete,
                    look_deadzone=look_deadzone,
                )

            # Carry hidden state forward (detached to truncate BPTT).
            next_h = metrics.pop("_next_hidden", None)
            if next_h is not None:
                hidden = next_h.detach() if hasattr(next_h, "detach") else next_h

            total_rows += length
            total_loss += float(metrics["loss"]) * length
            total_accuracy += float(metrics["accuracy"]) * length
            for key, val in metrics.items():
                if key in {"loss", "accuracy", "_next_hidden"}:
                    continue
                if key.startswith(_RAW_SUM_METRIC_PREFIXES):
                    raw_metric_totals[key] = raw_metric_totals.get(key, 0.0) + float(val)
                elif key.startswith(_AVERAGED_METRIC_PREFIXES):
                    averaged_metric_totals[key] = averaged_metric_totals.get(key, 0.0) + float(val) * length

            # Step optimizer every N chunks (across episodes, GRU unaffected).
            if training:
                accum_count += 1
                if accum_count >= _ACCUM_CHUNKS:
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
                    model.bc_step()
                    model.bc_zero_grad()
                    accum_count = 0
                    opt_steps += 1

                    # Accumulate running metrics for periodic reporting.
                    if step_callback and report_every > 0:
                        _report_rows += length
                        _report_loss += float(metrics["loss"]) * length
                        for key, val in metrics.items():
                            if key.startswith(_AVERAGED_METRIC_PREFIXES):
                                _report_avg_totals[key] = _report_avg_totals.get(key, 0.0) + float(val) * length
                            elif key.startswith(_RAW_SUM_METRIC_PREFIXES):
                                _report_raw_totals[key] = _report_raw_totals.get(key, 0.0) + float(val)

                        if opt_steps % report_every == 0:
                            rd = max(_report_rows, 1)
                            step_metrics = {"loss": _report_loss / rd, "n_rows": float(_report_rows), "opt_step": opt_steps}
                            for key, total in _report_avg_totals.items():
                                step_metrics[key] = total / rd
                            for key, total in _report_raw_totals.items():
                                step_metrics[key] = total
                            step_callback(step_metrics)
                            _report_rows = 0
                            _report_loss = 0.0
                            _report_avg_totals.clear()
                            _report_raw_totals.clear()

    # Flush any remaining accumulated gradients.
    if training and accum_count > 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_grad_norm)
        model.bc_step()

    denom = max(total_rows, 1)
    result: Dict[str, float] = {
        "loss": total_loss / denom,
        "accuracy": total_accuracy / denom,
        "n_rows": float(total_rows),
    }
    for key, total in raw_metric_totals.items():
        result[key] = total
    for key, total in averaged_metric_totals.items():
        result[key] = total / denom
    return result


# ---------------------------------------------------------------------------
# Prometheus pushgateway integration (optional).
# ---------------------------------------------------------------------------

_PROM_METRICS_TO_PUSH = (
    "val_mae_move", "val_mae_look", "val_mae_move_forward", "val_mae_move_strafe",
    "val_mae_look_yaw", "val_mae_look_pitch", "train_loss", "val_loss",
    "train_mae_move", "train_mae_look",
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
    rng = np.random.default_rng(config.seed)

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

    # Load precomputed .npy caches (produced by python -m quake_ai.rl.bc_collect)
    bc_data_dir = Path(config.bc_data_dir) if hasattr(config, "bc_data_dir") else Path(config.output_dir).parent
    train_cache = bc_data_dir / "precomputed_train"
    val_cache = bc_data_dir / "precomputed_val"
    if not train_cache.exists():
        raise RuntimeError(f"BC training data not found at {train_cache}. Run python -m quake_ai.rl.bc_collect first.")

    print(f"  [bc] Loading training data: {train_cache}")
    train_episodes = _load_precomputed(train_cache)
    val_episodes = _load_precomputed(val_cache) if val_cache.exists() else []

    sample_counts = {
        "train": sum(ep.n_samples for ep in train_episodes),
        "val": sum(ep.n_samples for ep in val_episodes),
    }

    # Accumulate action class counts
    import json as _json_counts
    counts_cache = train_cache / "class_counts.json"
    if counts_cache.exists():
        print(f"  [bc] Loading cached class counts: {counts_cache}")
        _raw = _json_counts.loads(counts_cache.read_text())
        class_counts = {h: np.array(v, dtype=np.float32) for h, v in _raw.items()}
    else:
        class_counts = _init_action_counts()
        for ep in train_episodes:
            _accumulate_action_counts(ep, class_counts)
        counts_cache.write_text(_json_counts.dumps({h: v.tolist() for h, v in class_counts.items()}))
        print(f"  [bc] Cached class counts: {counts_cache}")

    if sample_counts["train"] <= 0:
        raise RuntimeError("No training samples available")

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
            action_history_tokens=config.action_history_tokens,
            readout=config.readout,
        )

    weights = {
        head: torch.as_tensor(values, dtype=torch.float32, device=model.device)
        for head, values in _class_weights_from_counts(
            class_counts,
            power=config.class_weight_power,
            min_weight=config.class_weight_min,
            max_weight=config.class_weight_max,
        ).items()
    }

    # Parse per-head loss weights from JSON string if provided.
    import json as _json
    hlw: Dict[str, float] | None = None
    if config.head_loss_weights:
        hlw = _json.loads(config.head_loss_weights)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
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
        _variant_dir = _NAS_CHECKPOINTS + "\\" + Path(config.output_dir).name
        smbclient.makedirs(_variant_dir, exist_ok=True)
        _smb_available = True
        print(f"  [bc] NAS archive available: {_variant_dir}")
    except Exception:
        _smb_available = False
        print("  [bc] NAS archive not available — skipping offsite backup")

    # Resume from checkpoint if available.
    checkpoint_path = output / "bc_training_checkpoint.pt"
    if checkpoint_path.exists():
        import torch as _torch_resume
        ckpt = _torch_resume.load(checkpoint_path, map_location=model.device, weights_only=False)
        model.model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        history = ckpt.get("history", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        _best_move = ckpt.get("_best_move", float("inf"))
        _best_look = ckpt.get("_best_look", float("inf"))
        _best_max_reg = ckpt.get("_best_max_reg", float("inf"))
        _best_reg_epoch = ckpt.get("_best_reg_epoch", -1)
        _reg_violations = ckpt.get("_reg_violations", 0)
        # Optimizer state restored after first supervised step creates it.
        _resume_optimizer_state = ckpt.get("optimizer_state_dict")
        print(f"  [bc] Resuming from epoch {start_epoch} (best_val={best_val_loss:.4f} at epoch {best_epoch})")
    else:
        _resume_optimizer_state = None

    # torch.compile: tested but net negative for this model size (189K params).
    # The fused kernels don't help when individual ops are already microseconds,
    # and the compile wrapper adds overhead (val: 100s → 120s per epoch).
    # Revisit if model size increases significantly.

    # Per-step reporting: log every ~1024 samples (report_every optimizer steps).
    _report_every = max(1, 1024 // max(config.batch_size, 1)) if config.batch_size > 0 else 0
    _step_log: List[Dict[str, float]] = []

    def _on_step(step_metrics: Dict[str, float]) -> None:
        step_metrics["epoch"] = float(epoch)
        _step_log.append(step_metrics)
        mae_parts = [f"{k}={v:.4f}" for k, v in sorted(step_metrics.items()) if k.startswith("mae_")]
        print(f"  [bc]   step {int(step_metrics.get('opt_step', 0)):>5d}  "
              f"loss={step_metrics.get('loss', 0):.4f}  "
              f"{'  '.join(mae_parts)}")
        # Flush step log to disk every report interval for live monitoring.
        write_json(output / "bc_step_log.json", {"steps": _step_log})

    _active_lr = config.lr
    _lr_override_path = output / "lr_override.json"

    import math as _math
    import time as _time
    from datetime import datetime as _datetime, timezone as _tz

    for epoch in range(start_epoch, config.epochs):
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
            focal_gamma=config.focal_gamma,
            sparse_discrete=config.sparse_discrete,
            look_deadzone=config.look_deadzone,
            step_callback=_on_step,
            report_every=_report_every,
        )
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
            focal_gamma=config.focal_gamma,
            sparse_discrete=config.sparse_discrete,
            look_deadzone=config.look_deadzone,
        )
        # Clean train eval (model.eval mode, no dropout) on val-sized subset
        # for accurate generalization gap measurement.
        train_eval_metrics = _run_precomputed_supervised(
            model,
            train_episodes[:len(val_episodes)],
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            tbptt_limit=config.tbptt_limit,
            head_loss_weights=hlw,
            focal_gamma=config.focal_gamma,
            sparse_discrete=config.sparse_discrete,
            look_deadzone=config.look_deadzone,
        )
        _t_val_end = _time.monotonic()
        _train_secs = _t_train_end - _t_train_start
        _val_secs = _t_val_end - _t_val_start
        _wall_clock = _datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  [bc] timing: train={_train_secs:.1f}s  val={_val_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]")
        mae_str = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()) if k.startswith("mae_")
        )

        # Use val MAE sum for model selection when continuous-only (val_loss is
        # polluted by discrete head CE noise that never improves).
        val_mae_sum = val_metrics.get("mae_move", 0.0) + val_metrics.get("mae_look", 0.0)
        selection_metric = val_mae_sum if val_mae_sum > 0 else val_metrics["loss"]
        improved = selection_metric < best_val_loss

        train_eval_sum = train_eval_metrics.get("mae_move", 0.0) + train_eval_metrics.get("mae_look", 0.0)
        print(f"  [bc] Epoch {epoch + 1}/{config.epochs}  "
              f"train_eval={train_eval_sum:.4f}  "
              f"val={val_mae_sum:.4f}  "
              f"gap={val_mae_sum - train_eval_sum:+.4f}  "
              f"{'*' if improved else ''}  "
              f"{mae_str}")

        # Assemble and record per-epoch metrics.
        epoch_metrics: Dict[str, Any] = {
            "epoch": float(epoch),
            "train_secs": _train_secs,
            "val_secs": _val_secs,
            "wall_clock": _wall_clock,
        }
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
            _json.dumps({"epoch": epoch, "wall_clock": _wall_clock, "mode": "bc"}) + "\n"
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

        if config.regression_stop:
            # Regression-based stopping: track per-head bests and regression.
            val_move = val_metrics.get("mae_move", float("inf"))
            val_look = val_metrics.get("mae_look", float("inf"))
            _best_move = min(_best_move, val_move)
            _best_look = min(_best_look, val_look)
            move_reg = val_move - _best_move
            look_reg = val_look - _best_look

            # Checkpoint selection: best val MAE sum (same as non-regression mode).
            # Regression gate is purely for stopping, not model selection.
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
        else:
            if improved:
                best_val_loss = selection_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                model.save(output / "bc_best_model.pth")
            else:
                epochs_without_improvement += 1

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
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "_best_move": _best_move,
            "_best_look": _best_look,
            "_best_max_reg": _best_max_reg,
            "_best_reg_epoch": _best_reg_epoch,
            "_reg_violations": _reg_violations,
        }
        torch.save(ckpt_data, checkpoint_path)
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

        if config.regression_stop:
            if _reg_violations >= config.regression_patience:
                print(f"  [bc] Regression stop: {config.regression_patience} consecutive epochs "
                      f"above threshold {config.regression_threshold}. Best epoch: {best_epoch + 1}")
                break
        elif epochs_without_improvement >= config.patience:
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
        focal_gamma=config.focal_gamma,
        sparse_discrete=config.sparse_discrete,
    ) if val_episodes else {"loss": 0.0}

    summary: Dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": float(final_val_metrics["loss"]),
        "num_train_samples": int(sample_counts["train"]),
        "num_val_samples": int(sample_counts["val"]),
        "epochs_ran": len(history),
    }
    for key, value in final_val_metrics.items():
        if key == "_next_hidden":
            continue
        summary[f"final_val_{key}"] = float(value)

    write_json(output / "bc_history.json", {"history": history})
    write_json(output / "bc_summary.json", summary)
    write_experiment_manifest(output / "bc_manifest.json", asdict(config), summary)

    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
