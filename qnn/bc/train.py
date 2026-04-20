"""Behavior cloning trainer for the v0 Quake policy."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict
import time as _time

import numpy as np
import torch

from qnn.actions import ACTION_HEADS, DISCRETE_ACTION_HEADS
from qnn.schema import OBS_DIM
from qnn.model.policy import QNNPolicy
from qnn.utils.io import write_json
from qnn.utils.repro import set_global_seed, write_experiment_manifest

_DISCRETE_HEAD_NAMES = [head for head in ACTION_HEADS if head in DISCRETE_ACTION_HEADS]


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
    look_turn_alpha: float  # >0 upweights turning frames in look cosine loss; 0.0 = uniform
    look_cosine: bool  # true = L2 normalize + cosine loss for look head; false = tanh + smooth L1
    regression_stop: bool  # use regression-based stopping instead of patience
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


from qnn.bc.loop import MidEpochState as _MidEpochState, PrecomputedEpisode as _PrecomputedEpisode, run_epoch as _run_precomputed_supervised


def _selection_score(metrics: Mapping[str, float]) -> float:
    """Composite selection metric: angular look error + move MAE.

    Lower is better. Fire and switch F1 are trained but kept out of the
    selection score — their per-epoch noise masked slow look/move progress
    and we can slice checkpoints on any metric retrospectively from the
    per-epoch pt files anyway.
    """
    look_deg = float(metrics.get("mae_look_angle_deg", 0.0))
    move_mae = float(metrics.get("mae_move", 0.0))
    return look_deg + 10.0 * move_mae


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
                float(prev.get("train_mae_move", 0.0)) + float(prev.get("train_mae_look", 0.0)),
            )
        )
        prev_val_sum = float(prev.get("val_mae_move", 0.0)) + float(prev.get("val_mae_look", 0.0))
        val_regression = val_sum - prev_val_sum
        train_delta = train_proxy_sum - prev_train_proxy_sum
        if (
            val_regression > float(val_regression_threshold)
            and train_delta < -float(train_improve_threshold)
        ):
            reasons.append("val_regressed_train_improved")

    return train_proxy_sum, proxy_gap, reasons





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


def _load_precomputed(cache_dir: Path) -> list[_PrecomputedEpisode]:
    """Load precomputed episodes with real memory-mapped .npy arrays."""
    import json
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    episodes: list[_PrecomputedEpisode] = []
    if isinstance(manifest, dict) and manifest.get("format") == "sharded_v1":
        for shard in manifest.get("shards", []):
            obs_arrays = {
                key: np.load(cache_dir / fname, mmap_mode="r")
                for key, fname in shard["obs"].items()
            }
            action_arrays = {
                head: np.load(cache_dir / fname, mmap_mode="r")
                for head, fname in shard["actions"].items()
            }
            for arr in obs_arrays.values():
                _madvise_sequential(arr)
            for arr in action_arrays.values():
                _madvise_sequential(arr)
            start = 0
            for n_samples in shard.get("episode_lengths", []):
                end = start + int(n_samples)
                obs = {key: values[start:end] for key, values in obs_arrays.items()}
                actions = {head: values[start:end] for head, values in action_arrays.items()}
                episodes.append(_PrecomputedEpisode(obs=obs, actions=actions, n_samples=int(n_samples)))
                start = end
        return episodes

    for entry in manifest:
        obs = {key: np.load(cache_dir / fname, mmap_mode="r")
               for key, fname in entry["obs"].items()}
        actions = {head: np.load(cache_dir / fname, mmap_mode="r")
                   for head, fname in entry["actions"].items()}
        for arr in obs.values():
            _madvise_sequential(arr)
        for arr in actions.values():
            _madvise_sequential(arr)
        episodes.append(_PrecomputedEpisode(obs=obs, actions=actions, n_samples=entry["n_samples"]))
    return episodes



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
            action_history_tokens=config.action_history_tokens,
            readout=config.readout,
        )
    # look_cosine is a training behavior, not an architecture param.
    # Always honor the run config, not the checkpoint metadata.
    model.look_cosine = bool(config.look_cosine)

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
        from qnn.utils.checkpoint_converter import migrate_modality_embed
        ckpt = _torch_resume.load(checkpoint_path, map_location=model.device, weights_only=False)
        migrate_modality_embed(
            ckpt["model_state_dict"],
            optimizer=ckpt.get("optimizer_state_dict"),
        )
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
                      f"episode {_mid_epoch_resume.next_episode}")
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
            focal_gamma=config.focal_gamma,
            sparse_discrete=config.sparse_discrete,
            look_deadzone=config.look_deadzone,
            look_turn_alpha=config.look_turn_alpha,
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
            focal_gamma=config.focal_gamma,
            sparse_discrete=config.sparse_discrete,
            look_deadzone=config.look_deadzone,
            look_turn_alpha=config.look_turn_alpha,
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
                focal_gamma=config.focal_gamma,
                sparse_discrete=config.sparse_discrete,
                look_deadzone=config.look_deadzone,
                look_turn_alpha=config.look_turn_alpha,
                pin_memory=config.pin_memory,
                prefetch=config.prefetch,
            )
            _train_eval_secs = _time.monotonic() - _t_train_eval_start
            train_eval_sum = _selection_score(train_eval_metrics)
        _t_val_end = _time.monotonic()
        _train_secs = _t_train_end - _t_train_start
        _val_only_secs = _t_val_only_end - _t_val_start
        _val_secs = _t_val_end - _t_val_start
        _wall_clock = _datetime.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if train_eval_ran:
            print(
                f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  "
                f"train_eval={_train_eval_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]"
            )
        else:
            print(f"  [bc] timing: train={_train_secs:.1f}s  val={_val_only_secs:.1f}s  total={_train_secs + _val_secs:.1f}s  [{_wall_clock}]")
        mae_str = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()) if k.startswith("mae_")
        )

        # Surface discrete head tp/fp/fn in the epoch log line.
        _discrete_parts = []
        for _dh in _DISCRETE_HEAD_NAMES:
            _tp = float(val_metrics.get(f"tp_{_dh}", 0))
            _fp = float(val_metrics.get(f"fp_{_dh}", 0))
            _fn = float(val_metrics.get(f"fn_{_dh}", 0))
            _hlw_val = hlw.get(_dh, 1.0) if hlw else 1.0
            if _hlw_val > 0 and (_tp + _fp + _fn) > 0:
                _discrete_parts.append(f"{_dh}:tp={_tp:.0f}/fp={_fp:.0f}/fn={_fn:.0f}")
        _discrete_str = "  ".join(_discrete_parts)

        # Composite selection: angular look + move MAE + fire F1.
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
        epoch_line += (
            f"{mae_str}"
            f"{'  ' + _discrete_str if _discrete_str else ''}"
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
    final_model.look_cosine = bool(config.look_cosine)

    final_val_metrics = _run_precomputed_supervised(
        final_model,
        val_episodes,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        tbptt_limit=config.tbptt_limit,
        focal_gamma=config.focal_gamma,
        sparse_discrete=config.sparse_discrete,
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
    filtered_cfg = {k: v for k, v in bc_cfg.items() if k in valid_keys}
    results["bc"] = run_behavior_cloning(BCConfig(**filtered_cfg), seed_checkpoint=seed_checkpoint)
    stage_timings["bc"] = _time.monotonic() - started
    results["stage_timings"] = stage_timings
    return finalize_results(ctx, results, stage_timings)
