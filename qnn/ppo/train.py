"""PPO APPO training entry point for the Quake combat bot.

Usage (standalone):
    python -m qnn.ppo.train \\
        --algo=APPO --env=quake_combat \\
        --quake_executable=assets/bin/ppo_worker \\
        --quake_basedir=assets \\
        --quake_map_id=dm4 \\
        --num_workers=30 --rollout=256

Usage (programmatic):
    from qnn.ppo.train import register_quake_components, build_ppo_cfg, run_ppo
    register_quake_components()
    cfg = build_ppo_cfg(scenario="dm4", num_workers=8, ...)
    run_ppo(cfg)

Sample Factory hyperparameter defaults are calibrated to match the existing
PPO config (ppo_combat_bot_live.yaml) so that comparisons stay meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
    from sample_factory.envs.env_utils import register_env
    from sample_factory.train import make_runner
    from sample_factory.algo.utils.context import global_model_factory
    # SF 2.1.x uses the model_factory pattern; 2.0.x had register_custom_encoder
    try:
        from sample_factory.model.encoder import register_custom_encoder as _register_custom_encoder
        _HAS_REGISTER_CUSTOM_ENCODER = True
    except ImportError:
        _HAS_REGISTER_CUSTOM_ENCODER = False
except ImportError as exc:
    raise ImportError("sample-factory is required: pip install sample-factory>=2.0.0") from exc

import torch
import torch.nn.functional as F

from qnn.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS, HEAD_ORDER
from qnn.ppo.core import make_quake_core
from qnn.ppo.encoder import QuakeTransformerEncoder, make_quake_encoder
from qnn.ppo.env import make_quake_env
from qnn.run.metrics import effective_game_minutes_per_wall_minute
from qnn.utils.io import write_json


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_quake_encoder(cfg: Any, obs_space: Any):
    return make_quake_encoder(cfg, obs_space)


def _make_quake_core(cfg: Any, core_input_size: int):
    return make_quake_core(cfg, core_input_size)


def _allow_numpy_in_torch_load() -> None:
    """Allowlist numpy globals so ``torch.load(weights_only=True)`` works.

    PyTorch 2.6+ defaults weights_only=True, but SF checkpoints contain
    numpy scalars.  This must run in every process (main + SF subprocesses).
    """
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])


# Run at import time so SF learner subprocesses (which import this module
# indirectly via the model factory) also get the allowlist.
_allow_numpy_in_torch_load()


def _module_level_patches() -> None:
    """Re-apply patches at module import so they exist in subprocess copies.

    SF spawns learner / inference / rollout workers as separate processes.
    Each one imports ``qnn.ppo.train`` (via the model factory or the
    registered env), so attaching the patches at import time is the most
    reliable hook. ``register_quake_components`` is the main-process
    callsite; the subprocess imports execute this block automatically.

    Each patch self-guards via a Learner class attribute, so repeated
    calls are no-ops. Action-distribution + look-cosine patches depend
    on cfg fields that aren't yet bound at module import — they stay in
    ``register_quake_components``.
    """
    _patch_lenient_warm_start_load()
    _patch_save_best_keep()
    _patch_learner_record_summaries()


def _patch_sample_factory_checkpoint_loading() -> None:
    """Ensure numpy globals are allowed for SF checkpoint loading."""
    _allow_numpy_in_torch_load()


# Defer the call until after _patch_lenient_warm_start_load is defined.


def _patch_action_distribution_weights() -> None:
    """Teach SF's TupleActionDistribution to apply per-head loss weights.

    Weights are stored as a class attribute `_qnn_head_weights` (tensor of
    shape [n_heads], in action-space order — see qnn.actions.HEAD_ORDER).
    When set, the three aggregation methods (_calc_log_probs, entropy,
    kl_divergence) multiply each head's contribution before summing.

    Effect of weight=0 for a head:
      * log_prob sum excludes it → PPO ratio doesn't depend on that head
      * entropy sum excludes it  → no entropy-bonus pressure on that head
      * KL sum excludes it       → no reference-KL pressure on that head
    ⇒ zero gradient flows to that head's parameters.
    A weight of 0.5 gives a soft downweight, matching BC semantics.
    """
    try:
        from sample_factory.algo.utils.action_distributions import TupleActionDistribution
    except Exception:
        return
    if getattr(TupleActionDistribution, "_qnn_weighted", False):
        return

    def _weighted_per_head_values(values, weights):
        # values: list of per-head [batch] tensors; weights: [n_heads] tensor or None
        stacked = torch.cat([v.unsqueeze(dim=1) for v in values], dim=1)  # [batch, n_heads]
        if weights is not None:
            w = weights.to(stacked.device, stacked.dtype).unsqueeze(0)
            stacked = stacked * w
        return stacked.sum(dim=1)

    def _calc_log_probs(self, list_of_action_batches):
        log_probs = [d.log_prob(a) for d, a in zip(self.distributions, list_of_action_batches)]
        return _weighted_per_head_values(log_probs, TupleActionDistribution._qnn_head_weights)

    def entropy(self):
        ents = [d.entropy() for d in self.distributions]
        return _weighted_per_head_values(ents, TupleActionDistribution._qnn_head_weights)

    def kl_divergence(self, other):
        kls = [d.kl_divergence(o) for d, o in zip(self.distributions, other.distributions)]
        return _weighted_per_head_values(kls, TupleActionDistribution._qnn_head_weights)

    TupleActionDistribution._qnn_head_weights = None  # type: ignore[attr-defined]
    TupleActionDistribution._calc_log_probs = _calc_log_probs  # type: ignore[assignment]
    TupleActionDistribution.entropy = entropy  # type: ignore[assignment]
    TupleActionDistribution.kl_divergence = kl_divergence  # type: ignore[assignment]
    TupleActionDistribution._qnn_weighted = True  # type: ignore[attr-defined]


def _continuous_mean_slices() -> dict[str, slice]:
    slices: dict[str, slice] = {}
    offset = 0
    for head in HEAD_ORDER:
        size = ACTION_HEADS[head]
        if head in CONTINUOUS_ACTION_HEADS:
            slices[head] = slice(offset, offset + size)
            offset += 2 * size
        else:
            offset += size
    return slices


_CONTINUOUS_MEAN_SLICES = _continuous_mean_slices()


def _patch_look_cosine_parameterization() -> None:
    """Unit-normalize PPO look means at the action parameterization layer."""
    try:
        from sample_factory.model.action_parameterization import (
            ActionParameterizationContinuousNonAdaptiveStddev,
            ActionParameterizationDefault,
        )
    except Exception:
        return

    look_slice = _CONTINUOUS_MEAN_SLICES.get("look")
    if look_slice is None:
        return

    def _normalize_look_means(params: torch.Tensor) -> torch.Tensor:
        if params.ndim != 2 or params.shape[1] < look_slice.stop:
            return params
        normalized = params.clone()
        normalized[:, look_slice] = F.normalize(normalized[:, look_slice], dim=1)
        return normalized

    if not getattr(ActionParameterizationDefault, "_qnn_look_cosine_patched", False):
        _orig_default_forward = ActionParameterizationDefault.forward

        def _forward_default(self, actor_core_output):
            action_distribution_params, action_distribution = _orig_default_forward(self, actor_core_output)
            action_distribution_params = _normalize_look_means(action_distribution_params)
            action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
            return action_distribution_params, action_distribution

        from sample_factory.model.action_parameterization import get_action_distribution

        ActionParameterizationDefault.forward = _forward_default  # type: ignore[assignment]
        ActionParameterizationDefault._qnn_look_cosine_patched = True  # type: ignore[attr-defined]

    if not getattr(ActionParameterizationContinuousNonAdaptiveStddev, "_qnn_look_cosine_patched", False):
        _orig_nonadaptive_forward = ActionParameterizationContinuousNonAdaptiveStddev.forward

        def _forward_nonadaptive(self, actor_core_output):
            action_distribution_params, action_distribution = _orig_nonadaptive_forward(self, actor_core_output)
            action_distribution_params = _normalize_look_means(action_distribution_params)
            action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
            return action_distribution_params, action_distribution

        from sample_factory.model.action_parameterization import get_action_distribution

        ActionParameterizationContinuousNonAdaptiveStddev.forward = _forward_nonadaptive  # type: ignore[assignment]
        ActionParameterizationContinuousNonAdaptiveStddev._qnn_look_cosine_patched = True  # type: ignore[attr-defined]


def _patch_look_head_bias_init() -> None:
    """Bias the look head's initial mean output to [1, 0, 0] (neutral no-turn).

    The engine maps action.look to view-angle deltas via atan2(yaw, fwd) and
    atan2(pitch, fwd) (qnn_input.c).  At the origin [0,0,0] atan2 is
    singular: small random perturbations produce large, sign-random angle
    changes, which makes random-init PPO explore through chaotic view
    rotation rather than useful aim refinement.  Biasing the look-mean
    layer so the network starts near [1, 0, 0] — the "look straight ahead"
    point that also matches BC's no-turn label (qnn_collect_main.c) — moves
    the starting point off the singularity so sample noise translates into
    bounded angular perturbations (atan2(±0.2, 1.0) ≈ ±11°).
    """
    try:
        from sample_factory.model.action_parameterization import (
            ActionParameterizationContinuousNonAdaptiveStddev,
            ActionParameterizationDefault,
        )
    except Exception:
        return

    look_slice = _CONTINUOUS_MEAN_SLICES.get("look")
    if look_slice is None:
        return

    def _apply_look_bias(module: Any) -> None:
        linear = getattr(module, "distribution_linear", None)
        if linear is None or getattr(linear, "bias", None) is None:
            return
        if int(linear.bias.shape[0]) < look_slice.stop:
            return
        with torch.no_grad():
            linear.bias[look_slice].copy_(torch.tensor([1.0, 0.0, 0.0]))

    for cls in (ActionParameterizationDefault, ActionParameterizationContinuousNonAdaptiveStddev):
        if getattr(cls, "_qnn_look_bias_init_patched", False):
            continue
        _orig_init = cls.__init__

        def _make_patched_init(orig=_orig_init):
            def _patched_init(self, cfg, core_out_size, action_space):
                orig(self, cfg, core_out_size, action_space)
                _apply_look_bias(self)
            return _patched_init

        cls.__init__ = _make_patched_init()  # type: ignore[assignment]
        cls._qnn_look_bias_init_patched = True  # type: ignore[attr-defined]


def _install_head_loss_weights(cfg: Any) -> None:
    """Parse cfg.head_loss_weights JSON and plant it on TupleActionDistribution."""
    raw = getattr(cfg, "head_loss_weights", "") or ""
    if not raw.strip():
        return
    import json as _json
    from qnn.actions import HEAD_ORDER
    parsed = _json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"--head_loss_weights must be a JSON object, got {type(parsed).__name__}")
    unknown = set(parsed) - set(HEAD_ORDER)
    if unknown:
        raise RuntimeError(f"--head_loss_weights has unknown heads: {sorted(unknown)}")
    weights = torch.tensor(
        [float(parsed.get(h, 1.0)) for h in HEAD_ORDER], dtype=torch.float32,
    )
    from sample_factory.algo.utils.action_distributions import TupleActionDistribution
    TupleActionDistribution._qnn_head_weights = weights  # type: ignore[attr-defined]
    print(f"[quake_ppo] head_loss_weights installed: "
          f"{dict(zip(HEAD_ORDER, weights.tolist()))}")



def _experiment_dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "train_dir", ".")) / str(getattr(cfg, "experiment", "quake_combat"))


def _summary_dir(cfg: Any) -> Path:
    return Path(getattr(cfg, "train_dir", ".")) / ".summary"


_PPO_SCALAR_TAGS = {
    "mean_episode_return": "reward/reward",
    "episode_len_mean": "len/len",
    "true_objective_mean": "policy_stats/avg_true_objective",
    "frags_mean": "policy_stats/avg_frags",
    "deaths_mean": "policy_stats/avg_deaths",
    "damage_dealt_mean": "policy_stats/avg_damage_dealt",
    "damage_taken_mean": "policy_stats/avg_damage_taken",
    "accuracy": "policy_stats/avg_accuracy",
    "hit_count_mean": "policy_stats/avg_hits",
    "shots_fired_mean": "policy_stats/avg_shots_fired",
    "damage_per_death_mean": "policy_stats/avg_damage_per_death",
    "health_pickups_mean": "policy_stats/avg_health_pickups",
    "armor_pickups_mean": "policy_stats/avg_armor_pickups",
    "weapon_pickups_mean": "policy_stats/avg_weapon_pickups",
    "blind_fire_rate": "policy_stats/avg_blind_fire_rate",
    "stuck_rate": "policy_stats/avg_stuck_rate",
    "reward_total_mean": "policy_stats/avg_reward_total",
    "reward_frags_mean": "policy_stats/avg_reward_frags",
    "reward_deaths_mean": "policy_stats/avg_reward_deaths",
    "reward_ehp_mean": "policy_stats/avg_reward_ehp",
    "reward_edp_mean": "policy_stats/avg_reward_edp",
    "reward_tracking_mean": "policy_stats/avg_reward_tracking",
    "tracking_cos_mean": "policy_stats/avg_tracking_cos_mean",
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "entropy": "train/entropy",
    "kl_divergence": "train/kl_divergence",
    "fraction_clipped": "train/fraction_clipped",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/lr",
    "actual_learning_rate": "train/actual_lr",
    "fps": "perf/_fps",
    "sample_throughput": "perf/_sample_throughput",
}


def _read_latest_scalars(summary_dir: Path) -> Dict[str, Dict[str, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(summary_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    latest: Dict[str, Dict[str, float]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        tail = events[-1]
        latest[tag] = {
            "step": float(tail.step),
            "value": float(tail.value),
            "wall_time": float(tail.wall_time),
        }
    return latest


def _policy_summary_from_scalars(
    cfg: Any,
    policy_id: int,
    latest_scalars: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    steps_done = 0
    for tag, payload in latest_scalars.items():
        steps_done = max(steps_done, int(payload["step"]))
    for metric_name, tag in _PPO_SCALAR_TAGS.items():
        payload = latest_scalars.get(tag)
        if payload is not None:
            metrics[metric_name] = float(payload["value"])

    effective_minutes = effective_game_minutes_per_wall_minute(
        metrics.get("fps"),
        getattr(cfg, "quake_fixed_tick_hz", None),
    )
    if effective_minutes is not None:
        metrics["effective_game_minutes_per_wall_minute"] = float(effective_minutes)

    frags_mean = metrics.get("frags_mean")
    deaths_mean = metrics.get("deaths_mean")
    if frags_mean is not None and deaths_mean is not None:
        metrics["frag_delta_mean"] = float(frags_mean - deaths_mean)

    return {
        "policy_id": policy_id,
        "steps_done": steps_done,
        "metrics": metrics,
        "raw_scalars": {tag: float(payload["value"]) for tag, payload in sorted(latest_scalars.items())},
    }


def _select_reference_policy(policy_summaries: Dict[int, Dict[str, Any]]) -> int | None:
    if not policy_summaries:
        return None

    def _score(item: tuple[int, Dict[str, Any]]) -> tuple[float, int, int]:
        policy_id, payload = item
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        reward = metrics.get("true_objective_mean")
        if reward is None:
            reward = metrics.get("mean_episode_return")
        score = float(reward) if isinstance(reward, (int, float)) else float("-inf")
        return (score, int(payload.get("steps_done", 0)), -policy_id)

    return max(policy_summaries.items(), key=_score)[0]


def write_ppo_stage_artifacts(cfg: Any, status: Any, runner: Any | None = None) -> Dict[str, Any]:
    """Retain a compact PPO summary next to the experiment checkpoints."""
    experiment_dir = _experiment_dir(cfg)
    summary_root = _summary_dir(cfg)
    policy_summaries: Dict[int, Dict[str, Any]] = {}
    errors: list[str] = []

    if summary_root.exists():
        for child in sorted(summary_root.iterdir()):
            if not child.is_dir() or not child.name.isdigit():
                continue
            policy_id = int(child.name)
            try:
                policy_summary = _policy_summary_from_scalars(cfg, policy_id, _read_latest_scalars(child))
                if runner is not None:
                    env_steps = getattr(runner, "env_steps", {})
                    if isinstance(env_steps, dict):
                        policy_summary["steps_done"] = max(
                            int(policy_summary.get("steps_done", 0)),
                            int(env_steps.get(policy_id, 0)),
                        )
                policy_summaries[policy_id] = policy_summary
            except Exception as exc:  # pragma: no cover - defensive artifact export
                errors.append(f"{child}: {exc}")

    selected_policy_id = _select_reference_policy(policy_summaries)
    selected = policy_summaries.get(selected_policy_id, {}) if selected_policy_id is not None else {}
    selected_metrics = selected.get("metrics", {}) if isinstance(selected.get("metrics"), dict) else {}
    aggregate_fps = sum(
        float(payload.get("metrics", {}).get("fps", 0.0))
        for payload in policy_summaries.values()
        if isinstance(payload.get("metrics"), dict) and isinstance(payload.get("metrics", {}).get("fps"), (int, float))
    )
    aggregate_effective_minutes = effective_game_minutes_per_wall_minute(
        aggregate_fps if aggregate_fps > 0.0 else None,
        getattr(cfg, "quake_fixed_tick_hz", None),
    )

    summary: Dict[str, Any] = {
        "status": str(status),
        "experiment": str(getattr(cfg, "experiment", "ppo")),
        "train_dir": str(getattr(cfg, "train_dir", "")),
        "experiment_dir": str(experiment_dir),
        "summary_dir": str(summary_root),
        "policy_count": len(policy_summaries),
        "selected_policy_id": selected_policy_id,
        "steps_done": selected.get("steps_done"),
        "policies": {str(policy_id): payload for policy_id, payload in policy_summaries.items()},
    }
    summary.update({key: value for key, value in selected_metrics.items() if isinstance(value, (int, float))})
    if aggregate_fps > 0.0:
        summary["aggregate_fps"] = float(aggregate_fps)
    if aggregate_effective_minutes is not None:
        summary["effective_game_minutes_per_wall_minute"] = float(aggregate_effective_minutes)
    if errors:
        summary["errors"] = errors

    manifest = {
        "stage": "ppo",
        "status": str(status),
        "experiment": str(getattr(cfg, "experiment", "ppo")),
        "output_dir": str(experiment_dir),
        "summary_dir": str(summary_root),
        "selected_policy_id": selected_policy_id,
        "metrics": {key: value for key, value in summary.items() if isinstance(value, (int, float))},
        "policy_count": len(policy_summaries),
        "policies": {
            str(policy_id): {
                "steps_done": int(payload.get("steps_done", 0)),
                "metrics": dict(payload.get("metrics", {})),
            }
            for policy_id, payload in policy_summaries.items()
        },
        "source_summary_dirs": [str(path) for path in sorted(summary_root.iterdir()) if path.is_dir()] if summary_root.exists() else [],
    }
    if errors:
        manifest["errors"] = errors

    summary_path = experiment_dir / "ppo_summary.json"
    manifest_path = experiment_dir / "ppo_manifest.json"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)

    return {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "selected_policy_id": selected_policy_id,
        "metrics": manifest["metrics"],
    }


def _scrub_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python types so torch.save
    produces a checkpoint loadable with weights_only=True."""
    import numpy as np

    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _scrub_numpy(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _scrub_numpy(v)
    elif isinstance(obj, tuple):
        obj = tuple(_scrub_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        obj = int(obj)
    elif isinstance(obj, np.floating):
        obj = float(obj)
    elif isinstance(obj, np.ndarray):
        obj = obj.tolist()
    return obj


def _prepare_lenient_qnn_ckpt(ckpt: str, dest_dir: Path) -> str:
    """Pre-process a QNN-format checkpoint so it loads with current ModelConfig.

    Older BC checkpoints can carry meta.model in three legacy shapes:
      * unknown fields the current frozen ModelConfig rejects
        (e.g. legacy target-pointer flags like ``gru_target_query``,
        ``hard_target_feat``, ``linear_idx_prior``,
        ``gt_dist_target_feat``, ``prev_target_in_query``,
        ``weapon_in_target_query``, or ablation-run fields like
        ``attack_prior_mode``, ``attack_alignment_scale``)
      * missing fields that the current ModelConfig requires
        (post-v23 additions like ``d_target``,
        ``weapon_use_self_readout``, ``self_weapon_embed_in_self``)
      * renamed fields (``fire`` → ``attack`` in legacy
        ``head_bottleneck_dims``; the dict was later split into four
        per-action-head scalars ``d_move`` /
        ``d_look`` / ``d_attack`` / ``d_weapon``)

    For PPO warm-start we don't care about those flag values per se —
    the value head + action heads are random-init regardless, and the
    encoder + GRU weights are the real load target. Run the legacy
    meta through ``migrate_legacy_flat_meta`` (the canonical
    "synthesize defaults for missing fields + drop unknown" helper)
    and re-save under ``dest_dir/<basename>.ppo_compat.pth``.

    The source BC checkpoint dir is treated as read-only; the compat
    copy lives under the PPO run's checkpoint dir so concurrent runs
    against the same BC ckpt don't race on the same file.

    Returns the (possibly rewritten) path to load.
    """
    from qnn.model.network import ModelConfig
    from qnn.utils.checkpoint_converter import migrate_legacy_flat_meta
    from qnn.utils.io import trusted_torch_load

    payload = trusted_torch_load(ckpt, map_location="cpu")
    if not (isinstance(payload, dict) and "meta" in payload and isinstance(payload["meta"], dict)):
        return ckpt
    meta = dict(payload["meta"])
    model_meta = meta.get("model")
    if not isinstance(model_meta, dict):
        return ckpt
    # Try the fast path: meta.model is already current ModelConfig
    # shape. ``from_flat_dict`` filters unknown keys via the dataclass
    # field set; if the cleaned result equals the input, no rewrite is
    # needed.
    try:
        fast_cfg = ModelConfig.from_flat_dict(model_meta)
    except TypeError:
        fast_cfg = None
    if fast_cfg is not None and set(model_meta) <= set(fast_cfg.to_dict()):
        return ckpt
    # Slow path: meta.model is missing required fields, has
    # renamed keys, or carries unknown ones. Let
    # ``migrate_legacy_flat_meta`` reconstruct a clean nested meta
    # (it expects a flat dict-like map of all the same keys, which
    # matches our nested model_meta one-to-one). Merge top-level
    # training scalars (jump_pos_weight, attack_focal_*) back in so
    # QNNPolicy.load downstream sees them.
    migrated = migrate_legacy_flat_meta(dict(model_meta))
    if migrated is None:
        # Already-modern but ModelConfig rejected it — keep the
        # original behavior of filtering unknown fields only.
        if fast_cfg is None:
            raise RuntimeError(
                f"{ckpt}: meta.model neither matches current ModelConfig "
                "nor any recognized legacy schema"
            )
        meta["model"] = fast_cfg.to_dict()
    else:
        meta["model"] = migrated["model"]
        # Preserve top-level training scalars the source meta carries.
        for key in ("jump_pos_weight", "attack_focal_gamma",
                    "attack_focal_alpha", "attack_distance_sigma",
                    "jump_distance_sigma", "input_mask"):
            if key in meta:
                continue
            if key in migrated:
                meta[key] = migrated[key]
    payload["meta"] = meta
    dest_dir.mkdir(parents=True, exist_ok=True)
    compat_path = str(dest_dir / f"{Path(ckpt).stem}.ppo_compat.pth")
    _scrub_numpy(payload)
    torch.save(payload, compat_path)
    delta = sorted(set(model_meta) ^ set(meta["model"]))
    print(
        f"[quake_ppo] Normalized meta.model for {ckpt} (key delta {delta}); "
        f"wrote PPO-compat copy to {compat_path}"
    )
    return compat_path


def _warm_start_policy(
    pid: int,
    ckpt: str,
    exp_dir: Path,
) -> Path:
    """Seed a single policy dir from a checkpoint. Returns the dest path."""
    import shutil
    ckpt = _prepare_lenient_qnn_ckpt(ckpt, exp_dir / "warm_start")

    policy_dir = exp_dir / f"checkpoint_p{pid}"
    policy_dir.mkdir(parents=True, exist_ok=True)

    # SF discovers checkpoints by scanning `checkpoint_<train_step>_<env_steps>.pth`
    # in policy_dir. Any other filename is silently ignored and SF orthogonal-inits
    # instead, so always land the seed under the canonical step-0 name.
    #
    # Two seed formats are handled here:
    #   - QNN format (BC checkpoints, scripts/make_random_checkpoint.py output):
    #     payload is {"state_dict": ..., "meta": ...}. SF's learner expects
    #     {"train_step", "env_steps", "model", "optimizer", ...} and KeyErrors
    #     without them, so these must go through save_sf_format() which builds
    #     the matching structure (including a minimal Adam optimizer state).
    #   - SF format (previously-trained PPO checkpoints): payload already has
    #     {"model", "optimizer", "train_step", ...}. Raw-copy, with optional
    #     v17→v20 entity-vocab + self-scalar migrations if the seed predates
    #     the v21 wire-format split.
    from qnn.utils.checkpoint_converter import QNNPolicy, save_sf_format, migrate_entity_embed, migrate_self_scalars
    from qnn.utils.io import trusted_torch_load

    dest = policy_dir / "checkpoint_000000000_0.pth"
    payload = trusted_torch_load(ckpt, map_location="cpu")
    is_qnn_format = isinstance(payload, dict) and "state_dict" in payload and "meta" in payload
    is_sf_format = isinstance(payload, dict) and "model" in payload and "train_step" in payload

    if is_qnn_format:
        qnn_policy = QNNPolicy.load(ckpt, device="cpu")
        sf_dest = save_sf_format(qnn_policy, policy_dir)
        # save_sf_format names the file itself; rename to canonical step-0 form.
        if sf_dest != dest:
            shutil.move(str(sf_dest), str(dest))
        print(f"[quake_ppo] Policy {pid} warm-start converted (QNN -> SF): {ckpt} -> {dest}")
    elif is_sf_format:
        migrated = bool(migrate_entity_embed(payload["model"], optimizer=payload.get("optimizer")))
        migrated = bool(migrate_self_scalars(payload["model"], optimizer=payload.get("optimizer"))) or migrated
        if migrated:
            print(f"[quake_ppo] Migrated warm-start checkpoint tensors in {ckpt}")
            _scrub_numpy(payload)
            torch.save(payload, dest)
        else:
            shutil.copy2(ckpt, dest)
        print(f"[quake_ppo] Policy {pid} warm-start copied (SF): {ckpt} -> {dest}")
    else:
        raise RuntimeError(
            f"Warm-start checkpoint {ckpt} is neither QNN format (state_dict+meta) "
            f"nor SF format (model+train_step); keys={list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}"
        )

    if not dest.exists():
        raise RuntimeError(
            f"Warm-start copy missing at {dest}; SF would silently orthogonal-init"
        )
    return dest


def _ensure_warm_start_checkpoint(cfg: Any) -> Optional[Path]:
    """Seed SF checkpoint dirs from warm-start checkpoint(s).

    Supports both single-seed (all policies from one checkpoint) and
    multi-seed PBT (round-robin assignment from a list of checkpoints).

    Accepts either BC ``.pth`` (converted to SF format) or existing
    SF ``.pth`` (copied directly). The caller decides whether this launch
    is resuming an existing PPO experiment or seeding a fresh one.
    """
    # Multi-seed takes priority over single-seed.
    multi_raw = str(getattr(cfg, "quake_bc_checkpoints", "") or "").strip()
    multi_ckpts = [p.strip() for p in multi_raw.split(",") if p.strip()] if multi_raw else []
    single_ckpt = str(getattr(cfg, "quake_bc_checkpoint", "") or "").strip()

    if bool(getattr(cfg, "quake_resume", False)):
        print(f"[quake_ppo] Resume requested; using existing PPO checkpoints in {_experiment_dir(cfg)}")
        return None

    if not multi_ckpts and not single_ckpt:
        print("[quake_ppo] No warm-start checkpoint provided; SF will initialize policies randomly")
        return None

    cfg.load_checkpoint_kind = "latest"
    exp_dir = _experiment_dir(cfg)
    num_policies = int(getattr(cfg, "num_policies", 1))

    if multi_ckpts:
        print(f"[quake_ppo] Multi-seed warm-start: {len(multi_ckpts)} seed(s) across {num_policies} policies (round-robin)")
        first_path = None
        for pid in range(num_policies):
            ckpt = multi_ckpts[pid % len(multi_ckpts)]
            dest = _warm_start_policy(pid, ckpt, exp_dir)
            if pid == 0:
                first_path = dest
        return first_path
    else:
        print(f"[quake_ppo] Single-seed warm-start: {single_ckpt} → {num_policies} policies")
        first_path = None
        for pid in range(num_policies):
            dest = _warm_start_policy(pid, single_ckpt, exp_dir)
            if pid == 0:
                first_path = dest
        return first_path


def _patch_lenient_warm_start_load() -> None:
    """Allow SF's learner to load a partial warm-start checkpoint.

    BC checkpoints only carry the encoder + GRU + target_pointer weights
    we can map onto SF's actor-critic — the value head + flat action
    parameterization layer (and possibly any later additions to SF's
    own modules like obs_normalizer) need orthogonal init. SF defaults
    to ``strict=True``, which turns a warm-start into a hard crash.

    Patch ``Learner._load_state`` to load with ``strict=False`` and
    log the missing / unexpected keys instead of raising.
    """
    try:
        from sample_factory.algo.learning.learner import Learner
    except Exception:
        return
    if getattr(Learner, "_qnn_lenient_warm_start", False):
        return
    _orig_load_state = Learner._load_state

    def _lenient_load_state(self, checkpoint_dict, load_progress: bool = True):
        try:
            return _orig_load_state(self, checkpoint_dict, load_progress=load_progress)
        except RuntimeError as exc:
            msg = str(exc)
            if "Missing key(s)" not in msg and "Unexpected key(s)" not in msg:
                raise
            print(
                "[quake_ppo] strict warm-start load failed; retrying with "
                "strict=False — missing keys will use orthogonal init. "
                f"({msg.splitlines()[0]})"
            )
            missing, unexpected = self.actor_critic.load_state_dict(
                checkpoint_dict["model"], strict=False,
            )
            if missing:
                print(f"[quake_ppo] warm-start missing keys (orthogonal init): "
                      f"{sorted(missing)[:8]}{'…' if len(missing) > 8 else ''}")
            if unexpected:
                print(f"[quake_ppo] warm-start unexpected keys (ignored): "
                      f"{sorted(unexpected)[:8]}{'…' if len(unexpected) > 8 else ''}")
            # The original _load_state also restores optimizer / train_step /
            # env_steps; mirror that here so the rest of init proceeds.
            if load_progress:
                self.train_step = int(checkpoint_dict.get("train_step", 0))
                self.env_steps = int(checkpoint_dict.get("env_steps", 0))
                self.best_performance = float(checkpoint_dict.get("best_performance", -1e9))
            if checkpoint_dict.get("optimizer") is not None:
                # Skip optimizer restore on lenient load — param-group
                # lengths will not match after structural mismatch.
                print("[quake_ppo] warm-start optimizer state skipped (lenient mode)")
            # Match _orig_load_state's None return contract.
            return None

    Learner._load_state = _lenient_load_state
    Learner._qnn_lenient_warm_start = True


def _patch_save_best_keep() -> None:
    """Fix SF's hardcoded keep=1 in save_best to use cfg.keep_checkpoints."""
    from sample_factory.algo.learning.learner import Learner
    if getattr(Learner, "_qnn_save_best_patched", False):
        return

    def _save_best_keep_all(self, policy_id, metric, metric_value):
        if policy_id != self.policy_id:
            return False
        if metric_value - self.best_performance > 0.001:
            self.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.3f}"
            return self._save_impl("best", name_suffix, self.cfg.keep_checkpoints, verbose=False)
        return False

    Learner.save_best = _save_best_keep_all
    Learner._qnn_save_best_patched = True  # type: ignore[attr-defined]


def register_quake_components() -> None:
    """Register Quake env and encoder with Sample Factory (idempotent)."""
    _patch_save_best_keep()
    _patch_sample_factory_checkpoint_loading()
    _patch_learner_record_summaries()
    _patch_action_distribution_weights()
    _patch_lenient_warm_start_load()
    register_env("quake_combat", make_quake_env)
    if _HAS_REGISTER_CUSTOM_ENCODER:
        # SF < 2.1: direct registration helper
        _register_custom_encoder("quake_encoder", QuakeTransformerEncoder)  # type: ignore[name-defined]
    else:
        # SF 2.1+: register via model factory
        global_model_factory().register_encoder_factory(_make_quake_encoder)  # type: ignore[name-defined]
    global_model_factory().register_model_core_factory(_make_quake_core)  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


def add_quake_cli_args(parser: Any) -> None:
    """Add Quake-specific CLI arguments to an SF argument parser."""
    # Environment
    parser.add_argument("--quake_executable", type=str, default=None,
                        help="Path to the Quake worker binary")
    parser.add_argument("--quake_basedir", type=str, default=None,
                        help="QUAKE_BASEDIR — root directory containing id1/")
    parser.add_argument("--quake_native_workdir", type=str, default=None,
                        help="Working directory for Quake processes")
    parser.add_argument("--quake_map_id", type=str, default=None,
                        help="Map name for all workers")
    parser.add_argument("--quake_max_steps_per_episode", type=int, default=None,
                        help="Episode step limit")
    parser.add_argument("--quake_fixed_tick_hz", type=int, default=None,
                        help="Quake tick rate (ticks per second)")
    parser.add_argument("--quake_mode", type=str, default=None,
                        help="Reward mode passed to NativeWorldEnv")
    parser.add_argument("--quake_seed", type=int, default=None,
                        help="Base RNG seed; each worker adds its env_id")
    # Native args / options encoded as JSON strings
    parser.add_argument("--quake_native_args_json", type=str, default=None,
                        help='JSON array of extra Quake CLI args, e.g. ["-game","frikbotnex_train"]')
    parser.add_argument("--quake_options_json", type=str, default=None,
                        help="JSON object of Quake server options, e.g. {\"skill\":0}")
    parser.add_argument("--quake_procgen_json", type=str, default=None,
                        help="JSON object of procgen settings for single-map procgen runs")
    # Multi-scenario support (Step 8)
    parser.add_argument("--quake_scenario_config_json", type=str, default=None,
                        help="JSON array of scenario dicts with map_id/native_args/options")
    parser.add_argument("--reward_json_path", type=str, default=None,
                        help="Path to flat reward.json with reward weights")
    # Encoder architecture
    parser.add_argument("--quake_d_model", type=int, default=None,
                        help="Transformer token dimension (transformer only)")
    parser.add_argument("--quake_n_heads", type=int, default=None,
                        help="Number of attention heads (transformer only)")
    parser.add_argument("--quake_n_layers", type=int, default=None,
                        help="Number of transformer blocks (transformer only)")
    parser.add_argument("--quake_ffn_dim", type=int, default=None,
                        help="FFN inner dimension (transformer only)")
    parser.add_argument("--quake_attn_dropout", type=float, default=None,
                        help="Attention dropout rate (transformer only)")
    # BC warm-start
    parser.add_argument("--quake_bc_checkpoint", type=str, default=None,
                        help="Path to BC/PPO checkpoint for warm-start initialisation")
    parser.add_argument("--quake_bc_checkpoints", type=str, default=None,
                        help="Comma-separated BC/PPO checkpoints for multi-seed PBT warm-start")
    # Per-head loss shaping (same JSON schema as BC's --head_loss_weights).
    # Keys are action head names from qnn.actions.HEAD_ORDER; missing heads
    # default to 1.0.  Weight 0.0 on a head zeros its contribution to PPO
    # log-prob sum, entropy, and KL — so no gradient flows to that head.
    parser.add_argument("--head_loss_weights", type=str, default="",
                        help='JSON object of per-head weights, e.g. '
                             '\'{"move":0.0,"attack":0.0,"weapon":0.0}\' '
                             'to isolate the look head.')


def _validate_quake_cfg(cfg: Any) -> None:
    required_attrs = (
        "quake_executable",
        "quake_basedir",
        "quake_native_workdir",
        "quake_map_id",
        "quake_max_steps_per_episode",
        "quake_fixed_tick_hz",
        "quake_mode",
        "quake_seed",
        "quake_native_args_json",
        "quake_options_json",
        "quake_scenario_config_json",
        "quake_d_model",
        "quake_n_heads",
        "quake_n_layers",
        "quake_ffn_dim",
        "quake_attn_dropout",
        "reward_json_path",
    )
    missing = [name for name in required_attrs if getattr(cfg, name, None) is None]
    if missing:
        raise RuntimeError("Missing required Quake PPO config fields: " + ", ".join(sorted(missing)))
    if not str(cfg.reward_json_path).strip():
        raise RuntimeError("Quake PPO runs require reward_json_path")
    if str(cfg.quake_map_id) == "procgen" and not (cfg.quake_scenario_config_json or cfg.quake_procgen_json):
        raise RuntimeError("Procgen PPO runs require either quake_scenario_config_json or quake_procgen_json")


# ---------------------------------------------------------------------------
# Programmatic cfg builder
# ---------------------------------------------------------------------------


def build_ppo_cfg(
    *,
    scenario: str,
    num_workers: int,
    num_envs_per_worker: int,
    worker_num_splits: int,
    rollout: int,
    total_env_steps: int,
    output_dir: str,
    experiment: str,
    executable: str,
    basedir: str,
    native_workdir: str,
    native_args_json: str,
    options_json: str,
    procgen_json: str,
    scenario_config_json: str,
    mode: str,
    max_steps_per_episode: int,
    fixed_tick_hz: int,
    seed: int,
    device: str,
    encoder_hidden: int,
    d_gru: int,
    use_gru: bool,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ffn: int,
    attn_dropout: float,
    ppo_epochs: int,
    lr: float,
    entropy_coef: float,
    bc_kl_coef: float,
    clip_ratio: float,
    gamma: float,
    gae_lambda: float,
    max_grad_norm: float,
    value_coef: float,
    minibatch_size: int,
    policy_workers_per_policy: int,
    batched_sampling: bool,
    worker_inference: bool = False,
    worker_inference_device: str = "cpu",
    max_policy_lag: int = 30,
    with_wandb: bool = False,
    # Population-Based Training
    with_pbt: bool,
    num_policies: int,
    pbt_period_env_steps: int,
    pbt_start_mutation: int,
    pbt_replace_fraction: float,
    pbt_mutation_rate: float,
    pbt_optimize_gamma: bool,
    reward_json_path: str,
    init_checkpoint: str = "",
    init_checkpoints: Optional[List[str]] = None,
    resume: bool = False,
    head_loss_weights: str = "",
    initial_stddev: float = 0.2,
    extra_argv: Optional[List[str]] = None,
) -> Any:
    """Build a PPO cfg namespace without command-line parsing.

    Maps explicit PPO config fields to their Sample Factory equivalents so
    the pipeline can call PPO programmatically with the same resolved config.
    """
    register_quake_components()

    batch_size = num_workers * num_envs_per_worker * rollout
    num_batches_per_epoch = max(1, batch_size // minibatch_size) if minibatch_size else 1

    argv: List[str] = [
        "--algo=APPO",
        f"--env=quake_combat",
        f"--use_rnn={'True' if use_gru else 'False'}",
        f"--rnn_type=gru",
        f"--rnn_size={d_gru}",
        f"--num_workers={num_workers}",
        f"--num_envs_per_worker={num_envs_per_worker}",
        f"--worker_num_splits={worker_num_splits}",
        f"--rollout={rollout}",
        f"--recurrence={rollout}",
        f"--batch_size={batch_size}",
        f"--num_batches_per_epoch={num_batches_per_epoch}",
        f"--num_epochs={ppo_epochs}",
        f"--ppo_clip_ratio={clip_ratio}",
        f"--learning_rate={lr}",
        f"--lr_schedule=constant",
        f"--gamma={gamma}",
        f"--gae_lambda={gae_lambda}",
        f"--max_grad_norm={max_grad_norm}",
        f"--value_loss_coeff={value_coef}",
        f"--exploration_loss_coeff={entropy_coef}",
        f"--kl_loss_coeff={bc_kl_coef}",
        "--adaptive_stddev=False",
        "--continuous_tanh_scale=1.0",
        f"--initial_stddev={initial_stddev}",
        f"--policy_workers_per_policy={policy_workers_per_policy}",
        f"--batched_sampling={'True' if batched_sampling else 'False'}",
        f"--max_policy_lag={max_policy_lag}",
        "--keep_checkpoints=10",
        "--save_milestones_sec=600",
        f"--with_wandb={'True' if with_wandb else 'False'}",
        f"--experiment={experiment}",
        f"--train_dir={output_dir}",
        f"--device={'gpu' if device in ('gpu', 'cuda') else 'cpu'}",
        f"--train_for_env_steps={total_env_steps}",
        # Quake-specific
        f"--quake_executable={executable}",
        f"--quake_basedir={basedir}",
        f"--quake_native_workdir={native_workdir}",
        f"--quake_map_id={scenario}",
        f"--quake_max_steps_per_episode={max_steps_per_episode}",
        f"--quake_fixed_tick_hz={fixed_tick_hz}",
        f"--quake_mode={mode}",
        f"--quake_seed={seed}",
        f"--quake_native_args_json={native_args_json}",
        f"--quake_d_model={d_model}",
        f"--quake_n_heads={n_heads}",
        f"--quake_n_layers={n_layers}",
        f"--quake_ffn_dim={d_ffn}",
        f"--quake_attn_dropout={attn_dropout}",
    ]

    argv.append(f"--quake_options_json={options_json}")
    argv.append(f"--quake_procgen_json={procgen_json}")
    argv.append(f"--quake_scenario_config_json={scenario_config_json}")

    # Reward weights
    argv.append(f"--reward_json_path={reward_json_path}")

    # Population-Based Training
    argv.append(f"--num_policies={num_policies}")
    if with_pbt:
        argv.extend([
            "--with_pbt=True",
            f"--pbt_period_env_steps={pbt_period_env_steps}",
            f"--pbt_start_mutation={pbt_start_mutation}",
            f"--pbt_replace_fraction={pbt_replace_fraction}",
            f"--pbt_mutation_rate={pbt_mutation_rate}",
            f"--pbt_optimize_gamma={'True' if pbt_optimize_gamma else 'False'}",
        ])

    if init_checkpoints:
        argv.append(f"--quake_bc_checkpoints={','.join(init_checkpoints)}")
    elif init_checkpoint:
        argv.append(f"--quake_bc_checkpoint={init_checkpoint}")

    if head_loss_weights:
        argv.append(f"--head_loss_weights={head_loss_weights}")

    if extra_argv:
        argv.extend(extra_argv)

    parser, _ = parse_sf_args(argv=argv)
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser, argv=argv)
    cfg.quake_resume = bool(resume)
    cfg.worker_inference = bool(worker_inference)
    cfg.worker_inference_device = str(worker_inference_device).lower()
    # Our encoder already pre-scales scalar fields and passes categoricals through
    # embeddings; SF's RunningMeanStd is redundant. Disable it.
    cfg.normalize_input = False
    _validate_quake_cfg(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


def _set_ppo_report_interval(seconds: float = 60.0) -> None:
    """Increase Sample Factory's log spam interval from the hardcoded 5s default.

    Runner.__init__ hardcodes ``self.report_interval_sec = 5.0`` and then
    registers a periodic timer with that value.  We patch __init__ to
    overwrite both the attribute *and* the first timer's period after the
    original init completes.
    """
    try:
        from sample_factory.algo.runners import runner as runner_mod
        _orig_init = runner_mod.Runner.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.report_interval_sec = seconds
            # The first timer registered is the report timer.
            if self.timers:
                self.timers[0]._interval_sec = seconds

        if not getattr(_patched_init, "_quake_patched", False):
            _patched_init._quake_patched = True  # type: ignore[attr-defined]
            runner_mod.Runner.__init__ = _patched_init
    except Exception:
        pass


def _patch_learner_record_summaries() -> None:
    """Skip a summary cycle instead of crashing when a minibatch has zero
    valid samples (valid_ratios.min() errors on empty tensor).

    Hit under worker_inference=True when per-worker policy lag pushes some
    minibatches past max_policy_lag so all samples are filtered as invalid.
    """
    try:
        from sample_factory.algo.learning.learner import Learner
        if getattr(Learner, "_qnn_record_summaries_patched", False):
            return
        _orig_record = Learner._record_summaries

        def _safe_record(self, summary_vars, *args, **kwargs):
            try:
                return _orig_record(self, summary_vars, *args, **kwargs)
            except RuntimeError as exc:
                if "numel()" in str(exc) or "reduction dim" in str(exc):
                    return None
                raise

        Learner._record_summaries = _safe_record
        Learner._qnn_record_summaries_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass


def run_ppo(cfg: Any) -> Dict[str, Any]:
    """Launch PPO training and retain a compact summary artifact."""
    from sample_factory.algo.utils.misc import ExperimentStatus
    from qnn.ppo.observer import BestCheckpointArchiver

    register_quake_components()
    _install_head_loss_weights(cfg)
    _patch_look_cosine_parameterization()
    _patch_look_head_bias_init()
    _set_ppo_report_interval(60.0)
    _ensure_warm_start_checkpoint(cfg)

    # SF's argparse omits "linear_decay" from lr_schedule choices but the
    # scheduler code supports it.  Set it after argparse runs.
    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)

    # Worker inference: bypass centralized inference workers entirely.
    if getattr(cfg, "worker_inference", False):
        from qnn.ppo.worker_inference import WorkerInferenceSampler
        _orig_make_sampler = runner._make_sampler
        def _patched_make_sampler(sampler_cls, event_loop):
            print("[quake_ppo] Using WorkerInferenceSampler (bypassing inference workers)")
            return _orig_make_sampler(WorkerInferenceSampler, event_loop)
        runner._make_sampler = _patched_make_sampler
        print("[quake_ppo] Worker inference mode enabled")

    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    stage_artifacts = write_ppo_stage_artifacts(cfg, status, runner)
    return {
        "ppo_status": str(status),
        "train_dir": str(getattr(cfg, "train_dir", "")),
        "experiment_dir": str(_experiment_dir(cfg)),
        "ppo_summary_path": stage_artifacts["summary_path"],
        "ppo_manifest_path": stage_artifacts["manifest_path"],
        "selected_policy_id": stage_artifacts["selected_policy_id"],
        "metrics": stage_artifacts["metrics"],
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from sample_factory.algo.utils.misc import ExperimentStatus
    from qnn.ppo.observer import BestCheckpointArchiver

    register_quake_components()
    parser, _ = parse_sf_args()
    add_quake_cli_args(parser)
    cfg = parse_full_cfg(parser)
    _validate_quake_cfg(cfg)
    _ensure_warm_start_checkpoint(cfg)

    cfg.lr_schedule = "linear_decay"

    cfg, runner = make_runner(cfg)
    runner.register_observer(BestCheckpointArchiver(runner))

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()
    write_ppo_stage_artifacts(cfg, status, runner)



# Apply module-level patches now that all helpers are defined.
_module_level_patches()


if __name__ == "__main__":
    main()


# ── Runner entry point (called by run.router) ──────────────────────

def run(ctx) -> dict:
    """Runner entry point — delegates to ppo.pipeline."""
    from qnn.ppo.pipeline import run_pipeline
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
