"""Native single-process PPO trainer (Sample Factory dropped).

One trainer process owns N engine subprocesses stepped by a thread pool, one
batched ``act(B=N)`` per tick, host- or device-resident rollout buffers, and a
recurrent PPO update that reruns the model through the SAME sequence path BC
trains through. Native ``QNNPolicy`` checkpoints end-to-end — no format
conversion or grafted heads. Pipeline depth 1 is synchronous and zero-lag;
depth 2 bounds delayed-gradient training to exactly one update.

Design + rationale: agents/plans/ppo-rebuild.md.

Entry points:
  - ``run(ctx)``           — router entry (delegates to qnn.ppo.pipeline,
                             which handles seed resolution + post-train eval).
  - ``run_native_ppo(...)``— the trainer itself, called by the pipeline.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from qnn.env.reward import RewardWeights
from qnn.model.policy import QNNPolicy
from qnn.ppo.collector import RolloutCollector
from qnn.ppo.distributions import HeadDistribution, build_adapters
from qnn.ppo.learner import PPOUpdateConfig, ValueHead, ppo_update
from qnn.ppo.rollout import RolloutBuffer
from qnn.ppo.vec_env import VecQuakeEnv
from qnn.utils.artifacts import atomic_torch_save
from qnn.utils.io import read_json, trusted_torch_load

def _resolve_device(requested: str) -> torch.device:
    req = (requested or "cpu").lower()
    if req in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[ppo] CUDA requested but unavailable — falling back to CPU")
    return torch.device("cpu")


def _configure_ppo_blas_backend(device: torch.device) -> None:
    """Route ROCm PPO GEMMs through hipBLAS + Tensile.

    PyTorch's gfx12 heuristic currently selects hipBLASLt for many native
    learner shapes that it does not support, paying a failed dispatch before
    recovering through hipBLAS.  The public ``cublas`` preference names the
    portable CUDA API; on a ROCm build it selects the corresponding hipBLAS
    implementation.  Keep both environment overrides local to this PPO
    process: BC has much larger GEMMs and benefits from the trainer image's
    hipBLASLt defaults.
    """
    if device.type != "cuda" or not getattr(torch.version, "hip", None):
        return
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"
    os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
    selected = torch.backends.cuda.preferred_blas_library("cublas")
    print(f"[ppo] ROCm BLAS: {selected}, rocBLAS=Tensile, fused-addmm-lt=off")


def _configure_ppo_autocast(device: torch.device, requested: str) -> str:
    """Select the learner activation dtype without affecting CPU collection."""
    dtype = str(requested or "fp32").lower()
    if dtype not in {"fp32", "bf16"}:
        raise ValueError(f"learner_dtype must be 'fp32' or 'bf16', got {dtype!r}")
    selected = dtype if device.type == "cuda" else "fp32"
    os.environ["QNN_AUTOCAST_DTYPE"] = selected
    print(f"[ppo] learner autocast: {selected}")
    return selected


def _head_weight_map(raw: Any, default: Mapping[str, float]) -> Dict[str, float]:
    """Parse a per-head float map from config (dict or JSON string)."""
    if raw is None or raw == "":
        return dict(default)
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected a per-head mapping, got {type(raw).__name__}")
    return {str(k): float(v) for k, v in raw.items()}


def _scenarios_from_cfg(ppo_cfg: Mapping[str, Any]) -> List[Dict[str, Any]] | None:
    path = str(ppo_cfg.get("scenario_config_path", "") or "").strip()
    if not path:
        return None
    payload = read_json(path)
    scenarios = payload.get("scenarios", payload)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError(f"scenario_config_path must define a non-empty scenarios list: {path}")
    return scenarios


def _freeze_for_mode(
    policy: QNNPolicy,
    adapters: Mapping[str, HeadDistribution],
    trainable: str,
) -> List[torch.nn.Parameter]:
    """Set requires_grad per the trainability mode; return trainable params.

    ``heads``: only the trained heads' modules get policy gradients — the
    trunk (encoder/GRU/pointer) and every frozen head stay byte-identical,
    which is the behavior-preserving default for fine-tuning.
    ``fire_bias``: only the model-owned fire intercept trains; the conditional
    timing logits and weapon selection are frozen.
    ``full``: everything trains (later gens; drift monitored by eval).
    """
    net = policy.model
    if trainable == "full":
        for p in net.parameters():
            p.requires_grad_(True)
    elif trainable == "heads":
        for p in net.parameters():
            p.requires_grad_(False)
        for head, adapter in adapters.items():
            mod = getattr(net, adapter.module_name, None)
            if not isinstance(mod, torch.nn.Module):
                raise RuntimeError(
                    f"trained head {head!r} expects model module "
                    f"{adapter.module_name!r}"
                )
            for p in mod.parameters():
                p.requires_grad_(True)
    elif trainable == "fire_bias":
        for p in net.parameters():
            p.requires_grad_(False)
        fire_bias = getattr(getattr(net, "attack_head", None), "fire_bias", None)
        if not isinstance(fire_bias, torch.nn.Parameter):
            raise RuntimeError(
                "trainable='fire_bias' requires attack_head.fire_bias"
            )
        fire_bias.requires_grad_(True)
    else:
        raise ValueError(
            f"trainable must be 'fire_bias', 'heads', or 'full', got {trainable!r}"
        )
    return [p for p in net.parameters() if p.requires_grad]


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def run_native_ppo(
    ppo_cfg: dict[str, Any],
    asset_root: Path,
    worker_path: Path,
    requested_device: str,
) -> dict[str, Any]:
    """Train PPO on a native QNN checkpoint. Returns the summary dict."""
    collector_num_threads = int(ppo_cfg.get("collector_num_threads", 0))
    if collector_num_threads > 0:
        # PyTorch's default wide intra-op pool is beneficial at the retained
        # B=768 process topology but actively hurts small arena batches.  The
        # setting is process-global; engine workers are separate processes and
        # the GPU learner does not consume this CPU operator pool.
        torch.set_num_threads(collector_num_threads)
        print(f"[ppo] CPU collector intra-op threads: {torch.get_num_threads()}")
    device = _resolve_device(requested_device)
    _configure_ppo_blas_backend(device)
    learner_dtype = _configure_ppo_autocast(
        device, str(ppo_cfg.get("learner_dtype", "fp32")),
    )
    collector_dtype = str(ppo_cfg.get("collector_dtype", "fp32")).lower()
    if collector_dtype not in {"fp32", "bf16"}:
        raise ValueError(
            f"collector_dtype must be 'fp32' or 'bf16', got {collector_dtype!r}"
        )
    output_dir = Path(str(ppo_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(ppo_cfg.get("run_id", ""))
    seed = int(ppo_cfg.get("seed", 17))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── model ─────────────────────────────────────────────────────────
    state_path = output_dir / (f"ppo_state_{run_id}.pt" if run_id else "ppo_state.pt")
    resume = bool(ppo_cfg.get("resume", False)) and state_path.exists()
    init_ckpt = str(ppo_cfg.get("init_ckpt", "") or "")
    if not resume and not init_ckpt:
        raise RuntimeError(
            "Native PPO is the fine-tuning stage: provide checkpoint_path "
            "(a BC/RL seed) in run.json, or resume an existing run."
        )
    seed_path = init_ckpt
    state: dict[str, Any] | None = None
    if resume:
        state = trusted_torch_load(state_path, map_location="cpu")
        seed_path = str(state["seed_checkpoint"])
    # Polar look heads need the run's grid installed before any forward
    # (no code default since d24ee386) — same contract as the eval runner.
    # PPO runs carry the SEED's grid in config/look_grid.json (run init
    # copies it); reward_json_path is a sibling in the same config dir.
    _grid_path = Path(str(ppo_cfg["reward_json_path"])).parent / "look_grid.json"
    if _grid_path.exists():
        from qnn.model.look_bins import install_polar_grid
        _lg = json.loads(_grid_path.read_text())
        install_polar_grid(
            torch.tensor(_lg["mag_centers_rad"], dtype=torch.float32),
            torch.tensor(_lg["dir_centers_rad"], dtype=torch.float32)
            if "dir_centers_rad" in _lg else None,
        )

    policy = QNNPolicy.load(seed_path, device=str(device))
    policy.model.to(device)
    policy.autocast_dtype = learner_dtype
    # Project standard: op input mask is always on (feedback_input_mask_always_on).
    policy.input_mask = True
    # act() has no default decode facade — frozen heads execute the seed
    # generation's BC decode during rollouts. The loaded checkpoint carries
    # its own graph, so resolve the arch from THAT (no run-dir sniffing:
    # qnn.run.init copies seeds into runs/ppo/<name>/seed/, where no
    # probe.json exists).
    def _inject_decode(p: QNNPolicy) -> None:
        spec = getattr(p, "graph", None)
        attack_head = spec.head("attack") if spec is not None else None
        is_a25 = spec is not None and (
            spec.head("move_seg") is not None
            or (attack_head is not None and attack_head.type == "attack_with")
        )
        if not is_a25:
            raise RuntimeError(
                "cannot determine the seed checkpoint's decode arch from its "
                "graph (no move_seg head and no attack_with attack head); "
                "a24 is retired — only a25-arch seeds train under native PPO"
            )
        import qnn.model.decode_actions as _decode_mod
        p._decode_mod = _decode_mod

    # PPO must collect under the same frozen decode operating point retained
    # eval/export will use. Historically PPO installed only the geometry module,
    # so attack trained as a raw 9-way categorical while eval applied the fitted
    # fire vector (SG +4.5312 in the current pin). That made the learner optimize
    # a different decision boundary from the deployed bot.
    decode_regime = str(ppo_cfg.get("eval_decode_regime", "") or "").strip()

    def _inject_runtime_decode(p: QNNPolicy):
        _inject_decode(p)
        if not decode_regime:
            return None
        from qnn.model.decode_config import (
            apply_policy_decode_params,
            install_policy_decode_modules,
        )
        resolved = install_policy_decode_modules(p, decode_regime)
        if resolved is not None:
            apply_policy_decode_params(p, resolved)
        return resolved

    resolved_decode = _inject_runtime_decode(policy)

    # Promote decode-fit's family intercept into checkpoint-owned model state.
    # This is a fresh-run initializer only: resume restores the learned tensor
    # from ppo_state below.  Requiring every legacy external fire offset to be
    # zero prevents two hidden operating points from adding together.
    model_fire_bias_init = ppo_cfg.get("model_fire_bias_init")
    if model_fire_bias_init is not None:
        values = [float(x) for x in model_fire_bias_init]
        if len(values) != 8:
            raise ValueError(
                f"model_fire_bias_init must contain 8 values, got {len(values)}"
            )
        external = {
            "attack.bias": float(policy.attack_bias),
            "attack.bias_vec": list(policy.attack_bias_vec or [0.0] * 8),
            "attack.fire_bias_vec": list(policy.attack_fire_bias_vec or [0.0] * 8),
        }
        if (external["attack.bias"] != 0.0
                or any(float(x) != 0.0 for x in external["attack.bias_vec"])
                or any(float(x) != 0.0 for x in external["attack.fire_bias_vec"])):
            raise RuntimeError(
                "model_fire_bias_init requires zero external attack.bias, "
                "attack.bias_vec, and attack.fire_bias_vec; got "
                f"{external}"
            )
        fire_bias = getattr(getattr(policy.model, "attack_head", None),
                            "fire_bias", None)
        if not isinstance(fire_bias, torch.nn.Parameter):
            raise RuntimeError(
                "model_fire_bias_init requires an attack_with head carrying "
                "the model-owned fire_bias parameter"
            )
        if not resume:
            with torch.no_grad():
                fire_bias.copy_(fire_bias.new_tensor(values))
        print(
            "[ppo] model-owned fire calibration "
            f"{'restored on resume' if resume else 'initialized'}: {values}; "
            "external fire decode = zero"
        )

    # ── RL config ─────────────────────────────────────────────────────
    rl_head_weights = _head_weight_map(
        ppo_cfg.get("rl_head_weights"),
        default={"attack": 1.0, "move": 1.0},
    )
    attack_impulse_raw = ppo_cfg.get("rl_attack_impulse")
    attack_impulse = (
        None if attack_impulse_raw is None else int(attack_impulse_raw)
    )
    if attack_impulse is not None and resolved_decode is None:
        raise RuntimeError(
            "rl_attack_impulse requires eval_decode_regime: the fixed-weapon "
            "policy must train against an explicit deployed operating point"
        )
    adapters = build_adapters(
        rl_head_weights,
        attack_impulse=attack_impulse,
        attack_bias=float(policy.attack_bias),
        attack_bias_vec=policy.attack_bias_vec,
        attack_fire_bias_vec=policy.attack_fire_bias_vec,
    )
    if attack_impulse is not None:
        attack_adapter = adapters.get("attack")
        if attack_adapter is None:
            raise RuntimeError(
                "rl_attack_impulse is set but rl_head_weights does not enable "
                "the attack adapter"
            )
        print(
            "[ppo] decode-aligned fixed-weapon attack policy: "
            f"impulse={attack_impulse} "
            f"offset={getattr(attack_adapter, 'decode_offset', float('nan')):.6f} "
            f"decode={decode_regime}"
        )
    if not adapters:
        raise RuntimeError("rl_head_weights enables no heads — nothing to train")
    anchor_kl_coef = _head_weight_map(ppo_cfg.get("anchor_kl_coef"), default={})
    # Rung-3 trigger objective: fit P(attack | alignment) to a measured human
    # curve. Absent = off and the loss is byte-identical.
    pfire_coef = float(ppo_cfg.get("pfire_coef", 0.0))
    pfire_target = None
    if pfire_coef != 0.0:
        from qnn.ppo.pfire_target import PFireTarget
        cfg_block = ppo_cfg.get("pfire_target")
        if not cfg_block:
            raise RuntimeError(
                "pfire_coef is set but pfire_target is missing — the human "
                "curve the trigger objective is fit against must be named"
            )
        pfire_target = PFireTarget.from_config(cfg_block)
        print(f"[ppo] trigger objective on (coef {pfire_coef}): "
              f"{pfire_target.family} {pfire_target.skill_label} "
              f"<- {pfire_target.source}")
    fire_occupancy_coef = float(ppo_cfg.get("fire_occupancy_coef", 0.0))
    fire_occupancy_project = bool(ppo_cfg.get("fire_occupancy_project", False))
    fire_occupancy_project_max_delta = float(
        ppo_cfg.get("fire_occupancy_project_max_delta", 0.0)
    )
    if fire_occupancy_project_max_delta < 0.0:
        raise ValueError("fire_occupancy_project_max_delta must be >= 0")
    fire_occupancy_temperature = float(
        ppo_cfg.get("fire_occupancy_temperature", 1.0)
    )
    fire_occupancy_best_rel_tol_raw = ppo_cfg.get(
        "fire_occupancy_best_rel_tol"
    )
    fire_occupancy_best_rel_tol = (
        None if fire_occupancy_best_rel_tol_raw is None
        else float(fire_occupancy_best_rel_tol_raw)
    )
    if (fire_occupancy_best_rel_tol is not None
            and fire_occupancy_best_rel_tol < 0.0):
        raise ValueError("fire_occupancy_best_rel_tol must be >= 0")
    if fire_occupancy_temperature <= 0.0:
        raise ValueError(
            "fire_occupancy_temperature must be > 0, got "
            f"{fire_occupancy_temperature}"
        )
    fire_occupancy_target = None
    fire_occupancy_projection_targets = None
    if fire_occupancy_coef != 0.0 or fire_occupancy_project:
        tick_hz = float(ppo_cfg["fixed_tick_hz"])
        if attack_impulse is None:
            # General 9-way multi-weapon PPO (crest-finetune-allweapons
            # iteration 2): no single family cadence exists, so the scalar
            # fire_occupancy_target path is replaced by a per-weapon
            # projection-targets block. The per-tick occupancy LOSS
            # (fire_occupancy_coef) only ever scores ONE weapon's ticks
            # (basis="bolt", fixed-weapon only) — it has no multi-weapon
            # form, so it stays fixed-weapon-only here too.
            if fire_occupancy_coef != 0.0:
                raise RuntimeError(
                    "fire_occupancy_coef is set for the general (multi-"
                    "weapon) attack adapter: the per-tick occupancy LOSS "
                    "only supports a fixed-weapon policy — use "
                    "fire_occupancy_project + "
                    "fire_occupancy_projection_targets instead "
                    "(agents/plans/crest-finetune-allweapons.md)"
                )
            from qnn.ppo.pfire_target import load_fire_occupancy_projection_targets
            cfg_block = ppo_cfg.get("fire_occupancy_projection_targets")
            if not cfg_block:
                raise RuntimeError(
                    "fire_occupancy_project is set for the general attack "
                    "adapter but fire_occupancy_projection_targets is "
                    "missing — the per-weapon human cadence rulers must be "
                    "named"
                )
            fire_occupancy_projection_targets = (
                load_fire_occupancy_projection_targets(cfg_block)
            )
            for weapon, target in fire_occupancy_projection_targets.items():
                if target.tick_hz != tick_hz:
                    raise RuntimeError(
                        "fire occupancy tick_hz must match PPO "
                        f"fixed_tick_hz for {weapon}: {target.tick_hz} != "
                        f"{tick_hz}"
                    )
            print(
                "[ppo] per-weapon human fire occupancy projection on: "
                + ", ".join(
                    f"{w}={t.probability:.6f}/tick({t.basis})"
                    for w, t in sorted(fire_occupancy_projection_targets.items())
                )
                + f"; temperature={fire_occupancy_temperature}"
            )
        else:
            from qnn.ppo.pfire_target import FireOccupancyTarget
            cfg_block = ppo_cfg.get("fire_occupancy_target")
            if not cfg_block:
                raise RuntimeError(
                    "fire_occupancy_coef is set but fire_occupancy_target is "
                    "missing — the human cadence ruler must be named"
                )
            fire_occupancy_target = FireOccupancyTarget.from_config(cfg_block)
            if fire_occupancy_target.tick_hz != tick_hz:
                raise RuntimeError(
                    "fire occupancy tick_hz must match PPO fixed_tick_hz: "
                    f"{fire_occupancy_target.tick_hz} != {tick_hz}"
                )
            print(
                f"[ppo] human fire occupancy constraint on (coef "
                f"{fire_occupancy_coef}): {fire_occupancy_target.weapon} "
                f"p={fire_occupancy_target.probability:.6f} "
                f"({fire_occupancy_target.rate_per_s:.6f}/s) <- "
                f"{fire_occupancy_target.source}; "
                f"temperature={fire_occupancy_temperature}"
            )
    update_cfg = PPOUpdateConfig(
        clip_ratio=float(ppo_cfg.get("clip_ratio", 0.2)),
        ppo_epochs=int(ppo_cfg.get("ppo_epochs", 3)),
        minibatch_lanes=int(ppo_cfg.get("minibatch_lanes", 16)),
        value_coef=float(ppo_cfg.get("value_coef", 0.5)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        kl_target=float(ppo_cfg.get("kl_target", 0.02)),
        rl_head_weights=rl_head_weights,
        entropy_coef=_head_weight_map(ppo_cfg.get("entropy_coef"), default={}),
        temperatures=_head_weight_map(ppo_cfg.get("rl_temperatures"), default={}),
        anchor_kl_coef=anchor_kl_coef,
        pfire_coef=pfire_coef,
        pfire_target=pfire_target,
        fire_occupancy_coef=fire_occupancy_coef,
        fire_occupancy_target=fire_occupancy_target,
        fire_occupancy_temperature=fire_occupancy_temperature,
        fire_occupancy_project=fire_occupancy_project,
        fire_occupancy_project_max_delta=fire_occupancy_project_max_delta,
        fire_occupancy_projection_targets=fire_occupancy_projection_targets,
        value_af=bool(ppo_cfg.get("value_af", False)),
    )
    if update_cfg.value_af:
        print("[ppo] value_af on: V(s) reads self_arsenal_scalars[...,0] "
              "(remaining refire); trainer-owned, never a policy input")
    # Anchor for the KL fine-tune (rung 3): the frozen seed checkpoint —
    # ALWAYS reloaded from seed_path (resume-safe; the anchor never drifts
    # with the trained policy). Runs grad-free on the learner device.
    anchor_policy = None
    if any(float(v) != 0.0 for v in anchor_kl_coef.values()):
        anchor_policy = QNNPolicy.load(seed_path, device=str(device))
        anchor_policy.model.to(device)
        anchor_policy.model.eval()
        anchor_policy.model.requires_grad_(False)
        anchor_policy.input_mask = True
        print(f"[ppo] anchor KL on {anchor_kl_coef} (anchor = {seed_path})")

    # Crest-law discharge shaping (rung 3) — process backend only.
    crest_shaper = None
    if ppo_cfg.get("crest_reward"):
        from qnn.ppo.crest_reward import CrestRewardShaper
        crest_shaper = CrestRewardShaper.from_config(ppo_cfg["crest_reward"])
        print(f"[ppo] crest reward on: {dict(ppo_cfg['crest_reward'])}")
    gamma = float(ppo_cfg.get("gamma", 0.99))
    gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
    base_lr = float(ppo_cfg.get("policy_lr", 1e-4))
    lr_min_frac = float(ppo_cfg.get("lr_min_frac", 0.1))
    total_env_steps = int(ppo_cfg["total_env_steps"])
    rollout_steps = int(ppo_cfg.get("rollout_steps", 128))
    num_lanes = int(ppo_cfg["num_lanes"])
    checkpoint_interval = int(ppo_cfg.get("checkpoint_interval_iters", 10))
    pipeline_depth = int(ppo_cfg.get("pipeline_depth", 1))
    if pipeline_depth not in (1, 2):
        raise ValueError(
            f"pipeline_depth must be 1 (synchronous) or 2 (one-update lag), "
            f"got {pipeline_depth}"
        )
    pipeline_host_staging = bool(ppo_cfg.get("pipeline_host_staging", True))

    # ── env ───────────────────────────────────────────────────────────
    reward_weights = RewardWeights.from_json(str(ppo_cfg["reward_json_path"]))
    if str(ppo_cfg.get("env_backend", "process")) == "arena_grid":
        if crest_shaper is not None:
            raise RuntimeError(
                "crest_reward is implemented for the process backend only — "
                "arena_grid does not expose the pre-step obs/attack hook"
            )
        from qnn.ppo.arena_backend import ArenaGridBackend

        vec_env = ArenaGridBackend(
            num_lanes=num_lanes,
            server_executable=str(ppo_cfg["arena_server_binary"]),
            client_executable=str(ppo_cfg["arena_client_binary"]),
            basedir=str(asset_root),
            workdir=str(ppo_cfg.get("native_workdir", "") or "") or None,
            map_id=str(ppo_cfg["arena_map_id"]),
            matches_per_server=int(ppo_cfg["matches_per_server"]),
            seat_mode=str(ppo_cfg["seat_mode"]),
            base_port=int(ppo_cfg["arena_base_port"]),
            bot_skill=int(ppo_cfg["arena_bot_skill"]),
            max_steps_per_episode=int(ppo_cfg["max_steps_per_episode"]),
            fixed_tick_hz=int(ppo_cfg["fixed_tick_hz"]),
            reward_weights=reward_weights,
            direct_actions=True,
            observer_mode="virtual",
            scenario_id=f"arena-grid-{ppo_cfg['seat_mode']}",
        )
    else:
        vec_env = VecQuakeEnv(
            num_lanes=num_lanes,
            executable=str(worker_path),
            basedir=str(asset_root),
            workdir=str(ppo_cfg.get("native_workdir", "") or "") or None,
            map_id=str(ppo_cfg["map_id"]),
            native_args=list(ppo_cfg.get("native_args", [])),
            options=dict(ppo_cfg.get("options", {})),
            scenarios=_scenarios_from_cfg(ppo_cfg),
            procgen=ppo_cfg.get("procgen"),
            max_steps_per_episode=int(ppo_cfg["max_steps_per_episode"]),
            fixed_tick_hz=int(ppo_cfg["fixed_tick_hz"]),
            reward_weights=reward_weights,
            mode=str(ppo_cfg.get("mode", "pvp")),
            seed=seed,
            crest_shaper=crest_shaper,
        )

    history_path = output_dir / "ppo_history.json"
    summary: dict[str, Any] = {"run_id": run_id, "steps_done": 0}
    try:
        # ── value head (sized by a probe forward on real obs) ─────────
        obs0 = vec_env.reset()
        policy.model.eval()
        with torch.inference_mode():
            features, _, _, _, _ = policy._forward_tensors(dict(obs0), hidden=None)
        value_in_dim = int(features.shape[-1]) + (1 if update_cfg.value_af else 0)
        value_head = ValueHead(
            value_in_dim, int(ppo_cfg.get("value_head_hidden", 256)),
        ).to(device)

        # ── trainability + optimizer ──────────────────────────────────
        trainable = str(ppo_cfg.get("trainable", "heads"))
        params = _freeze_for_mode(policy, adapters, trainable)
        params += list(value_head.parameters())
        optimizer = torch.optim.Adam(
            params, lr=base_lr, fused=(device.type == "cuda"),
        )

        iteration, env_steps, best_score = 0, 0, float("-inf")
        return_ema: float | None = None
        if state is not None:
            policy.model.load_state_dict(state["model"])
            value_head.load_state_dict(state["value_head"])
            optimizer.load_state_dict(state["optimizer"])
            iteration = int(state["iteration"])
            env_steps = int(state["env_steps"])
            best_score = float(state["best_score"])
            return_ema = state.get("return_ema")
            print(f"[ppo] resumed at iteration {iteration} ({env_steps:,} env steps)")

        compile_all = bool(ppo_cfg.get("compile", True))
        compile_learner = bool(ppo_cfg.get("compile_learner", compile_all))
        compile_collector = bool(ppo_cfg.get("compile_collector", compile_all))

        if compile_learner and device.type == "cuda":
            from qnn.bc.train import compile_bc_hot_path
            compile_bc_hot_path(policy)

        # ── collect-time replica (worker-inference lesson, native form) ──
        # The d64 act forward is ~4× faster on CPU than ROCm eager. When
        # collect_device differs from the learner device, collection runs
        # on an in-process replica. Depth 1 syncs before every window and is
        # exactly on-policy; depth 2 publishes only after its concurrent
        # collect/learn pair and is bounded to one update of lag. The learner
        # (and its sequence path) stays on `device`.
        collect_device = torch.device(
            str(ppo_cfg.get("collect_device", "") or device))
        if collect_device != device:
            collect_policy = QNNPolicy.load(seed_path, device=str(collect_device))
            collect_policy.model.to(collect_device)
            collect_policy.autocast_dtype = collector_dtype
            collect_policy.input_mask = True
            _inject_runtime_decode(collect_policy)
            collect_value_head = ValueHead(
                value_in_dim, int(ppo_cfg.get("value_head_hidden", 256)),
            ).to(collect_device)

            def _sync_collect_replica() -> None:
                collect_policy.model.load_state_dict(policy.model.state_dict())
                collect_value_head.load_state_dict(value_head.state_dict())
                collect_policy.model.eval()
        else:
            collect_policy, collect_value_head = policy, value_head
            if collector_dtype != learner_dtype:
                raise ValueError(
                    "collector_dtype must equal learner_dtype when collection "
                    "and learning share one policy/device"
                )

            def _sync_collect_replica() -> None:
                pass

        _sync_collect_replica()

        # Collection is a fixed B=num_lanes flat forward. Compiling the whole
        # network (not the BC trainer's component-wise recipe) preserves the
        # fused cross-module CPU graph around embedding/encoder/GRU/heads;
        # B=128 profiling measured 11.8 -> 4.1 ms forward. Module.compile wraps
        # forward in place, so iteration-boundary load_state_dict sync remains
        # exact and checkpoint keys are unchanged.
        if compile_collector and collect_device.type == "cpu":
            torch._dynamo.config.suppress_errors = True
            collect_policy.model.compile(dynamic=False)
            print("[ppo] torch.compile: whole CPU collection model (dynamic=False)")

        collector = RolloutCollector(
            collect_policy, collect_value_head, vec_env, adapters,
            device=collect_device,
            seed=seed + iteration,  # fresh exploration stream on resume
            initial_obs=obs0,
            temperatures=update_cfg.temperatures,
            sample_temperatures=_head_weight_map(
                ppo_cfg.get("sample_temperatures"), default={},
            ),
            value_af=update_cfg.value_af,
        )
        d_gru = int(getattr(policy.model, "d_gru", 0))
        device_buffer_count = (
            1 if pipeline_depth == 2 and pipeline_host_staging else pipeline_depth
        )
        buffers = [
            RolloutBuffer(
                rollout_steps, num_lanes,
                heads={h: adapter.action_shape for h, adapter in adapters.items()},
                device=device,
                hidden_dim=d_gru,
            )
            for _ in range(device_buffer_count)
        ]
        if pipeline_depth == 2 and pipeline_host_staging:
            if collect_device.type != "cpu" or device.type == "cpu":
                raise ValueError(
                    "pipeline_host_staging requires CPU collection and a GPU learner"
                )
            collection_buffers = [
                RolloutBuffer(
                    rollout_steps, num_lanes,
                    heads={h: adapter.action_shape for h, adapter in adapters.items()},
                    device="cpu",
                    hidden_dim=d_gru,
                )
                for _ in range(2)
            ]
        else:
            collection_buffers = buffers
        mb_gen = torch.Generator().manual_seed(seed)

        history: List[dict[str, Any]] = list(
            read_json(history_path).get("history", [])
        ) if history_path.exists() else []

        def _set_lr() -> float:
            frac = min(env_steps / max(total_env_steps, 1), 1.0)
            lr_now = base_lr * (1.0 - (1.0 - lr_min_frac) * frac)
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            return lr_now

        def _learn_one(active_buffer: RolloutBuffer) -> tuple[Dict[str, float], float]:
            t0 = time.monotonic()
            metrics = ppo_update(
                policy, value_head, active_buffer, adapters, optimizer, update_cfg,
                mb_generator=mb_gen,
                anchor_model=anchor_policy.model if anchor_policy else None,
            )
            return metrics, time.monotonic() - t0

        def _record_iteration(
            stats: Dict[str, Any],
            t_collect: float,
            metrics: Dict[str, float],
            t_learn: float,
            lr_now: float,
            *,
            fps_denominator: float,
            pipeline_fields: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal iteration, env_steps, return_ema, best_score
            iteration += 1
            env_steps += int(stats["env_steps"])
            episodes = stats["episodes"]
            row: dict[str, Any] = {
                "iteration": iteration,
                "env_steps": env_steps,
                "lr": lr_now,
                "collect_s": round(t_collect, 3),
                "collect_act_s": round(float(stats.get("act_s", 0.0)), 3),
                "collect_env_s": round(float(stats.get("env_s", 0.0)), 3),
                **{
                    f"collect_{key}": round(float(value), 3)
                    for key, value in stats.items()
                    if key in {"buffer_s", "book_s", "bootstrap_s"}
                    or (key.startswith("env_") and key != "env_steps")
                },
                "learn_s": round(t_learn, 3),
                "fps": round(stats["env_steps"] / max(fps_denominator, 1e-9), 1),
                "episodes": len(episodes),
                **{k: round(v, 6) for k, v in metrics.items()},
            }
            if pipeline_fields:
                row.update(pipeline_fields)
            if episodes:
                ep_return = float(np.mean([e.return_value for e in episodes]))
                row["ep_return_mean"] = round(ep_return, 4)
                row["ep_len_mean"] = round(float(np.mean([e.length for e in episodes])), 1)
                for key in ("frag_delta", "frag_loss", "damage_dealt", "damage_taken"):
                    vals = [e.stats.get(key) for e in episodes if key in e.stats]
                    if vals:
                        row[f"ep_{key}_mean"] = round(float(np.mean(vals)), 4)
                return_ema = (
                    ep_return if return_ema is None
                    else 0.9 * return_ema + 0.1 * ep_return
                )
                row["return_ema"] = round(return_ema, 4)
            occupancy_best_ok = True
            if (fire_occupancy_best_rel_tol is not None
                    and fire_occupancy_target is not None):
                observed = row.get("fire_occupancy/q_mean")
                occupancy_best_ok = observed is not None and abs(
                    float(observed) - fire_occupancy_target.probability
                ) <= (fire_occupancy_best_rel_tol
                      * fire_occupancy_target.probability)
                row["fire_occupancy/best_gate"] = float(occupancy_best_ok)
            history.append(row)
            _atomic_write_json(history_path, {"history": history})
            print(
                f"[ppo] it {iteration}  steps {env_steps:,}/{total_env_steps:,}  "
                f"fps {row['fps']}  eps {len(episodes)}"
                + (f"  R {row.get('ep_return_mean')}" if episodes else "")
                + (
                    f"  kl {max(row.get(f'kl/{head}', 0.0) for head in adapters):.4f}"
                    if adapters else ""
                ),
                flush=True,
            )

            improved = (
                occupancy_best_ok
                and return_ema is not None
                and return_ema > best_score
            )
            if improved:
                best_score = float(return_ema)
                best_dir = output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save(best_dir / "best_model.pth", extra_meta={
                    "run_id": run_id,
                    "rl": {
                        "iteration": iteration,
                        "env_steps": env_steps,
                        "return_ema": best_score,
                        "seed_checkpoint": seed_path,
                        "trainable": trainable,
                        "rl_head_weights": rl_head_weights,
                        "rl_attack_impulse": attack_impulse,
                        "eval_decode_regime": decode_regime or None,
                        "decode_sha256": (
                            resolved_decode.sha256 if resolved_decode else None
                        ),
                        "anchor_kl_coef": anchor_kl_coef,
                        "pfire_coef": pfire_coef,
                        "pfire_target": (pfire_target.provenance()
                                         if pfire_target else None),
                        "fire_occupancy_coef": fire_occupancy_coef,
                        "fire_occupancy_target": (
                            fire_occupancy_target.provenance()
                            if fire_occupancy_target else None
                        ),
                        "fire_occupancy_projection_targets": (
                            {
                                w: t.provenance() for w, t in
                                fire_occupancy_projection_targets.items()
                            } if fire_occupancy_projection_targets else None
                        ),
                        "fire_occupancy_temperature": fire_occupancy_temperature,
                        "fire_occupancy_project": fire_occupancy_project,
                        "fire_occupancy_project_max_delta": (
                            fire_occupancy_project_max_delta
                        ),
                        "fire_occupancy_best_rel_tol": fire_occupancy_best_rel_tol,
                        "model_fire_bias_init": (
                            [float(x) for x in model_fire_bias_init]
                            if model_fire_bias_init is not None else None
                        ),
                        "crest_reward": dict(ppo_cfg.get("crest_reward") or {}),
                    },
                })
            if improved or iteration % checkpoint_interval == 0:
                atomic_torch_save({
                    "model": policy.model.state_dict(),
                    "value_head": value_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "env_steps": env_steps,
                    "best_score": best_score,
                    "return_ema": return_ema,
                    "seed_checkpoint": seed_path,
                }, state_path)

        def _collect_one(
            active_buffer: RolloutBuffer,
        ) -> tuple[Dict[str, Any], float, float]:
            active_buffer.reset()
            t0 = time.monotonic()
            stats = collector.collect(active_buffer)
            t_collect = time.monotonic() - t0
            t0 = time.monotonic()
            active_buffer.compute_gae(
                stats["bootstrap_value"], gamma=gamma, gae_lambda=gae_lambda,
            )
            return stats, t_collect, time.monotonic() - t0

        # ── the loop ──────────────────────────────────────────────────
        start_env_steps = env_steps
        t_start = time.monotonic()
        if env_steps >= total_env_steps:
            print(
                f"[ppo] target already complete at {env_steps:,} env steps; "
                "no additional rollout collected",
                flush=True,
            )
        elif pipeline_depth == 1:
            buffer = buffers[0]
            while env_steps < total_env_steps:
                _sync_collect_replica()
                stats, t_collect, t_gae = _collect_one(buffer)
                lr_now = _set_lr()
                metrics, t_learn = _learn_one(buffer)
                policy.model.eval()
                _record_iteration(
                    stats, t_collect, metrics, t_learn, lr_now,
                    fps_denominator=t_collect + t_gae + t_learn,
                    pipeline_fields={
                        "pipeline_depth": 1,
                        "pipeline_phase": "sync",
                        "collect_gae_s": round(t_gae, 3),
                        "policy_lag_updates": 0,
                    },
                )
        else:
            # Bounded delayed-gradient PPO. Fill one complete recurrent
            # unroll under immutable behavior generation k. While the learner
            # updates that buffer, collect exactly one successor unroll under
            # the still-immutable CPU replica. Only after both finish do we
            # publish the learner's new weights to the collector. Therefore a
            # consumed buffer is never more than one optimizer update stale.
            window_steps = rollout_steps * num_lanes
            windows_left = (
                total_env_steps - env_steps + window_steps - 1
            ) // window_steps
            collector_generation = iteration
            ready_index = 0
            ready_stats, ready_collect_s, ready_gae_s = _collect_one(
                collection_buffers[ready_index]
            )
            ready_stage_s = 0.0
            if pipeline_host_staging:
                t0 = time.monotonic()
                buffers[0].copy_from(collection_buffers[ready_index])
                ready_stage_s = time.monotonic() - t0
            ready_behavior_generation = collector_generation

            print(
                "[ppo] bounded pipeline: depth=2, immutable unrolls, "
                f"policy lag <= 1 update, host_staging={pipeline_host_staging}",
                flush=True,
            )
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ppo-learner",
            ) as learner_pool:
                for pipeline_window in range(windows_left):
                    learner_generation = iteration
                    policy_lag = learner_generation - ready_behavior_generation
                    if policy_lag not in (0, 1):
                        raise RuntimeError(
                            "bounded PPO pipeline violated its policy-lag "
                            f"contract: learner={learner_generation}, "
                            f"behavior={ready_behavior_generation}"
                        )

                    lr_now = _set_lr()
                    cycle_start = time.monotonic()
                    learn_future = learner_pool.submit(
                        _learn_one,
                        buffers[0] if pipeline_host_staging else buffers[ready_index],
                    )

                    next_window: tuple[
                        int, Dict[str, Any], float, float, float, int,
                    ] | None = None
                    if pipeline_window + 1 < windows_left:
                        next_index = 1 - ready_index
                        next_stats, next_collect_s, next_gae_s = _collect_one(
                            collection_buffers[next_index]
                        )
                        next_window = (
                            next_index,
                            next_stats,
                            next_collect_s,
                            next_gae_s,
                            0.0,
                            collector_generation,
                        )

                    metrics, t_learn = learn_future.result()
                    if next_window is not None and pipeline_host_staging:
                        t0 = time.monotonic()
                        buffers[0].copy_from(collection_buffers[next_window[0]])
                        next_stage_s = time.monotonic() - t0
                        next_window = (*next_window[:4], next_stage_s, next_window[5])
                    cycle_s = time.monotonic() - cycle_start
                    policy.model.eval()

                    if pipeline_window == 0:
                        phase = "fill"
                        denominator = (
                            ready_collect_s + ready_gae_s + ready_stage_s + t_learn
                        )
                    elif next_window is None:
                        phase = "drain"
                        denominator = (
                            ready_collect_s + ready_gae_s + ready_stage_s + t_learn
                        )
                    else:
                        phase = "steady"
                        denominator = cycle_s

                    _record_iteration(
                        ready_stats,
                        ready_collect_s,
                        metrics,
                        t_learn,
                        lr_now,
                        fps_denominator=denominator,
                        pipeline_fields={
                            "pipeline_depth": 2,
                            "pipeline_phase": phase,
                            "pipeline_cycle_s": round(cycle_s, 3),
                            "collect_gae_s": round(ready_gae_s, 3),
                            "collect_stage_s": round(ready_stage_s, 3),
                            "behavior_generation": ready_behavior_generation,
                            "learner_generation": learner_generation,
                            "policy_lag_updates": policy_lag,
                        },
                    )

                    if next_window is not None:
                        # Collection has stopped touching the replica and the
                        # learner has stopped mutating its model, so publishing
                        # the next generation is race-free.
                        _sync_collect_replica()
                        collector_generation = iteration
                        (
                            ready_index,
                            ready_stats,
                            ready_collect_s,
                            ready_gae_s,
                            ready_stage_s,
                            ready_behavior_generation,
                        ) = next_window

        wall = time.monotonic() - t_start
        trained_steps = env_steps - start_env_steps
        summary.update({
            "steps_done": env_steps,
            "iterations": iteration,
            "best_return_ema": best_score if best_score > float("-inf") else None,
            "train_fps": round(trained_steps / max(wall, 1e-9), 1),
            "wall_s": round(wall, 1),
            "history_path": str(history_path),
            "trainable": trainable,
            "learner_dtype": learner_dtype,
            "collector_dtype": collector_dtype,
            "pipeline_depth": pipeline_depth,
            "pipeline_host_staging": pipeline_host_staging,
            "rl_head_weights": rl_head_weights,
            "rl_attack_impulse": attack_impulse,
            "eval_decode_regime": decode_regime or None,
            "decode_sha256": resolved_decode.sha256 if resolved_decode else None,
            "anchor_kl_coef": anchor_kl_coef,
            "crest_reward": dict(ppo_cfg.get("crest_reward") or {}),
            "device": str(device),
        })
        _atomic_write_json(output_dir / "ppo_summary.json", summary)
    finally:
        vec_env.close()
    return summary


def run(ctx: Any) -> dict[str, Any]:
    """Runner entry point called by run.router (run.json mode == "ppo")."""
    from qnn.ppo.pipeline import run_pipeline
    return run_pipeline(ctx, post_train_eval=True, write_report=True)
