"""Checkpoint conversion between QNNPolicy and PPO warm-start formats.

Two conversion directions:
  BC/PPO → SF   : ``bc_to_sf()``  — warm-start APPO from a BC checkpoint.
  SF → BC/PPO   : ``sf_to_qnn()``  — convert SF checkpoint back to QNNPolicy
                                    format for evaluation.py.

SF 2.1.1 model state_dict layout:

  Transformer encoder:
    encoder.trunk.tokenizer.*                — input token projections + embeddings
    encoder.trunk.blocks.*                   — transformer block weights
    encoder.trunk.final_ln.*                 — final layer norm
    encoder.trunk.output_proj.*              — CLS+history → output_dim

  Shared:
    core.core.{weight_ih_l0, ...}            — single-layer GRU
    action_parameterization.distribution_linear.{weight,bias}
                                             — combined Linear(hidden, 69)
    critic_linear.{weight,bias}              — value head (hidden → 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from quake_ai.actions import ACTION_HEADS
from quake_ai.model.observation import (
    ACTION_HISTORY_DIM,
    ACTION_HISTORY_LEN,
    EVENT_ID_DIM,
    EVENT_SCALAR_DIM,
    MAX_EVENT_ATOMS,
    MAX_OBJECT_TOKENS,
    OBJECT_ID_DIM,
    OBJECT_SCALAR_DIM,
    SELF_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
)
from quake_ai.model.policy import QNNPolicy
from quake_ai.utils.io import trusted_torch_load

_HEAD_ORDER: list[str] = list(ACTION_HEADS.keys())
_HEAD_SIZES: list[int] = list(ACTION_HEADS.values())

# SF 2.1.1 key prefixes
_SF_ENCODER_PREFIX = "encoder.trunk"
_SF_GRU_PREFIX = "core.core"
_SF_VALUE_PREFIX = "critic_linear"
_SF_COMBINED_HEAD_KEY = "action_parameterization.distribution_linear"


def _is_sf_checkpoint(payload: Dict[str, Any]) -> bool:
    """Return True if payload looks like an SF checkpoint dict."""
    return "model" in payload and ("train_step" in payload or "env_steps" in payload)


def bc_to_sf(
    bc_path: str | Path,
    sf_model: torch.nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Copy weights from a BC/PPO checkpoint into an SF model state dict."""
    bc_policy = QNNPolicy.load(str(bc_path), device=device)
    bc_state = bc_policy.model.state_dict()
    sf_state = sf_model.state_dict()

    _copy_trunk(bc_state, sf_state, device)
    _copy_gru(bc_state, sf_state, device)
    _copy_value_head(bc_state, sf_state, device)
    _copy_bc_heads_to_sf_combined(bc_state, sf_state, device)

    return sf_state


def sf_to_qnn(
    sf_checkpoint_path: str | Path,
    obs_dim: int,
    trunk_hidden: int = 128,
    gru_hidden: int = 64,
    use_gru: bool = True,
    device: str = "cpu",
) -> QNNPolicy:
    """Load an SF checkpoint and return an QNNPolicy with copied weights."""
    payload = trusted_torch_load(str(sf_checkpoint_path), map_location="cpu")
    if not _is_sf_checkpoint(payload):
        raise ValueError(
            f"{sf_checkpoint_path} does not look like an SF checkpoint "
            f"(expected keys 'model' and 'train_step'/'env_steps')"
        )
    sf_state: Dict[str, torch.Tensor] = payload["model"]

    bc_policy = QNNPolicy(
        obs_dim=obs_dim,
        trunk_hidden=trunk_hidden,
        gru_hidden=gru_hidden,
        use_gru=use_gru and gru_hidden > 0,
        seed=0,
        device=device,
    )
    bc_state = bc_policy.model.state_dict()

    _copy_trunk(sf_state, bc_state, device, reverse=True)
    _copy_gru(sf_state, bc_state, device, reverse=True)
    _copy_value_head(sf_state, bc_state, device, reverse=True)
    _copy_sf_combined_to_bc_heads(sf_state, bc_state, device)

    bc_policy.model.load_state_dict(bc_state)
    bc_policy.model.to(bc_policy.device)
    return bc_policy


def save_sf_format(
    bc_policy: QNNPolicy,
    output_dir: str | Path,
    train_step: int = 0,
    env_steps: int = 0,
) -> Path:
    """Save an QNNPolicy as a minimal SF-compatible checkpoint."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    bc_state = bc_policy.model.state_dict()
    sf_style: Dict[str, torch.Tensor] = {}

    # Trunk
    for bc_key, tensor in bc_state.items():
        if bc_key.startswith("trunk."):
            sf_key = f"encoder.{bc_key}"
            sf_style[sf_key] = tensor.cpu()

    # GRU
    for gru_param in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        bc_key = f"gru.{gru_param}"
        if bc_key in bc_state:
            sf_style[f"{_SF_GRU_PREFIX}.{gru_param}"] = bc_state[bc_key].cpu()

    # Value head
    for suffix in ("weight", "bias"):
        bc_key = f"value_head.{suffix}"
        if bc_key in bc_state:
            sf_style[f"{_SF_VALUE_PREFIX}.{suffix}"] = bc_state[bc_key].cpu()

    # Combined policy head
    head_weights = [bc_state[f"policy_heads.{h}.weight"] for h in _HEAD_ORDER if f"policy_heads.{h}.weight" in bc_state]
    head_biases = [bc_state[f"policy_heads.{h}.bias"] for h in _HEAD_ORDER if f"policy_heads.{h}.bias" in bc_state]
    if head_weights:
        sf_style[f"{_SF_COMBINED_HEAD_KEY}.weight"] = torch.cat(head_weights, dim=0).cpu()
    if head_biases:
        sf_style[f"{_SF_COMBINED_HEAD_KEY}.bias"] = torch.cat(head_biases, dim=0).cpu()

    # SF 2.1.1 ActorCriticSharedWeights expects obs_normalizer and
    # returns_normalizer running stats.  Seed them at identity (mean=0,
    # var=1, count=1) so the first training steps fill them in.
    _add_sf_normalizer_buffers(sf_style)

    # Build a minimal Adam optimizer state dict.  SF creates a single flat
    # Adam over actor_critic.parameters(); load_state_dict validates that the
    # saved param_groups[0]['params'] length matches.  Parameters are all
    # non-normalizer tensors (normalizer buffers are registered_buffers, not
    # parameters).
    n_params = sum(
        1 for k in sf_style
        if not k.startswith("obs_normalizer.") and not k.startswith("returns_normalizer.")
    )
    payload_out = {
        "train_step": train_step,
        "env_steps": env_steps,
        "best_performance": -1e9,
        "model": sf_style,
        "optimizer": {
            "state": {},
            "param_groups": [{
                "lr": 0.00025,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "params": list(range(n_params)),
            }],
        },
    }
    meta = {
        "obs_dim": bc_policy.obs_dim,
        "trunk_hidden": bc_policy.trunk_hidden,
        "gru_hidden": bc_policy.gru_hidden,
        "use_gru": bc_policy.use_gru,
        "source": "bc_to_sf_converter",
    }
    ckpt_path = output / "checkpoint_000000000_0.pth"
    torch.save(payload_out, ckpt_path)
    (output / "checkpoint_000000000_0.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return ckpt_path


# ------------------------------------------------------------------
# SF normalizer buffers
# ------------------------------------------------------------------

# Observation space shapes that SF's RunningMeanStdInPlace normalizes.
# Must match QuakeEnv.observation_space exactly.
_OBS_SHAPES: Dict[str, tuple[int, ...]] = {
    "self_scalars": (SELF_SCALAR_DIM,),
    "self_weapon_id": (1,),
    "self_movement_id": (1,),
    "self_cluster_id": (1,),
    "object_ids": (MAX_OBJECT_TOKENS, OBJECT_ID_DIM),
    "object_scalars": (MAX_OBJECT_TOKENS, OBJECT_SCALAR_DIM),
    "object_mask": (MAX_OBJECT_TOKENS,),
    "event_ids": (MAX_EVENT_ATOMS, EVENT_ID_DIM),
    "event_scalars": (MAX_EVENT_ATOMS, EVENT_SCALAR_DIM),
    "event_owner": (MAX_EVENT_ATOMS,),
    "event_mask": (MAX_EVENT_ATOMS,),
    "spatial_ids": (SPATIAL_TOKEN_COUNT,),
    "spatial_scalars": (SPATIAL_TOKEN_COUNT, SPATIAL_SCALAR_DIM),
    "action_history": (ACTION_HISTORY_LEN, ACTION_HISTORY_DIM),
}


def _add_sf_normalizer_buffers(sf_state: Dict[str, torch.Tensor]) -> None:
    """Add zero-initialized SF normalizer entries so load_state_dict(strict=True) works."""
    # obs_normalizer.running_mean_std is a RunningMeanStdDictInPlace containing
    # a nn.ModuleDict named running_mean_std (hence the double prefix).
    for key, shape in _OBS_SHAPES.items():
        prefix = f"obs_normalizer.running_mean_std.running_mean_std.{key}"
        sf_state[f"{prefix}.running_mean"] = torch.zeros(shape, dtype=torch.float64)
        sf_state[f"{prefix}.running_var"] = torch.ones(shape, dtype=torch.float64)
        sf_state[f"{prefix}.count"] = torch.ones([1], dtype=torch.float64)

    # returns_normalizer (SF default: normalize_returns=True)
    sf_state["returns_normalizer.running_mean"] = torch.zeros([1], dtype=torch.float64)
    sf_state["returns_normalizer.running_var"] = torch.ones([1], dtype=torch.float64)
    sf_state["returns_normalizer.count"] = torch.ones([1], dtype=torch.float64)


# ------------------------------------------------------------------
# Low-level copy helpers
# ------------------------------------------------------------------


def _copy_weight(src: Dict[str, torch.Tensor], dst: Dict[str, torch.Tensor], src_key: str, dst_key: str, device: str) -> bool:
    """Copy a single weight tensor. Shapes must match exactly."""
    if src_key not in src or dst_key not in dst:
        return False
    s = src[src_key].to(device)
    if s.shape != dst[dst_key].shape:
        return False
    dst[dst_key].copy_(s)
    return True


def _copy_trunk(src: Dict, dst: Dict, device: str, reverse: bool = False) -> None:
    """Copy trunk weights between BC and SF state dicts.

    BC keys start with ``trunk.``, SF keys start with ``encoder.trunk.``.
    """
    bc_prefix = "trunk."
    sf_prefix = f"{_SF_ENCODER_PREFIX}."

    if reverse:
        for dst_key in list(dst.keys()):
            if not dst_key.startswith(bc_prefix):
                continue
            sf_key = sf_prefix + dst_key[len(bc_prefix):]
            _copy_weight(src, dst, sf_key, dst_key, device)
    else:
        for src_key in list(src.keys()):
            if not src_key.startswith(bc_prefix):
                continue
            sf_key = sf_prefix + src_key[len(bc_prefix):]
            _copy_weight(src, dst, src_key, sf_key, device)


def _copy_gru(src: Dict, dst: Dict, device: str, reverse: bool = False) -> None:
    for gru_param in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        bc_key = f"gru.{gru_param}"
        sf_key = f"{_SF_GRU_PREFIX}.{gru_param}"
        if reverse:
            _copy_weight(src, dst, sf_key, bc_key, device)
        else:
            _copy_weight(src, dst, bc_key, sf_key, device)


def _copy_value_head(src: Dict, dst: Dict, device: str, reverse: bool = False) -> None:
    for suffix in ("weight", "bias"):
        bc_key = f"value_head.{suffix}"
        sf_key = f"{_SF_VALUE_PREFIX}.{suffix}"
        if reverse:
            _copy_weight(src, dst, sf_key, bc_key, device)
        else:
            _copy_weight(src, dst, bc_key, sf_key, device)


def _copy_bc_heads_to_sf_combined(bc_state: Dict, sf_state: Dict, device: str) -> None:
    """Concatenate BC's 7 per-head Linear layers into SF's single combined Linear."""
    weights = [bc_state[f"policy_heads.{h}.weight"].to(device) for h in _HEAD_ORDER if f"policy_heads.{h}.weight" in bc_state]
    biases = [bc_state[f"policy_heads.{h}.bias"].to(device) for h in _HEAD_ORDER if f"policy_heads.{h}.bias" in bc_state]
    if not weights:
        return
    combined_w = torch.cat(weights, dim=0)
    combined_b = torch.cat(biases, dim=0)
    sf_w_key = f"{_SF_COMBINED_HEAD_KEY}.weight"
    sf_b_key = f"{_SF_COMBINED_HEAD_KEY}.bias"
    if sf_w_key in sf_state and combined_w.shape == sf_state[sf_w_key].shape:
        sf_state[sf_w_key].copy_(combined_w)
    if sf_b_key in sf_state and combined_b.shape == sf_state[sf_b_key].shape:
        sf_state[sf_b_key].copy_(combined_b)


def _copy_sf_combined_to_bc_heads(sf_state: Dict, bc_state: Dict, device: str) -> None:
    """Split SF's combined Linear into BC's 7 per-head Linear layers."""
    sf_w_key = f"{_SF_COMBINED_HEAD_KEY}.weight"
    sf_b_key = f"{_SF_COMBINED_HEAD_KEY}.bias"
    if sf_w_key not in sf_state:
        return
    combined_w = sf_state[sf_w_key].to(device)
    combined_b = sf_state[sf_b_key].to(device) if sf_b_key in sf_state else None
    row = 0
    for head, size in zip(_HEAD_ORDER, _HEAD_SIZES):
        w_key = f"policy_heads.{head}.weight"
        b_key = f"policy_heads.{head}.bias"
        if w_key in bc_state:
            bc_state[w_key].copy_(combined_w[row : row + size])
        if b_key in bc_state and combined_b is not None:
            bc_state[b_key].copy_(combined_b[row : row + size])
        row += size


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert BC/PPO checkpoints to/from SF format")
    sub = parser.add_subparsers(dest="cmd", required=True)

    to_sf = sub.add_parser("bc-to-sf", help="Convert a BC/PPO checkpoint to SF warm-start format")
    to_sf.add_argument("bc_path", help="Input BC/PPO checkpoint (.pth)")
    to_sf.add_argument("output_dir", help="Directory to write SF checkpoint")
    to_sf.add_argument("--device", default="cpu")

    to_qnn = sub.add_parser("sf-to-qnn", help="Convert an SF checkpoint to QNNPolicy format")
    to_qnn.add_argument("sf_path", help="Input SF checkpoint .pth")
    to_qnn.add_argument("output_path", help="Output .pth path for QNNPolicy")
    to_qnn.add_argument("--obs-dim", type=int, required=True)
    to_qnn.add_argument("--trunk-hidden", type=int, default=128)
    to_qnn.add_argument("--gru-hidden", type=int, default=128)
    to_qnn.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.cmd == "bc-to-sf":
        policy = QNNPolicy.load(args.bc_path, device=args.device)
        ckpt = save_sf_format(policy, args.output_dir)
        print(f"Saved SF-format warm-start checkpoint: {ckpt}")

    elif args.cmd == "sf-to-qnn":
        policy = sf_to_qnn(
            sf_checkpoint_path=args.sf_path,
            obs_dim=args.obs_dim,
            trunk_hidden=args.trunk_hidden,
            gru_hidden=args.gru_hidden,
            device=args.device,
        )
        policy.save(args.output_path)
        print(f"Saved QNNPolicy checkpoint: {args.output_path}")


if __name__ == "__main__":
    main()
