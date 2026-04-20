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

  Shared:
    core.core.{weight_ih_l0, ...}            — single-layer GRU
    action_parameterization.distribution_linear.{weight,bias}
                                             — combined Linear(hidden, sum mixed-head params)
    critic_linear.{weight,bias}              — value head (hidden → 1)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from qnn.actions import ACTION_HEADS, CONTINUOUS_ACTION_HEADS, HEAD_ORDER
from qnn.schema import (
    ACTION_HISTORY_DIM,
    ACTION_HISTORY_LEN,
    SELF_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
)
from qnn.model.policy import QNNPolicy
from qnn.utils.io import trusted_torch_load

_HEAD_ORDER = HEAD_ORDER
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
    trunk_hidden: int,
    gru_hidden: int,
    use_gru: bool,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_dim: int,
    action_history_tokens: int,
    attn_dropout: float,
    readout: str,
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
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        action_history_tokens=action_history_tokens,
        attn_dropout=attn_dropout,
        readout=readout,
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
    combined_w_parts = []
    combined_b_parts = []
    for head in _HEAD_ORDER:
        weight_key = f"policy_heads.{head}.weight"
        bias_key = f"policy_heads.{head}.bias"
        if weight_key not in bc_state or bias_key not in bc_state:
            continue
        head_weight = bc_state[weight_key]
        head_bias = bc_state[bias_key]
        if head in CONTINUOUS_ACTION_HEADS:
            log_std_key = f"continuous_log_std.{head}"
            log_std = bc_state[log_std_key] if log_std_key in bc_state else torch.full_like(head_bias, -1.0)
            combined_w_parts.extend([head_weight, torch.zeros_like(head_weight)])
            combined_b_parts.extend([head_bias, log_std])
        else:
            combined_w_parts.append(head_weight)
            combined_b_parts.append(head_bias)
    if combined_w_parts:
        sf_style[f"{_SF_COMBINED_HEAD_KEY}.weight"] = torch.cat(combined_w_parts, dim=0).cpu()
    if combined_b_parts:
        sf_style[f"{_SF_COMBINED_HEAD_KEY}.bias"] = torch.cat(combined_b_parts, dim=0).cpu()

    # Seed returns_normalizer at identity (the actor-critic constructs this
    # module regardless of cfg.normalize_input). We skip the obs_normalizer
    # keys — `cfg.normalize_input=False` is the QNN default, so the model
    # doesn't have an obs_normalizer module and these keys would be
    # "Unexpected" on load.
    sf_style["returns_normalizer.running_mean"] = torch.zeros([1], dtype=torch.float64)
    sf_style["returns_normalizer.running_var"] = torch.ones([1], dtype=torch.float64)
    sf_style["returns_normalizer.count"] = torch.ones([1], dtype=torch.float64)

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
        "n_heads": bc_policy.n_heads,
        "n_layers": bc_policy.n_layers,
        "ffn_dim": bc_policy.ffn_dim,
        "action_history_tokens": bc_policy.action_history_tokens,
        "attn_dropout": bc_policy.attn_dropout,
        "readout": bc_policy.readout,
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

# Observation space shapes — derived from the canonical schema.
from qnn.schema import OBS_SCHEMA

_OBS_SHAPES = OBS_SCHEMA


def migrate_modality_embed(state: Dict[str, torch.Tensor], expected_rows: int = 4,
                           optimizer: Dict[str, Any] | None = None) -> bool:
    """Shrink modality_embed from old 5-row layout to current 4-row layout.

    Old layout had NONE=0, SIGHT=1, PROXIMITY=2, SOUND=3, MEMORY=4 (5 rows
    total, but row 0 was a sentinel that the oracle never emitted, so it
    was never trained).  Current layout packs SIGHT=0, PROXIMITY=1, SOUND=2,
    MEMORY=3 (4 rows, all trained).  Migration drops row 0 and shifts the
    rest down.

    Works on both SF state dicts (key contains ``encoder.trunk.tokenizer.modality_embed.weight``)
    and BC state dicts (``trunk.tokenizer.modality_embed.weight``).

    If ``optimizer`` is provided, also shifts matching Adam momentum buffers
    so the optimizer state stays consistent with the model.

    Returns True if any tensor was migrated.
    """
    migrated = False
    param_keys = [k for k in state
                  if not k.startswith("obs_normalizer.") and not k.startswith("returns_normalizer.")]

    for idx, key in enumerate(param_keys):
        if not key.endswith("modality_embed.weight"):
            continue
        tensor = state[key]
        if tensor.shape[0] > expected_rows:
            # Drop the dead NONE row at index 0; keep rows 1..N.
            state[key] = tensor[1:1 + expected_rows].clone()
            migrated = True

        if optimizer is not None and idx in optimizer.get("state", {}):
            opt_entry = optimizer["state"][idx]
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in opt_entry and hasattr(opt_entry[buf_key], "shape"):
                    if opt_entry[buf_key].shape[0] > expected_rows:
                        opt_entry[buf_key] = opt_entry[buf_key][1:1 + expected_rows].clone()
                        migrated = True

    return migrated


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
    """Concatenate BC heads into SF's single combined Linear."""
    weights = []
    biases = []
    for head in _HEAD_ORDER:
        weight_key = f"policy_heads.{head}.weight"
        bias_key = f"policy_heads.{head}.bias"
        if weight_key not in bc_state or bias_key not in bc_state:
            continue
        head_weight = bc_state[weight_key].to(device)
        head_bias = bc_state[bias_key].to(device)
        if head in CONTINUOUS_ACTION_HEADS:
            log_std_key = f"continuous_log_std.{head}"
            log_std = (
                bc_state[log_std_key].to(device)
                if log_std_key in bc_state
                else torch.full_like(head_bias, -1.0)
            )
            weights.extend([head_weight, torch.zeros_like(head_weight)])
            biases.extend([head_bias, log_std])
        else:
            weights.append(head_weight)
            biases.append(head_bias)
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
    """Split SF's combined Linear into BC heads and continuous log-stds."""
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
        if head in CONTINUOUS_ACTION_HEADS:
            mean_rows = combined_w[row : row + size]
            std_rows = combined_w[row + size : row + (2 * size)]
            if w_key in bc_state:
                bc_state[w_key].copy_(mean_rows)
            if b_key in bc_state and combined_b is not None:
                bc_state[b_key].copy_(combined_b[row : row + size])
            log_std_key = f"continuous_log_std.{head}"
            if log_std_key in bc_state and combined_b is not None:
                bc_state[log_std_key].copy_(combined_b[row + size : row + (2 * size)])
            row += 2 * size
            _ = std_rows  # state-dependent std weights are intentionally ignored for QNNPolicy
            continue

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
    to_qnn.add_argument("--trunk-hidden", type=int, required=True)
    to_qnn.add_argument("--gru-hidden", type=int, required=True)
    to_qnn.add_argument("--use-gru", choices=["true", "false"], required=True)
    to_qnn.add_argument("--d-model", type=int, required=True)
    to_qnn.add_argument("--n-heads", type=int, required=True)
    to_qnn.add_argument("--n-layers", type=int, required=True)
    to_qnn.add_argument("--ffn-dim", type=int, required=True)
    to_qnn.add_argument("--action-history-tokens", type=int, required=True)
    to_qnn.add_argument("--attn-dropout", type=float, required=True)
    to_qnn.add_argument("--readout", required=True)
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
            use_gru=str(args.use_gru).lower() == "true",
            device=args.device,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_dim=args.ffn_dim,
            action_history_tokens=args.action_history_tokens,
            attn_dropout=args.attn_dropout,
            readout=args.readout,
        )
        policy.save(args.output_path)
        print(f"Saved QNNPolicy checkpoint: {args.output_path}")


if __name__ == "__main__":
    main()


# ── Unified checkpoint loading (any format) ────────────────────────

def is_sf_checkpoint(path: str | Path) -> bool:
    """Return True if *path* is a Sample Factory format checkpoint."""
    from qnn.utils.io import trusted_torch_load
    p = Path(path)
    if p.suffix != ".pth" or not p.exists():
        return False
    try:
        payload = trusted_torch_load(str(p), map_location="cpu")
        return isinstance(payload, dict) and "model" in payload and ("train_step" in payload or "env_steps" in payload)
    except Exception:
        return False


def load_sf_checkpoint_as_qnn(
    path: str | Path,
    device: str = "cpu",
    model_config: dict | None = None,
) -> "QNNPolicy":
    """Convert an SF checkpoint to a QNNPolicy in-memory."""
    import json as _json

    p = Path(path)
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise RuntimeError(f"SF checkpoint sidecar must be a JSON object: {sidecar}")
    elif model_config is not None:
        meta = dict(model_config)
        meta.setdefault("obs_dim", OBS_DIM)
    else:
        raise RuntimeError(
            f"SF checkpoint requires either a sidecar JSON ({sidecar}) or model_config"
        )

    required_keys = (
        "obs_dim", "trunk_hidden", "gru_hidden", "use_gru",
        "d_model", "n_heads", "n_layers", "ffn_dim",
        "action_history_tokens", "attn_dropout", "readout",
    )
    missing = [key for key in required_keys if key not in meta]
    if missing:
        raise RuntimeError(
            f"SF checkpoint sidecar is missing architecture fields ({', '.join(missing)}): {sidecar}"
        )

    return sf_to_qnn(
        sf_checkpoint_path=p,
        obs_dim=int(meta["obs_dim"]),
        trunk_hidden=int(meta["trunk_hidden"]),
        gru_hidden=int(meta["gru_hidden"]),
        use_gru=bool(meta["use_gru"]),
        device=device,
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        ffn_dim=int(meta["ffn_dim"]),
        action_history_tokens=int(meta["action_history_tokens"]),
        attn_dropout=float(meta["attn_dropout"]),
        readout=str(meta["readout"]),
    )


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    model_config: dict | None = None,
) -> "QNNPolicy":
    """Load a checkpoint in either QNN or SF format."""
    if is_sf_checkpoint(path):
        return load_sf_checkpoint_as_qnn(path, device=device, model_config=model_config)
    return QNNPolicy.load(str(path), device=device)
