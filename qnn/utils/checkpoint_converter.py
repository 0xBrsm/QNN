"""Read archived Sample Factory checkpoints as native QNNPolicy models.

The native PPO pipeline does not write or train Sample Factory checkpoints.
This module retains the SF → QNN direction only so historical checkpoints can
still be evaluated and used as seeds.

SF 2.1.1 model state_dict layout:

  Transformer encoder:
    encoder.obs_embedding.*                    — input embedding (projections + embeddings)
    encoder.encoder.blocks.*                   — transformer block weights
    encoder.encoder.final_ln.*                 — final layer norm

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
    OBS_DIM,
    SELF_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
)
from qnn.model.network import ModelConfig
from qnn.model.policy import QNNPolicy
from qnn.utils.io import trusted_torch_load
from qnn.vocab import (
    ENTITY_IDS,
    ENTITY_STREAM_COMBAT, ENTITY_STREAM_FULL,
    PROJECTILE_SCALAR_DIM, FULL_PROJECTILE_SCALAR_DIM,
)

_HEAD_ORDER = HEAD_ORDER
_HEAD_SIZES: list[int] = list(ACTION_HEADS.values())

# SF 2.1.1 key prefixes
_SF_ENCODER_PREFIX = "encoder.encoder"
_SF_GRU_PREFIX = "core.core"
_SF_VALUE_PREFIX = "critic_linear"
_SF_COMBINED_HEAD_KEY = "action_parameterization.distribution_linear"


def _is_sf_checkpoint(payload: Dict[str, Any]) -> bool:
    """Return True if payload looks like an SF checkpoint dict."""
    return "model" in payload and ("train_step" in payload or "env_steps" in payload)


def sf_to_qnn(
    sf_checkpoint_path: str | Path,
    *,
    obs_dim: int,
    model: "ModelConfig | None",
    device: str,
    graph: "Any | None" = None,
) -> QNNPolicy:
    """Load an SF checkpoint and return a QNNPolicy with copied weights.

    ``graph`` (a ``qnn.model.graph.GraphSpec``) must be passed when the
    PPO run was warm-started from a graph-described checkpoint — the QNN
    module is then rebuilt via ``build_network`` so the SF state-dict
    prefixes map onto the same token/encoder layout, and the converted
    checkpoint stays self-describing (``meta.model_graph``). When a graph
    is given the ModelConfig bridge is derived from it; a caller-supplied
    ``model`` is ignored (one source of truth — a stale flat config must
    not drive policy-layer behavior on a graph-built module).
    """
    payload = trusted_torch_load(str(sf_checkpoint_path), map_location="cpu")
    if not _is_sf_checkpoint(payload):
        raise ValueError(
            f"{sf_checkpoint_path} does not look like an SF checkpoint "
            f"(expected keys 'model' and 'train_step'/'env_steps')"
        )
    sf_state: Dict[str, torch.Tensor] = payload["model"]

    if graph is not None:
        model = None
    elif model is None:
        raise ValueError("sf_to_qnn needs either model or graph")

    # Loss-shaping knobs are training-time only — neutral values for a
    # converted checkpoint that exists to be evaluated/exported/re-seeded.
    bc_policy = QNNPolicy(
        obs_dim=obs_dim,
        model=model,
        graph=graph,
        jump_pos_weight=1.0,
        attack_focal_gamma=0.0,
        attack_focal_alpha=0.5,
        attack_distance_sigma=0.0,
        jump_distance_sigma=0.0,
        seed=0,
        device=device,
    )
    bc_state = bc_policy.model.state_dict()

    missed = _copy_encoder(sf_state, bc_state, device)
    missed += _copy_gru(sf_state, bc_state, device)
    if missed:
        raise RuntimeError(
            f"SF→QNN conversion left {len(missed)} trained weight(s) at random "
            f"init (no matching SF key/shape). Causes: converting a graph-"
            f"described run without its graph, an architecture mismatch, or a "
            f"pre-rename SF checkpoint whose keys need the legacy migrations "
            f"(reverse conversion does not apply them). First missed: {missed[:4]}"
        )
    _copy_value_head(sf_state, bc_state, device)
    if "move_head.mlp.0.weight" in bc_state:
        # SF trained a single flat action linear; there is no projection back
        # into bottleneck-MLP heads. The converted checkpoint carries the
        # TRAINED encoder, GRU,
        # and pointer — the action heads keep this policy's init and must be
        # re-fit (BC head-tune or distill) before the model is playable.
        print(
            "[sf_to_qnn] bottleneck-MLP action heads cannot receive SF's flat "
            "action linear — converted checkpoint has trained encoder/GRU/"
            "pointer but UNTRAINED action heads."
        )
    else:
        _copy_sf_combined_to_bc_heads(sf_state, bc_state, device)

    bc_policy.model.load_state_dict(bc_state)
    bc_policy.model.to(bc_policy.device)
    return bc_policy


def _copy_weight(src: Dict[str, torch.Tensor], dst: Dict[str, torch.Tensor], src_key: str, dst_key: str, device: str) -> bool:
    """Copy a single weight tensor. Shapes must match exactly."""
    if src_key not in src or dst_key not in dst:
        return False
    s = src[src_key].to(device)
    if s.shape != dst[dst_key].shape:
        return False
    dst[dst_key].copy_(s)
    return True


def _copy_encoder(
    src: Dict, dst: Dict, device: str,
) -> "list[str]":
    """Copy encoder weights from an SF state dict into a QNN state dict.

    BC keys split input embedding (``obs_embedding.*``) from transformer
    stack (``encoder.*``). SF wraps both under its actor-critic encoder:
    ``encoder.obs_embedding.*`` and ``encoder.encoder.*``.

    Returns the QNN keys that found no matching SF weight. A non-empty
    list means the converted policy would keep
    random-init weights where trained ones were expected (e.g. converting
    a graph-described run without its graph: every ``self_builders.*``
    key misses). Callers must fail loud on it.
    """
    pairs = (
        ("obs_embedding.", "encoder.obs_embedding."),
        # SF owns the pointer inside its encoder; copy it back too, else
        # converted full_5head policies keep a random-init pointer.
        ("target_pointer.", "encoder.target_pointer."),
        ("encoder.", f"{_SF_ENCODER_PREFIX}."),
    )

    missed: list[str] = []
    for qnn_prefix, sf_prefix in pairs:
        for dst_key in list(dst.keys()):
            if not dst_key.startswith(qnn_prefix):
                continue
            sf_key = sf_prefix + dst_key[len(qnn_prefix):]
            if not _copy_weight(src, dst, sf_key, dst_key, device):
                missed.append(dst_key)
    return missed


def _copy_gru(src: Dict, dst: Dict, device: str) -> "list[str]":
    """Copy GRU weights. Modern checkpoints key the recurrence
    ``temporal.gru.*`` (post-migrate_wrap_gru_in_temporal); pre-wrap ones
    use bare ``gru.*``. Returns QNN GRU keys that received nothing (empty
    when the model has no temporal slot).
    """
    missed: list[str] = []
    for gru_param in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        sf_key = f"{_SF_GRU_PREFIX}.{gru_param}"
        qnn_keys = [
            k for k in (f"temporal.gru.{gru_param}", f"gru.{gru_param}") if k in dst
        ]
        for qnn_key in qnn_keys:
            if not _copy_weight(src, dst, sf_key, qnn_key, device):
                missed.append(qnn_key)
    return missed


def _copy_value_head(src: Dict, dst: Dict, device: str) -> None:
    for suffix in ("weight", "bias"):
        bc_key = f"value_head.{suffix}"
        sf_key = f"{_SF_VALUE_PREFIX}.{suffix}"
        _copy_weight(src, dst, sf_key, bc_key, device)


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

    parser = argparse.ArgumentParser(
        description="Convert archived Sample Factory checkpoints to QNN format",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    to_qnn = sub.add_parser("sf-to-qnn", help="Convert an SF checkpoint to QNNPolicy format")
    to_qnn.add_argument("sf_path", help="Input SF checkpoint .pth")
    to_qnn.add_argument("output_path", help="Output .pth path for QNNPolicy")
    to_qnn.add_argument("--obs-dim", type=int, required=True)
    to_qnn.add_argument(
        "--model-json",
        default=None,
        help="Path to a model.json (ModelConfig-compatible) for flat runs. "
             "Omit for graph-described runs — the model graph is read from "
             "a --graph-json file or the warm-start seed's sidecar.",
    )
    to_qnn.add_argument(
        "--graph-json",
        default=None,
        help="Path to a JSON file holding the run's model graph "
             "(meta.model_graph of the warm-start seed checkpoint).",
    )
    to_qnn.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.cmd == "sf-to-qnn":
        graph = None
        model_cfg = None
        if args.graph_json:
            from qnn.model.graph import GraphSpec
            with open(args.graph_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Accept either a bare graph or a full checkpoint sidecar meta.
            graph = GraphSpec.from_dict(raw.get("model_graph", raw))
        elif args.model_json:
            with open(args.model_json, "r", encoding="utf-8") as f:
                model_cfg = ModelConfig.from_dict(json.load(f))
        else:
            parser.error("sf-to-qnn needs --model-json (flat run) or --graph-json (graph run)")
        policy = sf_to_qnn(
            sf_checkpoint_path=args.sf_path,
            obs_dim=args.obs_dim,
            model=model_cfg,
            device=args.device,
            graph=graph,
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
    *,
    device: str,
    model_config: "dict | ModelConfig | None" = None,
) -> "QNNPolicy":
    """Convert an SF checkpoint to a QNNPolicy in-memory.

    ``model_config`` is a ``ModelConfig`` (or model.json-style dict, which
    is converted via ``ModelConfig.from_dict``). When omitted, a sidecar
    JSON next to ``path`` with the same structure as a QNN checkpoint
    meta block is read; the sidecar must contain a ``"model"`` field plus
    ``"obs_dim"``. A sidecar ``"model_graph"`` always wins — the QNN
    module is rebuilt through the graph so SF weights map onto the same
    token/encoder layout (graph-described runs would otherwise convert
    with every self-token weight silently left at random init).
    """
    import json as _json

    p = Path(path)
    graph = None
    obs_dim = OBS_DIM
    model: "ModelConfig | None" = None
    sidecar = p.with_suffix(".json")
    meta: "dict | None" = None
    if sidecar.exists():
        meta = _json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise RuntimeError(f"SF checkpoint sidecar must be a JSON object: {sidecar}")
        if meta.get("model_graph") is not None:
            from qnn.model.graph import GraphSpec
            graph = GraphSpec.from_dict(meta["model_graph"])
        if "obs_dim" in meta:
            obs_dim = int(meta["obs_dim"])

    if graph is None:
        if model_config is not None:
            model = (
                model_config
                if isinstance(model_config, ModelConfig)
                else ModelConfig.from_dict(model_config)
            )
        elif meta is not None and "model" in meta:
            model = ModelConfig.from_dict(meta["model"])
        else:
            raise RuntimeError(
                f"SF checkpoint requires a sidecar JSON with 'model'/'model_graph' "
                f"({sidecar}) or an explicit model_config"
            )

    return sf_to_qnn(
        sf_checkpoint_path=p,
        obs_dim=obs_dim,
        model=model,
        device=device,
        graph=graph,
    )


def sniff_entity_stream(state: "Dict[str, torch.Tensor]") -> str | None:
    """Which entity stream a state dict was trained on, from module shape.

    The one shape that separates the generations: the projectile
    projection's in-dim (combat 7 vs full 8 — the a26 stream keeps the
    trailing recency scalar). Returns None when the state dict carries no
    obs embedding at this key (SF layouts, pre-rename legacy checkpoints —
    those lines never trained the full stream, so there is nothing to
    sniff). Any other in-dim fails loud: it is not a stream this line can
    rebuild.
    """
    weight = state.get("obs_embedding.proj_projectile.weight")
    if weight is None:
        return None
    in_dim = int(weight.shape[1])
    if in_dim == FULL_PROJECTILE_SCALAR_DIM:
        return ENTITY_STREAM_FULL
    if in_dim == PROJECTILE_SCALAR_DIM:
        return ENTITY_STREAM_COMBAT
    raise ValueError(
        f"obs_embedding.proj_projectile.weight has in-dim {in_dim}; expected "
        f"{PROJECTILE_SCALAR_DIM} (combat stream) or "
        f"{FULL_PROJECTILE_SCALAR_DIM} (full a26 stream) — unknown entity "
        "stream, refusing to guess a model layout"
    )


def checkpoint_model_graph(path: str | Path) -> dict[str, Any] | None:
    """The checkpoint's declarative model graph, if its sidecar carries one.

    Reads the sidecar JSON next to the ``.pth`` (no torch load) and returns
    the raw ``meta.model_graph`` dict. Tolerant by design: missing/empty
    path, missing sidecar, unreadable/corrupt JSON, or a non-dict
    ``model_graph`` all return None (legacy flat checkpoint).
    """
    if not str(path):
        return None
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    graph = meta.get("model_graph") if isinstance(meta, dict) else None
    return graph if isinstance(graph, dict) else None


def load_checkpoint(
    path: str | Path,
    *,
    device: str,
    model_config: "dict | ModelConfig | None" = None,
) -> "QNNPolicy":
    """Load a checkpoint in either QNN or SF format.

    Every checkpoint is self-describing: graph-described checkpoints rebuild
    via ``meta["model_graph"]``; canonical flat checkpoints via ``meta["model"]``.
    ``QNNPolicy.load`` picks the path from the embedded meta — no probe.json /
    bench-factory rehydration (the legacy HEADS reload path was retired).
    """
    if is_sf_checkpoint(path):
        policy = load_sf_checkpoint_as_qnn(path, device=device, model_config=model_config)
    else:
        policy = QNNPolicy.load(str(path), device=device, model_factory=None)
    policy.contract = resolve_checkpoint_contract(path)
    return policy


def resolve_checkpoint_contract(path: str | Path) -> "dict | None":
    """Resolve a checkpoint's model↔engine contract, backfilling if absent.

    The checkpoint is the SOURCE OF TRUTH. Returns ``meta["contract"]`` verbatim
    when present; otherwise BACKFILLS from the generation→contract registry
    (:mod:`qnn.contracts`) for the recognized generation. Returns ``None`` (and
    warns) when the checkpoint carries no contract AND its generation is
    unrecognized — never invents a value.

    The QNN ``meta`` block is read from ``payload["meta"]`` (QNN/BC format) or,
    for SF warm-start checkpoints (no embedded meta), from the sibling
    ``.json`` sidecar that holds the same ``{"model", "obs_dim", ...}`` schema.
    Reads only those schema markers; never inspects an ONNX graph, tensor
    shapes, or the filename.
    """
    import json as _json
    import warnings

    from qnn.contracts import backfill_contract

    p = Path(path)
    payload = trusted_torch_load(p, map_location="cpu")
    meta: Dict[str, Any] | None = None
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        meta = payload["meta"]
    else:
        # SF warm-start checkpoints carry the QNN config in a sibling .json
        # sidecar (see load_sf_checkpoint_as_qnn), not in payload["meta"].
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            try:
                loaded = _json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (ValueError, OSError):
                meta = None
    if meta is None:
        warnings.warn(
            f"{p}: no QNN meta (payload['meta'] or sidecar .json); cannot resolve contract.",
            stacklevel=2,
        )
        return None
    existing = meta.get("contract")
    if isinstance(existing, dict):
        return dict(existing)
    backfilled = backfill_contract(meta)
    if backfilled is None:
        warnings.warn(
            f"{path}: checkpoint has no 'contract' block and its generation is "
            "unrecognized by the contract registry — leaving contract unset. "
            "Stamp it explicitly with tools/stamp_checkpoint.py before export.",
            stacklevel=2,
        )
    return backfilled
