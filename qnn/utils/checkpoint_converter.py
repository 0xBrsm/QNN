"""Checkpoint conversion between QNNPolicy and PPO warm-start formats.

Two conversion directions:
  BC/PPO → SF   : ``bc_to_sf()``  — warm-start APPO from a BC checkpoint.
  SF → BC/PPO   : ``sf_to_qnn()``  — convert SF checkpoint back to QNNPolicy
                                    format for evaluation.py.

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
    SELF_SCALAR_DIM,
    SPATIAL_SCALAR_DIM,
    SPATIAL_TOKEN_COUNT,
)
from qnn.model.network import ModelConfig
from qnn.model.policy import QNNPolicy
from qnn.utils.io import trusted_torch_load
from qnn.vocab import ENTITY_IDS

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


def bc_to_sf(
    bc_path: str | Path,
    sf_model: torch.nn.Module,
    *,
    device: str,
) -> Dict[str, Any]:
    """Copy weights from a BC/PPO checkpoint into an SF model state dict."""
    bc_policy = QNNPolicy.load(str(bc_path), device=device)
    bc_state = bc_policy.model.state_dict()
    sf_state = sf_model.state_dict()

    _copy_encoder(bc_state, sf_state, device)
    _copy_gru(bc_state, sf_state, device)
    _copy_value_head(bc_state, sf_state, device)
    _copy_bc_heads_to_sf_combined(bc_state, sf_state, device)

    return sf_state


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

    missed = _copy_encoder(sf_state, bc_state, device, reverse=True)
    missed += _copy_gru(sf_state, bc_state, device, reverse=True)
    if missed:
        raise RuntimeError(
            f"SF→QNN conversion left {len(missed)} trained weight(s) at random "
            f"init (no matching SF key/shape). Causes: converting a graph-"
            f"described run without its graph, an architecture mismatch, or a "
            f"pre-rename SF checkpoint whose keys need the legacy migrations "
            f"(reverse conversion does not apply them). First missed: {missed[:4]}"
        )
    _copy_value_head(sf_state, bc_state, device, reverse=True)
    if "move_head.mlp.0.weight" in bc_state:
        # Mirror of save_sf_format's forward warning: SF trains a single flat
        # action linear; there is no projection back into bottleneck-MLP
        # heads. The converted checkpoint carries the TRAINED encoder, GRU,
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

    # ─── Encoder ──────────────────────────────────────────────────
    # BC trunk lives under ``obs_embedding.*`` + ``encoder.*``; SF's
    # actor-critic wraps both under its own ``encoder.*`` namespace.
    # ``target_pointer.*`` is the MLP score module (``score.0.*`` /
    # ``score.2.*``) — port its keys so the SF QuakeTransformerEncoder's
    # owned TargetPointer warm-starts rather than orthogonal-initializing.
    for bc_key, tensor in bc_state.items():
        if bc_key.startswith("obs_embedding."):
            sf_key = f"encoder.{bc_key}"
            sf_style[sf_key] = tensor.cpu()
        elif bc_key.startswith("encoder."):
            sf_key = f"{_SF_ENCODER_PREFIX}.{bc_key[len('encoder.'):]}"
            sf_style[sf_key] = tensor.cpu()
        elif bc_key.startswith("target_pointer."):
            sf_key = f"encoder.{bc_key}"
            sf_style[sf_key] = tensor.cpu()
        # Legacy attention-style pointer keys (query_proj.*,
        # weapon_query_embed.*, idx_prior_scale) are silently dropped:
        # the pre-MLP pointer doesn't load into the new MLP score module.
        # Pre-promotion checkpoints are not compatible — see
        # ModelConfig.d_target for the new pointer's sole knob.

    # ─── GRU ───────────────────────────────────────────────────────
    # Pre-temporal-wrap checkpoints emit ``gru.*``; modern checkpoints
    # (post-migrate_wrap_gru_in_temporal) emit ``temporal.gru.*``. SF
    # expects ``core.core.*``.
    for gru_param in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        for bc_prefix in ("temporal.gru.", "gru."):
            bc_key = f"{bc_prefix}{gru_param}"
            if bc_key in bc_state:
                sf_style[f"{_SF_GRU_PREFIX}.{gru_param}"] = bc_state[bc_key].cpu()
                break

    # ─── Value head ────────────────────────────────────────────────
    # Modern BC doesn't train a value head (combat-objective phase 1);
    # leave SF's critic_linear at random init. Earlier checkpoints
    # carried a ``value_head.*`` Linear → keep the old mapping for them.
    for suffix in ("weight", "bias"):
        bc_key = f"value_head.{suffix}"
        if bc_key in bc_state:
            sf_style[f"{_SF_VALUE_PREFIX}.{suffix}"] = bc_state[bc_key].cpu()

    # ─── Action head ───────────────────────────────────────────────
    # Modern BC heads (move/look/attack/weapon) are bottleneck MLPs
    # (Linear → GELU → Linear) under ``{head}_head.mlp.*``, while SF
    # expects a single combined Linear over all heads at
    # ``action_parameterization.distribution_linear.*``. There is no
    # bit-perfect projection, so when we detect the modern layout we
    # skip the action head copy and let SF orthogonal-init it — the
    # encoder + GRU + target_pointer warm-start is still useful for
    # PPO. Older checkpoints with the flat ``policy_heads.*`` layout
    # keep the legacy concat path.
    # Modern heads are wired together — one canonical lookup catches
    # all four. (move_head.mlp.0.weight is the move head's input
    # projection; if it's present, the BC checkpoint was trained with
    # bottleneck-MLP heads end to end.)
    has_modern_heads = "move_head.mlp.0.weight" in bc_state
    if has_modern_heads:
        print(
            "[bc_to_sf] BC checkpoint uses bottleneck-MLP heads "
            "(move_head/look_head/attack_head/weapon_head); SF's flat "
            "action_parameterization can't represent them — leaving the SF "
            "action head and critic_linear at orthogonal init. Encoder + GRU + "
            "target_pointer weights are warm-started."
        )
    else:
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
    # Arch fields are sourced from the policy's ModelConfig (the
    # canonical home is now ``bc_policy.config`` — a frozen
    # ModelConfig dataclass; the older flat attributes are gone).
    meta = {
        "obs_dim": bc_policy.obs_dim,
        "model": bc_policy.config.to_dict(),
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


def migrate_legacy_flat_meta(meta: Dict[str, Any]) -> Dict[str, Any] | None:
    """Translate a pre-ModelConfig flat checkpoint meta into the modern nested format.

    Pre-ModelConfig checkpoints stored architecture as a flat top-level dict
    (`d_model`, `d_gru`, `look_bypass_gru`, ...). Modern ``QNNPolicy.load``
    expects ``meta["model"]`` to be a ModelConfig-shaped sub-dict and requires
    ``jump_pos_weight`` at the top level. Two flavors are recognized:

    - **v22** (move_categorical=False fields like ``target_bypass_gru``/
      ``readout`` absent; ``head_bottleneck_dims`` plural present): direct
      field copy with sensible defaults for post-v22-only flags.
    - **v17** (``target_bypass_gru`` / ``move_categorical`` / ``readout``
      present; no weapon head trained): bare-Linear heads, weapon head
      disabled (ONNX wrapper substitutes constant weapon_logits).

    Returns the rewritten meta dict, or ``None`` if the input is already in
    modern form (i.e. ``meta["model"]`` is present) or doesn't look like a
    recognized legacy schema.

    Limitations:
    - For v17: synthesized weapon_logits are constant zeros at export.
    - ``jump_pos_weight`` / ``attack_focal_gamma`` defaults are inert at
      inference but won't match training-time values.
    """
    if "model" in meta:
        return None
    # Oldest flat checkpoints (pre `d_*` rename, commit 5dc0d41) stored the
    # transformer/GRU widths under their pre-rename names. Normalize those
    # aliases up front so the recognition check and field reads below — which
    # use the modern `d_*` names — accept them. (`trunk_hidden` is not consumed
    # by the modern ModelConfig, so it is intentionally dropped.)
    meta = dict(meta)
    for _old, _new in (("ffn_dim", "d_ffn"), ("gru_hidden", "d_gru")):
        if _old in meta and _new not in meta:
            meta[_new] = meta[_old]
    required = ("d_model", "n_heads", "n_layers", "d_ffn", "attn_dropout",
                "use_gru", "d_gru", "look_bypass_gru")
    if any(k not in meta for k in required):
        return None

    is_v17 = "target_bypass_gru" in meta or "move_categorical" in meta or "readout" in meta

    if is_v17:
        model_cfg = {
            "d_model":                  int(meta["d_model"]),
            "n_heads":                  int(meta["n_heads"]),
            "n_layers":                 int(meta["n_layers"]),
            "d_ffn":                  int(meta["d_ffn"]),
            "attn_dropout":             float(meta["attn_dropout"]),
            "use_gru":                  bool(meta["use_gru"]),
            "d_gru":               int(meta["d_gru"]),
            # v17 had no weapon head — state_dict migration allow-lists
            # weapon_head.* / weapon_embed.* as missing keys at load time.
            "use_weapon_head":          False,
            "weapon_switch_confidence": 0.5,
            "weapon_switch_margin":     0.0,
            "weapon_context_from_obs":  False,
            "look_bypass_gru":          bool(meta["look_bypass_gru"]),
            # MLP target pointer hidden width — d_model is the historical
            # default. v17 had no MLP pointer; weights are random-init at
            # load time for the new score module.
            "d_target":                 int(meta["d_model"]),
            # Inert when use_weapon_head=False (weapon_head module isn't
            # built); set to canonical so the spec is well-formed.
            "weapon_sources":           ["self_readout", "target_feat"],
            "self_weapon_embed_in_self": False,
            "d_move":      0,
            "d_look":      0,
            "d_attack":    0,
            "d_weapon":    0,
            "head_activation":          "none",
        }
        return {
            "obs_dim":          int(meta.get("obs_dim", 0)),
            "jump_pos_weight":  1.0,
            "attack_focal_gamma": 0.0,
            "attack_focal_alpha": 0.5,
            "model":            model_cfg,
        }

    # v22 schema: most modern fields already present, only the post-v22
    # additions need synthesized defaults.  Older schemas stored the
    # per-head MLP widths under names that have since been retired:
    #   * v22:        head_bottleneck_dims (dict, "fire" head name)
    #   * v23 era:    head_d_hidden        (dict, "attack" head name)
    # The modern schema is four scalar fields (d_move,
    # d_look, d_attack, d_weapon).
    bottleneck = meta.get("head_bottleneck_dims") or meta.get("head_d_hidden")
    if isinstance(bottleneck, dict):
        # Legacy "fire" head key was renamed "attack" to match the QW
        # source's BUTTON_ATTACK / self.button0 naming.
        bd = {k: int(v) for k, v in bottleneck.items()}
        if "fire" in bd and "attack" not in bd:
            bd["attack"] = bd.pop("fire")
        d_move = int(bd.get("move", 0))
        d_look = int(bd.get("look", 0))
        d_attack = int(bd.get("attack", 0))
        d_weapon = int(bd.get("weapon", 0))
    elif bottleneck is not None:
        v = int(bottleneck)
        d_move = d_look = d_attack = d_weapon = v
    else:
        # Modern direct schema — already four scalars in the metadata.
        d_move = int(meta.get("d_move", 0))
        d_look = int(meta.get("d_look", 0))
        d_attack = int(meta.get("d_attack", 0))
        d_weapon = int(meta.get("d_weapon", 0))

    # Weapon-selector composition. New checkpoints carry weapon_sources
    # directly; pre-rename ones carry the weapon_use_gru / weapon_use_self_readout
    # bools (with the older weapon_use_cls_readout alias). Historical defaults
    # were both True → canonical [gru, self_readout, target_feat].
    if "weapon_sources" in meta:
        weapon_sources = list(meta["weapon_sources"])
    else:
        _w_gru = bool(meta.get("weapon_use_gru", True))
        _w_self = bool(meta.get("weapon_use_self_readout", meta.get("weapon_use_cls_readout", True)))
        weapon_sources = (
            (["gru"] if _w_gru else [])
            + (["self_readout"] if _w_self else [])
            + ["target_feat"]
        )

    model_cfg = {
        "d_model":                   int(meta["d_model"]),
        "n_heads":                   int(meta["n_heads"]),
        "n_layers":                  int(meta["n_layers"]),
        "d_ffn":                   int(meta["d_ffn"]),
        "attn_dropout":              float(meta["attn_dropout"]),
        "use_gru":                   bool(meta["use_gru"]),
        "d_gru":                int(meta["d_gru"]),
        "use_weapon_head":           bool(meta.get("use_weapon_head", True)),
        "weapon_switch_confidence":  float(meta.get("weapon_switch_confidence", 0.65)),
        "weapon_switch_margin":      float(meta.get("weapon_switch_margin", 0.15)),
        "weapon_sources":            weapon_sources,
        "weapon_context_from_obs":   bool(meta.get("weapon_context_from_obs", False)),
        "look_bypass_gru":           bool(meta["look_bypass_gru"]),
        # MLP target pointer hidden width — falls back to d_model when
        # the source meta predates the MLP pointer (pre-promotion
        # checkpoints carried query_proj weights instead). state_dict
        # migration drops the pre-MLP target_pointer.* keys and the new
        # MLP score module is random-init at load.
        "d_target":                  int(
            meta.get("d_target", meta.get("d_model"))
        ),
        "self_weapon_embed_in_self": bool(meta.get("self_weapon_embed_in_self", False)),
        "d_move":             d_move,
        "d_look":             d_look,
        "d_attack":           d_attack,
        "d_weapon":           d_weapon,
        "head_activation":           str(meta.get("head_activation", "relu")),
    }
    # fire_* keys are the pre-rename names for the attack-head focal /
    # distance knobs. Honor them when the modern key is absent.
    return {
        "obs_dim":             int(meta.get("obs_dim", 0)),
        "jump_pos_weight":     float(meta.get("jump_pos_weight", 1.0)),
        "attack_focal_gamma":  float(
            meta.get("attack_focal_gamma", meta.get("fire_focal_gamma", 0.0))
        ),
        # attack_focal_alpha=0.5 is neutral (equivalent to no class
        # rebalancing); safe migration default for pre-toggle ckpts.
        "attack_focal_alpha":  float(
            meta.get("attack_focal_alpha", meta.get("fire_focal_alpha", 0.5))
        ),
        "attack_distance_sigma": float(
            meta.get("attack_distance_sigma", meta.get("fire_distance_sigma", 0.0))
        ),
        "jump_distance_sigma": float(meta.get("jump_distance_sigma", 0.0)),
        "model":               model_cfg,
    }


def migrate_v17_move_heads(state: Dict[str, torch.Tensor]) -> bool:
    """Fold v17's split move_fb_head + move_lr_head into the v20 unified move_head.

    v17 trained two 3-class Linear heads (fb, lr) and never modeled the
    ud (jump) axis.  v20 uses a single Linear(head_in, 9) reshaped to
    (3 axes × 3 classes).  Layout: rows 0-2 = fb, 3-5 = lr, 6-8 = ud.

    Migration packs the v17 weights into rows 0-5 of the unified head and
    bias-locks the ud axis to class 1 ("none") so v17 inference produces
    deterministic no-jump decisions instead of random ud logits.  The
    split-head tensors are then deleted from the state dict so they don't
    show up as unexpected keys at load time.

    Operates on BC state dicts (top-level ``move_fb_head.*`` /
    ``move_lr_head.*`` keys).  Returns True iff a migration ran.
    """
    fb_w = state.get("move_fb_head.weight")
    fb_b = state.get("move_fb_head.bias")
    lr_w = state.get("move_lr_head.weight")
    lr_b = state.get("move_lr_head.bias")
    if fb_w is None or fb_b is None or lr_w is None or lr_b is None:
        return False

    # Sanity-check shapes: (3, head_in) and (3,) for each axis.
    if fb_w.shape != lr_w.shape or fb_w.shape[0] != 3:
        return False

    head_in = fb_w.shape[1]
    unified_w = torch.zeros((9, head_in), dtype=fb_w.dtype)
    unified_w[0:3] = fb_w
    unified_w[3:6] = lr_w
    # ud rows stay zero — bias-only decision.

    unified_b = torch.zeros((9,), dtype=fb_b.dtype)
    unified_b[0:3] = fb_b
    unified_b[3:6] = lr_b
    # Lock ud softmax to class 1 (none) regardless of features.
    unified_b[6:9] = torch.tensor([-100.0, 100.0, -100.0], dtype=fb_b.dtype)

    state["move_head.weight"] = unified_w
    state["move_head.bias"] = unified_b
    for key in ("move_fb_head.weight", "move_fb_head.bias",
                "move_lr_head.weight", "move_lr_head.bias"):
        state.pop(key, None)
    return True


def migrate_drop_fire_align_scalar(state: Dict[str, torch.Tensor]) -> bool:
    """Strip the trailing alignment-scalar column from pre-v21 fire heads.

    Pre-v21 fire heads were Linear(fire_in + 1, …) — the +1 was the
    cosine of look_predict against the target-anchored look_prior,
    concatenated as the last feature dim.  Settled-null in ablation
    and removed from the architecture; v17/v20-era checkpoints still
    carry the 129-wide first-layer weight (128 fused features + 1
    alignment scalar) even though their training treated the scalar
    as essentially dead weight.

    Migration drops the last input column from the attack head's first
    Linear so it matches the current 128-wide architecture.  Covers
    both single-Linear (`attack_head.weight`) and bottlenecked
    (`attack_head.0.weight`) layouts.  Bias is per-output, unaffected.
    Returns True iff a migration ran.

    Detection is by shape, not by meta flag: v17 doesn't record a
    `fire_use_align_scalar` field even though it trained with the
    scalar wired in.  An odd `in_dim` is the unambiguous tell — the
    even base_features_dim makes that impossible without the +1.

    Runs AFTER `migrate_rename_fire_head_to_attack_head`, so the
    keys this scans are already in the new `attack_head.*` layout.
    """
    migrated = False
    for key in ("attack_head.weight", "attack_head.0.weight"):
        w = state.get(key)
        if w is None or w.ndim != 2:
            continue
        in_dim = w.shape[1]
        if in_dim % 2 != 1:
            continue  # already even — no trailing align column to drop
        state[key] = w[:, :-1].contiguous()
        migrated = True
    return migrated


def migrate_drop_weapon_embed_self(state: Dict[str, torch.Tensor]) -> bool:
    """Strip the dedicated `weapon_embed_self` table from v24-era checkpoints.

    v24 added a separate impulse-indexed `obs_embedding.weapon_embed_self`
    Embedding(WEAPON_HEAD_SIZE+1, d_model) for the current-held-weapon
    contribution on the self token.  That contribution now routes through
    the shared `entity_embed` (weapons live at rows 3..10 by design — see
    qnn.vocab), so the dedicated table is no longer instantiated.

    Deletes the stale key so load_state_dict(strict=True) doesn't trip
    on an unexpected param.  Returns True iff a deletion ran.  The held-
    weapon representation is bootstrapped from entity_embed's already-
    trained weapon rows (learned from world weapon tokens), so no warm-
    init copy is needed.
    """
    migrated = False
    for key in list(state.keys()):
        if key.endswith("weapon_embed_self.weight"):
            del state[key]
            migrated = True
    return migrated


def migrate_obs_embedding_self_token_builder(state: Dict[str, torch.Tensor]) -> bool:
    """Rename canonical self_proj tensors into TokenBuilder's scalar Linear.

    Option B moved the production monolithic self token from
    ``ObsEmbedding.self_proj`` to ``ObsEmbedding.self_token_builder`` while
    preserving a single full-width Linear over ``self_scalars``. The parameter
    mapping is therefore exact:

      ``*.obs_embedding.self_proj.weight`` -> ``*.obs_embedding.self_token_builder.projs.0.weight``
      ``*.obs_embedding.self_proj.bias``   -> ``*.obs_embedding.self_token_builder.projs.0.bias``
    """
    rewrites = {
        "self_proj.weight": "self_token_builder.projs.0.weight",
        "self_proj.bias": "self_token_builder.projs.0.bias",
    }
    migrated = False
    for key in list(state.keys()):
        for old_suffix, new_suffix in rewrites.items():
            if not key.endswith(old_suffix):
                continue
            new_key = key[: -len(old_suffix)] + new_suffix
            if new_key not in state:
                state[new_key] = state[key]
            del state[key]
            migrated = True
            break
    return migrated


def migrate_drop_action_history(state: Dict[str, torch.Tensor]) -> bool:
    """Strip the action_history obs-embedding pieces from pre-rip-out checkpoints.

    Pre-rip-out embeddings carried:
      - obs_embedding.action_proj.{weight,bias}: Linear(8, d_model)
      - obs_embedding.action_pos_embed.weight:   Embedding(8, d_model)
      - obs_embedding.kind_embed.weight:         Embedding(4, d_model)
        — kinds 0..3 = self / entity / spatial / action.

    The new obs embedding drops the action-history branch entirely and sizes
    kind_embed at (3, d_model), keeping the same row order for the first
    three kinds.  This migration deletes the dead action_proj / pos_embed
    keys and truncates kind_embed to its first three rows so load_state_dict
    can apply the v17/v20-era checkpoint without shape mismatches on the
    kind embedding.  Returns True iff any change was made.
    """
    migrated = False
    prefixes = ("obs_embedding.", "encoder.obs_embedding.", "tokenizer.", "encoder.tokenizer.")
    for prefix in prefixes:
        for suffix in (
            "action_proj.weight",
            "action_proj.bias",
            "action_pos_embed.weight",
        ):
            key = f"{prefix}{suffix}"
            if key in state:
                del state[key]
                migrated = True

        kind_key = f"{prefix}kind_embed.weight"
        kind_w = state.get(kind_key)
        if kind_w is not None and kind_w.shape[0] == 4:
            state[kind_key] = kind_w[:3].clone()
            migrated = True

    return migrated


def migrate_rename_trunk_to_encoder(state: Dict[str, torch.Tensor]) -> bool:
    """Rename ``trunk.*`` state-dict keys to ``encoder.*``.

    The transformer-stack module was renamed ``TransformerTrunk`` →
    ``TransformerEncoder`` (and its enclosing attribute ``self.trunk`` →
    ``self.encoder``) to match transformer-canonical naming. Pre-rename
    checkpoints carry keys like ``trunk.obs_embedding.self_proj.weight`` and
    ``trunk.blocks.0.*``; this rewrites them to the ``encoder.*`` prefix.
    Prefix-based (only at start-of-key) so it doesn't touch unrelated
    occurrences.
    """
    migrated = False
    prefix = "trunk."
    replacement = "encoder."
    for key in list(state.keys()):
        if key.startswith(prefix):
            new_key = replacement + key[len(prefix):]
            state[new_key] = state.pop(key)
            migrated = True
    return migrated


def migrate_rename_tokenizer_to_obs_embedding(state: Dict[str, torch.Tensor]) -> bool:
    """Rename temporary ``tokenizer`` state-dict paths to ``obs_embedding``.

    A short-lived feature branch named the input-embedding module
    ``tokenizer``. The settled public name is ``obs_embedding``. Rewrite both
    top-level keys (``tokenizer.*``) and nested keys
    (``encoder.tokenizer.*`` / ``trunk.tokenizer.*`` after trunk migration).
    """
    migrated = False
    for key in list(state.keys()):
        new_key = key
        if new_key.startswith("tokenizer."):
            new_key = "obs_embedding." + new_key[len("tokenizer."):]
        new_key = new_key.replace(".tokenizer.", ".obs_embedding.")
        if new_key != key:
            state[new_key] = state.pop(key)
            migrated = True
    return migrated


def migrate_hoist_encoder_obs_embedding(state: Dict[str, torch.Tensor]) -> bool:
    """Move pre-split ``encoder.obs_embedding.*`` keys to top-level ``obs_embedding.*``.

    The transformer stack remains under ``encoder.*``; only the raw
    observation embedding was lifted out so it can be swapped independently
    from the stack in bench/probe networks.
    """
    migrated = False
    prefix = "encoder.obs_embedding."
    replacement = "obs_embedding."
    for key in list(state.keys()):
        if key.startswith(prefix):
            new_key = replacement + key[len(prefix):]
            state[new_key] = state.pop(key)
            migrated = True
    return migrated


def migrate_rename_fire_head_to_attack_head(state: Dict[str, torch.Tensor]) -> bool:
    """Rename ``fire_head.*`` state-dict keys to ``attack_head.*``.

    The binary attack-bit head was historically named ``fire_head`` even
    after the broader fire→attack rename (486af0b / b0279fe) renamed the
    action axis, head string ("attack"), and logits dict key. The
    attribute name lagged behind. Pre-rename checkpoints carry keys like
    ``fire_head.weight`` / ``fire_head.0.weight`` / ``fire_head.2.bias``;
    this rewrites the prefix so load_state_dict matches ``self.attack_head``.

    Prefix-based — only rewrites keys starting with ``fire_head.``.
    Must run BEFORE ``migrate_drop_fire_align_scalar``, which expects
    the new ``attack_head.*`` layout.
    """
    migrated = False
    prefix = "fire_head."
    replacement = "attack_head."
    for key in list(state.keys()):
        if key.startswith(prefix):
            new_key = replacement + key[len(prefix):]
            state[new_key] = state.pop(key)
            migrated = True
    return migrated


def migrate_wrap_gru_in_temporal(state: Dict[str, torch.Tensor]) -> bool:
    """Rename ``gru.*`` state-dict keys to ``temporal.gru.*``.

    The GRU now lives inside the ``Temporal`` component so the orchestrator
    can stay seq/flat-agnostic. Pre-refactor checkpoints carry keys like
    ``gru.weight_ih_l0`` / ``gru.bias_hh_l0``; this rewrites the prefix so
    load_state_dict matches ``self.temporal.gru.*``.

    Prefix-based — only rewrites keys starting with ``gru.`` exactly.
    """
    migrated = False
    prefix = "gru."
    replacement = "temporal.gru."
    for key in list(state.keys()):
        if key.startswith(prefix):
            new_key = replacement + key[len(prefix):]
            state[new_key] = state.pop(key)
            migrated = True
    return migrated


def migrate_wrap_heads_in_components(state: Dict[str, torch.Tensor]) -> bool:
    """Wrap head MLPs and aux params into per-head Component containers.

    Heads (move/look/attack/weapon) used to live as flat attributes on
    ``Network`` — ``self.move_head`` was a Linear or Sequential directly,
    and weapon_embed / attack_alignment_scale[_emb] were siblings of the
    heads. The component refactor moves each head's machinery under its
    own ``nn.Module`` container (``MoveHead``, ``LookHead``, ``AttackHead``,
    ``WeaponHead``). This shifts the state-dict layout:

      Before                              After
      -------                             -----
      move_head.*                         move_head.mlp.*
      look_head.*                         look_head.mlp.*
      attack_head.*                       attack_head.mlp.*
      weapon_head.*                       weapon_head.mlp.*
      weapon_embed.*                      weapon_head.embed.*
      attack_alignment_scale_emb.*        attack_head.alignment_scale_emb.*
      attack_alignment_scale  (scalar)    attack_head.alignment_scale

    Runs AFTER ``migrate_rename_fire_head_to_attack_head`` so it sees
    the ``attack_head.*`` layout (not ``fire_head.*``) and AFTER
    ``migrate_drop_fire_align_scalar`` so the trimmed shapes are already
    in place under the new key.
    """
    PREFIX_REWRITES = (
        ("attack_alignment_scale_emb.", "attack_head.alignment_scale_emb."),
        ("weapon_embed.",               "weapon_head.embed."),
        ("move_head.",                  "move_head.mlp."),
        ("look_head.",                  "look_head.mlp."),
        ("attack_head.",                "attack_head.mlp."),
        ("weapon_head.",                "weapon_head.mlp."),
    )
    EXACT_REWRITES = {
        "attack_alignment_scale": "attack_head.alignment_scale",
    }
    migrated = False
    for old_key in list(state.keys()):
        if old_key in EXACT_REWRITES:
            new_key = EXACT_REWRITES[old_key]
            state[new_key] = state.pop(old_key)
            migrated = True
            continue
        for old_prefix, new_prefix in PREFIX_REWRITES:
            if old_key.startswith(old_prefix):
                new_key = new_prefix + old_key[len(old_prefix):]
                state[new_key] = state.pop(old_key)
                migrated = True
                break
    return migrated


def _expanded_self_scalar_weight(tensor: torch.Tensor) -> torch.Tensor:
    """Expand self-proj columns from 14 old scalars to 16 v21 scalars."""
    old_sg = tensor[:, 2:3]
    old_ng = tensor[:, 3:4]
    return torch.cat(
        [
            tensor[:, 0:2],
            old_sg * 0.5,
            old_sg,
            old_ng * 0.5,
            old_ng,
            tensor[:, 4:],
        ],
        dim=1,
    )


def _expanded_self_scalar_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Expand normalizer vectors from the old 14-scalar self layout."""
    return torch.cat(
        [
            tensor[0:2],
            tensor[2:3],
            tensor[2:3],
            tensor[3:4],
            tensor[3:4],
            tensor[4:],
        ],
        dim=0,
    )


def _pad_self_scalar_weight_for_attack_finished(tensor: torch.Tensor) -> torch.Tensor:
    """Append a zero column for the new attack_finished scalar at idx 16.

    Pre-c38a5a26 checkpoints have self_proj input dim 16; commit c38a5a26
    added attack_finished as idx 16, bumping SELF_SCALAR_DIM 16 → 17.
    Zero-padding the column means the migrated model treats that input as
    dead weight (semantically: trained without that signal), so forward
    outputs are unchanged for the same observation.
    """
    zeros = torch.zeros(
        (tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device,
    )
    return torch.cat([tensor, zeros], dim=1)


def migrate_self_attack_finished_scalar(
    state: Dict[str, torch.Tensor], optimizer: Dict[str, Any] | None = None,
) -> bool:
    """Pad pre-c38a5a26 self_proj weights from 16 → 17 input dims."""
    migrated = False
    param_keys = [
        key for key in state
        if not key.startswith("obs_normalizer.") and not key.startswith("returns_normalizer.")
    ]
    for idx, key in enumerate(param_keys):
        tensor = state[key]
        if not (key.endswith("self_proj.weight") and tensor.ndim == 2 and tensor.shape[1] == 16):
            continue
        state[key] = _pad_self_scalar_weight_for_attack_finished(tensor)
        migrated = True

        if optimizer is not None and idx in optimizer.get("state", {}):
            opt_entry = optimizer["state"][idx]
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in opt_entry and hasattr(opt_entry[buf_key], "shape"):
                    buf = opt_entry[buf_key]
                    if buf.ndim == 2 and buf.shape[1] == 16:
                        opt_entry[buf_key] = _pad_self_scalar_weight_for_attack_finished(buf)
                        migrated = True

    for key, tensor in list(state.items()):
        if ".self_scalars." in key and hasattr(tensor, "shape") and tensor.shape == torch.Size([16]):
            zeros = torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
            state[key] = torch.cat([tensor, zeros], dim=0)
            migrated = True

    return migrated


def migrate_self_scalars(state: Dict[str, torch.Tensor], optimizer: Dict[str, Any] | None = None) -> bool:
    """Migrate pre-v21 self scalar layout to 7 binary inventory flags."""
    migrated = False
    param_keys = [
        key for key in state
        if not key.startswith("obs_normalizer.") and not key.startswith("returns_normalizer.")
    ]
    for idx, key in enumerate(param_keys):
        tensor = state[key]
        if not (key.endswith("self_proj.weight") and tensor.ndim == 2 and tensor.shape[1] == 14):
            continue
        state[key] = _expanded_self_scalar_weight(tensor)
        migrated = True

        if optimizer is not None and idx in optimizer.get("state", {}):
            opt_entry = optimizer["state"][idx]
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in opt_entry and hasattr(opt_entry[buf_key], "shape"):
                    buf = opt_entry[buf_key]
                    if buf.ndim == 2 and buf.shape[1] == 14:
                        opt_entry[buf_key] = _expanded_self_scalar_weight(buf)
                        migrated = True

    for key, tensor in list(state.items()):
        if ".self_scalars." in key and hasattr(tensor, "shape") and tensor.shape == torch.Size([14]):
            state[key] = _expanded_self_scalar_vector(tensor)
            migrated = True

    return migrated


def _permute_entity_rows_to_impulse(old: torch.Tensor) -> torch.Tensor | None:
    """Map a pre-impulse-order entity embed tensor to the v22 layout.

    Pre-impulse layouts (rows 0-2 = NONE/PLAYER/WEAPON, rows 3..N-1 follow):
      v17 (42 rows): no SUPER_SHOTGUN, no SUPER_NAILGUN.
        AXE=3, SG=4, NG=5, GL=6, RL=7, LG=8, AMMO=9 .. TRAIN=41.
      v20 (43 rows): SUPER_SHOTGUN appended at the tail (=42).
      pre-recollect 44 (interim): SUPER_NAILGUN appended at 43.

    v22 layout (this commit): weapons in Quake impulse order, contiguous.
      AXE=3, SG=4, SSG=5, NG=6, SNG=7, GL=8, RL=9, LG=10, AMMO=11 .. TRAIN=43.

    Returns the permuted (44, D) tensor.  If the old tail rows for SSG /
    SNG don't exist, those indices are seeded from the family parent
    (SHOTGUN / NAILGUN).
    """
    n_old = old.shape[0]
    if n_old not in (42, 43, 44):
        return None
    new = torch.zeros((44, *old.shape[1:]), dtype=old.dtype)
    # NONE..SHOTGUN (indices 0-4) carry over unchanged.
    new[0:5] = old[0:5]
    # SUPER_SHOTGUN: from old idx 42 if present, else seed from SHOTGUN.
    new[5] = old[42] if n_old >= 43 else old[4]
    # NAILGUN: old idx 5 → new idx 6.
    new[6] = old[5]
    # SUPER_NAILGUN: from old idx 43 if present, else seed from NAILGUN.
    new[7] = old[43] if n_old >= 44 else old[5]
    # GL, RL, LG: old indices 6..8 → new indices 8..10.
    new[8:11] = old[6:9]
    # AMMO..TRAIN: old indices 9..41 → new indices 11..43.
    new[11:44] = old[9:42]
    return new


def migrate_entity_embed(state: Dict[str, torch.Tensor], optimizer: Dict[str, Any] | None = None) -> bool:
    """Permute pre-impulse-order entity_embed rows into the v22 layout.

    Detection is by row count: 42- or 43-row tensors are v17 / v20 with
    SSG-only; both need permutation.  44-row tensors are assumed to
    already be in the new impulse-ordered layout (no shipped checkpoint
    has the brief interim 44-row old layout from commit 2fa6432b).
    """
    migrated = False
    target_rows = len(ENTITY_IDS)  # 44

    param_keys = [
        key for key in state
        if not key.startswith("obs_normalizer.") and not key.startswith("returns_normalizer.")
    ]
    for idx, key in enumerate(param_keys):
        if not key.endswith("entity_embed.weight"):
            continue
        tensor = state[key]
        if tensor.shape[0] == target_rows:
            continue
        permuted = _permute_entity_rows_to_impulse(tensor)
        if permuted is None:
            continue
        state[key] = permuted
        migrated = True

        if optimizer is not None and idx in optimizer.get("state", {}):
            opt_entry = optimizer["state"][idx]
            for buf_key in ("exp_avg", "exp_avg_sq"):
                buf = opt_entry.get(buf_key)
                if buf is None or not hasattr(buf, "shape"):
                    continue
                if buf.shape[0] == target_rows:
                    continue
                permuted_buf = _permute_entity_rows_to_impulse(buf)
                if permuted_buf is not None:
                    opt_entry[buf_key] = permuted_buf
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


def _copy_encoder(
    src: Dict, dst: Dict, device: str, reverse: bool = False,
) -> "list[str]":
    """Copy encoder weights between BC and SF state dicts.

    BC keys split input embedding (``obs_embedding.*``) from transformer
    stack (``encoder.*``). SF wraps both under its actor-critic encoder:
    ``encoder.obs_embedding.*`` and ``encoder.encoder.*``.

    In ``reverse`` mode returns the BC keys that found no matching SF
    weight — a non-empty list means the converted policy would keep
    random-init weights where trained ones were expected (e.g. converting
    a graph-described run without its graph: every ``self_builders.*``
    key misses). Callers must fail loud on it.
    """
    pairs = (
        ("obs_embedding.", "encoder.obs_embedding."),
        # SF owns the pointer inside its encoder (QuakeTransformerEncoder
        # builds it; save_sf_format ports it there) — copy it back too,
        # else converted full_5head policies keep a random-init pointer.
        ("target_pointer.", "encoder.target_pointer."),
        ("encoder.", f"{_SF_ENCODER_PREFIX}."),
    )

    if reverse:
        missed: list[str] = []
        for bc_prefix, sf_prefix in pairs:
            for dst_key in list(dst.keys()):
                if not dst_key.startswith(bc_prefix):
                    continue
                sf_key = sf_prefix + dst_key[len(bc_prefix):]
                if not _copy_weight(src, dst, sf_key, dst_key, device):
                    missed.append(dst_key)
        return missed
    else:
        for bc_prefix, sf_prefix in pairs:
            for src_key in list(src.keys()):
                if not src_key.startswith(bc_prefix):
                    continue
                sf_key = sf_prefix + src_key[len(bc_prefix):]
                _copy_weight(src, dst, src_key, sf_key, device)


def _copy_gru(src: Dict, dst: Dict, device: str, reverse: bool = False) -> "list[str]":
    """Copy GRU weights. Modern checkpoints key the recurrence
    ``temporal.gru.*`` (post-migrate_wrap_gru_in_temporal); pre-wrap ones
    use bare ``gru.*``. In ``reverse`` mode returns the BC GRU keys that
    received nothing (empty when the model has no temporal slot).
    """
    missed: list[str] = []
    for gru_param in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
        sf_key = f"{_SF_GRU_PREFIX}.{gru_param}"
        if reverse:
            bc_keys = [
                k for k in (f"temporal.gru.{gru_param}", f"gru.{gru_param}") if k in dst
            ]
            for bc_key in bc_keys:
                if not _copy_weight(src, dst, sf_key, bc_key, device):
                    missed.append(bc_key)
        else:
            for bc_prefix in ("temporal.gru.", "gru."):
                if f"{bc_prefix}{gru_param}" in src:
                    _copy_weight(src, dst, f"{bc_prefix}{gru_param}", sf_key, device)
                    break
    return missed


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

    if args.cmd == "bc-to-sf":
        policy = QNNPolicy.load(args.bc_path, device=args.device)
        ckpt = save_sf_format(policy, args.output_dir)
        print(f"Saved SF-format warm-start checkpoint: {ckpt}")

    elif args.cmd == "sf-to-qnn":
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
