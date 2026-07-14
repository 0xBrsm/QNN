"""Combat-objective BC network.

This is the ``nn.Module`` compute graph for the model — the encoder,
temporal (recurrence), target pointer, and the four output heads (move,
look, attack, weapon). It is the "model" in the SL sense.

The training-time wrapper (optimizers, loss shaping, sampling,
checkpoint I/O) lives in :mod:`qnn.model.policy` as ``QNNPolicy``.

Forward uses a reshape-once-at-entry / reshape-once-at-exit pattern: the
orchestrator detects sequence vs flat input from the obs shape, flattens
to ``(T*B, ...)`` if needed, runs every component on flat tensors, then
reshapes outputs back to ``(T, B, ...)`` at the end. The Temporal
component owns the seq-vs-flat branching internally (it has to — it's
the one component that needs the time axis to apply reset_mask
per-step).

Forward returns
``(features, logits_dict, values, next_hidden, target_logits)``;
ablation modules (e.g. ``qnn.model.bench.*``) must respect the same contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping

import torch
from torch import nn

from qnn.actions import MOVE_AXES, MOVE_AXIS_CLASSES
from qnn.model.attack_head import AttackHead, AttackHeadInput
from qnn.model.look_head import LookHead, LookHeadInput
from qnn.model.move_head import MoveHead, MoveHeadInput
from qnn.model.target import TargetPointer, TargetPointerInput
from qnn.model.temporal import Temporal, TemporalInput
from qnn.model.transformer import ObsEmbedding, TransformerEncoder
from qnn.model.weapon_head import WeaponHead, WeaponHeadInput
from qnn.vocab import TOKEN_ACTOR


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Canonical model architecture config — sole source of truth for arch.

    All fields are required; defaults live only in model.json. The
    dataclass is frozen so a constructed config can't drift from its
    serialized form. ``head_activation`` is "none" or "gelu" (ReLU was
    removed). Per-action-head MLP intermediate widths are explicit
    scalars (``d_move`` etc.); ``0`` disables the intermediate
    layer for that head.

    The target pointer is the MLP variant — see
    :mod:`qnn.model.target`. ``d_target`` is its MLP hidden width and
    the only architectural knob. Legacy attention-style variants
    (cls/GRU query, weapon-id query shift, hard-argmax / gt-dist /
    prev-target probes, learnable idx prior) live exclusively in
    :mod:`qnn.model.bench` for ablation.
    """
    d_model: int
    n_heads: int
    n_layers: int
    d_ffn: int
    attn_dropout: float
    use_gru: bool
    d_gru: int
    use_weapon_head: bool
    weapon_switch_confidence: float
    weapon_switch_margin: float
    # Weapon-selector input composition as an ordered edge list, not bools.
    # Each member names a source the selector cat is built from, in order:
    #   "gru"          — temporal hidden (contributes only when a temporal
    #                     slot is active; silently dropped otherwise).
    #   "self_readout" — encoder self/CLS readout.
    #   "target_feat"  — pointer-blended target feature (zeros when the
    #                     target slot is off, but still occupies its width).
    # Canonical is ("gru", "self_readout", "target_feat"). At least one of
    # {"gru", "self_readout"} must be present — target_feat alone is too thin.
    weapon_sources: tuple[str, ...]
    weapon_context_from_obs: bool
    look_bypass_gru: bool
    d_target: int
    self_weapon_embed_in_self: bool
    d_move: int
    d_look: int
    d_attack: int
    d_weapon: int
    head_activation: str

    @classmethod
    def from_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Build from a model.json-style mapping.

        Strips the legacy ``encoder_hidden`` alias of ``d_model``. Any
        other unknown key or any missing required field raises
        TypeError — every architectural flag must be set explicitly in
        model.json.
        """
        data = dict(raw)
        data.pop("encoder_hidden", None)
        data["weapon_sources"] = cls._resolve_weapon_sources(data)
        if data.get("head_activation") not in ("none", "gelu", "relu"):
            raise ValueError(
                f"head_activation must be 'none', 'gelu', or 'relu', got {data.get('head_activation')!r}"
            )
        return cls(**data)

    @staticmethod
    def _resolve_weapon_sources(data: "dict[str, Any]") -> "tuple[str, ...]":
        """Pop weapon-source keys from ``data`` and return the source tuple.

        New configs carry ``weapon_sources`` directly. Pre-rename configs
        (the 396 serialized run model.json) carry the old
        ``weapon_use_gru`` / ``weapon_use_self_readout`` bools (with the
        even-older ``weapon_use_cls_readout`` alias of the latter) — those
        are migrated to the ordered edge list here so existing checkpoints
        keep loading. Legacy keys are always stripped so they never reach
        the dataclass constructor.
        """
        legacy_gru = data.pop("weapon_use_gru", None)
        legacy_self = data.pop("weapon_use_self_readout", None)
        legacy_cls = data.pop("weapon_use_cls_readout", None)
        if "weapon_sources" in data:
            return tuple(data["weapon_sources"])
        # Migrate from bools. Historical defaults were both True (canonical
        # [gru, self_readout, target_feat]); only an explicit False drops a source.
        use_gru = True if legacy_gru is None else bool(legacy_gru)
        self_val = legacy_self if legacy_self is not None else legacy_cls
        use_self = True if self_val is None else bool(self_val)
        sources = []
        if use_gru:
            sources.append("gru")
        if use_self:
            sources.append("self_readout")
        sources.append("target_feat")
        return tuple(sources)

    @classmethod
    def from_flat_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Like ``from_dict`` but extracts the model fields from a larger
        flat config dict (e.g. a PPO config that merges train + model
        keys). Missing required model fields still raise TypeError.
        """
        keys = {f.name for f in fields(cls)} | {
            "encoder_hidden",
            "weapon_use_gru", "weapon_use_self_readout", "weapon_use_cls_readout",
        }
        subset = {k: v for k, v in raw.items() if k in keys}
        return cls.from_dict(subset)

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)


# Logits-dict keys — string identifiers used to address each head's
# output across the BC/PPO/eval call sites.
MOVE_HEAD = "move"
LOOK_HEAD = "look"
ATTACK_HEAD = "attack"
WEAPON_HEAD = "weapon"
MOVE_HAZARD_HEAD = "move_hazard"  # a25 WHEN/termination head (opt-in)

# Output sizes, exported for callers that build padded buffers or
# downstream layers against these sizes. Heads define their own
# OUT_DIM internally; these are the public face.
MOVE_HEAD_SIZE = MOVE_AXES * MOVE_AXIS_CLASSES  # 9 logits
LOOK_HEAD_SIZE = 3  # 3D direction vector
ATTACK_HEAD_SIZE = 1  # binary logit

# Offset of the relative-XYZ block inside an actor's per-token scalar vector.
# Mirrors qnn.bc.target_labeler._ACTOR_REL_OFFSET; duplicated here so the model
# layer doesn't import from BC.
_ACTOR_REL_OFFSET = 3
# Offset of the team scalar inside an actor's per-token scalar vector.
# Mirrors qnn.bc.target_labeler._ACTOR_TEAM_OFFSET. Used to derive enemy_mask
# for the target pointer.
_ACTOR_TEAM_OFFSET = 16
_TEAM_TEAMMATE_VALUE = 1.0


class _Off:
    """Sentinel singleton: pass ``Off`` for any per-slot override in
    ``Network.__init__`` to disable that slot — Network will skip the
    slot's forward call and (for head slots) omit its entry from the
    logits dict. ``None`` means "build the canonical component";
    an ``nn.Module`` means "use this one as the override".
    """

    _instance: "_Off | None" = None

    def __new__(cls) -> "_Off":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Off"

    def __bool__(self) -> bool:
        return False


Off = _Off()


def _flatten_obs(
    obs: Dict[str, torch.Tensor],
) -> tuple[tuple[int, int] | None, Dict[str, torch.Tensor]]:
    """Detect sequence-vs-flat from obs, return (seq_shape, flat_obs).

    ``vel`` (or legacy ``self_scalars``) carries the canonical ndim signal:
    2D means flat ``(B, ...)``, 3D means sequence ``(T, B, ...)``. When
    sequence, every obs tensor is reshaped to ``(T*B, ...)``. When flat,
    obs is returned unchanged. ``seq_shape`` is ``(T, B)`` or ``None``.
    """
    sample = obs.get("vel")
    if sample is None:
        sample = obs["self_scalars"]  # legacy fallback
    if sample.ndim == 3:
        seq_len = int(sample.shape[0])
        batch_size = int(sample.shape[1])
        flat = {
            key: value.reshape(seq_len * batch_size, *value.shape[2:])
            for key, value in obs.items()
        }
        return (seq_len, batch_size), flat
    return None, obs


def _restore_outputs(
    features_flat: torch.Tensor,
    logits_flat: Dict[str, torch.Tensor],
    values_flat: torch.Tensor,
    target_logits_flat: torch.Tensor,
    seq_shape: tuple[int, int],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Reshape every flat output back to ``(T, B, ...)`` for sequence callers."""
    seq_len, batch_size = seq_shape
    features = features_flat.reshape(seq_len, batch_size, -1)
    logits: Dict[str, torch.Tensor] = {
        key: value.reshape(seq_len, batch_size, *value.shape[1:])
        for key, value in logits_flat.items()
    }
    values = values_flat.reshape(seq_len, batch_size)
    target_logits = target_logits_flat.reshape(seq_len, batch_size, target_logits_flat.shape[-1])
    return features, logits, values, target_logits


def _weapon_source_dim(
    source: str, *, d_model: int, d_gru: int, has_temporal: bool,
) -> int:
    """Width one weapon-selector source contributes to the selector cat.

    ``gru`` contributes ``d_gru`` only when a temporal slot is active (and 0
    otherwise — it's silently dropped from the cat); ``self_readout`` and
    ``target_feat`` are each ``d_model`` wide.
    """
    if source == "gru":
        return int(d_gru) if has_temporal else 0
    if source in ("self_readout", "target_feat") or source.startswith("token:"):
        # a token:<name> source reads one encoder self-token output (d_model wide)
        return int(d_model)
    raise ValueError(f"unknown weapon source {source!r}")


def slot_dims(
    *,
    d_model: int,
    d_gru: int,
    has_temporal: bool,
    has_target_pointer: bool,
    has_weapon_head: bool,
    weapon_sources: "tuple[str, ...]",
) -> dict[str, int]:
    """Single authority for the dim contract Network's slots are built with.

    Pure function over *resolved node widths*, not a ModelConfig — ``d_model``
    is the encoder/obs-embedding ``out_dim``, ``d_gru`` is the temporal slot's
    ``out_dim`` (pass 0 when the slot is off). ``Network.__init__`` calls this
    with its built nodes' ``out_dim`` so nodes — not config scalars — are the
    dim source of truth; bench builders that must size an override head before
    constructing Network call it with their config-derived widths.

    ``has_target_pointer`` controls whether the motor feature vector carries the
    pointer's ``target_feat`` (``d_model`` wide). When the pointer slot is Off
    there is no target — the block is dropped entirely rather than fed as a
    ``d_model``-wide zeros pad the heads have to learn to ignore (and which, with
    a weapon head present, sits *between* the readout and weapon_context where no
    prefix slice can drop it). "If target is off, it's off."

    Keys:
      base_features_dim — input to motor heads excluding weapon_context.
      motor_in          — input to MoveHead / LookHead / AttackHead.
      weapon_in         — input to WeaponHead's classifier.
    """
    d_gru = int(d_gru) if has_temporal else 0
    d_model = int(d_model)
    weapon_ctx_dim = d_model if has_weapon_head else 0
    readout_dim = d_gru if has_temporal else d_model
    target_dim = d_model if has_target_pointer else 0
    base_features_dim = readout_dim + target_dim
    motor_in = base_features_dim + weapon_ctx_dim
    weapon_in = sum(
        _weapon_source_dim(src, d_model=d_model, d_gru=d_gru, has_temporal=has_temporal)
        for src in weapon_sources
    )
    return {
        "d_model": d_model,
        "d_gru": d_gru,
        "base_features_dim": base_features_dim,
        "motor_in": motor_in,
        "weapon_in": weapon_in,
    }


class Network(nn.Module):
    """The compute graph: encoder + temporal + target pointer + heads.

    Built by ``QNNPolicy`` (the training-time wrapper). Forward returns the
    five-tuple ``(features, logits_dict, values, next_hidden, target_logits)``.

    Per-slot overrides
    ------------------
    Each of ``obs_embedding``, ``encoder``, ``temporal``, ``target_pointer``,
    ``move_head``, ``look_head``, ``attack_head``, ``weapon_head`` accepts:

      * ``None`` (default) — Network builds the canonical component from
        ``ModelConfig`` (respecting the existing ``use_gru`` / ``use_weapon_head``
        flags for backward compatibility).
      * An ``nn.Module`` instance — Network uses it as-is. Use
        ``slot_dims(...)`` (passing the override's ``out_dim``) to size the
        override correctly.
      * ``Off`` (sentinel defined below) — slot disabled. ``Network.forward``
        substitutes zero tensors where the slot's output would have fed
        downstream, and (for head slots) omits the slot's logits-dict entry.
        ``obs_embedding`` and ``encoder`` cannot be ``Off``.

    ObsEmbedding vs encoder
    -----------------------
    ``obs_embedding(obs) → EncoderInput`` builds the token sequence (CLS +
    self subtokens + [spatial +] entities) with explicit slot slices.
    ``encoder(EncoderInput) → EncoderOutput`` runs (or skips) attention
    over those tokens and slices ``self_readout`` / ``entity_outs`` out.
    Swap them independently for token-layout vs attention-style
    ablations.
    """

    def __init__(
        self,
        obs_dim: int,
        model: ModelConfig,
        *,
        obs_embedding: nn.Module | None = None,
        encoder: nn.Module | None = None,
        temporal: "nn.Module | Off | None" = None,
        target_pointer: "nn.Module | Off | None" = None,
        move_head: "nn.Module | Off | None" = None,
        look_head: "nn.Module | Off | None" = None,
        attack_head: "nn.Module | Off | None" = None,
        weapon_head: "nn.Module | Off | None" = None,
        move_hazard_head: "nn.Module | Off | None" = Off,
    ) -> None:
        super().__init__()
        if obs_embedding is Off:
            raise ValueError("obs_embedding slot cannot be disabled (Off)")
        if encoder is Off:
            raise ValueError("encoder slot cannot be disabled (Off)")
        self.obs_dim = int(obs_dim)
        self.config = model
        self.obs_embedding = obs_embedding if obs_embedding is not None else ObsEmbedding(
            d_model=int(model.d_model),
            self_weapon_embed_in_self=bool(model.self_weapon_embed_in_self),
        )
        self.encoder = encoder if encoder is not None else TransformerEncoder(
            d_model=int(model.d_model),
            n_heads=int(model.n_heads),
            n_layers=int(model.n_layers),
            d_ffn=int(model.d_ffn),
            dropout=float(model.attn_dropout),
        )
        # Token width is the encoder's declared out_dim — the resolved node,
        # not the config scalar, is the dim source of truth (an override
        # encoder may carry a different width). A passthrough encoder
        # (PreAttnEncoder) emits the obs-embedding's tokens unchanged and
        # declares no out_dim, so fall back to the obs-embedding's width.
        self.d_model = int(getattr(self.encoder, "out_dim", None) or self.obs_embedding.out_dim)
        self.use_gru = bool(model.use_gru and model.d_gru > 0)
        self.use_weapon_head = bool(model.use_weapon_head)
        # look_bypass_gru is a v17-fidelity load-time flag.  v20+ always sets
        # this False — when True (only via QNNPolicy.load on a v17 checkpoint)
        # the look head is fed cat(self_readout, target_feat) instead of
        # cat(gru_flat, target_feat), matching the features it was trained on.
        self.look_bypass_gru = bool(model.look_bypass_gru and self.use_gru)
        # Weapon-selector composition is an ordered edge list (see
        # ModelConfig.weapon_sources). The "gru" source contributes only when
        # a temporal slot is active; the others always contribute their width.
        self.weapon_sources = tuple(model.weapon_sources)
        # A token:<name> source reads one encoder self-token output as the readout;
        # resolve names → self_block indices from the obs embedding's token order.
        self._self_token_index = {
            n: i for i, n in enumerate(getattr(self.obs_embedding, "self_token_names", ()))
        }
        _valid_sources = {"gru", "self_readout", "target_feat"}
        _bad = [
            s for s in self.weapon_sources
            if s not in _valid_sources and not s.startswith("token:")
        ]
        if _bad:
            raise ValueError(
                f"weapon_sources contains unknown source(s) {_bad}; "
                f"valid sources are {sorted(_valid_sources)} or 'token:<name>'"
            )
        for s in self.weapon_sources:
            if s.startswith("token:") and s[len("token:"):] not in self._self_token_index:
                raise ValueError(
                    f"weapon source {s!r} → unknown self-token; obs embedding has "
                    f"{list(self._self_token_index)}"
                )
        if not ({"gru", "self_readout"} & set(self.weapon_sources)) and not any(
            s.startswith("token:") for s in self.weapon_sources
        ):
            raise ValueError(
                "weapon head needs a readout — 'gru' / 'self_readout' or a "
                f"'token:<name>' source; got {self.weapon_sources!r} "
                "(target_feat alone is too thin)"
            )
        self.weapon_context_from_obs = bool(model.weapon_context_from_obs)
        self.d_target = int(model.d_target)
        self.d_move = int(model.d_move)
        self.d_look = int(model.d_look)
        self.d_attack = int(model.d_attack)
        self.d_weapon = int(model.d_weapon)
        self.head_activation = model.head_activation

        # Resolve slot activation FIRST so dim computation reflects what
        # downstream consumers will actually receive. Off explicitly disables;
        # None defers to the canonical config flag (use_gru / use_weapon_head);
        # an nn.Module override always activates the slot.
        self._has_temporal = (temporal is not Off) and (temporal is not None or self.use_gru)
        self._has_target_pointer = target_pointer is not Off
        self._has_weapon_head = (weapon_head is not Off) and (weapon_head is not None or self.use_weapon_head)
        self._has_move_head = move_head is not Off
        self._has_look_head = look_head is not Off
        self._has_attack_head = attack_head is not Off
        # a25 hazard head: opt-in only (no canonical fallback). Present iff
        # build_network passed a real module — default Off keeps every existing
        # Network construction (which never passes it) hazard-free.
        self._has_move_hazard_head = isinstance(move_hazard_head, nn.Module)

        # Build the upstream slots (temporal, target pointer) FIRST so the
        # downstream dim contract reads their declared out_dim rather than a
        # config scalar — nodes are the dim source of truth. The canonical
        # GRU is still *sized* from config d_gru; self.d_gru is then read back
        # from whatever node landed in the slot (canonical or override).
        if self._has_temporal:
            self.temporal = (
                Temporal(d_model=self.d_model, hidden_dim=int(model.d_gru))
                if temporal is None else temporal
            )
            self.d_gru = int(self.temporal.out_dim)
        else:
            self.d_gru = 0

        if self._has_target_pointer:
            if target_pointer is None:
                self.target_pointer = TargetPointer(
                    d_model=self.d_model,
                    d_target=self.d_target,
                )
            else:
                self.target_pointer = target_pointer

        # Effective dim contract — single authority in slot_dims, fed the
        # resolved node widths (d_model from encoder, d_gru from temporal slot).
        dims = slot_dims(
            d_model=self.d_model,
            d_gru=self.d_gru,
            has_temporal=self._has_temporal,
            has_target_pointer=self._has_target_pointer,
            has_weapon_head=self._has_weapon_head,
            weapon_sources=self.weapon_sources,
        )
        motor_in = dims["motor_in"]
        weapon_in = dims["weapon_in"]

        if self._has_weapon_head:
            if weapon_head is None:
                self.weapon_head = WeaponHead(
                    selector_dim=weapon_in,
                    d_model=self.d_model,
                    d_hidden=self.d_weapon,
                    activation=self.head_activation,
                    context_from_obs=self.weapon_context_from_obs,
                )
            else:
                self.weapon_head = weapon_head

        if self._has_move_head:
            self.move_head = (
                MoveHead(in_dim=motor_in, d_hidden=self.d_move, activation=self.head_activation)
                if move_head is None else move_head
            )
        if self._has_look_head:
            self.look_head = (
                LookHead(in_dim=motor_in, d_hidden=self.d_look, activation=self.head_activation)
                if look_head is None else look_head
            )
        if self._has_attack_head:
            self.attack_head = (
                AttackHead(
                    in_dim=motor_in,
                    d_hidden=self.d_attack,
                    activation=self.head_activation,
                )
                if attack_head is None else attack_head
            )

        if self._has_move_hazard_head:
            # No canonical fallback — always an override built by build_network.
            self.move_hazard_head = move_hazard_head

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _with_weapon_context(
        features: torch.Tensor,
        weapon_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if weapon_context is None:
            return features
        return torch.cat([features, weapon_context], dim=-1)

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        reset_ts: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_shape, flat_obs = _flatten_obs(obs)
        sample = flat_obs.get("vel")
        if sample is None:
            sample = flat_obs["self_scalars"]
        batch_flat = int(sample.shape[0])

        enc_out = self.encoder(self.obs_embedding(flat_obs))
        self_readout = enc_out.self_readout

        if self._has_temporal:
            temporal_out = self.temporal(TemporalInput(
                flat_pool=self_readout,
                hidden=hidden,
                reset_mask=reset_mask,
                seq_shape=seq_shape,
                reset_ts=reset_ts,
            ))
            gru_flat = temporal_out.flat_out
            next_hidden = temporal_out.next_hidden
        else:
            gru_flat = None
            next_hidden = torch.zeros((batch_flat, 0), dtype=sample.dtype, device=sample.device)

        # Target pointer (slot). The canonical MLP pointer scores each entity
        # token independently and pools by the resulting softmax; enemy_mask
        # restricts the softmax to actor entities that are not on the
        # player's team. When the slot is Off, target_logits / target_feat
        # become zeros and downstream consumers continue with
        # shape-preserving substitutes.
        actor_mask_flat = (flat_obs["entity_types"].long() == TOKEN_ACTOR)
        team_flat = flat_obs["entity_scalars_raw"][..., _ACTOR_TEAM_OFFSET]
        enemy_mask_flat = actor_mask_flat & (team_flat != _TEAM_TEAMMATE_VALUE)
        if self._has_target_pointer:
            tp_out = self.target_pointer(TargetPointerInput(
                entity_outs=enc_out.entity_outs,
                entity_mask=enc_out.entity_mask,
                enemy_mask=enemy_mask_flat,
                self_readout=self_readout,
            ))
            target_logits = tp_out.target_logits
            target_feat = tp_out.target_feat
        else:
            # No pointer slot → no target. target_logits stays a shape-preserving
            # zeros (entity-anchored priors / weapon target_feat source still read
            # it), but target_feat is NOT fabricated into the motor feature cat —
            # see slot_dims(has_target_pointer=...). "If target is off, it's off."
            n_entities = int(enc_out.entity_outs.shape[1])
            zero_kw = {"dtype": self_readout.dtype, "device": self_readout.device}
            target_logits = torch.zeros((batch_flat, n_entities), **zero_kw)
            target_feat = torch.zeros((batch_flat, self.d_model), **zero_kw)

        # Motor feature base = readout (+ target_feat only when a pointer exists).
        readout_flat = gru_flat if self._has_temporal else self_readout
        if self._has_target_pointer:
            features_base_flat = torch.cat([readout_flat, target_feat], dim=-1)
        else:
            features_base_flat = readout_flat

        entity_scalars_flat = flat_obs["entity_scalars_raw"]
        entity_rel_flat = entity_scalars_flat[..., _ACTOR_REL_OFFSET:_ACTOR_REL_OFFSET + 3]

        weapon_out = None
        if self._has_weapon_head:
            # Selector cat — assembled by iterating weapon_sources in declared
            # order. The "gru" source only exists when temporal is on; it's
            # dropped from the cat otherwise (matching its 0-width dim).
            _ws_available: dict[str, torch.Tensor] = {
                "self_readout": self_readout,
                "target_feat": target_feat,
            }
            if self._has_temporal:
                _ws_available["gru"] = gru_flat
            for _src in self.weapon_sources:
                if _src.startswith("token:"):
                    if enc_out.self_block is None:
                        raise ValueError(
                            f"weapon source {_src!r} needs per-token encoder outputs, "
                            "but the encoder did not emit self_block"
                        )
                    _ws_available[_src] = enc_out.self_block[
                        :, self._self_token_index[_src[len("token:"):]], :
                    ]
            weapon_selector_flat = torch.cat(
                [_ws_available[src] for src in self.weapon_sources if src in _ws_available],
                dim=-1,
            )
            weapon_out = self.weapon_head(WeaponHeadInput(
                selector=weapon_selector_flat,
                obs_weapon_id=flat_obs["self_weapon_id"] if self.weapon_context_from_obs else None,
            ))
        weapon_context = weapon_out.context if weapon_out is not None else None

        move_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
        move_out = self.move_head(MoveHeadInput(features=move_features_flat)) if self._has_move_head else None

        # a25 hazard head (opt-in): reads the SAME motor feature vector as the
        # move head plus the semi-Markov decode state (held_class/dwell_age),
        # which arrive as flat_obs fields (train: precomputed act_move columns;
        # deploy: threaded from the engine's move-decode state). Local import
        # keeps the canonical Network module free of a top-level bench dependency.
        move_hazard_out = None
        # The move-hazard (a25 WHEN-law) head is a TRAINING auxiliary: it consumes
        # move_held_class / move_dwell_age, which the BC loader derives from
        # act_move and exist only at train time. It is never a deploy head — the
        # exported/live move decode uses the tabulated semi-Markov hazard, and the
        # deploy graph is 5-head (see tools/export_onnx CORE_OUTPUT_NAMES). So run
        # it only when its derived inputs are present; absent (export / live
        # inference) it is cleanly skipped, letting a full_6head checkpoint export
        # as a 5-head deploy graph.
        if self._has_move_hazard_head and "move_held_class" in flat_obs:
            move_hazard_out = self.move_hazard_head(self.move_hazard_head.Input(
                cls_feat=move_features_flat,
                held_class=flat_obs["move_held_class"],
                dwell_age=flat_obs["move_dwell_age"],
            ))

        # Look = target-anchored prior + learned residual. v17 checkpoints
        # set look_bypass_gru=True so the look head sees the same features
        # it was trained on (cat(self_readout, target_feat)).
        if self._has_temporal and self.look_bypass_gru:
            bypass_base = (
                torch.cat([self_readout, target_feat], dim=-1)
                if self._has_target_pointer else self_readout
            )
            look_features_flat = self._with_weapon_context(bypass_base, weapon_context)
        else:
            look_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
        if self._has_look_head:
            look_out = self.look_head(LookHeadInput(
                features=look_features_flat,
                target_logits=target_logits,
                entity_rel=entity_rel_flat,
                actor_mask=actor_mask_flat,
            ))
        else:
            look_out = None
        # look_prior is consumed by prior-style attack-head variants;
        # canonical AttackHead ignores it. Substitute zeros when look is off.
        look_prior_for_attack = (
            look_out.look_prior if look_out is not None
            else torch.zeros((batch_flat, 3), dtype=self_readout.dtype, device=self_readout.device)
        )

        attack_features_flat = self._with_weapon_context(features_base_flat, weapon_context)
        if self._has_attack_head:
            attack_out = self.attack_head(AttackHeadInput(
                features=attack_features_flat,
                look_prior=look_prior_for_attack,
                weapon_id=flat_obs["self_weapon_id"],
                target_logits=target_logits,
                entity_scalars=entity_scalars_flat,
                actor_mask=actor_mask_flat,
                self_scalars=flat_obs.get("self_scalars"),
            ))
        else:
            attack_out = None

        logits_flat: Dict[str, torch.Tensor] = {}
        if move_out is not None:
            logits_flat[MOVE_HEAD] = move_out.logits
        if move_hazard_out is not None:
            # Per-axis release-hazard logits — consumed by the loss (phase 3) and
            # the move WHEN-decode; not an argmax-sampled action vector.
            logits_flat[MOVE_HAZARD_HEAD] = move_hazard_out.hazard_logits
        if look_out is not None:
            logits_flat[LOOK_HEAD] = look_out.look_predict
            # Underscored keys are loss-only; not used for inference / sampling.
            logits_flat["_look_prior"] = look_out.look_prior
            logits_flat["_look_delta"] = look_out.look_delta
            # Distributional head outputs (binned / polar) forwarded
            # generically so a new look head's LOSS can live with the head:
            # QNNPolicy dispatches to look_head.look_loss, which reads these.
            for _f in ("look_bins", "look_mag_logits", "look_dir_logits"):
                _v = getattr(look_out, _f, None)
                if _v is not None:
                    logits_flat["_" + _f] = _v
        if attack_out is not None:
            logits_flat["_attack_prior"] = attack_out.prior_logit
            logits_flat["_attack_delta"] = attack_out.delta_attack
            logits_flat[ATTACK_HEAD] = attack_out.attack_logit
        if weapon_out is not None:
            logits_flat[WEAPON_HEAD] = weapon_out.logits
            # Loss-only underscored key (not used for inference/sampling) so a
            # bench weapon head's LOSS can live entirely in bench — mirrors look.
            if getattr(weapon_out, "when_logit", None) is not None:
                logits_flat["_weapon_switch_when"] = weapon_out.when_logit

        features_flat = move_features_flat  # downstream consumers ignore the weapon-ctx dim
        values_flat = torch.zeros((batch_flat,), dtype=sample.dtype, device=sample.device)

        if seq_shape is None:
            return (
                features_flat, logits_flat, values_flat,
                next_hidden, target_logits,
            )
        features, logits, values, target_logits = _restore_outputs(
            features_flat, logits_flat, values_flat,
            target_logits, seq_shape,
        )
        return features, logits, values, next_hidden, target_logits
