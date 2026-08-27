"""Combat-objective BC network.

This is the ``nn.Module`` compute graph for the model — the encoder,
temporal (recurrence), target pointer, and the canonical a28 heads
(look, attack selector, move_seg, jump; look_seg / attack_future are
bench-probe slots). It is the "model" in the SL sense.

a28 invariant: every head consumes EXACTLY its declared graph inputs.
There is one shared feature cat — (readout | target_feat [| intent]) —
and heads prefix-slice it to their declared width. Nothing is appended
past the declared edges (weapon_ctx, the one historical implicit input,
is gone; the selector emits logits only). The attack selector composes
its OWN cat from its declared edges, optionally with the realized-
alignment tail (``aim``) — also declared, also nothing implicit.

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
from qnn.model.attack_with_head import AttackSelectorInput
from qnn.model.target import TargetPointer, TargetPointerInput
from qnn.model.temporal import Temporal, TemporalInput
from qnn.model.transformer import ObsEmbedding, TransformerEncoder
from qnn.vocab import TOKEN_ACTOR


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Flat policy-layer bridge of the graph spec's node parameters.

    a28: the graph spec (``qnn.model.graph``) is the sole author of the
    architecture; this dataclass carries the handful of scalars the
    policy layer and Network internals still read. All fields required,
    no legacy aliases, no migration — a pre-a28 model.json fails loud.
    """
    d_model: int
    n_heads: int
    n_layers: int
    d_ffn: int
    attn_dropout: float
    use_gru: bool
    d_gru: int
    # Selector input composition as an ordered edge list (from the attack
    # selector's declared graph inputs): "gru" / "self_readout" /
    # "target_feat". Empty when no selector head is present.
    weapon_sources: tuple[str, ...]
    d_target: int
    head_activation: str

    @classmethod
    def from_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Build from a model.json-style mapping. Unknown keys or missing
        required fields raise TypeError — every field is explicit."""
        data = dict(raw)
        data["weapon_sources"] = tuple(data.get("weapon_sources", ()))
        if data.get("head_activation") not in ("none", "gelu", "relu"):
            raise ValueError(
                f"head_activation must be 'none', 'gelu', or 'relu', got {data.get('head_activation')!r}"
            )
        return cls(**data)

    @classmethod
    def from_flat_dict(cls, raw: "Mapping[str, Any]") -> "ModelConfig":
        """Like ``from_dict`` but extracts the model fields from a larger
        flat config dict (e.g. a PPO config that merges train + model
        keys). Missing required model fields still raise TypeError.
        """
        keys = {f.name for f in fields(cls)}
        subset = {k: v for k, v in raw.items() if k in keys}
        return cls.from_dict(subset)

    def to_dict(self) -> "dict[str, Any]":
        return asdict(self)


# Logits-dict keys — string identifiers used to address each head's
# output across the BC/PPO/eval call sites. MOVE_HEAD survives as the
# ACTION-STREAM key only (act_move labels feed move_seg target derivation
# and the decode state); there is no per-tick move head.
MOVE_HEAD = "move"
LOOK_HEAD = "look"
ATTACK_HEAD = "attack"
ATTACK_FIRE_BIAS = "_attack_fire_bias"  # model-owned fire-only intercept (8,)
MOVE_SEG_HEAD = "move_seg"        # a25 segment (class x duration) head
LOOK_SEG_HEAD = "look_seg"        # a25 look segment head (bench slot)
JUMP_HEAD = "jump"                # a25 2-class land-jump head
ATTACK_FUTURE_HEAD = "attack_future"  # a27 MTP aux head (training-only, bench slot)
MOVE_TICK_HEAD = "move_tick"      # BENCH: revived per-tick move head (cell C3 of
                                  # agents/plans/seg-vs-frame-decision.md). Distinct
                                  # from MOVE_HEAD, which is the action-stream key.

# Output sizes, exported for callers that build padded buffers or
# downstream layers against these sizes. Heads define their own
# OUT_DIM internally; these are the public face.
MOVE_HEAD_SIZE = MOVE_AXES * MOVE_AXIS_CLASSES  # 9 move-class logits (action space)
LOOK_HEAD_SIZE = 3  # 3D direction vector
ATTACK_HEAD_SIZE = 9  # categorical no-attack + Quake impulses 1..8

# Offset of the relative-XYZ block inside an actor's per-token scalar vector.
# Mirrors qnn.bc.target_labeler._ACTOR_REL_OFFSET; duplicated here so the model
# layer doesn't import from BC.
_ACTOR_REL_OFFSET = 3
# Offset of the team scalar inside an actor's per-token scalar vector.
# Mirrors qnn.bc.target_labeler._ACTOR_TEAM_OFFSET. Used to derive enemy_mask
# for the target pointer.
_ACTOR_TEAM_OFFSET = 16
_TEAM_TEAMMATE_VALUE = 1.0

# Realized-alignment block width — the spec's AIM_DIM, restated here because the
# model layer must not import the graph layer (build.py owns the crossing and a
# unit test pins the two equal).
AIM_BLOCK_DIM = 17

# A″ forward-projected alignment tail (EDGE_AIM2) — restated from the spec's
# AIM2_HORIZONS_TICKS / AIM2_DIM for the same layering reason (a unit test
# pins the two equal; see qnn.model.graph.spec for the full rationale).
AIM2_HORIZONS_TICKS: tuple[int, ...] = (2, 5, 10, 16)
AIM2_EXTRA_DIM = 8 * len(AIM2_HORIZONS_TICKS)
AIM2_BLOCK_DIM = AIM_BLOCK_DIM + AIM2_EXTRA_DIM


@torch.no_grad()
def alignment_edge_block(
    alignment: torch.Tensor,        # (R, 8) per-weapon expected crest payout
    alignment_prev: torch.Tensor,   # (R, 8) previous tick's alignment (zeros = none)
    has_target: torch.Tensor,       # (R,)  1.0 when an enemy actor is perceived
    dtype: torch.dtype,
) -> torch.Tensor:
    """``(R, 17)`` alignment feature block (EDGE_AIM, rung-3 A′ form).

    Columns, in order:

    0..7   ``alignment[k]`` — expected crest payout of firing weapon k this
           tick (``lead_aim.weapon_alignment``: per-enemy lead-law hbw →
           ``exp(−ALIGNMENT_GAMMA·hbw)``, pooled by the detached target
           pointer's belief). The reward's own state variable, per weapon.
    8..15  ``Δalignment[k]`` — realized one-tick backward difference, zeroed
           when EITHER endpoint had no target (payouts are strictly positive
           with a target, so all-zero rows are unambiguous) — the trend that
           makes fire-at-the-trough legible: fire when alignment is high and
           Δ has stopped improving.
    16     ``has_target`` — explicit gate bit (zero-disambiguation insurance).

    Previous-tick threading: the SEQ path derives ``alignment_prev`` by
    shifting within the window (reset rows zeroed); the flat/act path reads
    caller state (``prepare_act_state``'s ``alignment_prev`` lanes, updated
    in place from the ``_alignment`` aux logit each tick).

    GRADIENT ISOLATION unchanged from the original aim edge: no_grad
    computation off detached pointer logits — the selector MLP's own
    parameters are the only ones that learn through this edge.
    """
    valid = (
        (alignment_prev.amax(dim=-1, keepdim=True) > 0)
        & (alignment.amax(dim=-1, keepdim=True) > 0)
    ).to(alignment.dtype)
    delta = (alignment - alignment_prev) * valid
    return torch.cat(
        [alignment, delta, has_target.reshape(-1, 1)], dim=-1,
    ).to(dtype)


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
    """Width one selector source contributes to the selector cat.

    ``gru`` requires an active temporal slot (the spec forbids a dangling
    gru edge, so a graph-built model never hits the 0 branch — it exists
    for direct Network construction in tests); ``self_readout`` and
    ``target_feat`` are each ``d_model`` wide.
    """
    if source == "gru":
        return int(d_gru) if has_temporal else 0
    if source in ("self_readout", "target_feat"):
        return int(d_model)
    raise ValueError(f"unknown weapon source {source!r}")


def slot_dims(
    *,
    d_model: int,
    d_gru: int,
    has_temporal: bool,
    has_target_pointer: bool,
    weapon_sources: "tuple[str, ...]" = (),
    intent_dim: int = 0,
    aim_dim: int = 0,
) -> dict[str, int]:
    """Single authority for the dim contract Network's slots are built with.

    Pure function over *resolved node widths*, not a ModelConfig — ``d_model``
    is the encoder/obs-embedding ``out_dim``, ``d_gru`` is the temporal slot's
    ``out_dim`` (pass 0 when the slot is off). ``Network.__init__`` calls this
    with its built nodes' ``out_dim`` so nodes — not config scalars — are the
    dim source of truth; bench builders that must size an override head before
    constructing Network call it with their config-derived widths.

    The shared feature cat is (readout | target_feat [| intent]) and that is
    ALL of it — heads prefix-slice to their declared width, and nothing is
    appended past the declared edges (a28 removed weapon_ctx, the one
    historical undeclared block).

    ``has_target_pointer`` controls whether the feature cat carries the
    pointer's ``target_feat`` (``d_model`` wide). When the pointer slot is Off
    there is no target — the block is dropped entirely rather than fed as a
    ``d_model``-wide zeros pad the heads have to learn to ignore.
    "If target is off, it's off."

    ``intent_dim`` > 0 splices a shared ATTACK-INTENT block onto the end of
    the feature cat (coordination program hook; 0 in the canonical graph).
    A head that declares intent must also declare target.feat when the
    pointer exists — the prefix slice can only drop the tail.

    Keys:
      base_features_dim  — pre-intent features. The intent PRODUCER reads this
                           (it cannot consume its own output).
      coord_features_dim — base + intent. What intent CONSUMERS read; equal to
                           base_features_dim in the canonical graph.
      motor_in           — alias of coord_features_dim (the full shared cat).
      weapon_in          — the attack selector's DECLARED-EDGE cat (pre-aim).
      weapon_coord_in    — weapon_in + aim_dim. What a selector declaring the
                           realized-alignment edge reads; equal to weapon_in in
                           the canonical graph.
      intent_dim         — width of the shared intent block (0 = off).
      aim_dim            — width of the selector's alignment tail (0 = off).

    ``aim_dim`` > 0 splices the realized-alignment block onto the END of the
    SELECTOR's cat only (not the shared motor cat) — see EDGE_AIM and
    :func:`aim_alignment_block`.
    """
    d_gru = int(d_gru) if has_temporal else 0
    d_model = int(d_model)
    intent_dim = max(0, int(intent_dim))
    aim_dim = max(0, int(aim_dim))
    readout_dim = d_gru if has_temporal else d_model
    target_dim = d_model if has_target_pointer else 0
    base_features_dim = readout_dim + target_dim
    coord_features_dim = base_features_dim + intent_dim
    weapon_in = sum(
        _weapon_source_dim(src, d_model=d_model, d_gru=d_gru, has_temporal=has_temporal)
        for src in weapon_sources
    )
    return {
        "d_model": d_model,
        "d_gru": d_gru,
        "base_features_dim": base_features_dim,
        "coord_features_dim": coord_features_dim,
        "intent_dim": intent_dim,
        "aim_dim": aim_dim,
        "motor_in": coord_features_dim,
        "weapon_in": weapon_in,
        "weapon_coord_in": weapon_in + aim_dim,
    }


class Network(nn.Module):
    """The compute graph: encoder + temporal + target pointer + heads.

    Built by ``QNNPolicy`` (the training-time wrapper). Forward returns the
    five-tuple ``(features, logits_dict, values, next_hidden, target_logits)``.

    Per-slot overrides
    ------------------
    ``obs_embedding`` / ``encoder`` / ``temporal`` / ``target_pointer``
    accept ``None`` (build the canonical component from ``ModelConfig``),
    an ``nn.Module`` (use as-is; size overrides with ``slot_dims``), or
    ``Off`` (slot disabled; ``obs_embedding``/``encoder`` cannot be Off).

    Head slots (``look_head``, ``attack_head``, ``move_seg_head``,
    ``look_seg_head``, ``jump_head``, ``attack_future_head``,
    ``move_tick_head``) take an
    ``nn.Module`` built by ``qnn.model.graph.build_network`` or ``Off``
    (default) — there are no canonical in-Network head fallbacks; the
    graph spec is the sole author of head construction.

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
        look_head: "nn.Module | Off" = Off,
        attack_head: "nn.Module | Off" = Off,
        move_seg_head: "nn.Module | Off" = Off,
        look_seg_head: "nn.Module | Off" = Off,
        jump_head: "nn.Module | Off" = Off,
        attack_future_head: "nn.Module | Off" = Off,
        move_tick_head: "nn.Module | Off" = Off,
        intent_source: str | None = None,
        aim_edge: bool = False,
        aim2_edge: bool = False,
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
        # Selector composition is an ordered edge list from the attack
        # selector's declared graph inputs (see ModelConfig.weapon_sources).
        self.weapon_sources = tuple(model.weapon_sources)
        _valid_sources = {"gru", "self_readout", "target_feat"}
        _bad = [s for s in self.weapon_sources if s not in _valid_sources]
        if _bad:
            raise ValueError(
                f"weapon_sources contains unknown source(s) {_bad}; valid sources are "
                f"{sorted(_valid_sources)}"
            )
        self.d_target = int(model.d_target)
        self.head_activation = model.head_activation
        # Gradient-isolated attack-intent block (spec intent node;
        # agents/plans/attack-intent-feedforward.md). "sg_softmax" detaches
        # the selector softmax; "prev_attack" one-hots the teacher-forced
        # previous-tick attack class from obs["attack_intent_prev"]. Both
        # append a 9-wide tail to the shared feature cat for the heads that
        # DECLARE the intent edge; non-declaring heads prefix-slice it off.
        if intent_source is not None and intent_source not in ("sg_softmax", "prev_attack"):
            raise ValueError(f"unknown intent_source {intent_source!r}")
        self.intent_source = intent_source
        # Realized-alignment edge into the SELECTOR (EDGE_AIM;
        # agents/plans/coordination-objective-probes.md §B-i). Appends
        # aim_alignment_block to the selector's own input cat. Gradient-isolated
        # by construction — see that function.
        self.aim_edge = bool(aim_edge)
        # A″ forward-projected extension (EDGE_AIM2; crest-ceiling-handoff.md
        # "Candidate next steps" §3). Appends AIM2_EXTRA_DIM more columns
        # after the base alignment_edge_block — see forward(). Requires
        # aim_edge (the base block) to also be on; build_network always sets
        # both together (spec.aim_edge is true whenever spec.aim2_edge is),
        # this is a defensive check for direct Network construction (tests).
        self.aim2_edge = bool(aim2_edge)
        if self.aim2_edge and not self.aim_edge:
            raise ValueError("aim2_edge requires aim_edge (it extends the base block)")

        # Resolve slot activation FIRST so dim computation reflects what
        # downstream consumers will actually receive. Off explicitly disables;
        # None defers to the canonical config flag (use_gru); an nn.Module
        # override always activates the slot. Heads are opt-in modules only —
        # build_network is the sole author of head construction.
        self._has_temporal = (temporal is not Off) and (temporal is not None or self.use_gru)
        self._has_target_pointer = target_pointer is not Off
        self._has_look_head = isinstance(look_head, nn.Module)
        self._has_attack_head = isinstance(attack_head, nn.Module)
        if self._has_attack_head and not hasattr(attack_head, "attack_loss"):
            raise ValueError(
                "attack_head must be a categorical selector module owning its "
                "attack_loss (qnn.model.attack_with_head)")
        if self.intent_source == "sg_softmax" and not self._has_attack_head:
            raise ValueError("intent_source 'sg_softmax' requires an attack selector head")
        if self.aim_edge and not self._has_attack_head:
            raise ValueError("aim_edge requires an attack selector head")
        self._has_move_seg_head = isinstance(move_seg_head, nn.Module)
        self._has_look_seg_head = isinstance(look_seg_head, nn.Module)
        self._has_jump_head = isinstance(jump_head, nn.Module)
        self._has_attack_future_head = isinstance(attack_future_head, nn.Module)
        self._has_move_tick_head = isinstance(move_tick_head, nn.Module)
        # a27 MTP aux head switch (agents/plans/mtp-attack-future-probe.md).
        # A PLAIN attribute, not a buffer: it must never enter the state dict
        # (a checkpoint would then carry an export-time value into training)
        # and it must be flippable on a loaded model. Export sets it False so
        # the aux MLP is never traced into the ONNX graph.
        self.aux_training_heads: bool = True

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

        if self._has_look_head:
            self.look_head = look_head
        if self._has_attack_head:
            self.attack_head = attack_head
        if self._has_move_seg_head:
            self.move_seg_head = move_seg_head
        if self._has_look_seg_head:
            self.look_seg_head = look_seg_head
        if self._has_jump_head:
            self.jump_head = jump_head
        if self._has_attack_future_head:
            # ASSIGNED LAST, and this ordering is load-bearing: _init_weights
            # walks self.modules() in registration order, so appending the aux
            # head at the end leaves every other module's xavier_uniform_ draw
            # bit-identical to the control arm at the same seed. Moving this
            # block up would reshuffle the RNG stream and make the two probe
            # arms incomparable. Ordering is HALF the invariant — the head's
            # builder also restores the RNG across its nn.Linear constructor
            # draw, which lands before _init_weights runs
            # (attack_future_head._build_attack_future). Both are pinned by
            # tests/model/test_attack_future_head.py.
            self.attack_future_head = attack_future_head
        if self._has_move_tick_head:
            # ASSIGNED AFTER EVERY OTHER HEAD, same reason as the aux head
            # above: _init_weights walks self.modules() in registration order,
            # so a bench head appended at the end leaves every other module's
            # xavier draw bit-identical to the control arm at the same seed.
            # Its builder also restores the RNG across its nn.Linear
            # constructor draw (move_tick_head._build_move_tick). Both halves
            # are pinned by tests/model/test_move_tick_arm.py.
            self.move_tick_head = move_tick_head

        if self.aim_edge:
            # Static weapon-trajectory table for the alignment block. A
            # NON-PERSISTENT buffer: it carries no learned state, so it must
            # never enter the state dict (a checkpoint would then pin a
            # physics table that qnn.bc.weapon_physics owns), but it must
            # follow .to(device)/.to(dtype) with the module. Registering a
            # buffer draws no RNG, so arms stay init-comparable.
            from qnn.bc.weapon_physics import build_model_weapon_scalars
            self.register_buffer(
                "_aim_weapon_physics",
                torch.from_numpy(build_model_weapon_scalars()).float(),
                persistent=False,
            )

        self._init_weights()

    @property
    def wants_prev_attack(self) -> bool:
        """True when the graph needs ``obs['attack_intent_prev']``.

        Sole remaining consumer: the ``prev_attack`` intent node. (The
        alignment edge computed per-weapon since the rung-3 A′ redesign, so
        its old prev-attack weapon key — and the one-tick staleness that came
        with it — is gone.) QNNPolicy teacher-forces the column at train/val.
        """
        return self.intent_source == "prev_attack"

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # P1 feasibility mask (agents/plans/attack-finished-masking-refactor.md):
    # owned+ammo (readiness > 0.1; the 0.1 floor marks owned-but-empty) AND the
    # refire cooldown elapsed (af == remaining-cooldown/TIME_SCALE ≈ 0 ⇒ ready).
    # Both sources are dequant OUTPUTS that survive resident compaction, so the
    # mask builds identically at train, serve, and ONNX-export time.
    _FEAS_OWNED_AMMO = 0.1 + 1e-4
    _FEAS_AF_READY = 1e-4

    @staticmethod
    def _weapon_feasibility_mask(flat_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        readiness = flat_obs["self_weapon_readiness"]        # (B*, 8) axe..LG
        af = flat_obs["self_arsenal_scalars"][..., 0:1]      # (B*, 1) remaining cooldown /TIME_SCALE
        feas8 = (readiness > Network._FEAS_OWNED_AMMO) & (af <= Network._FEAS_AF_READY)
        neg = torch.where(feas8, af.new_zeros(()), af.new_full((), -1e9))  # (B*, 8)
        # class 0 (no_attack) is always feasible → never masked.
        return torch.cat([torch.zeros_like(af), neg], dim=-1)             # (B*, 9)

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

        # The shared feature cat = readout (+ target_feat only when a pointer
        # exists). This is the WHOLE of it — every non-selector head reads a
        # prefix of exactly this tensor (a28: nothing undeclared is appended).
        readout_flat = gru_flat if self._has_temporal else self_readout
        if self._has_target_pointer:
            features_base_flat = torch.cat([readout_flat, target_feat], dim=-1)
        else:
            features_base_flat = readout_flat

        selector_out = None
        if self._has_attack_head:
            # Selector cat — assembled by iterating weapon_sources in declared
            # order. The "gru" source only exists when temporal is on; it's
            # dropped from the cat otherwise (matching its 0-width dim).
            _ws_available: dict[str, torch.Tensor] = {
                "self_readout": self_readout,
                "target_feat": target_feat,
            }
            if self._has_temporal:
                _ws_available["gru"] = gru_flat
            _selector_parts = [
                _ws_available[src] for src in self.weapon_sources if src in _ws_available
            ]
            if self.aim_edge:
                # Alignment TAIL block (rung-3 A′): per-weapon expected crest
                # payout + realized trend. Computed for ALL 8 weapons, so no
                # intent keying is needed (the old block's prev-attack key and
                # its one-tick staleness are gone).
                from qnn.model.lead_aim import weapon_alignment
                _alignment_flat, _has_t = weapon_alignment(
                    flat_obs["entity_scalars_raw"],
                    flat_obs["entity_types"],
                    # STOP-GRADIENT on the pointer (the intent-edge discipline).
                    target_logits.detach(),
                    self._aim_weapon_physics,
                )
                if seq_shape is not None:
                    # In-window shift: prev row per lane, zeroed at t=0 and on
                    # episode-start rows (reset_mask True ⇒ the previous tick
                    # belongs to a different episode).
                    _T, _B = seq_shape
                    _a_seq = _alignment_flat.reshape(_T, _B, 8)
                    _prev = torch.cat(
                        [torch.zeros_like(_a_seq[:1]), _a_seq[:-1]], dim=0)
                    if reset_mask is not None:
                        _rm = reset_mask.reshape(_T, _B, 1).to(_prev.dtype)
                        _prev = _prev * (1.0 - _rm)
                    _alignment_prev = _prev.reshape(-1, 8)
                else:
                    _ap = flat_obs.get("alignment_prev")
                    _alignment_prev = (
                        _ap.reshape(-1, 8).to(_alignment_flat.dtype)
                        if _ap is not None
                        else torch.zeros_like(_alignment_flat)
                    )
                _selector_parts.append(alignment_edge_block(
                    _alignment_flat, _alignment_prev, _has_t,
                    readout_flat.dtype,
                ))
                if self.aim2_edge:
                    # Forward-projected tail (EDGE_AIM2): per-weapon crest
                    # payout predicted at +k ticks, constant-velocity
                    # extrapolated from the SAME current-position-anchor law
                    # (qnn.model.lead_aim.weapon_alignment_projected — no
                    # second geometry). Stateless (no prev-tick carry needed,
                    # unlike the backward delta above), so no act()-time
                    # loopback field is required beyond what EDGE_AIM already
                    # threads.
                    from qnn.model.lead_aim import weapon_alignment_projected
                    _self_vel_flat = flat_obs["self_motion_scalars"][..., 0:3]
                    _projected_flat = weapon_alignment_projected(
                        flat_obs["entity_scalars_raw"],
                        flat_obs["entity_types"],
                        _self_vel_flat,
                        target_logits.detach(),
                        self._aim_weapon_physics,
                        AIM2_HORIZONS_TICKS,
                    )
                    _selector_parts.append(_projected_flat.to(readout_flat.dtype))
            weapon_selector_flat = torch.cat(_selector_parts, dim=-1)
            _feas_mask = (
                self._weapon_feasibility_mask(flat_obs)
                if getattr(self.attack_head, "wants_feasibility_mask", False)
                else None
            )
            selector_out = self.attack_head(AttackSelectorInput(
                selector=weapon_selector_flat,
                feasibility_mask=_feas_mask,
            ))

        # Gradient-isolated attack-intent tail (spec intent node). Appended
        # AFTER (readout | target_feat) so non-declaring heads prefix-slice
        # it off; both sources carry NO gradient path to the attack head.
        if self.intent_source == "sg_softmax":
            intent_flat = torch.softmax(
                selector_out.logits.detach().to(features_base_flat.dtype), dim=-1)
            features_coord_flat = torch.cat([features_base_flat, intent_flat], dim=-1)
        elif self.intent_source == "prev_attack":
            prev = flat_obs.get("attack_intent_prev")
            if prev is None:
                raise ValueError(
                    "intent_source 'prev_attack' needs obs['attack_intent_prev'] "
                    "(teacher-forced shifted act_attack at train/val; the bot's "
                    "own previous attack class at act time)")
            intent_flat = torch.nn.functional.one_hot(
                prev.reshape(-1).long().clamp(0, 8), num_classes=9,
            ).to(features_base_flat.dtype)
            features_coord_flat = torch.cat([features_base_flat, intent_flat], dim=-1)
        else:
            features_coord_flat = features_base_flat

        # Heads. Every non-selector head reads the shared feature cat and
        # prefix-slices to its declared inputs.
        look_out = self.look_head(features_coord_flat) if self._has_look_head else None

        move_seg_out = None
        if self._has_move_seg_head:
            move_seg_out = self.move_seg_head(features_coord_flat)

        look_seg_out = None
        if self._has_look_seg_head:
            look_seg_out = self.look_seg_head(features_coord_flat)

        jump_out = None
        if self._has_jump_head:
            jump_out = self.jump_head(features_coord_flat)

        move_tick_out = None
        if self._has_move_tick_head:
            move_tick_out = self.move_tick_head(features_coord_flat)

        attack_future_out = None
        if self._has_attack_future_head and self.aux_training_heads:
            # a27 MTP aux head. Skipped entirely when aux_training_heads is
            # off (export) so the aux MLP is never traced into the ONNX graph.
            attack_future_out = self.attack_future_head(features_coord_flat)

        logits_flat: Dict[str, torch.Tensor] = {}
        if move_seg_out is not None:
            logits_flat[MOVE_SEG_HEAD] = move_seg_out
        if move_tick_out is not None:
            logits_flat[MOVE_TICK_HEAD] = move_tick_out
        if look_seg_out is not None:
            logits_flat[LOOK_SEG_HEAD] = look_seg_out
        if jump_out is not None:
            logits_flat[JUMP_HEAD] = jump_out
        if attack_future_out is not None:
            logits_flat[ATTACK_FUTURE_HEAD] = attack_future_out
        if look_out is not None:
            logits_flat[LOOK_HEAD] = look_out.look_predict
            # Underscored keys are loss-only; not used for inference / sampling.
            # Distributional head outputs (polar mag/dir) forwarded generically
            # so the look head's LOSS lives with the head: QNNPolicy dispatches
            # to look_head.look_loss, which reads these.
            # look_hold_logit / look_features are the XM head's (look_head_xm):
            # its loss hook re-forwards the turn MLP with K noise draws, so the
            # head's own SLICED features ride the same generic channel.
            for _f in ("look_bins", "look_mag_logits", "look_dir_logits",
                       "look_hold_logit", "look_features"):
                _v = getattr(look_out, _f, None)
                if _v is not None:
                    logits_flat["_" + _f] = _v
        if selector_out is not None:
            logits_flat[ATTACK_HEAD] = selector_out.logits
            # Expand rather than bake this into selector_out.logits: selection
            # must remain a function of the raw nine-way head only.  Publishing
            # it in the output mapping also makes collect-policy replicas and
            # learner recomputation consume their own synchronized parameter.
            logits_flat[ATTACK_FIRE_BIAS] = self.attack_head.fire_bias.to(
                selector_out.logits.dtype,
            ).reshape(1, -1).expand(batch_flat, -1)
            if self.aim_edge:
                # Aux (underscored = never sampled): this tick's alignment
                # vector, read back by QNNPolicy.act to update the caller's
                # alignment_prev state lanes in place.
                logits_flat["_alignment"] = _alignment_flat

        features_flat = features_coord_flat
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
