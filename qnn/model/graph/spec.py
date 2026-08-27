"""GraphSpec — the declarative model-assembly config ("model graph").

One JSON-shaped document describes the whole model: tokens as dicts of
scalar/vocab field names, encoder/temporal/pointer nodes with their
parameters, and heads with parameters plus named input edges. Every
pipeline (BC, bench probes, eval, PPO, ONNX export) assembles the model
from this one spec through :func:`qnn.model.graph.build.build_network`.

Design doc: ``src/docs/model-graph.md``. Field names resolve against the
catalogs in :mod:`qnn.model.tokens.obs_fields`; nothing here holds
tensors or modules — the spec is pure data with a strict JSON
round-trip (unknown keys raise, missing required keys raise).

a28 scope notes:

* The canonical heads are ``look`` (polar), ``attack`` (attack_with
  selector), ``move_seg``, ``jump``; ``look_seg`` / ``attack_future`` /
  ``move_tick`` are bench-probe slots. The native PPO trainer owns its
  critic; it is not a node.
* ``move_tick`` is the deliberately-revived per-tick move head — the
  cell-C3 bench arm of agents/plans/seg-vs-frame-decision.md, NOT the
  pre-a28 canonical ``move`` head coming back. The old slot NAME stays
  retired precisely so no pre-a28 graph document can partially validate
  on this line.
* Every input edge is EXPLICIT: heads declare ``gru`` or
  ``self_readout`` (+ ``target.feat``). There is no polymorphic
  ``readout`` edge and no implicit input — what a head consumes is
  exactly its declared ``inputs`` list (the a28 rule; weapon_ctx was
  the one implicit edge and it is gone).
* Pre-a28 graphs (``move``/``weapon``/``move_hazard`` heads, ``readout``
  / ``token.*`` / ``scalar.*`` edges) do NOT load on this line — eval
  and export them from their own branches.
* All heads share one activation (the ``ModelConfig`` bridge needs a
  single ``head_activation``); mixed activations raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from qnn.model.tokens.obs_fields import SCALAR_FIELDS, VOCAB_FIELDS
from qnn.vocab import (
    ENTITY_STREAMS as _ENTITY_STREAMS,
    ENTITY_STREAM_COMBAT as _ENTITY_STREAM_COMBAT,
)
from qnn.schema import (
    PROBE_SPATIAL_SOURCES as _PROBE_SOURCES,
    SPATIAL_SOURCE_EGO,
    SPATIAL_SOURCES as _SPATIAL_SOURCES,
    SPATIAL_TOKEN_COUNT,
)

GRAPH_VERSION = 1

# Reserved token kinds the builder constructs outside TokenBuilder.
TOKEN_KIND_FIELDS = "fields"
TOKEN_KIND_CLS = "cls"
TOKEN_KIND_SPATIAL = "spatial"

TOKEN_KIND_ENTITIES = "entities"
_TOKEN_KINDS = (TOKEN_KIND_FIELDS, TOKEN_KIND_CLS, TOKEN_KIND_SPATIAL, TOKEN_KIND_ENTITIES)
_SELF_KINDS = (TOKEN_KIND_FIELDS, TOKEN_KIND_CLS)

# Edge names — the COMPLETE edge vocabulary. A head's ``inputs`` list is
# the exhaustive statement of what it consumes; the build wires exactly
# these, nothing more (no implicit context, no silent fallback).
EDGE_SELF_READOUT = "self_readout"  # encoder CLS readout
EDGE_GRU = "gru"                    # temporal hidden (requires a temporal node)
EDGE_TARGET_FEAT = "target.feat"    # pointer-blended target feature
EDGE_INTENT = "intent"              # gradient-isolated attack-intent block
                                    # (requires an intent node; consumers only)
EDGE_AIM = "aim"                    # realized-alignment block off the pointer
                                    # (requires a target pointer; selector only)
EDGE_AIM2 = "aim2"                  # aim + forward-projected alignment tail
                                    # (requires a target pointer; selector only;
                                    # mutually exclusive with EDGE_AIM — see AIM2_DIM)
_READOUT_EDGES = (EDGE_GRU, EDGE_SELF_READOUT)
_TAIL_EDGES = (EDGE_AIM, EDGE_AIM2)

# Intent-node sources (agents/plans/attack-intent-feedforward.md). Both are
# gradient-isolated from the attack head by construction:
#   sg_softmax  — stop_grad(softmax(attack logits)), same-tick expectation
#   prev_attack — one-hot of the PREVIOUS tick's actual attack class
#                 (teacher-forced from act_attack at train/val)
INTENT_SOURCES = ("sg_softmax", "prev_attack")
INTENT_DIM = 9   # 9-way attack class block, both sources

# Alignment block (EDGE_AIM; fire-at-alignment rung 3 arm A′ — supersedes the
# §B-i 3-column form). 17 per-frame scalars computed in the network forward
# from obs + STOP-GRADIENT pointer logits, appended to the attack selector's
# own input cat:
#   0..7   alignment[k]   expected crest payout of firing weapon k this tick
#                         (lead-law hbw → exp(−ALIGNMENT_GAMMA·hbw), pooled by
#                         the pointer's belief over enemy actors)
#   8..15  Δalignment[k]  realized one-tick backward difference (zeroed at
#                         episode starts and when either endpoint lacks a
#                         target)
#   16     has_target     1.0 when an enemy actor is perceived, else 0.0
# Rows with no enemy are zero-filled with has_target = 0 so "aligned" and
# "nothing there" are distinguishable. See qnn.model.network.alignment_edge_block
# and qnn.model.lead_aim.weapon_alignment.
AIM_DIM = 17

# A″ forward-projected alignment tail (EDGE_AIM2; crest-ceiling-handoff.md
# "Candidate next steps" §3). Extends the EDGE_AIM block (columns 0..16
# unchanged, same layout) with per-weapon crest payouts PREDICTED at +k
# ticks for k in AIM2_HORIZONS_TICKS, constant-velocity extrapolated from the
# SAME current-position-anchor hbw law (qnn.model.lead_aim
# weapon_alignment_projected — no second geometry). Horizons span the
# confirmed SG refire window (10 ticks, crest-ceiling-handoff.md "The
# ceiling is feature-bound") through the classic RL/GL 0.8s cooldown
# (16 ticks @ 20 Hz), bracketed by a short near-term tick (2) and a
# half-window midpoint (5):
#   17..24  alignment[k] at +2 ticks  (0.10s — near-term / fast-cadence families)
#   25..32  alignment[k] at +5 ticks  (0.25s — half SG window)
#   33..40  alignment[k] at +10 ticks (0.50s — full SG refire, confirmed)
#   41..48  alignment[k] at +16 ticks (0.80s — full RL/GL refire)
# No backward deltas on the projected columns — the forward payout already
# supersedes the trend proxy the k=0 deltas exist for.
AIM2_HORIZONS_TICKS: tuple[int, ...] = (2, 5, 10, 16)
AIM2_EXTRA_DIM = 8 * len(AIM2_HORIZONS_TICKS)
AIM2_DIM = AIM_DIM + AIM2_EXTRA_DIM

# Edge name ↔ flat ``ModelConfig.weapon_sources`` name. The graph spells
# the pointer edge ``"target.feat"``; the flat config spells it
# ``"target_feat"`` — this table owns the rename at every crossing.
WEAPON_EDGE_TO_SOURCE = {
    EDGE_GRU: "gru",
    EDGE_SELF_READOUT: "self_readout",
    EDGE_TARGET_FEAT: "target_feat",
}
WEAPON_SOURCE_TO_EDGE = {v: k for k, v in WEAPON_EDGE_TO_SOURCE.items()}
_WEAPON_EDGES = tuple(WEAPON_EDGE_TO_SOURCE)

# ``attack_future`` (a27 MTP aux) reads the same base feature vector as the
# other non-selector heads; it stays outside _MOTOR_HEADS only so the
# shared-inputs rule doesn't force it into lock-step with look when a bench
# arm varies one of them.
_HEAD_NAMES = ("look", "attack", "move_seg", "jump", "look_seg", "attack_future",
               "move_tick")
# Head type carrying the XM best-of-K knobs (qnn.model.look_head_xm). Named here
# because the spec layer gates its optional keys by TYPE, not by slot name.
HEAD_TYPE_XM_TANGENT = "xm_tangent"
_MOTOR_HEADS = ("look",)
_ACTIVATIONS = ("none", "gelu", "relu")

class GraphSpecError(ValueError):
    """Raised for any structurally invalid graph spec."""


def _require(raw: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in raw:
        raise GraphSpecError(f"{ctx}: missing required key {key!r}")
    return raw[key]


def _reject_unknown(raw: Mapping[str, Any], allowed: tuple[str, ...], ctx: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise GraphSpecError(f"{ctx}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _str_tuple(value: Any, ctx: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise GraphSpecError(f"{ctx}: expected a list of strings, got {value!r}")
    return tuple(value)


# ── Nodes ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TokenSpec:
    """One token of the obs embedding.

    ``fields`` tokens are built declaratively by ``TokenBuilder``; the
    reserved kinds (``cls``, ``spatial``, ``entities``) are constructed
    by the embedding outside the field catalog. Field order inside a
    token is fixed by the spec layout (scalars → kind_tag → vocab →
    readiness → ammo_pools → vocab_sum) so the parameter layout is
    deterministic.
    """
    name: str
    kind: str = TOKEN_KIND_FIELDS
    scalars: tuple[str, ...] = ()
    vocab: tuple[str, ...] = ()
    vocab_sum: tuple[str, ...] = ()
    readiness: bool = False
    ammo_pools: bool = False
    kind_tag: bool = False
    # Spatial-kind only: where the band panoramas come from. "ego" is the
    # per-tick wire atlas; "probe_grid" fuses the k nearest precomputed
    # map probes (loader-supplied probe_atlas/probe_offsets fields);
    # "pooled9" reduces the ego atlas to the v1 9-sector depth summary
    # (capacity-class bench arm — qnn.model.spatial_pool).
    source: str = SPATIAL_SOURCE_EGO
    k: int = 0
    # Probe-source only: which of the 11 atlas elevation bands the probe
    # tokens carry (default () = all 11). Used to prune bands a cheaper
    # source now owns — e.g. probe_grid_nf drops the steep floor bands the
    # ego ring supplies, shrinking the spatial sequence. Band indices index
    # SPATIAL_FIELDS order (0=-75° … 10=+75°).
    probe_bands: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "TokenSpec":
        ctx = f"tokens[{name!r}]"
        _reject_unknown(
            raw,
            ("kind", "scalars", "vocab", "vocab_sum", "readiness", "ammo_pools",
             "kind_tag", "source", "k", "probe_bands"),
            ctx,
        )
        kind = str(raw.get("kind", TOKEN_KIND_FIELDS))
        if kind not in _TOKEN_KINDS:
            raise GraphSpecError(f"{ctx}: unknown kind {kind!r}; allowed: {list(_TOKEN_KINDS)}")
        spec = cls(
            name=name,
            kind=kind,
            scalars=_str_tuple(raw.get("scalars", ()), ctx),
            vocab=_str_tuple(raw.get("vocab", ()), ctx),
            vocab_sum=_str_tuple(raw.get("vocab_sum", ()), ctx),
            readiness=bool(raw.get("readiness", False)),
            ammo_pools=bool(raw.get("ammo_pools", False)),
            kind_tag=bool(raw.get("kind_tag", False)),
            source=str(raw.get("source", SPATIAL_SOURCE_EGO)),
            k=int(raw.get("k", 0)),
            probe_bands=tuple(int(b) for b in raw.get("probe_bands", ()) or ()),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        ctx = f"tokens[{self.name!r}]"
        if self.kind != TOKEN_KIND_SPATIAL and (
                self.source != SPATIAL_SOURCE_EGO or self.k != 0):
            raise GraphSpecError(f"{ctx}: source/k are spatial-token knobs")
        if self.kind == TOKEN_KIND_SPATIAL:
            if self.source not in _SPATIAL_SOURCES:
                raise GraphSpecError(
                    f"{ctx}: unknown source {self.source!r}; "
                    f"allowed: {list(_SPATIAL_SOURCES)}"
                )
            if self.source in _PROBE_SOURCES and not 1 <= self.k <= 8:
                raise GraphSpecError(f"{ctx}: {self.source} needs k in 1..8, got {self.k}")
            if self.source not in _PROBE_SOURCES and self.k != 0:
                raise GraphSpecError(f"{ctx}: k is a probe_grid knob")
            if self.probe_bands:
                if self.source not in _PROBE_SOURCES:
                    raise GraphSpecError(f"{ctx}: probe_bands is a probe-source knob")
                if (len(set(self.probe_bands)) != len(self.probe_bands)
                        or not all(0 <= b < SPATIAL_TOKEN_COUNT for b in self.probe_bands)):
                    raise GraphSpecError(
                        f"{ctx}: probe_bands must be distinct indices in "
                        f"0..{SPATIAL_TOKEN_COUNT - 1}, got {self.probe_bands}"
                    )
        if self.kind != TOKEN_KIND_FIELDS:
            if (self.scalars or self.vocab or self.vocab_sum
                    or self.readiness or self.ammo_pools or self.kind_tag):
                raise GraphSpecError(f"{ctx}: kind={self.kind!r} tokens carry no fields")
            return
        if not (self.scalars or self.vocab or self.vocab_sum
                or self.readiness or self.ammo_pools or self.kind_tag):
            raise GraphSpecError(f"{ctx}: fields token declares no fields")
        for n in self.scalars:
            if n not in SCALAR_FIELDS:
                raise GraphSpecError(f"{ctx}: unknown scalar field {n!r}")
        for n in self.vocab:
            if n not in VOCAB_FIELDS or VOCAB_FIELDS[n].reduce != "none":
                raise GraphSpecError(f"{ctx}: {n!r} is not a single-id vocab field")
        for n in self.vocab_sum:
            if n not in VOCAB_FIELDS or VOCAB_FIELDS[n].reduce != "sum":
                raise GraphSpecError(f"{ctx}: {n!r} is not a summed vocab field")

    def to_dict(self) -> dict[str, Any]:
        if self.kind != TOKEN_KIND_FIELDS:
            out = {"kind": self.kind}
            if self.source != SPATIAL_SOURCE_EGO:
                out["source"] = self.source
                out["k"] = self.k
                if self.probe_bands:
                    out["probe_bands"] = list(self.probe_bands)
            return out
        out: dict[str, Any] = {}
        if self.scalars:
            out["scalars"] = list(self.scalars)
        if self.vocab:
            out["vocab"] = list(self.vocab)
        if self.vocab_sum:
            out["vocab_sum"] = list(self.vocab_sum)
        if self.readiness:
            out["readiness"] = True
        if self.ammo_pools:
            out["ammo_pools"] = True
        if self.kind_tag:
            out["kind_tag"] = True
        return out


def monolithic_self_token() -> TokenSpec:
    """The pre-split monolithic self token, in TokenSpec form.

    Kept as the single source for the legacy flat layout's field order
    (asserted against the token catalog by a unit test). The field
    tuple it induces via ``qnn.model.graph.embedding.token_fields`` is
    asserted equal to ``qnn.model.tokens.obs_fields.canonical_self_fields``
    by a unit test (tokens must not import graph, so the two definitions
    cannot share code directly — drift fails the test instead).
    """
    return TokenSpec(
        name="self",
        scalars=("self_scalars",),
        vocab=("armor_type", "movement_id"),
        vocab_sum=("powerup_all",),
        kind_tag=True,
    )


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    type: str  # "transformer" | "passthrough"
    d_model: int
    n_heads: int = 0
    n_layers: int = 0
    d_ffn: int = 0
    attn_dropout: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EncoderSpec":
        ctx = "encoder"
        _reject_unknown(raw, ("type", "d_model", "n_heads", "n_layers", "d_ffn", "attn_dropout"), ctx)
        spec = cls(
            type=str(_require(raw, "type", ctx)),
            d_model=int(_require(raw, "d_model", ctx)),
            n_heads=int(raw.get("n_heads", 0)),
            n_layers=int(raw.get("n_layers", 0)),
            d_ffn=int(raw.get("d_ffn", 0)),
            attn_dropout=float(raw.get("attn_dropout", 0.0)),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        if self.type not in ("transformer", "passthrough"):
            raise GraphSpecError(f"encoder: unknown type {self.type!r}")
        if self.d_model <= 0:
            raise GraphSpecError("encoder: d_model must be positive")
        if self.type == "transformer" and not (
            self.n_heads > 0 and self.n_layers > 0 and self.d_ffn > 0
        ):
            raise GraphSpecError("encoder: transformer needs n_heads/n_layers/d_ffn > 0")
        if self.type == "passthrough" and (self.n_heads or self.n_layers or self.d_ffn):
            raise GraphSpecError("encoder: passthrough carries no attention params")

    def to_dict(self) -> dict[str, Any]:
        if self.type == "passthrough":
            return {"type": self.type, "d_model": self.d_model}
        return {
            "type": self.type, "d_model": self.d_model, "n_heads": self.n_heads,
            "n_layers": self.n_layers, "d_ffn": self.d_ffn, "attn_dropout": self.attn_dropout,
        }


@dataclass(frozen=True, slots=True)
class TemporalSpec:
    type: str  # "gru"
    d_gru: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TemporalSpec":
        ctx = "temporal"
        _reject_unknown(raw, ("type", "d_gru"), ctx)
        spec = cls(type=str(_require(raw, "type", ctx)), d_gru=int(_require(raw, "d_gru", ctx)))
        if spec.type != "gru":
            raise GraphSpecError(f"{ctx}: unknown type {spec.type!r}")
        if spec.d_gru <= 0:
            raise GraphSpecError(f"{ctx}: d_gru must be positive")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "d_gru": self.d_gru}


@dataclass(frozen=True, slots=True)
class PointerSpec:
    name: str
    type: str  # "mlp" | "gt"
    d_target: int = 0

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "PointerSpec":
        ctx = f"pointers[{name!r}]"
        _reject_unknown(raw, ("type", "d_target"), ctx)
        spec = cls(name=name, type=str(_require(raw, "type", ctx)), d_target=int(raw.get("d_target", 0)))
        if spec.type not in ("mlp", "gt"):
            raise GraphSpecError(f"{ctx}: unknown type {spec.type!r}")
        if spec.type == "mlp" and spec.d_target <= 0:
            raise GraphSpecError(f"{ctx}: mlp pointer needs d_target > 0")
        if spec.type == "gt" and spec.d_target:
            raise GraphSpecError(f"{ctx}: gt pointer carries no d_target")
        if name != "target":
            raise GraphSpecError(f"{ctx}: unknown pointer (v1 supports 'target')")
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.type == "mlp":
            out["d_target"] = self.d_target
        return out


@dataclass(frozen=True, slots=True)
class IntentSpec:
    """Gradient-isolated attack-intent block (see EDGE_INTENT)."""
    source: str  # one of INTENT_SOURCES

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "IntentSpec":
        ctx = "intent"
        _reject_unknown(raw, ("source",), ctx)
        spec = cls(source=str(_require(raw, "source", ctx)))
        if spec.source not in INTENT_SOURCES:
            raise GraphSpecError(
                f"{ctx}: unknown source {spec.source!r}; allowed: {list(INTENT_SOURCES)}")
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source}


@dataclass(frozen=True, slots=True)
class HeadNodeSpec:
    """One output head: a registered type, parameters, and input edges."""
    name: str
    type: str
    inputs: tuple[str, ...]
    d_hidden: int = 0
    activation: str = "gelu"
    # categorical attack selector — feasibility masking + rare-positive shaping
    # (agents/plans/attack-finished-masking-refactor.md). All default-off ⇒
    # graphs without these keys build byte-identically to before.
    feasibility_mask: bool = False
    focal_gamma: float = 0.0
    pos_weight: float = 0.0
    water_ud: bool = False
    # look_seg's corpus tick rate (qnn.model.look_seg_bins.bins_for_hz
    # selector). 0 = unspecified — every probe.json predating this field, whose
    # checkpoints resolve to LEGACY_HZ (10) at build time
    # (look_seg_bins.resolve_hz), never guessed from the corpus. Not
    # shape-bearing (JOINT is fixed across fits) so it does not gate any other
    # head's tensor widths.
    hz: int = 0
    # XM look head only (qnn.model.look_head_xm): best-of-K candidate count and
    # noise-latent width. 0 = unspecified (the ``hz`` precedent) → the head's own
    # defaults. d_noise is SHAPE-BEARING (it widens the turn MLP's input), so
    # both round-trip through to_dict when set.
    k_explore: int = 0
    d_noise: int = 0
    # attack_with selector only — DART-style train-time channel dropout on the
    # head's own CONTEXT sub-slice (every declared edge except EDGE_AIM), so the
    # truthful alignment channel is the best predictor left under teacher
    # forcing (agents/plans/coordination-objective-probes.md §B-i, arm b1_dart).
    # 0.0 = off ⇒ byte-identical to a graph without the key. Not shape-bearing,
    # but it selects the training objective (the k_explore precedent), so it
    # round-trips through to_dict.
    dart_p: float = 0.0
    # Fire-at-alignment objective (agents/plans/fire-at-alignment-objective.md,
    # rung 1): attack_with selector only. Marginal-preserving CE reweight of
    # POSITIVE (fire) frames by the corpus's align_hbw sidecar
    # (qnn.bc.cache_align_hbw) — aligned fires get more CE weight, wild fires
    # less, total positive mass held constant so the reweight shifts WHICH
    # frames the head fires on, never HOW OFTEN (a28 already under-fires
    # 45-52%). 0.0 = off ⇒ byte-identical to a graph without the key (the
    # dart_p precedent). Requires the corpus's align_hbw action array when
    # set (qnn.bc.container._required_actions_for_config fails loud at
    # startup if missing).
    align_weight_gamma: float = 0.0
    # Action-label contract this head's target is derived from
    # (qnn.model.action_labels).  Empty = not a labelled selector.
    #
    # This exists because the 9-class attack-with selector is ONE module
    # that has served three different label semantics across generations,
    # and the binding used to be implied by the SLOT NAME: the `weapon`
    # slot read act_weapon, the `attack` slot read act_attack, and nothing
    # checked that the chosen column carried data.  A27 retired act_weapon,
    # so an a27 probe naming the `weapon` slot trained the selector on an
    # all-zero column — every positive masked to ignore_index, no error.
    # Declaring the contract makes the binding reviewable in the config and
    # checkable at build time.
    label: str = ""
    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "HeadNodeSpec":
        ctx = f"heads[{name!r}]"
        allowed = ("type", "inputs", "d_hidden", "activation")
        if name == "attack" and raw.get("type") == "attack_with":
            allowed += ("feasibility_mask", "focal_gamma", "pos_weight", "label",
                        "dart_p", "align_weight_gamma")
        if name == "jump":
            allowed += ("pos_weight",)
        if name == "attack_future":
            # Escape hatch only (see attack_future_head): scales the event-class
            # training terms, never the suffstats. The horizon/bucket grid is
            # pinned in qnn.model.attack_future_bins, not spec'd here.
            allowed += ("pos_weight",)
        if name == "move_seg":
            allowed += ("water_ud",)
        if name == "move_tick":
            # ud POS-class reweighting for the ~4% positive rate (the jump
            # head's knob, head-owned instead of the retired policy-level
            # jump_pos_weight). See qnn.model.move_tick_head.
            allowed += ("pos_weight",)
        if name == "look_seg":
            allowed += ("hz",)
        if raw.get("type") == HEAD_TYPE_XM_TANGENT:
            allowed += ("k_explore", "d_noise")
        _reject_unknown(raw, allowed, ctx)
        spec = cls(
            name=name,
            type=str(_require(raw, "type", ctx)),
            inputs=_str_tuple(_require(raw, "inputs", ctx), ctx),
            d_hidden=int(raw.get("d_hidden", 0)),
            activation=str(raw.get("activation", "gelu")),
            feasibility_mask=bool(raw.get("feasibility_mask", False)),
            focal_gamma=float(raw.get("focal_gamma", 0.0)),
            pos_weight=float(raw.get("pos_weight", 0.0)),
            water_ud=bool(raw.get("water_ud", False)),
            hz=int(raw.get("hz", 0)),
            k_explore=int(raw.get("k_explore", 0)),
            d_noise=int(raw.get("d_noise", 0)),
            dart_p=float(raw.get("dart_p", 0.0)),
            align_weight_gamma=float(raw.get("align_weight_gamma", 0.0)),
            label=str(raw.get("label", "")),
        )
        spec._validate()
        return spec

    @property
    def resolved_label(self) -> str:
        """The action-label contract this head trains against.

        a28: always the explicitly declared ``label`` — the legacy
        (slot, type) implication table is gone with the pre-a28 loaders.
        Selector heads REQUIRE it (validated below).
        """
        return self.label

    def _validate(self) -> None:
        ctx = f"heads[{self.name!r}]"
        if self.name not in _HEAD_NAMES:
            raise GraphSpecError(f"{ctx}: unknown head; allowed: {list(_HEAD_NAMES)}")
        if self.hz < 0:
            raise GraphSpecError(f"{ctx}: hz must be >= 0 (0 = unspecified), got {self.hz}")
        if self.type != HEAD_TYPE_XM_TANGENT and (self.k_explore or self.d_noise):
            raise GraphSpecError(
                f"{ctx}: k_explore/d_noise are {HEAD_TYPE_XM_TANGENT!r} knobs")
        if self.k_explore < 0 or self.d_noise < 0:
            raise GraphSpecError(
                f"{ctx}: k_explore/d_noise must be >= 0 (0 = unspecified), "
                f"got {self.k_explore}/{self.d_noise}")
        is_selector = self.name == "attack" and self.type == "attack_with"
        if self.dart_p and not is_selector:
            raise GraphSpecError(f"{ctx}: dart_p is an 'attack_with' selector knob")
        if not 0.0 <= self.dart_p < 1.0:
            raise GraphSpecError(
                f"{ctx}: dart_p must be in [0, 1) (0 = off), got {self.dart_p}")
        if self.align_weight_gamma and not is_selector:
            raise GraphSpecError(
                f"{ctx}: align_weight_gamma is an 'attack_with' selector knob")
        if self.align_weight_gamma < 0.0:
            raise GraphSpecError(
                f"{ctx}: align_weight_gamma must be >= 0 (0 = off), got "
                f"{self.align_weight_gamma}")
        if is_selector and not self.label:
            # a28: the legacy (slot, type)→label implication table is gone;
            # a selector's action-label contract must be declared.
            raise GraphSpecError(f"{ctx}: selector heads must declare 'label'")
        if self.label:
            # Fail loud on an unknown contract: the alternative is a head
            # that silently trains against whatever column its slot name
            # happens to imply, which is the failure this field exists to
            # prevent.
            from qnn.model import action_labels
            try:
                contract = action_labels.contract(self.label)
            except KeyError as exc:
                raise GraphSpecError(f"{ctx}: {exc.args[0]}") from None
            # The contract's kind must agree with the head's structural role.
            if contract.selector != is_selector:
                want = "a selector" if is_selector else "a non-selector"
                raise GraphSpecError(
                    f"{ctx}: head is {'a' if is_selector else 'not a'} "
                    f"categorical selector but label {self.label!r} is "
                    f"{'a selector' if contract.selector else 'not a selector'} "
                    f"contract ({contract.label}); expected {want} contract")
        if self.activation not in _ACTIVATIONS:
            raise GraphSpecError(f"{ctx}: activation must be one of {list(_ACTIVATIONS)}")
        if not self.inputs:
            raise GraphSpecError(f"{ctx}: inputs must be non-empty")
        if len(set(self.inputs)) != len(self.inputs):
            raise GraphSpecError(f"{ctx}: duplicate input edge")
        # One edge vocabulary for every head: an explicit readout
        # (gru | self_readout), optionally target.feat, optionally intent
        # (non-selector heads only — the selector is the intent PRODUCER) or
        # aim (SELECTOR ONLY for now — the alignment block is appended to the
        # selector's own cat, not to the shared motor cat).
        # Declared order is positional (readout first) because Network feeds
        # heads one shared feature cat in that layout.
        allowed = (_WEAPON_EDGES + _TAIL_EDGES if is_selector
                   else _WEAPON_EDGES + (EDGE_INTENT,))
        bad = [e for e in self.inputs if e not in allowed]
        if bad:
            raise GraphSpecError(
                f"{ctx}: unknown edge(s) {bad}; allowed: {list(allowed)}")
        _tail = [e for e in self.inputs if e in _TAIL_EDGES]
        if len(_tail) > 1:
            raise GraphSpecError(
                f"{ctx}: at most one of {list(_TAIL_EDGES)} may be declared "
                f"(aim2 already extends aim); got {list(self.inputs)}")
        if _tail and self.inputs[-1] != _tail[0]:
            raise GraphSpecError(
                f"{ctx}: {_tail[0]!r} must be declared last (tail block); "
                f"got {list(self.inputs)}")
        readouts = [e for e in self.inputs if e in _READOUT_EDGES]
        if len(readouts) != 1:
            raise GraphSpecError(
                f"{ctx}: exactly one readout edge required ({EDGE_GRU!r} or "
                f"{EDGE_SELF_READOUT!r}); got {list(self.inputs)}")
        if self.inputs[0] not in _READOUT_EDGES:
            raise GraphSpecError(
                f"{ctx}: the readout edge must be declared first; got {list(self.inputs)}")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type, "inputs": list(self.inputs),
            "d_hidden": self.d_hidden, "activation": self.activation,
        }
        if self.name == "attack" and self.type == "attack_with":
            if self.feasibility_mask:
                out["feasibility_mask"] = self.feasibility_mask
            if self.focal_gamma:
                out["focal_gamma"] = self.focal_gamma
            if self.pos_weight:
                out["pos_weight"] = self.pos_weight
            if self.label:
                out["label"] = self.label
            if self.dart_p:
                out["dart_p"] = self.dart_p
            if self.align_weight_gamma:
                out["align_weight_gamma"] = self.align_weight_gamma
        if self.name in ("jump", "attack_future", "move_tick") and self.pos_weight:
            out["pos_weight"] = self.pos_weight
        if self.name == "move_seg" and self.water_ud:
            # Shape-bearing: a water_ud head is 3×JOINT wide. Dropping this
            # on serialization made checkpoints unloadable (p3/p3b reload
            # crash — 90-wide checkpoint vs 60-wide rebuilt head).
            out["water_ud"] = self.water_ud
        if self.name == "look_seg" and self.hz:
            out["hz"] = self.hz
        if self.type == HEAD_TYPE_XM_TANGENT:
            # d_noise is shape-bearing (turn-MLP in-width) and k_explore selects
            # the training objective; a checkpoint rebuilt without them would
            # not match its own state dict (the move_seg.water_ud lesson).
            if self.k_explore:
                out["k_explore"] = self.k_explore
            if self.d_noise:
                out["d_noise"] = self.d_noise
        return out


# ── The graph ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GraphSpec:
    tokens: tuple[TokenSpec, ...]
    encoder: EncoderSpec
    temporal: TemporalSpec | None
    pointers: tuple[PointerSpec, ...]
    heads: tuple[HeadNodeSpec, ...]
    intent: IntentSpec | None = None
    graph_version: int = GRAPH_VERSION
    # Entity-stream selector (qnn.vocab.ENTITY_STREAMS): "combat" (default,
    # the A27 actor/projectile stream) or "full" (the a26-line stream —
    # recency dims, live item/mover tokens, 4-way modality vocab). Optional
    # spec key; a26 checkpoints predate it, so QNNPolicy.load sniffs the
    # state dict (proj_projectile in-dim) and upgrades the default.
    entity_stream: str = _ENTITY_STREAM_COMBAT

    # -- accessors ----------------------------------------------------

    @property
    def self_tokens(self) -> tuple[TokenSpec, ...]:
        return tuple(t for t in self.tokens if t.kind in _SELF_KINDS)

    @property
    def has_spatial(self) -> bool:
        return any(t.kind == TOKEN_KIND_SPATIAL for t in self.tokens)

    @property
    def pointer(self) -> PointerSpec | None:
        """The target pointer, if declared (v1: 'target' is the only pointer)."""
        for ptr in self.pointers:
            if ptr.name == "target":
                return ptr
        return None

    def head(self, name: str) -> HeadNodeSpec | None:
        for h in self.heads:
            if h.name == name:
                return h
        return None

    @property
    def aim_edge(self) -> bool:
        """True when the attack selector declares the realized-alignment edge
        (``aim``) OR its forward-projected extension (``aim2`` — both need
        the base 17-wide block; see ``aim2_edge`` for the extra tail)."""
        selector = self.head("attack")
        return selector is not None and (
            EDGE_AIM in selector.inputs or EDGE_AIM2 in selector.inputs)

    @property
    def aim2_edge(self) -> bool:
        """True when the attack selector declares the forward-projected
        alignment edge (``aim2``)."""
        selector = self.head("attack")
        return selector is not None and EDGE_AIM2 in selector.inputs

    # -- construction -------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphSpec":
        ctx = "graph"
        _reject_unknown(raw, ("graph_version", "tokens", "encoder", "temporal",
                              "pointers", "heads", "intent", "entity_stream"), ctx)
        version = int(_require(raw, "graph_version", ctx))
        if version != GRAPH_VERSION:
            raise GraphSpecError(f"{ctx}: unsupported graph_version {version}")
        tokens_raw = _require(raw, "tokens", ctx)
        if not isinstance(tokens_raw, Mapping) or not tokens_raw:
            raise GraphSpecError(f"{ctx}: tokens must be a non-empty object")
        tokens = tuple(TokenSpec.from_dict(n, t) for n, t in tokens_raw.items())
        temporal_raw = raw.get("temporal")
        pointers_raw = raw.get("pointers", {})
        if not isinstance(pointers_raw, Mapping):
            raise GraphSpecError(f"{ctx}: pointers must be an object")
        heads_raw = _require(raw, "heads", ctx)
        if not isinstance(heads_raw, Mapping) or not heads_raw:
            raise GraphSpecError(f"{ctx}: heads must be a non-empty object")
        intent_raw = raw.get("intent")
        spec = cls(
            tokens=tokens,
            encoder=EncoderSpec.from_dict(_require(raw, "encoder", ctx)),
            temporal=TemporalSpec.from_dict(temporal_raw) if temporal_raw is not None else None,
            pointers=tuple(PointerSpec.from_dict(str(n), p) for n, p in pointers_raw.items()),
            heads=tuple(HeadNodeSpec.from_dict(str(n), h) for n, h in heads_raw.items()),
            intent=IntentSpec.from_dict(intent_raw) if intent_raw is not None else None,
            graph_version=version,
            entity_stream=str(raw.get("entity_stream", _ENTITY_STREAM_COMBAT)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        ctx = "graph"
        if self.entity_stream not in _ENTITY_STREAMS:
            raise GraphSpecError(
                f"{ctx}: unknown entity_stream {self.entity_stream!r}; "
                f"allowed: {list(_ENTITY_STREAMS)}"
            )
        names = [t.name for t in self.tokens]
        if len(set(names)) != len(names):
            raise GraphSpecError(f"{ctx}: duplicate token name")
        kinds = [t.kind for t in self.tokens]
        if kinds.count(TOKEN_KIND_ENTITIES) != 1:
            raise GraphSpecError(f"{ctx}: exactly one entities token required")
        if kinds.count(TOKEN_KIND_SPATIAL) > 1 or kinds.count(TOKEN_KIND_CLS) > 1:
            raise GraphSpecError(f"{ctx}: at most one spatial and one cls token")
        if not self.self_tokens:
            raise GraphSpecError(f"{ctx}: at least one self token (cls/fields) required")
        # Layout: [cls?, fields.., spatial?, entities] — reserved tail ordering.
        expected_tail = [k for k in kinds if k in (TOKEN_KIND_SPATIAL, TOKEN_KIND_ENTITIES)]
        if kinds[-len(expected_tail):] != expected_tail or expected_tail[-1] != TOKEN_KIND_ENTITIES:
            raise GraphSpecError(f"{ctx}: tokens must be ordered [cls?, fields.., spatial?, entities]")
        if TOKEN_KIND_CLS in kinds and kinds[0] != TOKEN_KIND_CLS:
            raise GraphSpecError(f"{ctx}: cls token must be first")

        pointer_names = [p.name for p in self.pointers]
        if len(set(pointer_names)) != len(pointer_names):
            raise GraphSpecError(f"{ctx}: duplicate pointer")

        head_names = [h.name for h in self.heads]
        if len(set(head_names)) != len(head_names):
            raise GraphSpecError(f"{ctx}: duplicate head")
        activations = {h.activation for h in self.heads}
        if len(activations) > 1:
            raise GraphSpecError(f"{ctx}: heads must share one activation, got {sorted(activations)}")

        # Edge resolution: no dangling edges, no silent drops.
        intent_consumers = []
        for h in self.heads:
            hctx = f"heads[{h.name!r}]"
            if EDGE_TARGET_FEAT in h.inputs and self.pointer is None:
                raise GraphSpecError(f"{hctx}: {EDGE_TARGET_FEAT!r} requires a target pointer node")
            if EDGE_GRU in h.inputs and self.temporal is None:
                raise GraphSpecError(f"{hctx}: {EDGE_GRU!r} requires a temporal node")
            if EDGE_AIM in h.inputs and self.pointer is None:
                # The alignment block is pooled by the pointer softmax; with no
                # pointer there is nothing to compute it from.
                raise GraphSpecError(
                    f"{hctx}: {EDGE_AIM!r} requires a target pointer node")
            if EDGE_AIM2 in h.inputs and self.pointer is None:
                raise GraphSpecError(
                    f"{hctx}: {EDGE_AIM2!r} requires a target pointer node")
            if EDGE_INTENT in h.inputs:
                intent_consumers.append(h)
                if self.intent is None:
                    raise GraphSpecError(f"{hctx}: {EDGE_INTENT!r} requires an intent node")
                # Prefix layout is (readout | target.feat | intent): a head
                # cannot consume intent while skipping target.feat when the
                # pointer exists (the prefix slice can only drop the TAIL).
                if self.pointer is not None and EDGE_TARGET_FEAT not in h.inputs:
                    raise GraphSpecError(
                        f"{hctx}: {EDGE_INTENT!r} with a pointer declared "
                        f"requires {EDGE_TARGET_FEAT!r} too (prefix layout)")
                if EDGE_INTENT != h.inputs[-1]:
                    raise GraphSpecError(
                        f"{hctx}: {EDGE_INTENT!r} must be declared last (tail block)")
        if self.intent is not None:
            if not intent_consumers:
                raise GraphSpecError(
                    f"{ctx}: intent node declared but no head consumes {EDGE_INTENT!r}")
            if self.intent.source == "sg_softmax" and self.head("attack") is None:
                raise GraphSpecError(
                    f"{ctx}: intent source 'sg_softmax' requires an attack selector head")
        # Network feeds ONE shared feature cat (readout | target_feat) to
        # every non-selector head; heads that drop target.feat prefix-slice
        # it off. That requires every head to agree on WHICH readout edge
        # the shared cat starts with (the selector composes its own cat but
        # shares the same edge vocabulary, so it participates in the check).
        readout_edges = {e for h in self.heads for e in h.inputs if e in _READOUT_EDGES}
        if len(readout_edges) > 1:
            raise GraphSpecError(
                f"{ctx}: heads must agree on one readout edge (Network builds "
                f"one shared feature cat); got { {h.name: h.inputs for h in self.heads} }"
            )
        # A motor head cannot opt out of target_feat while the pointer is
        # present (network.py builds one shared cat; the prefix slice can
        # only drop the TAIL). Reject the inexpressible graph instead of
        # silently building different wiring than declared.
        motor = [h for h in self.heads if h.name in _MOTOR_HEADS]
        if motor and self.pointer is not None and EDGE_TARGET_FEAT not in motor[0].inputs:
            raise GraphSpecError(
                f"{ctx}: a target pointer is declared, so motor heads must "
                f"include {EDGE_TARGET_FEAT!r} in inputs (Network always cats "
                f"target_feat into the shared motor features when the pointer "
                f"exists); drop the pointer to ablate it"
            )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "graph_version": self.graph_version,
            "tokens": {t.name: t.to_dict() for t in self.tokens},
            "encoder": self.encoder.to_dict(),
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "pointers": {p.name: p.to_dict() for p in self.pointers},
            "heads": {h.name: h.to_dict() for h in self.heads},
        }
        if self.intent is not None:
            # Shape-bearing (intent widens consumer heads): must round-trip.
            out["intent"] = self.intent.to_dict()
        if self.entity_stream != _ENTITY_STREAM_COMBAT:
            # Shape-bearing (like move_seg.water_ud): a full-stream model
            # rebuilt without it would build the combat entity wiring and
            # fail its own state dict. Combat stays key-free so existing
            # graph documents round-trip byte-identically.
            out["entity_stream"] = self.entity_stream
        return out
