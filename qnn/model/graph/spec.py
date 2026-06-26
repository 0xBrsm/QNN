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

v1 scope notes:

* Heads are the four engine heads (``move``/``look``/``attack``/
  ``weapon``). The PPO critic is Sample Factory-owned, not a node.
* Motor-head edges are ``readout`` (+ ``target.feat``); the v17
  ``look_bypass_gru`` layout is not expressible — those checkpoints
  keep loading through the legacy flat-``ModelConfig`` path.
* All heads share one activation (the ``ModelConfig`` bridge needs a
  single ``head_activation``); mixed activations raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from qnn.model.tokens.obs_fields import SCALAR_FIELDS, VOCAB_FIELDS

GRAPH_VERSION = 1

# Reserved token kinds the builder constructs outside TokenBuilder.
TOKEN_KIND_FIELDS = "fields"
TOKEN_KIND_CLS = "cls"
TOKEN_KIND_SPATIAL = "spatial"
TOKEN_KIND_ENTITIES = "entities"
_TOKEN_KINDS = (TOKEN_KIND_FIELDS, TOKEN_KIND_CLS, TOKEN_KIND_SPATIAL, TOKEN_KIND_ENTITIES)
_SELF_KINDS = (TOKEN_KIND_FIELDS, TOKEN_KIND_CLS)

# Edge names. Motor heads (move/look/attack) read the shared motor
# feature vector; the weapon selector composes its own cat.
EDGE_READOUT = "readout"            # gru hidden if temporal else self_readout
EDGE_SELF_READOUT = "self_readout"  # encoder CLS readout
EDGE_GRU = "gru"                    # temporal hidden (requires a temporal node)
EDGE_TARGET_FEAT = "target.feat"    # pointer-blended target feature
_MOTOR_EDGES = (EDGE_READOUT, EDGE_TARGET_FEAT)

# Weapon-head edge name ↔ flat ``ModelConfig.weapon_sources`` name. The
# graph spells the pointer edge ``"target.feat"``; the flat config spells
# it ``"target_feat"`` — this table owns the rename at every crossing
# (spec validation, ``_weapon_sources``, ``graph_from_model_config``).
WEAPON_EDGE_TO_SOURCE = {
    EDGE_GRU: "gru",
    EDGE_SELF_READOUT: "self_readout",
    EDGE_TARGET_FEAT: "target_feat",
}
WEAPON_SOURCE_TO_EDGE = {v: k for k, v in WEAPON_EDGE_TO_SOURCE.items()}
_WEAPON_EDGES = tuple(WEAPON_EDGE_TO_SOURCE)

# A head may also read a single declared token as its readout via a
# ``token.<name>`` edge (e.g. ``token.arsenal``) — resolved to the encoder's
# per-self-token output at that token's index. The build maps it to a
# ``token:<name>`` weapon source; existence (``<name>`` is a real self-token) is
# checked at GraphSpec level where the token list is known.
EDGE_TOKEN_PREFIX = "token."


def _is_token_edge(edge: str) -> bool:
    return edge.startswith(EDGE_TOKEN_PREFIX)


def _token_edge_name(edge: str) -> str:
    return edge[len(EDGE_TOKEN_PREFIX):]

# ``move_hazard`` (a25) is a motor head: it reads the shared motor feature
# vector (readout [+ target.feat]) like move/look/attack. Its additional
# semi-Markov inputs (held_class / dwell_age) are NOT graph edges — they are
# decode-state obs fields the Network passes straight through ``flat_obs`` (see
# qnn.model.bench.a25.move_hazard_head and network.py), so the edge contract
# here is unchanged.
_HEAD_NAMES = ("move", "look", "attack", "weapon", "move_hazard")
_MOTOR_HEADS = ("move", "look", "attack", "move_hazard")
_ACTIVATIONS = ("none", "gelu", "relu")

_WEAPON_DECODE_KEYS = ("sticky_confidence", "sticky_margin")


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
    readiness → vocab_sum) so the parameter layout is deterministic.
    """
    name: str
    kind: str = TOKEN_KIND_FIELDS
    scalars: tuple[str, ...] = ()
    vocab: tuple[str, ...] = ()
    vocab_sum: tuple[str, ...] = ()
    readiness: bool = False
    kind_tag: bool = False

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "TokenSpec":
        ctx = f"tokens[{name!r}]"
        _reject_unknown(raw, ("kind", "scalars", "vocab", "vocab_sum", "readiness", "kind_tag"), ctx)
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
            kind_tag=bool(raw.get("kind_tag", False)),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        ctx = f"tokens[{self.name!r}]"
        if self.kind != TOKEN_KIND_FIELDS:
            if self.scalars or self.vocab or self.vocab_sum or self.readiness or self.kind_tag:
                raise GraphSpecError(f"{ctx}: kind={self.kind!r} tokens carry no fields")
            return
        if not (self.scalars or self.vocab or self.vocab_sum or self.readiness or self.kind_tag):
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
            return {"kind": self.kind}
        out: dict[str, Any] = {}
        if self.scalars:
            out["scalars"] = list(self.scalars)
        if self.vocab:
            out["vocab"] = list(self.vocab)
        if self.vocab_sum:
            out["vocab_sum"] = list(self.vocab_sum)
        if self.readiness:
            out["readiness"] = True
        if self.kind_tag:
            out["kind_tag"] = True
        return out


def monolithic_self_token(include_weapon_id: bool) -> TokenSpec:
    """The production monolithic self token, in TokenSpec form.

    Single source for the canonical flat layout's self token
    (``graph_from_model_config`` derives its token from this). The field
    tuple it induces via ``qnn.model.graph.embedding.token_fields`` is
    asserted equal to ``qnn.model.tokens.obs_fields.canonical_self_fields``
    by a unit test (tokens must not import graph, so the two definitions
    cannot share code directly — drift fails the test instead).
    """
    return TokenSpec(
        name="self",
        scalars=("self_scalars",),
        vocab=("armor_type", "movement_id")
        + (("weapon_id",) if include_weapon_id else ()),
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
class HeadNodeSpec:
    """One output head: a registered type, parameters, and input edges."""
    name: str
    type: str
    inputs: tuple[str, ...]
    d_hidden: int = 0
    activation: str = "gelu"
    context_from_obs: bool = True            # weapon only
    # Stored as sorted (key, value) pairs — hashable and immutable like the
    # rest of the spec. Construct with a plain dict; __post_init__ normalizes
    # either form so direct construction compares equal to a from_dict
    # round-trip.
    decode: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decode",
            tuple(sorted((str(k), float(v)) for k, v in dict(self.decode).items())),
        )

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> "HeadNodeSpec":
        ctx = f"heads[{name!r}]"
        allowed = ("type", "inputs", "d_hidden", "activation")
        if name == "weapon":
            allowed += ("context_from_obs", "decode")
        _reject_unknown(raw, allowed, ctx)
        decode_raw = raw.get("decode", {})
        if not isinstance(decode_raw, Mapping):
            raise GraphSpecError(f"{ctx}: decode must be an object")
        _reject_unknown(decode_raw, _WEAPON_DECODE_KEYS, f"{ctx}.decode")
        spec = cls(
            name=name,
            type=str(_require(raw, "type", ctx)),
            inputs=_str_tuple(_require(raw, "inputs", ctx), ctx),
            d_hidden=int(raw.get("d_hidden", 0)),
            activation=str(raw.get("activation", "gelu")),
            context_from_obs=bool(raw.get("context_from_obs", True)),
            decode=dict(decode_raw),
        )
        spec._validate()
        return spec

    def _validate(self) -> None:
        ctx = f"heads[{self.name!r}]"
        if self.name not in _HEAD_NAMES:
            raise GraphSpecError(f"{ctx}: unknown head; allowed: {list(_HEAD_NAMES)}")
        if self.activation not in _ACTIVATIONS:
            raise GraphSpecError(f"{ctx}: activation must be one of {list(_ACTIVATIONS)}")
        if not self.inputs:
            raise GraphSpecError(f"{ctx}: inputs must be non-empty")
        if len(set(self.inputs)) != len(self.inputs):
            raise GraphSpecError(f"{ctx}: duplicate input edge")
        if self.name == "weapon":
            # weapon may also read a single token as its readout (token.<name>);
            # the token's existence is checked at GraphSpec level.
            bad = [e for e in self.inputs
                   if e not in _WEAPON_EDGES and not _is_token_edge(e)]
            allowed_edges = list(_WEAPON_EDGES) + [f"{EDGE_TOKEN_PREFIX}<name>"]
        else:
            allowed_edges = _MOTOR_EDGES
            bad = [e for e in self.inputs if e not in allowed_edges]
        if bad:
            raise GraphSpecError(f"{ctx}: unknown edge(s) {bad}; allowed: {list(allowed_edges)}")
        if self.name in _MOTOR_HEADS and EDGE_READOUT not in self.inputs:
            raise GraphSpecError(f"{ctx}: motor heads must read {EDGE_READOUT!r}")
        if self.name == "weapon" and not (
            ({EDGE_GRU, EDGE_SELF_READOUT} & set(self.inputs))
            or any(_is_token_edge(e) for e in self.inputs)
        ):
            raise GraphSpecError(
                f"{ctx}: weapon needs a readout — one of {EDGE_GRU!r} / "
                f"{EDGE_SELF_READOUT!r} or a {EDGE_TOKEN_PREFIX}<name> token edge"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type, "inputs": list(self.inputs),
            "d_hidden": self.d_hidden, "activation": self.activation,
        }
        if self.name == "weapon":
            out["context_from_obs"] = self.context_from_obs
            if self.decode:
                out["decode"] = {k: v for k, v in self.decode}
        return out


# ── The graph ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GraphSpec:
    tokens: tuple[TokenSpec, ...]
    encoder: EncoderSpec
    temporal: TemporalSpec | None
    pointers: tuple[PointerSpec, ...]
    heads: tuple[HeadNodeSpec, ...]
    graph_version: int = GRAPH_VERSION

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

    # -- construction -------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphSpec":
        ctx = "graph"
        _reject_unknown(raw, ("graph_version", "tokens", "encoder", "temporal", "pointers", "heads"), ctx)
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
        spec = cls(
            tokens=tokens,
            encoder=EncoderSpec.from_dict(_require(raw, "encoder", ctx)),
            temporal=TemporalSpec.from_dict(temporal_raw) if temporal_raw is not None else None,
            pointers=tuple(PointerSpec.from_dict(str(n), p) for n, p in pointers_raw.items()),
            heads=tuple(HeadNodeSpec.from_dict(str(n), h) for n, h in heads_raw.items()),
            graph_version=version,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        ctx = "graph"
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
        self_token_names = {t.name for t in self.self_tokens}
        for h in self.heads:
            hctx = f"heads[{h.name!r}]"
            if EDGE_TARGET_FEAT in h.inputs and self.pointer is None:
                raise GraphSpecError(f"{hctx}: {EDGE_TARGET_FEAT!r} requires a target pointer node")
            if EDGE_GRU in h.inputs and self.temporal is None:
                raise GraphSpecError(f"{hctx}: {EDGE_GRU!r} requires a temporal node")
            for e in h.inputs:
                if _is_token_edge(e) and _token_edge_name(e) not in self_token_names:
                    raise GraphSpecError(
                        f"{hctx}: token edge {e!r} → unknown self-token "
                        f"{_token_edge_name(e)!r}; declared self-tokens: "
                        f"{sorted(self_token_names)}"
                    )
        # Network feeds ONE motor feature vector to all motor heads.
        motor = [h for h in self.heads if h.name in _MOTOR_HEADS]
        if motor and len({h.inputs for h in motor}) != 1:
            raise GraphSpecError(
                f"{ctx}: motor heads must declare identical inputs "
                f"(Network builds one shared motor feature vector); got "
                f"{ {h.name: h.inputs for h in motor} }"
            )
        # Network's motor feature vector is cat(readout, target_feat)
        # whenever a pointer node exists — a motor head cannot opt out of
        # target_feat while the pointer is present (network.py builds one
        # shared cat; there is no per-head slice). Reject the inexpressible
        # graph instead of silently building different wiring than declared.
        if motor and self.pointer is not None and EDGE_TARGET_FEAT not in motor[0].inputs:
            raise GraphSpecError(
                f"{ctx}: a target pointer is declared, so motor heads must "
                f"include {EDGE_TARGET_FEAT!r} in inputs (Network always cats "
                f"target_feat into the shared motor features when the pointer "
                f"exists); drop the pointer to ablate it"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_version": self.graph_version,
            "tokens": {t.name: t.to_dict() for t in self.tokens},
            "encoder": self.encoder.to_dict(),
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "pointers": {p.name: p.to_dict() for p in self.pointers},
            "heads": {h.name: h.to_dict() for h in self.heads},
        }
