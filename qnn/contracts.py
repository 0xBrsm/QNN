"""Model↔engine CONTRACT resolution — keyed on the CHECKPOINT, never the ONNX.

The contract is versioned on three independent axes (see
``src/docs/contracts/README.md``):

  * **wire**       — the ONNX I/O tensor signature (``engine_norm.WIRE_CONTRACT_ID``)
  * **semantics**  — normalization scales + vocab id mappings
                     (``engine_norm.SEMANTICS_CONTRACT_ID``)
  * **arch**       — model internals / weight layout (checkpoint-side only)

A checkpoint carries its contract under ``meta["contract"] = {"wire", "semantics",
"arch"}``. The checkpoint is the SOURCE OF TRUTH — nothing here (or in the
exporter) is ever inferred from an ONNX graph, an input/output name, a tensor
shape, or a filename.

This module is intentionally **torch-free**: it imports only ``engine_norm``
(the live contract ids) so it can be used by tooling (``tools/stamp_checkpoint``,
``tools/export_onnx``) and by the torch-free portions of the checkpoint
converter without dragging torch in.

What lives here:
  * :data:`GENERATION_CONTRACTS` — the explicit generation→native-contract
    registry, keyed on the SAME schema markers the checkpoint converter uses to
    recognize a generation.
  * :func:`recognize_generation` — classify a checkpoint ``meta`` dict into a
    generation tag using those markers.
  * :func:`backfill_contract` — given a ``meta`` dict with no ``contract`` block,
    return the native contract for its recognized generation (or ``None`` if the
    generation is unrecognized — never invent a value).
  * :func:`arch_id_from_model_config` — derive a stable arch id from a
    ModelConfig-shaped dict, for the save-time stamp.
  * :func:`current_contract` — the contract a freshly-saved modern checkpoint is
    born with.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from qnn import engine_norm as _en
from qnn.vocab import ENTITY_STREAM_COMBAT, ENTITY_STREAM_FULL, ENTITY_STREAMS

__all__ = [
    "Contract",
    "GENERATION_CONTRACTS",
    "MODERN",
    "GEN_V17",
    "GEN_V22",
    "recognize_generation",
    "backfill_contract",
    "arch_id_from_model_config",
    "current_contract",
    "wire_sig_from_graph",
    "semantics_sig",
    "render_version",
]

# Onnx TensorProto elem_type → name (for wire_sig_from_graph).
_ONNX_ELEM = {1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16", 6: "INT32",
              7: "INT64", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE", 12: "UINT32", 13: "UINT64"}


def wire_sig_from_graph(model) -> str:
    """16-hex fingerprint of an ONNX model's I/O signature: input
    (name|dtype|shape) in graph order, then output names. A fingerprint of the
    artifact — NOT a contract classifier (the contract id is declared, never
    derived from this). Lets a forgotten/stale `wire_contract` be told apart."""
    import hashlib
    parts = []
    for vi in model.graph.input:
        tt = vi.type.tensor_type
        dt = _ONNX_ELEM.get(tt.elem_type, str(tt.elem_type))
        shp = tuple(d.dim_value for d in tt.shape.dim)
        parts.append(f"{vi.name}|{dt}|{shp}")
    parts.append("--outputs--")
    parts.append(",".join(o.name for o in model.graph.output))
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def semantics_sig() -> str:
    """16-hex fingerprint of the SEMANTICS contract — normalization scales,
    per-field (scale, transform), item bit masks, item-amount tables, and the
    vocab tables/sizes. See src/docs/contracts/semantics/semantics.2.md."""
    import hashlib
    from qnn import engine_norm as en
    from qnn import vocab as vb
    parts: list[str] = []
    for k in ("MAX_HEALTH", "MAX_ARMOR_EFFECT", "MAX_SHELLS", "MAX_NAILS",
              "MAX_ROCKETS", "MAX_CELLS", "MAX_VELOCITY", "TIME_SCALE", "DIST_SCALE",
              "ITEMS_WEAPON_MASK", "ITEMS_ARMOR_MASK", "ITEMS_POWERUP_MASK",
              "ITEMS_MEANINGFUL", "MOVEMENT_GROUND", "MOVEMENT_AIR",
              "MOVEMENT_WATER_LOW", "MOVEMENT_WATER_MID", "MOVEMENT_WATER_HIGH"):
        parts.append(f"{k}={getattr(en, k)}")
    tables = [("SELF", en.SELF_FIELDS),
              ("ENTITY_COMMON", en.ENTITY_COMMON_FIELDS)]
    tables += [(f"ENTITY_{t}", tbl) for t, tbl in en.ENTITY_FIELDS.items()]
    for tname, tbl in tables:
        for f in tbl:
            parts.append(f"{tname}.{f.name}|scale={f.scale}|tf={f.transform}")
    # Item `amount` value-semantics (engine_norm.ITEM_AMOUNT_*). Item/mover are
    # not in the wire.13 model obs, but they stay live for rewards/store and for
    # the wire.11/wire.12 codecs that share the bin, so their dequant is
    # fingerprinted here — a silent drift stays catchable.
    parts.append("ITEM_AMOUNT_MULT=" + ",".join(f"{x:.8g}" for x in en.ITEM_AMOUNT_MULT))
    parts.append("ITEM_AMOUNT_CONST=" + ",".join(f"{x:.8g}" for x in en.ITEM_AMOUNT_CONST))
    # Spatial VALUE semantics: how depth-atlas codes map to world geometry.
    # These change meaning while leaving tensor shapes intact — exactly the
    # silent-miscalibration class this signature exists to catch. The
    # spatial tensor's shape and row identity (band/yaw counts, band-id
    # names) are wire-scoped: a change there cannot bind at load, so the
    # WIRE axis covers it and it is deliberately NOT hashed here.
    parts.append("ATLAS_DEPTH_LEVELS=" + ",".join(str(x) for x in en.ATLAS_DEPTH_LEVELS))
    parts.append(f"ATLAS_MISS_CODE={en.ATLAS_MISS_CODE}")
    parts.append("ATLAS_ELEV_DEG=" + ",".join(str(x) for x in en.ATLAS_ELEV_DEG))
    parts.append(
        f"ATLAS_RANGES={en.ATLAS_HORIZ_RANGE:.8g},{en.ATLAS_VERT_RANGE:.8g}"
    )
    for k in ("ENTITY_VOCAB_SIZE", "ACTION_VOCAB_SIZE", "MODALITY_VOCAB_SIZE",
              "COMBAT_MODALITY_VOCAB_SIZE", "MAX_PLAYER_INDICES",
              "TOKEN_PROJECTILE", "TOKEN_ACTOR", "TOKEN_ITEM", "TOKEN_MOVER",
              "PROJECTILE_SCALAR_DIM", "ACTOR_SCALAR_DIM",
              "ITEM_SCALAR_DIM", "MOVER_SCALAR_DIM", "MAX_ENTITY_EVENTS",
              "MAX_TOKEN_OBJECTS"):
        parts.append(f"{k}={getattr(vb, k)}")
    for vname in (
        "ENTITY_IDS", "ACTION_IDS", "MODALITY_IDS", "COMBAT_MODALITY_IDS",
    ):
        tbl = getattr(vb, vname)
        parts.append(f"{vname}=" + ",".join(f"{k}:{v}" for k, v in tbl.items()))
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]

# A contract triple. Plain dict (JSON-friendly, stored verbatim in meta).
Contract = Dict[str, str]

# Generation tags. These name the converter-recognized GENERATIONS — the same
# ones migrate_legacy_flat_meta / QNNPolicy.load distinguish.
MODERN = "modern"   # nested ModelConfig (meta["model"] present) — full_4head lineage
GEN_V17 = "v17"     # flat meta, v17 markers (no weapon head)
GEN_V22 = "v22"     # flat meta, v22 markers (recognized but not v17)

# Arch ids for the legacy flat generations. Modern checkpoints derive their arch
# from the ModelConfig (see arch_id_from_model_config); the legacy generations
# have no ModelConfig, so their arch id is the generation tag itself.
_ARCH_V17 = "v17"
_ARCH_V22 = "v22"

# ── Generation → NATIVE contract registry ────────────────────────────────────
# Each converter-recognized generation maps to the contract it was NATIVELY born
# with. wire/semantics ids are pinned to the historical release (NOT to the live
# engine_norm constants) so backfilling an archived checkpoint records what it
# actually shipped with, not whatever the current HEAD happens to be.
#
#   modern / full_4head → wire.13 / semantics.2   (arch derived from ModelConfig)
#   v17, v22            → wire.7 / semantics.1
#
# wire.13 is the A27 HEAD contract: wire.11 action decode plus the depth atlas
# and the pure actor/projectile combat entity stream (semantics.2). The
# exporter ALWAYS bakes the stateful move decode + the attack decode + their
# recurrent state I/O into the graph (tools/export_onnx.py ExportWrapper), so a
# freshly-trained / re-exported full_4head model IS a wire.13 graph (native
# 29-obs split + the in-graph decided `move`/`attack` + the move_state/
# attack_state loop-back pairs) and is stamped wire.13.
#
# RECLAIMED NUMBERS: an in-graph move shape was briefly wire.10 (vs engine-argmax
# wire.9) during a24 dev; wire.10 was never released and stays BURNED (never
# reuse). wire.9 (in-graph move, engine-side attack) was the a24 contract until the
# attack decode moved in-graph → wire.11. a26's wire.12 (depth atlas + full
# entity stream, semantics.1) reclaimed the number the A27 combat shape briefly
# held; the combat shape moved to wire.13 to avoid the collision. The
# in-development HEAD is wire.13; the bin's live codec set is
# {wire.11, wire.12, wire.13} (wire.7 / wire.9 retired but recognized).
#
# (Older generations — wire.1–.6 — exist but the converter does not recognize
#  them; they are deliberately absent here. recognize_generation returns None and
#  backfill_contract refuses rather than guessing.)
GENERATION_CONTRACTS: Dict[str, Contract] = {
    # wire.13.2 = wire.11 action decode + finalized 24x11 nibble-packed
    # depth-atlas obs + the A27 pure-combat entity stream (semantics.2).
    # Distinct from a26's wire.12.x (same atlas, FULL entity stream,
    # semantics.1); the bin runs wire.11/.12.x/.13.x side by side but this
    # codebase only exports wire.13.x natively.
    # (The a27 rc1 line's 72-wide unpacked atlas is wire.13.1; it is not a
    # GENERATION of its own — same modern ModelConfig — so it has no row
    # here. The exporter picks the id from the checkpoint's atlas width,
    # see tools/export_onnx.py:_native_obs_for_model.)
    MODERN:  {"wire": "wire.13.2", "semantics": "semantics.2", "arch": "full_4head"},
    GEN_V17: {"wire": "wire.7",  "semantics": "semantics.1", "arch": _ARCH_V17},
    GEN_V22: {"wire": "wire.7",  "semantics": "semantics.1", "arch": _ARCH_V22},
}

# Defensive parity: the MODERN registry row pins HEAD wire/semantics, which must
# equal the live engine_norm ids (HEAD == the modern native contract). If HEAD
# bumps a contract axis, add a new generation row and update this — don't let the
# registry silently drift from the constants the save-time stamp uses.
assert GENERATION_CONTRACTS[MODERN]["wire"] == _en.WIRE_CONTRACT_ID, (
    "contracts.py MODERN wire id is stale vs engine_norm.WIRE_CONTRACT_ID "
    f"({GENERATION_CONTRACTS[MODERN]['wire']} != {_en.WIRE_CONTRACT_ID}); "
    "a wire bump needs a new generation row, not a silent registry edit."
)
assert GENERATION_CONTRACTS[MODERN]["semantics"] == _en.SEMANTICS_CONTRACT_ID, (
    "contracts.py MODERN semantics id is stale vs "
    f"engine_norm.SEMANTICS_CONTRACT_ID ({GENERATION_CONTRACTS[MODERN]['semantics']} "
    f"!= {_en.SEMANTICS_CONTRACT_ID})."
)

# The a26-line FULL entity-stream contract (recency dims, live item/mover
# tokens, 4-way modality vocab — qnn.vocab.ENTITY_STREAM_FULL, the WS6 port).
# NOT engine_norm.WIRE_CONTRACT_ID/SEMANTICS_CONTRACT_ID — those are this
# line's COMBAT-stream ids; a26 owns its native wire/semantics ids on its own
# branch (feat/a26) and this line only needs to STAMP them correctly on a
# full-stream graph, not build the a26 codec. Pinned literally (like
# WIRE_CONTRACT_ID_ATLAS_LEGACY) rather than derived, since there is no
# live a26 engine_norm module on this branch to import the ids from.
WIRE_CONTRACT_ID_FULL_STREAM = "wire.12.2"
SEMANTICS_CONTRACT_ID_FULL_STREAM = "semantics.1"

# entity_stream -> (wire, semantics) for a freshly-saved MODERN checkpoint.
# Keyed on qnn.vocab.ENTITY_STREAMS so an unknown stream fails loud instead of
# silently falling back to the combat pair.
_ENTITY_STREAM_CONTRACT: Dict[str, tuple] = {
    ENTITY_STREAM_COMBAT: (_en.WIRE_CONTRACT_ID, _en.SEMANTICS_CONTRACT_ID),
    ENTITY_STREAM_FULL: (WIRE_CONTRACT_ID_FULL_STREAM, SEMANTICS_CONTRACT_ID_FULL_STREAM),
}


# ── Compact a/s/w version render (display / provenance only) ─────────────────
# The per-axis ids in GENERATION_CONTRACTS remain the source of truth the engine
# matches; this is just a sane one-string render, e.g. `a24b.s1.w9`. `a` is the
# arch generation; the in-development `full_4head` arch carries a `b` (bench /
# pre-release) suffix until it's blessed as a numbered release.
_ARCH_A_TAG: Dict[str, str] = {
    _ARCH_V17:    "a17",
    _ARCH_V22:    "a22",
    "full_4head": "a24b",
}


def render_version(contract: Contract) -> str:
    """Render a contract triple as the canonical ``a{arch}.s{sem}.w{wire}`` string
    (e.g. ``a24b.s1.w9``). Display/provenance only — NOT a codec-selection key.
    An unmapped arch id falls back to the raw id so an unknown lineage is visible,
    not silently mis-tagged.

    The wire/semantics tags are the id with its family prefix stripped, NOT the
    last dot-segment: a minor-versioned id like ``wire.12.1`` must render
    ``w12.1``, not ``w1`` (which would read as ``wire.1``, a different and much
    older contract). Same rule for semantics."""
    a = _ARCH_A_TAG.get(contract["arch"], contract["arch"])
    s = contract["semantics"].removeprefix("semantics.")
    w = contract["wire"].removeprefix("wire.")
    return f"{a}.s{s}.w{w}"


def recognize_generation(meta: Dict[str, Any]) -> Optional[str]:
    """Classify a checkpoint ``meta`` dict into a converter-recognized generation.

    Keyed on the SAME schema markers the checkpoint converter already uses (see
    :func:`qnn.utils.checkpoint_converter.migrate_legacy_flat_meta`):

      * a nested ``meta["model"]`` block → :data:`MODERN`.
      * otherwise a flat legacy meta with the v17/v22 ModelConfig fields:
          - v17 markers (``target_bypass_gru`` / ``move_categorical`` /
            ``readout``) → :data:`GEN_V17`.
          - else, when the required flat-arch fields are present → :data:`GEN_V22`.

    Returns ``None`` when the meta is neither modern nor a recognized legacy flat
    schema (e.g. a pre-v17 generation the converter doesn't migrate). Never
    guesses.
    """
    if "model" in meta:
        return MODERN

    # Mirror migrate_legacy_flat_meta's alias normalization + required-field gate
    # so we recognize exactly the same set it can migrate (no more, no less).
    probe = dict(meta)
    for _old, _new in (("ffn_dim", "d_ffn"), ("gru_hidden", "d_gru")):
        if _old in probe and _new not in probe:
            probe[_new] = probe[_old]
    required = ("d_model", "n_heads", "n_layers", "d_ffn", "attn_dropout",
                "use_gru", "d_gru", "look_bypass_gru")
    if any(k not in probe for k in required):
        return None

    is_v17 = ("target_bypass_gru" in probe or "move_categorical" in probe
              or "readout" in probe)
    return GEN_V17 if is_v17 else GEN_V22


def backfill_contract(meta: Dict[str, Any]) -> Optional[Contract]:
    """Return the NATIVE contract for ``meta``'s recognized generation.

    Looks up :data:`GENERATION_CONTRACTS` by :func:`recognize_generation`. Returns
    ``None`` (caller should warn, never invent) when the generation is
    unrecognized. The returned dict is a fresh copy safe to store in meta.

    For the MODERN generation the arch id is refined from the embedded
    ModelConfig (``arch_id_from_model_config``) rather than the registry's generic
    ``"full_4head"`` placeholder, so a non-weapon-head modern checkpoint records
    its true arch.
    """
    gen = recognize_generation(meta)
    if gen is None:
        return None
    contract = dict(GENERATION_CONTRACTS[gen])
    if gen == MODERN:
        model_cfg = meta.get("model")
        if isinstance(model_cfg, dict):
            contract["arch"] = arch_id_from_model_config(model_cfg)
    return contract


def arch_id_from_model_config(model_cfg: Dict[str, Any]) -> str:
    """Derive a stable arch id from a ModelConfig-shaped dict.

    The arch axis pins model internals / weight layout. The stable, observable
    discriminator across the modern lineage is the head set: the canonical model
    is ``full_4head`` (move + look + attack + weapon) when ``use_weapon_head`` is
    set, otherwise ``3head`` (no weapon head, e.g. v17-shaped modern configs).
    Kept deliberately coarse — a finer arch id would couple to GRU width / d_model
    which are already captured by ``wire_sig``/``arch`` fingerprint at export.
    """
    use_weapon = bool(model_cfg.get("use_weapon_head", True))
    return "full_4head" if use_weapon else "3head"


def current_contract(
    model_cfg: Optional[Dict[str, Any]] = None,
    *,
    entity_stream: str = ENTITY_STREAM_COMBAT,
) -> Contract:
    """The contract a freshly-saved MODERN checkpoint is born with.

    wire/semantics are keyed on ``entity_stream`` (the checkpoint's
    ``GraphSpec.entity_stream``, NOT inferred from anything else): combat gets
    the LIVE engine_norm ids (what this line's exporter produces); full gets
    the a26-line pair (``WIRE_CONTRACT_ID_FULL_STREAM`` /
    ``SEMANTICS_CONTRACT_ID_FULL_STREAM``). A checkpoint built with
    ``entity_stream="full"`` trains on full-stream data — stamping the combat
    pair on it silently mismatched its actual training contract. arch is
    derived from the ModelConfig when supplied, else the canonical
    ``full_4head``.
    """
    if entity_stream not in ENTITY_STREAMS:
        raise ValueError(
            f"unknown entity_stream {entity_stream!r}; allowed: {list(ENTITY_STREAMS)}")
    arch = arch_id_from_model_config(model_cfg) if isinstance(model_cfg, dict) else "full_4head"
    wire, semantics = _ENTITY_STREAM_CONTRACT[entity_stream]
    return {
        "wire": wire,
        "semantics": semantics,
        "arch": arch,
    }
