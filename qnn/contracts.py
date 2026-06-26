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
    vocab tables/sizes. See src/docs/contracts/semantics/semantics.1.md."""
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
    tables = [("SELF", en.SELF_FIELDS), ("SPATIAL", en.SPATIAL_FIELDS),
              ("ENTITY_COMMON", en.ENTITY_COMMON_FIELDS)]
    tables += [(f"ENTITY_{t}", tbl) for t, tbl in en.ENTITY_FIELDS.items()]
    for tname, tbl in tables:
        for f in tbl:
            parts.append(f"{tname}.{f.name}|scale={f.scale}|tf={f.transform}")
    parts.append("ITEM_AMOUNT_MULT=" + ",".join(f"{x:.8g}" for x in en.ITEM_AMOUNT_MULT))
    parts.append("ITEM_AMOUNT_CONST=" + ",".join(f"{x:.8g}" for x in en.ITEM_AMOUNT_CONST))
    for k in ("ENTITY_VOCAB_SIZE", "ACTION_VOCAB_SIZE", "MODALITY_VOCAB_SIZE",
              "MAX_PLAYER_INDICES", "TOKEN_PROJECTILE", "TOKEN_ACTOR", "TOKEN_ITEM",
              "TOKEN_MOVER", "PROJECTILE_SCALAR_DIM", "ACTOR_SCALAR_DIM",
              "ITEM_SCALAR_DIM", "MOVER_SCALAR_DIM", "MAX_ENTITY_EVENTS",
              "MAX_TOKEN_OBJECTS"):
        parts.append(f"{k}={getattr(vb, k)}")
    parts.append(f"SPATIAL_TOKEN_COUNT={en.SPATIAL_TOKEN_COUNT}")
    for vname in ("ENTITY_IDS", "ACTION_IDS", "MODALITY_IDS", "SPATIAL_SECTOR_IDS"):
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
#   modern / full_4head → wire.9  / semantics.1   (arch derived from ModelConfig)
#   v17, v22            → wire.7  / semantics.1
#
# (Older generations — wire.1–.6 — exist but the converter does not recognize
#  them; they are deliberately absent here. recognize_generation returns None and
#  backfill_contract refuses rather than guessing.)
GENERATION_CONTRACTS: Dict[str, Contract] = {
    MODERN:  {"wire": "wire.9", "semantics": "semantics.1", "arch": "full_4head"},
    GEN_V17: {"wire": "wire.7", "semantics": "semantics.1", "arch": _ARCH_V17},
    GEN_V22: {"wire": "wire.7", "semantics": "semantics.1", "arch": _ARCH_V22},
}

# Defensive parity: the MODERN registry row pins wire.9/semantics.1, which must
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
    not silently mis-tagged. ``wire``/``semantics`` are already int-suffixed
    (``wire.9`` → ``w9``)."""
    a = _ARCH_A_TAG.get(contract["arch"], contract["arch"])
    s = contract["semantics"].rsplit(".", 1)[-1]
    w = contract["wire"].rsplit(".", 1)[-1]
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


def current_contract(model_cfg: Optional[Dict[str, Any]] = None) -> Contract:
    """The contract a freshly-saved MODERN checkpoint is born with.

    wire/semantics come from the LIVE engine_norm constants (this is what the
    running code produces); arch is derived from the ModelConfig when supplied,
    else the canonical ``full_4head``.
    """
    arch = arch_id_from_model_config(model_cfg) if isinstance(model_cfg, dict) else "full_4head"
    return {
        "wire": _en.WIRE_CONTRACT_ID,
        "semantics": _en.SEMANTICS_CONTRACT_ID,
        "arch": arch,
    }
