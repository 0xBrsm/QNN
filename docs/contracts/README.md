# Observation/Action Contract Registry

This is the index for the QNN model↔engine contract. It replaces the single
monolithic "Token Specification vN" (which bundled four independent things into
one number and made it impossible to tell which change would break a model
load). The contract is now tracked on **three independent axes**, each versioned
in its own doc series, with this file as the cross-reference map.

## The three axes

| Axis | What it pins | Analogy | Failure if mismatched | Owner | Docs |
|------|-------------|---------|----------------------|-------|------|
| **Wire** | The ONNX **I/O tensor signature** — input + output names, dtypes, shapes. The *syntax* of the exchange. | A **file format** (PNG) | **Loud** on inputs (`Session.Run: Invalid input name`); silent-wrong on outputs — a parse error | the model **declares** it; a **codec** in the bin implements it | [`wire/`](wire/) |
| **Semantics** | The *meaning* behind those tensors — normalization scales + vocab id mappings | What the bytes *denote* | **Silent corruption** (well-formed, wrong meaning) | engine `engine_norm` + `vocab` must equal the model's baked dequant/embeds | [`semantics/`](semantics/) |
| **Arch** | Model internals & weight layout (subtoken splits, head structure, GRU width) | — | Old checkpoint won't load into new model code | the checkpoint converter — **live bin ignores it** (the `.onnx` is self-contained) | [`arch/`](arch/) |

**Wire vs codec.** The *wire contract* is the spec (what the bytes look like); a
*codec* is the bin-side code that emits/decodes it. Like **PNG vs libpng**: the
model declares its format; a codec implements it. We version the **contract**,
not the codec — a model stays self-describing forever, and a codec can be
rewritten without re-versioning. The codec is named after the contract it
implements ("the `wire.7` codec").

**Wire vs semantics = syntax vs meaning.** Wire is the grammar (shape of the
data) → violating it is a *parse error*, loud. Semantics is what well-formed
data means (scales, vocab) → violating it is a *misinterpretation*, silent. The
two failure modes line up exactly with the two axes.

Each axis bumps independently. The live id of record for a *freshly trained*
model is the constant in `src/qnn/engine_norm.py` (`WIRE_CONTRACT_ID`,
`SEMANTICS_CONTRACT_ID`). But the **source of truth for any given model is its
CHECKPOINT**: `model.save()` writes `meta["contract"] = {"wire", "semantics",
"arch"}` (born self-versioned from the live constants), and `tools/export_onnx.py`
stamps *those exact ids* — read from the loaded checkpoint, never re-derived
from the `engine_norm` constants or sniffed from the ONNX graph — into the ONNX
metadata, alongside a `wire_sig`/`semantics_sig` fingerprint. The bin then
selects the matching codec and refuses on a semantics mismatch without sniffing
input names. Archived checkpoints with no `contract` block are BACKFILLED from
the generation→contract registry (`qnn.contracts`, keyed on the converter's own
schema markers) at load, or stamped offline with `tools/stamp_checkpoint.py`; an
unrecognized generation is left unset (never guessed) and export FAILS loudly
rather than falling back to a constant.

## Registry — release / checkpoint generation → contract

`wire.N` is a single monotonic line across the **full history** so a future
codec slots into a named slot rather than forcing a renumber. (Two lineages live
on that line: `wire.1–2` are the early flat world-frame tensors; `wire.3+` are
the `engine_norm` field-table lineage — a doc-level note, not part of the id.)
**"Artifact?"** = a checkpoint or ONNX still survives that runs on it. **Band** =
support feasibility (below).

| Release | Bundled doc | Wire | Semantics | Arch (gen) | Ckpt gen | Artifact? | Band |
|---------|-------------|------|-----------|------------|----------|-----------|------|
| 0.1–0.2 | `obs-spec-v1/v2` | `wire.1` | `semantics.0a` | flat MLP/`MLPGRUPolicy` | — | No | C |
| ~0.3 | `obs-spec-v3` | `wire.2` | `semantics.0a` | early `QNNPolicy` (transformer) | — | No | C |
| 0.4–0.9 | `token-spec-v5/6/7` | `wire.3` | `semantics.0b` | cluster/QTOK | v5–v7 | No | C |
| 0.10 | `token-spec-v8` | `wire.4` | `semantics.0c` | three-store oracle | v8 | No | B |
| 0.11 | `token-spec-v9` | `wire.5` | `semantics.0d` | per-type entity stream | v9 | No | B |
| 0.14–0.15 | `token-spec-v10` | `wire.6` | `semantics.0d` | move[3]/jump-collapse | v10 | No | B |
| 0.17 | `token-spec.md` (v11) | **`wire.7`** | **`semantics.1`** | v11 packed | **v17, v22** | **Yes** | **A** |
| 0.21 | (no doc bump) | `wire.8` | `semantics.1` | native split | — | No (never exported) | A |
| HEAD | v11 + `look_delta` | **`wire.9`** | **`semantics.1`** | `full_4head` | **v24** | **Yes** | **A** |

Distinct contracts across the full history: **9 wire × 5 semantics**. With a
surviving runnable artifact: **2 wire** (`wire.7`, `wire.9`) **× 1 semantics**
(`semantics.1`). `wire.8` is a reconstructed id — the native exporter postdates
`look_delta`, so no 43-input graph was ever exported.

## Support bands & the wire.7 floor

A field is **DEAD** when the current `qnn_snapshot_t` / emit path can no longer
produce it. The current engine has removed three field families, which sets a
hard floor on faithful support:

- **`action_history`** — removed at v11; no history ring in the snapshot.
- **recall / MENTAL token channel** — removed; only inert residue remains
  (`qnn_store.h` `mem`, `MODALITY_MEMORY`), nothing emits the token.
- **`cluster_id` / `route_embed`** — gone from the snapshot→emit path (survives
  only in the standalone nav subsystem).

| Band | Range | Dead fields | Verdict | Codec cost |
|------|-------|-------------|---------|------------|
| **A — faithful** | `wire.7`→`wire.9` (v11→HEAD) | none | Fully loadable; covers every surviving artifact (v17, v22, v24, `full_4head`) | 2 wire + 1 semantics (`wire.7`, `wire.9` built; `wire.8` notional) |
| **B — approximation** | `wire.4`–`.6` (v8→v10) | recall, action_history (+cluster @ `wire.4`) | Zero-fill only → degraded; no surviving model; converter doesn't recognize pre-v17 | +3 wire, +2 semantics |
| **C — impossible** | `wire.1`–`.3` (flat + v5–7) | cluster, route, recall, action_history (+ different architecture) | No migration path; flat era is a separate tensor architecture | +3 wire, +2 semantics |

**The faithful-load floor is `wire.7` (v11).** Below it, models require
information the engine no longer produces. We build codecs for Band A only;
Bands B/C are documented (so a found checkpoint is immediately diagnosable) but
not implemented.

## Bump policy

- **Wire** (`WIRE_CONTRACT_ID`): bump on any I/O tensor added / removed / retyped
  / reshaped → next integer on the `wire.N` line; add `wire/wire.N.md`.
- **Semantics** (`SEMANTICS_CONTRACT_ID`): bump on any scale constant or vocab
  id-mapping change, even with the wire byte-identical; add `semantics/N.md`.
- **Arch**: bump on a model-internals change that breaks checkpoint loading; add
  an `arch/` entry; cross-reference the `checkpoint_converter` migration.

When reserving a historical slot, write the doc even if no codec is built — the
census-level summary + dead-field annotation is enough for diagnosis.

## See also

- `src/qnn/engine_norm.py` — the live contract ids + field table (the ids a new model is born with).
- `src/qnn/contracts.py` — torch-free generation→contract registry + backfill/arch-id helpers.
- `tools/export_onnx.py` — stamps the checkpoint's contract into the ONNX (+ `_wire_sig`/`_semantics_sig` fingerprints).
- `tools/stamp_checkpoint.py` — add/backfill a `meta["contract"]` block on an archived `.pth`.
- `qnn/utils/checkpoint_converter.py` — the arch migration chain + `resolve_checkpoint_contract` (load-time backfill).
- `docs/archive/` — the historical bundled snapshots (`token-spec-v*`, `obs-spec-v*`).
