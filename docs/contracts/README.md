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

**Semantics also pins the tick, not just the value.** The timeliness clause
([`semantics/semantics.1.md`](semantics/semantics.1.md), "Temporal alignment")
requires every protocol frontend to deliver server-time-aligned self-state —
`obs(t)` reflects own commands through t−1, the training alignment. It is
behavioral (not in `semantics_sig`), so it is enforced per frontend by a
conformance layer (NQ: `qnn_predict.c`; QW: native prediction) plus a
certification capture, not at model load.

**Two more things ride the export but are NOT versioned axes.** The **decode
config** (the decode/guard *behavior* layer — sampling, polar look, sticky
weapon, aim-prior, hazard) and **`state.loopback`** (recurrent-state carrying)
are both stamped into the ONNX at export time for provenance, but neither is a
contract axis: decode is decoupled from training and A/B'd by pointing at a
different JSON (no id, no retrain), and `state.loopback` rides the wire
contract. See [Decode config](#decode-config--the-decodeguard-layer) and
[`state.loopback`](#state-loopback--recurrent-state).

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

## `state.loopback` — recurrent state {#state-loopback--recurrent-state}

A **fourth, declarative** stamp (not a versioned axis — it rides the wire
contract) tells the engine how to carry **recurrent state** across ticks
*generically and opaquely*. The engine has **zero semantic knowledge** of any
state tensor: it does not special-case `hidden` / `move_state` / `move_state_rng`,
and it does **not** sniff input names to decide what to carry. It reads this
declaration and builds a generic loop-back table.

`tools/export_onnx.py` stamps `state.loopback` into ONNX metadata; the engine
parses it at load (`qnn_loopback_parse` in `src/engine/common/qnn_onnx.c`).

**Grammar.** Entries `;`-separated; fields `,`-separated `key=value`; init-CSV
lanes space-separated:

```
in=<input> , out=<output> , init=<policy> , reset=<policy>   ; …next entry…
```

| field | values | meaning |
|-------|--------|---------|
| `in`  | tensor name | the recurrent INPUT the engine binds each tick from its carried buffer |
| `out` | tensor name | the OUTPUT whose result is copied back into `in`'s buffer for next tick (distinct name — ONNX forbids an input/output sharing a name) |
| `init` | `zeros` \| `entropy` \| `<csv lanes>` | buffer init at load (and on episode reset where `reset=episode`). `entropy` = seed once at load from wall-clock entropy; `<csv>` = write the space-separated lane values (tiled across the buffer) |
| `reset` | `episode` \| `persist` | re-apply `init` on episode reset (`episode`) or keep the carried value across episodes (`persist`, e.g. an RNG stream) |

The engine, per entry: at load allocates the buffer by the **ORT-reported
shape/dtype** of `in` and applies `init`; each tick binds the buffer as the `in`
input, runs, copies the `out` result back; on episode reset re-applies `init`
only where `reset=episode`.

**Adding a future state tensor is an export/contract change with ZERO engine
change** — stamp another entry, no C edit.

**Load validation (hard refusal).** The engine refuses if a declared `in`/`out`
is missing from the graph I/O, if a graph input is neither an obs input nor a
declared loop-back `in`, or if the `move` action output disagrees with the
wire-version stamp (decided `move` vs raw `move_logits`). The wire version gates
**only** that action interpretation — never state carrying.

The current HEAD (`wire.12`, inherited unchanged from `wire.11`) declaration is:

```
in=hidden,out=next_hidden,init=zeros,reset=episode;
in=move_state,out=move_state_out,init=1 1 1 1 1 -1 -1 0 0 0 0,reset=episode;
in=move_state_rng,out=move_state_rng_out,init=entropy,reset=persist
```

See [`wire/wire.9.md`](wire/wire.9.md#stateloopback--generic-recurrent-state-carrying).

## Decode config — the decode/guard layer {#decode-config--the-decodeguard-layer}

Decode is the model's **behavioral readout** — how raw head outputs become an
action: sampling vs argmax, the polar-look hybrid, the sticky-weapon gate, the
aim-prior blend, the move hazard/dwell supplement. It is **not a contract axis**
and carries no `wire`/`semantics`/`arch`-style id, for one reason: **decode is
decoupled from training** (you never backprop through it), so it is an
EXPORT-time artifact, not part of the trained graph. You A/B a decode change by
pointing the exporter at a different JSON — no retrain, no model-code change.

A decode config (a JSON resolved by `src/qnn/model/decode_config.py`, shipped
with its generation) is the self-describing, run-pinned record of that layer:

| field | meaning |
|-------|---------|
| `decode_module` | dotted import path — the gen-coupled decode geometry (must read the head params) |
| `guard_module`  | dotted path \| `"none"` — exposes `guard_attack_logit_for_export` + `policy_decode_action_postprocess` |
| `params`        | flat `str→scalar\|list` map (`look.*`, `weapon.*`, `guard.*`, `weapon_ban`) |
| `look_grid`     | run-relative path to the polar look grid (null = code default) |
| `move_hazard`   | path to the hazard/dwell table JSON (null = none / head-driven) |

**Two numbers, distinct** (this is the usual point of confusion):
`decode_version` is the **schema** version of the JSON format itself (currently
`1`); `version` is a **named build id** for provenance only (e.g. `a24rc2`),
tied to the arch lineage — it is not enforced and never gates a load.

**Provenance.** The exporter stamps the resolved config's **sha256 + the repo
git sha** into the ONNX `metadata_props` (under the `decode.` namespace), so the
exact decode/guard source of any shipped model is recoverable. The one
training-fixed constraint `decode_config` enforces at resolve time: the look
grid's **bin count** must equal the head's trained output width (center values
are free to tune; the count is not — a count mismatch is an incompatible
decode/checkpoint pairing).

**Where the decode CODE lives** (a cross-gen split, not a version): the
cross-gen-stable readout primitives (sampling, attack-bit, per-axis move) are in
`src/qnn/model/decode.py`; each generation's *choices* (its decode geometry and
any guard) live in that generation's own decode module, resolved through the
config's `decode_module` / `guard_module`. A future generation replaces the gen
module, not the base.

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
| (a24) | native split + in-graph MOVE decode | `wire.9` | `semantics.1` | `full_4head` | v24 | superseded by `wire.11` | A |
| a24 | native split + in-graph MOVE **and ATTACK** decode | **`wire.11`** | **`semantics.1`** | `full_4head` | **v24** | **Yes** | **A** |
| a26 | 72×11 **unpacked** spatial depth atlas + learned band IDs | **`wire.12.1`** | **`semantics.1`** | `full_4head` | `20260722-98wtxv` | **Yes** | A |
| HEAD | 24×11 **packed** spatial depth atlas + learned band IDs | **`wire.12.2`** | **`semantics.1`** | `full_4head` | `20260722-98wtxv` | **Yes** | A |

Distinct contracts across the full history: **12 wire × 5 semantics**. With a
surviving runnable artifact: **5 wire** (`wire.7`, `wire.9`, `wire.11`,
`wire.12.1`, `wire.12.2`) **× 1 semantics** (`semantics.1`) — and all five have a
registered codec in the bin (`QNN_CODECS` in `src/engine/common/qnn_onnx.c`), so
one client serves the whole load set. `wire.8` is a reconstructed id — the native exporter postdates
`look_delta`, so no 43-input graph was ever exported. `wire.11` = `wire.9` (native
44-obs split + in-graph decided `move`/`weapon` + the move-decode state pair) **+
the in-graph ATTACK decode** (decided `attack` bit instead of `fire_logit`, plus
the `attack_state` hold-tail loop-back) — so ALL four actions decode in-graph and
the engine is decode-agnostic. `wire.11` REPLACES `wire.9` for the a24 gen
(re-export); `wire.10` stays burned. The legacy `wire.7` logit path is unchanged.
See [`wire/wire.11.md`](wire/wire.11.md). Spatial-v2 replaces the spatial
observation block with a single center-ray depth atlas. The atlas GRID moved
once while the line was in flight and BOTH resolutions have deployed artifacts,
so each is its own contract — see [`wire.12.md`](wire/wire.12.md):

| id | atlas grid | on the wire | line |
|---|---|---|---|
| `wire.12.1` | 11 bands × 72 yaw cells | 792 B, one 4-bit code per byte | a26 rc1 |
| `wire.12.2` | 11 bands × 24 yaw cells | 132 B, two codes per byte | HEAD |

`wire.12.2` is the finalized frontier: it passed a fresh production-engine
reconstruction gate after the earlier supporting-plane layouts failed.

> **Bare `wire.12` is RETIRED — never re-use it.** Both families were stamped
> `wire.12` before the frontier was settled, so the id cannot select a codec
> without inspecting tensor shapes. The bin briefly did exactly that; it now
> refuses the bare id and names the fix. Re-stamp with
> `tools/stamp_onnx.py --wire wire.12.1|wire.12.2 --model-version <tier>`.
> The retired set is `QNN_RETIRED_WIRES` in `qnn_onnx.c`; a parity test
> (`tests/test_engine_norm_parity.py`) asserts no id is both registered and
> retired, and that every id Python can stamp has a codec.

> **`wire.9` number reclaimed.** During active a24 development the in-graph
> move-decode shape was briefly numbered `wire.10`, distinct from an
> engine-side-argmax `wire.9`. That `wire.10` was never finalized as a release
> and the old engine-argmax `wire.9` has no surviving artifact, so the number
> was collapsed — we don't bump the wire number until a shape is finalized, so
> the in-graph migration stayed **under** `wire.9`. There is no `wire.10`.

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
| **A — faithful** | `wire.7`→`wire.12` (v11→HEAD) | none | Covers every surviving artifact plus spatial-v2 HEAD (v17, v22, v24, `full_4head`) | surviving artifacts: `wire.7`, `wire.11`, `wire.12` |
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
- `tools/export_onnx.py` — `build_contract_manifest()` assembles the flat manifest (the checkpoint's contract ids + `_wire_sig`/`_semantics_sig` fingerprints); `_stamp_metadata()` writes it into the ONNX metadata (authoritative) and re-renders it via `_contract_document()` into a structured `<model>.contract.json` sidecar (the human-readable three-axis view — `wire`/`semantics`/`arch` sections, each with `id` + `sig`; cannot drift, it is a pure re-render of the stamped manifest).
- `src/qnn/model/decode_config.py` — resolves a decode-config JSON → decode/guard modules + params; computes the config sha256 the exporter stamps. NOT a contract axis (decode is decoupled from training). The per-generation config templates ship with their generation.
- `tools/stamp_checkpoint.py` — add/backfill a `meta["contract"]` block on an archived `.pth`.
- `qnn/utils/checkpoint_converter.py` — the arch migration chain + `resolve_checkpoint_contract` (load-time backfill).
- `research/archive/` — the historical bundled snapshots (`token-spec-v*`, `obs-spec-v*`).
