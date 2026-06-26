# Wire contract `wire.9` — native split + in-graph MOVE decode (current)

The current wire contract — what `tools/export_onnx.py` produces for a
`full_4head` model at HEAD. `wire.9` = [`wire.8`](wire.8.md) + `look_delta`
(the native 44-obs split) **+** the recurrent **MOVE-decode state** threaded as
I/O, with `move` being the **decided 3-axis class** (the a24 stateful decode runs
in-graph) rather than raw logits. Stamped as `wire_contract=wire.9` in model
metadata.

> **Number reclaimed.** During active a24 development the in-graph move-decode
> shape was briefly numbered `wire.10`, distinct from an engine-side-argmax
> `wire.9`. That `wire.10` was never finalized as a release and the old
> engine-argmax `wire.9` has no surviving artifact, so the number was collapsed:
> we don't bump the wire number until a shape is finalized, so the in-graph
> migration stayed **under** `wire.9`. There is no `wire.10`. Net wire set with a
> surviving artifact: `wire.7`, `wire.9`.

- **Lineage:** `engine_norm` field-table (native split). **Band A** (faithful).
- **Semantics:** [`semantics.1`](../semantics/semantics.1.md). **Arch:** `full_4head`.
- **Codec:** built (`QNN_CODEC_WIRE_9` in `qnn_onnx.c` — the native obs pack +
  the decided-`move` / decided-weapon decode; the move difference vs the legacy
  logit-move path is keyed on the resolved **wire-version** `ctx->wire_major` —
  ACTION interpretation only, never state).
- **Artifacts:** `/tmp/qnn_v24*.onnx`, the deployed bot on `\\pi.local\qnn`,
  `runs/bc/bench/**` checkpoints.
- **Source of truth:** `src/qnn/engine_norm.py` + `NATIVE_INPUTS` /
  `_output_names` in `tools/export_onnx.py`; C side `src/engine/common/qnn_onnx.c`.

## The in-graph MOVE decode

The **a24 stateful MOVE decode** — the `fb`/`lr` sticky gate, switch-back
watermark, dwell-hazard release, stop-onset suppression, and the
continuous-weapon fire hold-tail — once ran **engine-side** (a
`qnn_onnx_decode_core` state machine + the `decode.move_*` metadata params).
It now runs **IN-GRAPH** (the export wrapper bakes
`qnn.model.decode_a24.move_decode_step_graph`, with the taus/eps/hazard table as
graph constants). The engine carries no move state machine; it just:

1. reads the **decided** `move` classes off the graph and packs the wire byte, and
2. threads the recurrent move-decode state (`move_state` / `move_state_rng`)
   frame-to-frame — but **generically**, via the [`state.loopback`](#stateloopback--generic-recurrent-state-carrying)
   contract dimension, **exactly** like the GRU `hidden` / `next_hidden` pair and
   with **zero engine knowledge** of what these tensors mean.

The legacy logit-move generation (`wire.7`) instead runs **plain per-axis
argmax** in the engine (no sticky gate) — the original pre-2026-06-10 behavior
(a17/a22 predate the sticky gate). This is gated on the resolved **wire-version
major** (`ctx->wire_major`), the ONE thing the wire version gates — **ACTION
interpretation only, never state carrying**.

## `state.loopback` — generic recurrent-state carrying

The engine does **not** know what `hidden`, `move_state`, or `move_state_rng`
mean, and does **not** sniff input names to decide what state to carry. Instead
the model stamps a **`state.loopback`** declaration into ONNX metadata, and the
engine builds a generic, opaque loop-back table from it. Each entry pairs a
recurrent INPUT with the OUTPUT that produces its next-tick value, plus an
init + reset policy:

```
in=hidden,out=next_hidden,init=zeros,reset=episode;
in=move_state,out=move_state_out,init=1 1 1 1 1 -1 -1 0 0 0 0,reset=episode;
in=move_state_rng,out=move_state_rng_out,init=entropy,reset=persist
```

(entries `;`-separated, fields `,`-separated, init-CSV lanes space-separated).
The engine, for each entry: at load allocates the buffer by the **ORT-reported
shape/dtype** of `in`, applies `init`; each tick binds the buffer as the `in`
input, runs, copies the `out` result back into the buffer; on episode reset
re-applies `init` **only where `reset=episode`**. `init=entropy` seeds once at
load from wall-clock entropy and (with `reset=persist`) keeps its stream across
episodes — the RNG case.

**Adding a future state tensor is an export/contract change with ZERO engine
change.** See the [`state.loopback` contract dimension](../README.md#state-loopback--recurrent-state)
for the full grammar and policies.

**Load validation.** The engine refuses (hard load error) if a declared
loop-back `in`/`out` is missing from the graph I/O, if a graph input is neither
an obs input nor a declared loop-back input, or if the `move` action output
disagrees with the wire-version stamp (a `wire.9` decided-`move` stamp on a
`move_logits` graph, or vice versa).

## Inputs — 44 obs (+ `hidden`) + the move-state pair

The 44 native obs tensors are the [`wire.8`](wire.8.md) native split **+**
`look_delta` (input #13, `float16 (3,)` — `look[t-1]−look[t-2]`, the realized
look-vec change, ~0 under steady turn; inference-wire-only — dropped before the
NPY cache and re-derived from the `look` column at BC preload). Per-field native
dtypes, model-side dequant; leading axis is dynamic `batch`. The full obs field
table is `src/qnn/engine_norm.py` (`SELF_FIELDS` / `SPATIAL_FIELDS` /
`ENTITY_*_FIELDS`). `hidden` (float32 (64,), GRU state in) is appended last.

Two recurrent move-decode state inputs are appended after `hidden`:

| name | dtype | shape | meaning |
|------|-------|-------|---------|
| `move_state` | float32 | (B, 11) | flat MOVE-decode state in: `[prev_move(3), dwell_age(2), swb_banned(2), swb_w(2), rng_float(unused), fire_hold(1)]` (layout = `qnn.model.decode_a24` `MOVE_STATE_DIM`) |
| `move_state_rng` | int64 | (B,) | xorshift32 state in (uint32 held in int64; a float32 round-trip would lose it) |

**Episode reset / init** are declared by the `state.loopback` entry, NOT
hardcoded: `move_state` re-inits to the lanes `[1,1,1, 1,1, -1,-1, 0,0, 0, 0]`
(bit-for-bit with `move_decode_reset_flat`) on episode reset; `move_state_rng`
is seeded **once at load** (`init=entropy`) and **persists across reset**
(`reset=persist`, so episodes don't replay the same hazard switch timings).

## Outputs (7)

| name | dtype | shape | meaning |
|------|-------|-------|---------|
| `move` | int64 | (B,3) | **DECIDED** per-axis class (fb/lr/jump) `{0:neg,1:none,2:pos}`; the a24 stateful decode ran in-graph. Engine packs these into the press byte (`QNN_PackInputMask`) |
| `look` | float32 | (B,3) | sampled look unit vector (polar mag×dir, expmap in-graph) |
| `fire_logit` | float32 | (B,1) | attack logit — engine still decodes attack (sigmoid+threshold) + the continuous-weapon hold-tail from this, for every wire format |
| `weapon` | int64 | (B,1) | decided impulse 1..8 (sticky gate in-graph, Pattern A) — OPTIONAL (present iff weapon head) |
| `next_hidden` | float32 | (B,64) | GRU state out |
| `move_state_out` | float32 | (B,11) | updated MOVE-decode state (thread back to `move_state` next tick) |
| `move_state_rng_out` | int64 | (B,) | advanced xorshift32 state (thread back to `move_state_rng`) |

`move_state_out` / `move_state_rng_out` are **distinct names** from the
`move_state` / `move_state_rng` inputs (ONNX forbids an input and output sharing
a name) — same pattern as `hidden` → `next_hidden`.

## Notes

- **`fire_logit` stays a logit, attack stays engine-side.** The in-graph decode
  computes an attack bit + hold-tail (in `move_state`) but the export discards
  that and emits `fire_logit`, so the engine remains the single attack-decode
  site across all wire formats (the f687cb1d continuous-fire hold-tail fix
  applies to every generation).
- **New-vs-legacy `move` interpretation is gated on the wire-version stamp**
  (`ctx->wire_major` in `qnn_onnx.c`): wire.9 → in-graph decided `move`; wire.7 →
  per-axis argmax of `move_logits`. State carrying is **separate** and
  fully generic (the `state.loopback` table) — the engine no longer sniffs a
  `move_state` input to decide behavior.
- The `decode.move_*` metadata (taus / eps / hazard table / stop_onset) is still
  **stamped for provenance** but the engine no longer reads it — those params
  bake into the graph as constants.
