# Wire contract `wire.11` — native split + in-graph MOVE **and ATTACK** decode (current)

The current wire contract — what `tools/export_onnx.py` produces for a
`full_4head` model at HEAD. `wire.11` = [`wire.9`](wire.9.md) **+ the in-graph
ATTACK decode**: the `attack` output is the **decided fire/no-fire bit** (int64
`(B,1)`) rather than the raw `fire_logit` float, and the recurrent **attack-decode
state** (`attack_state`) is threaded as I/O.
Stamped as `wire_contract=wire.11` in model metadata.

> **CORRECTION (2026-08-26) — the hold-tail was never in-graph.** This doc
> originally described `attack_state` as carrying the continuous-weapon
> **hold-tail**, and said the engine runs none of its own. Neither was ever true
> of the shipped code: `qnn_onnx_apply_continuous_hold_tail` has run
> **engine-side for every wire generation**, `wire.11` included, and
> `attack_state` today carries `weapon.af_lockout` (4 lanes: `y`,
> `locked_weapon`, `af_prev`, `dt` — `ATTACK_STATE_DIM`). The in-graph attack
> decode is memoryless. The tail is now gated per model by the
> `decode.attack.hold_tail_sec` stamp (0 = off, the default for every fresh
> export); an artifact exported before that key existed — every `wire.11`
> artifact — inherits the historical 0.25 s. Sentences below that assert
> otherwise are left as written for the record; this note supersedes them.

> **`wire.11` REPLACES `wire.9`.** The two share the native obs format and the
> in-graph move/weapon decode; they differ in exactly one output (`fire_logit` →
> decided `attack`). The whole a24-rc generation migrates by re-export
> (coordinated model+engine deploy) — there is no `wire.9`/`wire.11` coexistence.
> `wire.10` stays **burned** (it was a never-released in-graph-move number during
> a24 dev; never reuse). Net live wire set: `wire.7`, `wire.11`.

- **Lineage:** `engine_norm` field-table (native split). **Band A** (faithful).
- **Semantics:** [`semantics.1`](../semantics/semantics.1.md) (unchanged — attack
  was never a semantics item: no scale, no vocab). **Arch:** `full_4head`.
- **Codec:** `QNN_CODEC_WIRE_11` in `qnn_onnx.c` — the native obs pack (shared
  `wire9_emit`) + the decided-`move`/`weapon`/`attack` decode; the attack
  difference vs `wire.9` (decided bit vs engine-side `sigmoid(fire_logit)>thr`) is
  keyed on the resolved **wire-version** `ctx->wire_major >= 11` — ACTION
  interpretation only, never state.
- **Source of truth:** `src/qnn/engine_norm.py` (`WIRE_CONTRACT_ID`) +
  `_output_names`/`_state_loopback_decl` in `tools/export_onnx.py`; C side
  `src/engine/common/qnn_onnx.c`.

## The in-graph ATTACK decode

> **CORRECTION (2026-08-27) — this section describes the retired a24 decode
> law, not what HEAD exports.** `tools/export_onnx.py` now refuses any decode
> config whose `decode_module` is not `qnn.model.decode_actions` — a24's
> Bernoulli attack step is unreachable from export. The a25/a28 attack decode
> (`qnn.model.decode_actions.attack_with_decode` /
> `attack_with_decode_step`) is a **deterministic, greedy** joint argmax over
> the 9-way `attack_logits` (class 0 = no-attack, 1..8 = attack that weapon),
> not a Bernoulli draw, and `attack.bias`/`attack_bias` no longer exists (see
> [`decode-config-defaults.md`](../../decode-config-defaults.md)). `attack_rng`
> is still threaded as I/O for wire/state parity but is **INERT** — nothing
> reads it — per `qnn.model.decode_actions.attack_decode_reset_flat`'s own
> docstring. `attack_state`'s 4 lanes carry `weapon.af_lockout`, as the note
> above already says. The paragraph below is left as written for the
> historical record of what `wire.11` shipped with at a24; it does not
> describe the current decode law.

The attack decode is **SAMPLED**: `attack_bit ~ Bernoulli(sigmoid((fire_logit +
attack_bias) / temp))` off attack's **own xorshift rng** (`attack_rng`), plus the
continuous-weapon fire **hold-tail**. Attack was historically eval-sampled but
deploy-greedy (the engine ran `sigmoid > decode.attack_threshold`, an
inconsistency) — `wire.11` finally makes deploy **sample** to match eval, the
settled regime (`sigmoid` is the BCE-learned P(attack|state); human attack is
stochastic, a hard threshold is robotic). It runs **IN-GRAPH**
(`qnn.model.bench.a24.decode.attack_decode_step_graph`, baked by the export
wrapper with `attack_bias`/`temperature` as graph constants, the draw off
`attack_rng`, the hold-tail in `attack_state`). The competence guard still drives
unsafe-fire rows' logit far below the cut **before** the in-graph sigmoid, so
guarded rows decode to `attack=0`. The engine just:

1. reads the **decided** (sampled) `attack` bit and ORs it into the Quake press
   byte (`QNN_PackInputMask`, bit 0), and
2. threads the recurrent `attack_state` (hold-tail, reset=episode) + `attack_rng`
   (the draw stream, init=entropy/reset=persist) frame-to-frame **generically**
   via the [`state.loopback`](wire.9.md#stateloopback--generic-recurrent-state-carrying)
   contract dimension — exactly like `hidden`/`move_state`/`move_state_rng`, with
   **zero engine knowledge** of what they mean (a pure export/contract change).

`attack_threshold` is **retired** at `wire.11` (there is no engine-side threshold
to configure); `attack.bias` stays and is applied inside the in-graph decode. The
legacy `wire.7` generation still decodes attack engine-side from `fire_logit`
(gated `ctx->wire_major < 11`) — unchanged.

## Why this is the terminal decode-driven wire

With `wire.11`, **all four action heads** (`move`, `look`, `weapon`, `attack`)
are decided in-graph and the graph emits decided outputs the engine reads
verbatim. The engine carries **no action decode** beyond the legacy `wire.7`
path — it is decode-agnostic. Consequently a **decode-regime** change (attack
temperature, gain/α tuning, guard changes, even attack greedy↔sampled — the draw
already rides `attack_rng` in-graph) rides entirely in the ONNX graph and **does
not bump the wire** — the I/O signature is stable. `wire.11` is the last
decode-driven wire bump.

## Outputs (sampled export)

```
move             int64 (B,3)   decided fb/lr/jump class {0:neg,1:none,2:pos}
look             float (B,3)   unit look vector
attack           int64 (B,1)   DECIDED (sampled) fire bit {0,1}  ← was fire_logit (float)
weapon           int64 (B,1)   decided impulse 1..8              (weapon head present)
next_hidden          …         GRU loop-back
move_state_out   float (B,9)   move-decode loop-back             (was (B,11) at wire.9)
move_state_rng_out   …         move-decode rng loop-back
attack_state_out float (B,1)   attack hold-tail loop-back        ← NEW
attack_rng_out   int64 (B,)    attack sampling-rng loop-back     ← NEW
```

`state.loopback` adds
`in=attack_state,out=attack_state_out,init=zeros,reset=episode` and
`in=attack_rng,out=attack_rng_out,init=entropy,reset=persist`.

## Status

Model/export/contract side: **done + verified** (export stamps `a24rc1q.s1.w11`,
emits decided `attack`, ORT trunk parity holds; move+attack decode-parity green).
Engine codec (`QNN_CODEC_WIRE_11`): implemented, **pending a live trainer check**
(build the client with `src/engine/build/build_nq_client.sh`).
