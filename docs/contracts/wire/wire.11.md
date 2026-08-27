# Wire contract `wire.11` — native split + in-graph MOVE **and ATTACK** decode (current)

The current wire contract — what `tools/export_onnx.py` produces for a
`full_4head` model at HEAD. `wire.11` = [`wire.9`](wire.9.md) **+ the in-graph
ATTACK decode**: the `attack` output is the **decided fire/no-fire bit** (int64
`(B,1)`) rather than the raw `fire_logit` float, and the recurrent **attack-decode
state** (`attack_state` — the continuous-weapon hold-tail) is threaded as I/O.
Stamped as `wire_contract=wire.11` in model metadata.

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
