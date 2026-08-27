# Wire contracts `wire.13.1` / `wire.13.2` — depth atlas + A27 pure-combat stream

> **Bare `wire.13` is RETIRED and must never be re-used.** The atlas grid moved
> while this line was in flight — exactly as it did one contract earlier on
> [`wire.12`](wire.12.md) — and artifacts of BOTH resolutions were stamped
> `wire.13` before the frontier was settled, including the deployed `a27rc1a`.
> That made the id ambiguous: it could not select a codec without inspecting
> the graph's tensor shapes, which is exactly the guessing the contract
> registry exists to forbid. The two resolutions are now separate contracts;
> the bin refuses the bare id and prints the re-stamp command. (Same treatment
> as bare `wire.12` and the burned `wire.10`.)
>
> Re-stamp an older artifact with:
> ```
> python tools/stamp_onnx.py <model>.onnx --wire wire.13.1 \
>     --semantics semantics.2 --arch full_movearch --model-version a27rc1a
> ```
> `--model-version` matters: without it the compact `version` string loses its
> RC tier and renders the generic arch tag.

`wire.13.x` is a DISTINCT wire line from [`wire.12.x`](wire.12.md): a26
reclaimed wire.12 for the depth atlas + FULL entity stream (`semantics.1`), so
the A27 pure-combat shape moved off that number. All of wire.11, wire.12.1,
wire.12.2, wire.13.1 and wire.13.2 are live codecs in the one `nq_client` bin
(see [Coexisting contracts](#coexisting-contracts)).

Both `wire.13` contracts share the same action decode, recurrent-state
loopbacks, self block and combat entity block. They differ ONLY in the spatial
observation — the same tensor NAME (`spatial_atlas`) at a different grid
resolution and packing.

| property | `wire.13.1` | `wire.13.2` |
|---|---|---|
| line | a27 rc1 — `a27rc1a` (deployed) | HEAD — what the exporter produces |
| lineage | [`wire.11`](wire.11.md) action decode + the [`wire.12`](wire.12.md) depth atlas + the A27 pure-combat entity stream | same |
| semantics | [`semantics.2`](../semantics/semantics.2.md) | same |
| spatial attention tokens | 11 elevation bands | 11 elevation bands |
| atlas cells | 792: 11 bands × 72 five-degree yaw cells | 264: 11 bands × 24 fifteen-degree yaw cells |
| `spatial_atlas` tensor | (1, 11, 72) u8 — one code per byte | (1, 11, 12) u8 — two 4-bit codes per byte |
| spatial wire bytes | 792 | 132 |
| observation inputs | 29: 12 self + 1 spatial + 16 entity | same |
| codec | `QNN_CODEC_WIRE_13_1` | `QNN_CODEC_WIRE_13_2` |
| obs spec | `QNN_OBS_SPEC_COMBAT_ATLAS_LEGACY` | `QNN_OBS_SPEC_COMBAT_ATLAS_PACKED` |

The remainder of this page documents `wire.13.2`, the finalized frontier.

| property | value |
|---|---|
| entity stream offset | 158 |

The recurrent-state loopbacks are inherited from `wire.11`. The action output
is deliberately different: `wire.13.x` replaces the separate decided binary
`attack` and decided `weapon` outputs with one categorical `attack` output.
The observation delta replaces the 11 raycast-scalar spatial inputs with one
input:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `spatial_atlas` | (1, 11, 12) | u8 | packed center-ray depth codes: low nibble = even yaw cell, high nibble = odd yaw cell; 0–14 index the log depth ladder, 15 = no hit within the band's range |

## Action output

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `move` | (B, 3) | i64 | decided forward/right/up classes in `0..2` |
| `look` | (B, 3) | f32 | decided view-relative look vector |
| `attack` | (B, 1) | i64 | `0` = no attack; `1..8` = select and fire that Quake impulse |

There is no `weapon` output and no parallel binary fire output. The engine
derives both button 0 and the impulse from `attack`. Logits-level parity export
uses `attack_logits (B, 9)` for the same categorical head.

## A27 combat entity stream

The entity stream admits only `PROJECTILE=0` and `ACTOR=1`. Item and mover
rows are absent; their static engine state remains available to geometry,
events, rewards, and future slower layers. Both combat types carry a modality
ID from the two-row combat vocabulary: `SIGHT=0`, `PROXIMITY=1`.

Both channels are current-frame observations. `SIGHT` means the entity is in
the configured view cone and passes the world trace. `PROXIMITY` means it is
in the current engine PVS but did not qualify for SIGHT. SIGHT wins on overlap.
The POC source for PROXIMITY is engine ground truth; a higher layer may later
replace that producer without changing this wire.

The variable per-token scalar payloads are:

| token | wire payload | bytes | model scalar width |
|-------|--------------|------:|-------------------:|
| projectile | `rel[3] i16`, `vel[3] i16` | 12 | 7, including derived `dist` |
| actor | `half_extents[3] u8`, `rel[3] i16`, `vel[3] i16`, `path[3] i16`, `path_dist u16`, `eta f16`, `facing u8`, `team u8`, `score u8` | 28 | 18, including derived `dist` |

There is no recency input and no item/mover-only `amount`, `regen`, or `state`
input. The model's actor and projectile projections are both live attention
inputs; only actor rows are valid target-pointer candidates.

The depth ladder (`engine_norm.ATLAS_DEPTH_LEVELS`) is 0, 8, 16, 24, 36,
52, 72, 100, 136, 184, 248, 336, 456, 620, 1016 units — resolution
concentrated at movement-critical near range. Adjacent yaw codes are packed
into one byte in the flat wire, collected corpus, and ONNX tensor. The engine
keeps an unpacked 24-cell row only as tick-local ray-query scratch. (`wire.13.1`
stores the same codes one per byte, unpacked.)

## Grid identity

Rows are elevation bands centered at −75° … +75° in 15° steps
(`qnn.vocab.SPATIAL_BAND_IDS`, `Elev_n75` … `Elev_p75`); columns are 15°
yaw cells counter-clockwise from the player's view yaw, cell 0 =
forward, tiling the full circle. The frame is yaw-only; pitch stays
self-state (`view_pitch`). Each band's radial range is
min(1024, 128/|sin elev|) — the same 1024-unit horizontal / 128-unit
vertical contract as v1. Band identity is fixed wire order plus a
learned 11-entry band-ID embedding model-side; yaw-cell identity is
scalar position inside the band token's projection.

Coverage is deliberately **not** the full sphere: cells reach ±82.5°
elevation, leaving two unsampled 7.5°-half-angle polar cones (the v1
exact ground/ceiling columns were retired with the sector layout). The
standing floor is still observable — hull-1 contact reads as depth 0
across the entire −75° band — but geometry confined to within ~7.5° of
straight up/down is invisible to this block, and the reconstruction
gate's truth field samples the same band centers, so it does not score
the poles either.

## Model-facing values

`SpatialDequantizer` expands each packed band to 48 float scalars:

```text
[depth / 1000 × 24 cells, hit × 24 cells]
```

Decoded depth is clamped to the band's range limit, and a miss decodes
to exactly that limit with `hit = 0` — quantization can never place a
surface past the instrument's range. This yields 11 spatial attention
tokens (sequence 28 = 1 self + 11 spatial + 16 entity for the
monolithic-self graph). The yaw shape is fixed by the contract; a different
yaw count requires a coordinated wire change and recollect — which is precisely
why `wire.13.1` and `wire.13.2` are two contracts and not one.

## Geometry source

The world hull 1 is carved once per map into its complete solid-boundary
face set. Each cell is the exact first intersection of the cell's center
direction with that face set (`QNN_CarveQueryRay`) — a projection of
known geometry, not a discovery trace: no BSP traversal, no clipping
degeneracy, and a face plane through the origin (the standing player's
floor) hits at distance zero when entered. Brush movers are carved once
in local space and translated to their live origins. Hull 1 already
includes the player's collision-box expansion, so depths are exact
player clearances. The XY-grid broad-phase is byte-purity-proven against
the linear scan (`QNN_SPATIAL_LINEAR=1` disables it; identical sidecars).

## Reconstruction diagnostic

Set `QNN_SPATIAL_DIAG` during demo playback to write the production
quantized codes (and pre-quantization distances) beside independent
dense hull-1 traces on the same grid. The scorer reconstructs
directional depth from the codes alone:

```bash
python -m qnn.diag.spatial_reconstruction run \
  --worker assets/bin/qw_demo_worker --assets assets \
  --demo path/to/demo.qwd --sidecar records.jsonl

python -m qnn.diag.spatial_reconstruction score \
  povdmm4.jsonl dm2.jsonl dm6.jsonl \
  --max-mae 25 --max-missed-obstacle-rate 0.03 --max-level-mae 110 \
  --max-false-block-rate-all 0.01 \
  --max-blocked-early-gt-32-rate 0.03
```

Layout `atlas` scores the production codes (the gate target);
`atlas_float` scores the pre-quantization distances (the
representation's upper bound, isolating quantization loss). False blocks
report both denominators (`…_given_open` and `…_all`).

The atlas payload is byte-identical to `wire.12.x`'s, so the gate evidence is
NOT re-derived here — see the two result tables in
[`wire.12.md`](wire.12.md#reconstruction-diagnostic): the 792-direction table
is the `wire.13.1` payload and the 264-direction table is the `wire.13.2`
payload. Every gate threshold passes with margin on both. Layout history: the
five-profile supporting-plane payload (revs 4–6) was rejected here —
extrapolating finite polygon support through broad angular volumes
reconstructs open space too conservatively; see the plan doc for the full
audit. The diagnostic does not prove complete map topology, policy use of the
features, or generalization beyond the sampled maps.

## Removed v1 inputs

`spatial_dir`, `spatial_nearest_dist`, `spatial_mean_dist`,
`spatial_openness`, `spatial_traversable`, `spatial_dropoff`,
`spatial_solid_frac`, `spatial_water_frac`,
`spatial_slime_frac`, and `spatial_lava_frac` no longer exist, as do the
interim five-profile inputs (`spatial_normal_yaw`, `spatial_normal_z`,
`spatial_clearance`).

## Coexisting contracts

`wire.13.1` and `wire.13.2` are registered ALONGSIDE
[`wire.12.1`/`wire.12.2`](wire.12.md) (depth atlas + FULL entity stream,
`semantics.1`), [`wire.11`](wire.11.md) and [`wire.9`](wire.9.md) (v1 raycast
spatial, `semantics.1`) and the legacy [`wire.7`](wire.7.md) in the one
`nq_client` bin — seven codecs, so one client serves v17/v22, a24, a24/a25, a26
and a27 artifacts.

The loaded model's `wire_contract` stamp selects the codec (`qnn_onnx.c`
`qnn_codec_by_id`) and NOTHING else: selection is a pure id lookup, never an
inspection of the graph's tensor shapes. That is why the bare ids are retired
rather than sniffed. The selected codec's OBS SPEC then declares both what the
engine computes and what the packer fills: `spatial` (ATLAS for `wire.13.2`,
ATLAS_LEGACY for `wire.13.1`) and `entity` (COMBAT for both) are pushed at load
via `QNN_IOSetSpatialMode` / `QNN_IOSetEntityMode`, so `QNN_IOEmit` and the
oracle produce EXACTLY this contract's obs block per tick.

A wire.13 model needs a bin that registers these codecs; bins predating them
will refuse the load loudly (no silent misread). The `amount`/`regen`/`state`
and `recency` fields still exist as engine state for geometry, events, rewards
and the wire.9/.11/.12.x codecs — they are simply not part of the wire.13
combat obs.
