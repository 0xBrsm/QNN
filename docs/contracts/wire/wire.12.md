# Wire contracts `wire.12.1` / `wire.12.2` — spatial depth atlas

> **Bare `wire.12` is RETIRED and must never be re-used.** The atlas grid moved
> once while this line was in flight, and artifacts of BOTH resolutions were
> stamped `wire.12` before the frontier was settled. That made the id ambiguous:
> it could not select a codec without inspecting the graph's tensor shapes,
> which is exactly the guessing the contract registry exists to forbid. The two
> resolutions are now separate contracts; the bin refuses the bare id and prints
> the re-stamp command. (Same treatment as the burned `wire.10`.)
>
> Re-stamp an older artifact with:
> ```
> python tools/stamp_onnx.py <model>.onnx --wire wire.12.1 \
>     --semantics semantics.1 --arch full_4head --model-version a26rc1a
> ```
> `--model-version` matters: without it the compact `version` string loses its
> RC tier and renders the generic arch tag.

Both contracts share `wire.11`'s action decode, recurrent-state loopbacks, self
block and entity block. They differ ONLY in the spatial observation — the same
tensor NAME (`spatial_atlas`) at a different grid resolution and packing.

| property | `wire.12.1` | `wire.12.2` |
|---|---|---|
| line | a26 rc1 (deployed) | HEAD — what the exporter produces |
| lineage | [`wire.11`](wire.11.md) plus spatial-tokens-v2 observations | same |
| semantics | [`semantics.1`](../semantics/semantics.1.md) | same |
| spatial attention tokens | 11 elevation bands | 11 elevation bands |
| atlas cells | 792: 11 bands × 72 five-degree yaw cells | 264: 11 bands × 24 fifteen-degree yaw cells |
| `spatial_atlas` tensor | (1, 11, 72) u8 — one code per byte | (1, 11, 12) u8 — two 4-bit codes per byte |
| spatial wire bytes | 792 | 132 |
| observation inputs | 34: 13 self + 1 spatial + 20 entity | same |
| codec | `QNN_CODEC_WIRE_12_1` | `QNN_CODEC_WIRE_12_2` |
| obs spec | `QNN_OBS_SPEC_ATLAS_LEGACY` | `QNN_OBS_SPEC_ATLAS_PACKED` |

The remainder of this page documents `wire.12.2`, the finalized frontier.

| property | value |
|---|---|
| fixed observation frame | 864 bytes (maximum payload 848; optional pose tail 848–863) |
| collected tick record | 896 bytes (16-byte header + 864-byte observation + 16-byte action) |
| entity stream offset | 159 |

The action decode and recurrent-state loopbacks are unchanged from
`wire.11`. The observation delta replaces the 11 raycast-scalar spatial
inputs with one input:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `spatial_atlas` | (1, 11, 12) | u8 | packed center-ray depth codes: low nibble = even yaw cell, high nibble = odd yaw cell |

The depth ladder (`engine_norm.ATLAS_DEPTH_LEVELS`) is 0, 8, 16, 24, 36,
52, 72, 100, 136, 184, 248, 336, 456, 620, 1016 units — resolution
concentrated at movement-critical near range. Adjacent yaw codes are packed
into one byte in the flat wire, collected corpus, and ONNX tensor. The engine
keeps an unpacked 24-cell row only as tick-local ray-query scratch.

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
monolithic-self graph). The yaw shape is fixed by `wire.12`; a different yaw
count requires a coordinated wire change and recollect.

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

Gate results for this final wire payload, 50 frames × 264 directions per map
(fresh production engine dumps, not offline subsampling):

| map | 3D MAE | missed obstacles | false blocks (all) | blocked early >32u | level-yaw MAE | recall at 32u | recall at 128u |
|---|---:|---:|---:|---:|---:|---:|---:|
| povdmm4 | 4.23 | 0.38% | 0.03% | 1.80% | 26.06 | 98.75% | 98.20% |
| dm2 | 3.58 | 0.53% | 0.02% | 0.45% | 10.82 | 98.20% | 96.89% |
| dm6 | 6.13 | 0.58% | 0.07% | 1.64% | 21.40 | 95.24% | 95.95% |
| combined | 4.65 | 0.50% | 0.04% | 1.29% | 19.43 | 97.61% | 96.91% |

Every threshold passes with margin. The retained 72/36/24 offline frontier
predicted this result; the table above is the required confirmation using true
24-column emission. Layout history: the five-profile supporting-plane payload
(revs 4–6) was rejected here —
extrapolating finite polygon support through broad angular volumes
reconstructs open space too conservatively; see the plan doc for the
full audit. The diagnostic does not prove complete map topology, policy
use of the features, or generalization beyond the sampled maps.

## Removed v1 inputs

`spatial_dir`, `spatial_nearest_dist`, `spatial_mean_dist`,
`spatial_openness`, `spatial_traversable`, `spatial_dropoff`,
`spatial_solid_frac`, `spatial_water_frac`,
`spatial_slime_frac`, and `spatial_lava_frac` no longer exist, as do the
interim five-profile inputs (`spatial_normal_yaw`, `spatial_normal_z`,
`spatial_clearance`).

Pre-v12 caches cannot be fed by this engine build. The retained `wire.11`
codec remains available for matching models, but `wire.11` and `wire.12`
observations are not interchangeable. A coordinated model, engine, and
recollect is required for `wire.12`.
