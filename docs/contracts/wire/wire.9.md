# Wire contract `wire.9` — native split, 44 inputs (current)

The current wire contract — what `tools/export_onnx.py` produces and what the
deployed **v24 / `full_4head`** model is. `wire.9` = [`wire.8`](wire.8.md) +
`look_delta`. Stamped as `wire_contract=wire.9` in model metadata.

- **Lineage:** `engine_norm` field-table (native split). **Band A** (faithful).
- **Semantics:** [`semantics.1`](../semantics/semantics.1.md). **Arch:** `full_4head`.
- **Codec:** built (the current `QNN_ONNX_INPUTS` table in `qnn_onnx.c`).
- **Artifacts:** `/tmp/qnn_v24*.onnx`, the deployed bot on `\\pi.local\qnn`,
  `runs/head_probe/**` checkpoints.
- **Source of truth:** `src/qnn/engine_norm.py` (`SELF_FIELDS`/`SPATIAL_FIELDS`/
  `ENTITY_*_FIELDS`) and `NATIVE_INPUTS` in `tools/export_onnx.py`; C side
  `src/engine/common/qnn_io.h` + `qnn_onnx.c`.

## Inputs — 44 obs tensors (+ `hidden`)

Per-field native dtypes; leading axis is dynamic `batch`. Per-row shape excludes
batch. This ordered list (minus `hidden`) plus the output names is the
`wire_sig` basis.

### Self block (13)
| # | name | dtype | shape | meaning |
|---|------|-------|-------|---------|
| 1 | `self_health` | uint8 | () | STAT_HEALTH (scale 100) |
| 2 | `self_effective_armor` | uint8 | () | round(raw_armor × type_factor) (scale 160) |
| 3 | `self_ammo_shells` | uint8 | () | scale 100 |
| 4 | `self_ammo_nails` | uint8 | () | scale 200 |
| 5 | `self_ammo_rockets` | uint8 | () | scale 100 |
| 6 | `self_ammo_cells` | uint8 | () | scale 100 |
| 7 | `self_vel` | int16 | (3,) | view-frame velocity, clamped ±2000 |
| 8 | `self_attack_finished` | float16 | () | cooldown seconds (scale 60) |
| 9 | `self_weapon_id` | uint8 | () | ENTITY_IDS-encoded (embedding) |
| 10 | `self_movement_id` | uint8 | () | 0 ground / 1 air / 2-4 water (embedding) |
| 11 | `self_items` | int32 | () | raw `cl.items` bitfield |
| 12 | `view_pitch` | int8 | () | pitch_deg/90, ~[-1,1] |
| 13 | `look_delta` | float16 | (3,) | look[t-1]−look[t-2]; realized look-vec change (~0 under steady turn). **`wire.9`-only** (absent in `wire.8`) |

### Spatial block (11 — 9 sectors)
| name | dtype | shape | meaning |
|------|-------|-------|---------|
| `spatial_dir` | int8 | (9,3) | view-frame unit dir (scale 127) |
| `spatial_nearest_dist` | int32 | (9,) | raw units (scale 1000); widened from u16 (tracer rejects UInt16 input) |
| `spatial_mean_dist` | int32 | (9,) | raw units (scale 1000); widened from u16 |
| `spatial_openness` | uint8 | (9,) | [0,1] (scale 255) |
| `spatial_clearance` | uint8 | (9,) | scale 255 |
| `spatial_traversable` | uint8 | (9,) | scale 255 |
| `spatial_dropoff` | uint8 | (9,) | scale 255 |
| `spatial_solid_frac` | uint8 | (9,) | scale 255 |
| `spatial_water_frac` | uint8 | (9,) | scale 255 |
| `spatial_slime_frac` | uint8 | (9,) | scale 255 |
| `spatial_lava_frac` | uint8 | (9,) | scale 255 |

### Entity block (20 — 16 slots)
| name | dtype | shape | meaning |
|------|-------|-------|---------|
| `entity_types` | int8 | (16,) | TOKEN_* tag, −1 empty |
| `entity_subject_id` | uint8 | (16,) | ENTITY_IDS (embedding) |
| `entity_modality_id` | uint8 | (16,) | MODALITY_IDS (embedding) |
| `entity_player_id` | uint8 | (16,) | actor identity (embedding) |
| `entity_event_count` | uint8 | (16,) | 0..4 |
| `entity_event_actions` | uint8 | (16,4) | ACTION_IDS (embedding) |
| `entity_event_sources` | uint8 | (16,4) | ENTITY_IDS (embedding) |
| `entity_half_extents` | uint8 | (16,3) | bbox half-sizes (scale 1000, saturating) |
| `entity_rel` | int16 | (16,3) | view-frame position (scale 1000) |
| `entity_vel` | int16 | (16,3) | view-frame velocity (clamped ±2000) |
| `entity_path` | int16 | (16,3) | navmesh waypoint dir (scale 1000) |
| `entity_path_dist` | int32 | (16,) | path length (scale 1000); widened from u16 |
| `entity_eta` | float16 | (16,) | seconds (scale 60) |
| `entity_recency` | float16 | (16,) | seconds (scale 60) |
| `entity_facing` | uint8 | (16,) | [0,1] (scale 255) |
| `entity_team` | uint8 | (16,) | {0,1} |
| `entity_score` | uint8 | (16,) | [0,1] (scale 255) |
| `entity_amount` | uint8 | (16,) | raw pickup amount (item-amount transform) |
| `entity_regen` | float16 | (16,) | seconds (scale 60) |
| `entity_state` | uint8 | (16,) | [0,1] (scale 255) |

### State tensor
`hidden` — float32 (64,) — GRU recurrent state in. Appended last. The GRU width
(64) is an **arch** property, not part of `wire_sig`.

## Outputs (5)
| name | dtype | shape | meaning |
|------|-------|-------|---------|
| `move_logits` | float32 | (B,3,3) | per-axis (fb/lr/ud) logits; Gumbel-perturbed in the sampled export so the bin's argmax = a sample |
| `look` | float32 | (B,3) | sampled look unit vector (polar mag×dir Gumbel-sampled + expmap in-graph) |
| `fire_logit` | float32 | (B,1) | attack logit (name retained through the fire→attack rename) |
| `weapon_logits` | float32 | (B,8) | 8 weapon classes (impulse 1..8 → class 0..7) |
| `next_hidden` | float32 | (B,64) | GRU state out |

## Notes
- **`look_delta` is inference-wire-only.** It is emitted on the live/ONNX wire
  but **dropped before the NPY cache** (`qnn.bc.collect`); BC preload re-derives
  it from the `look` column, so training and inference compute the identical
  quantity and the cache schema is unaffected.
- The `wire.8`→`wire.9` delta is *exactly* `+look_delta` (verified against
  commit `7147e8f4` and the live `/tmp/qnn_v24.onnx` graph) — nothing else
  changed, and outputs are identical.
