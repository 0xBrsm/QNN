# Wire contract `wire.7` — packed scalars, 13 inputs (legacy, v17/v22)

The packed `obs_buffer_v1` wire as of token-spec ~v11, consumed by the **v17 and
v22** checkpoints. This is the **faithful-load floor**: it is the oldest contract
that can still be reconstructed from the current engine (see Loadability). A
codec for `wire.7` is a **Band A build target** so v17/v22 play through the same
bin as the current model.

- **Lineage:** `engine_norm` field-table, *packed float32* form (pre native split).
- **Semantics:** [`semantics.1`](../semantics/semantics.1.md) (unchanged from `wire.9`).
- **Arch:** v11 packed generation.
- **Codec:** not yet built (Band A target).
- **Artifacts:** `/tmp/qnn_v17.onnx`, `/tmp/qnn_v22.onnx`; both on `\\pi.local\qnn`.
- **Authoritative source:** the deployed v17/v22 ONNX graphs (tensor signature)
  + the Python packed layout `wire.py:SELF_FIELDS`/`OBS_SCHEMA` at commit
  `b0f75210^`. **Not** the C `qnn_onnx.c` at that commit — that file was already
  the half-migrated native path (references struct fields that don't exist yet).

## Inputs — 13 tensors (v17 and v22 byte-identical)
| # | name | dtype | shape | meaning |
|---|------|-------|-------|---------|
| 0 | `self_scalars` | float32 | (17,) | health, armor, 7 weapon one-hots (SG,SSG,NG,SNG,GL,RL,LG), ammo×4 (shells,nails,rockets,cells), vel[3] view-frame, attack_finished(s). Slot 16 = attack_finished |
| 1 | `self_weapon_id` | int64 | (1,) | held weapon, **ENTITY_IDS-encoded** (NONE=0…THUNDERBOLT=10), not impulse |
| 2 | `self_armor_type_id` | int64 | (1,) | 0/green/yellow/red |
| 3 | `self_movement_id` | int64 | (1,) | 0 ground / 1 air / 2-4 water |
| 4 | `self_powerup_ids` | int64 | (5,) | active powerup subject-ids, zero-padded |
| 5 | `entity_types` | int64 | (16,) | per-slot {PROJECTILE 0, ACTOR 1, ITEM 2, MOVER 3}, −1 empty |
| 6 | `entity_scalars_raw` | float32 | (16,19) | per-slot raw scalars, zero-padded to 19 (ACTOR width) |
| 7 | `entity_ids` | int64 | (16,3) | (subject_id, modality_id, player_id); player_id ACTOR-only |
| 8 | `entity_event_actions` | int64 | (16,4) | ACTION_IDS per event |
| 9 | `entity_event_sources` | int64 | (16,4) | ENTITY_IDS source per event |
| 10 | `entity_event_counts` | int64 | (16,) | valid events per slot (0-4) |
| 11 | `spatial_scalars` | float32 | (9,13) | 9 sectors × (dir3, nearest_dist/1000, mean_dist/1000, openness, clearance, traversable, dropoff, solid/water/slime/lava_frac) |
| 12 | `hidden` | float32 | (64,) | GRU recurrent state in |

## Outputs (5)
Same 5 as `wire.9`: `move_logits` (B,3,3), `look` (B,3), `fire_logit` (B,1),
`weapon_logits` (B,8), `next_hidden` (B,64). **v17 vs v22:** v22's
`weapon_logits` second dim is concrete `8`; v17's is a symbolic export artifact —
semantically 8 either way. **All inputs are byte-identical between v17 and v22.**

## Loadability — faithful (zero dead fields)
Every `wire.7` input is still emitted or derivable from the current
`qnn_snapshot_t`, with the encoding (`semantics.1`) unchanged:
- weapon one-hots ← `self_items` bits 0–6,12 (`ITEMS_WEAPON_MASK`)
- `self_armor_type_id` ← `self_items` bits 13–15 (`ITEMS_ARMOR_MASK`)
- `self_powerup_ids` ← `self_items` bits 19–22 (`ITEMS_POWERUP_MASK`)
- entity `dist` ← `|rel| / DIST_SCALE` (and is still emitted in the entity scalars)

## Codec caveats (for the eventual build)
- **int32 → int64 promotion.** The on-disk/packed ID fields are int32; the ONNX
  graph inputs are int64 — the codec must promote.
- **v17 `weapon_logits` dynamic dim** — canonicalize to `[B,8]` on ingest.
- **`self_scalars` is 17-wide** for both v17 and v22 (post-`attack_finished`). A
  pre-v17 16-wide packed model would be a different contract (`wire.6` or below)
  with a −4-byte offset shift — out of the load set.

The packed-float32 layout differs from `semantics.1`'s native-width application
only in *transport*; the field meanings and scales are identical, which is why
`wire.7` shares `semantics.1` with `wire.9`.
