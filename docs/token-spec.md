# Token Specification v9

This document defines the token contract between the C worker and the Python
model. It describes the wire format (`obs_buffer_v1`), the Python adapter
output, and the transformer input contract.

The canonical Python source of truth for obs shapes is `OBS_SCHEMA` in
`qnn/schema.py`. The C wire layout is in `qnn_io.h`.

## Design Principles

- **Unified egocentric frame.** All spatial data is view-frame:
  forward/right/up relative to the player's gaze.
- **One embedding table per concept.** `entity_embed`, `modality_embed`,
  `action_embed` are shared across all token types.
- **Per-type entity projections.** Each entity type (projectile, actor, item,
  mover) has its own scalar projection layer. The wire format uses type-tagged
  variable-length tokens; the Python adapter densifies these into fixed arrays.
- **Projection inside the model.** The adapter emits raw scalars
  (`entity_scalars_raw`); type-specific linear projections happen inside the
  transformer `Tokenizer`, not in the transport layer.
- **Pre-computed derived scalars.** Distance, path distance, and ETA are
  shipped explicitly so the model does not waste capacity on sqrt or route
  planning.

---

## Wire Format (obs_buffer_v1)

Binary obs buffer, 4096 bytes, little-endian. Layout from `qnn_io.h`:

| Offset | Field | Type | Shape | Bytes |
|--------|-------|------|-------|-------|
| 0 | self_scalars | float32 | [14] | 56 |
| 56 | self_weapon_id | int32 | [1] | 4 |
| 60 | self_armor_type_id | int32 | [1] | 4 |
| 64 | self_powerup_ids | int32 | [5] | 20 |
| 84 | self_movement_id | int32 | [1] | 4 |
| 88 | spatial_scalars | float32 | [9, 13] | 468 |
| 556 | action_history | float32 | [8, 8] | 256 |
| 812 | entity_stream | variable | see below | ~1825 max |

Total max: ~2637 bytes. Buffer oversized to 4096 for safety.

---

## Python Adapter Output

`unpack_obs_buffer()` in `wire.py` parses the wire format and returns a
dict of numpy arrays. The entity stream is densified into fixed-size arrays by
`densify_entity_tokens()`. All keys and shapes are defined in `OBS_SCHEMA`.

| Key | Shape | Dtype | Source |
|-----|-------|-------|--------|
| self_scalars | (14,) | float32 | fixed section |
| self_weapon_id | (1,) | int32 | fixed section |
| self_armor_type_id | (1,) | int32 | fixed section |
| self_powerup_ids | (5,) | int32 | fixed section |
| self_movement_id | (1,) | int32 | fixed section |
| entity_types | (16,) | int32 | densified, -1 = empty |
| entity_scalars_raw | (16, 19) | float32 | densified, zero-padded |
| entity_ids | (16, 3) | int32 | densified |
| entity_event_actions | (16, 4) | int32 | densified |
| entity_event_sources | (16, 4) | int32 | densified |
| entity_event_counts | (16,) | uint8 | densified |
| spatial_scalars | (9, 13) | float32 | fixed section |
| action_history | (8, 8) | float32 | fixed section |

---

## Self Token

14 scalars + 4 embedding lookups.

### Scalars (from `qnn_self_token_t`)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0 | health | /100 | 1.0 = full, >1.0 = megahealth |
| 1 | armor | /100 | effective armor value |
| 2 | weapon_sg | 0/0.5/1.0 | 0=none, 0.5=SG, 1.0=SSG |
| 3 | weapon_ng | 0/0.5/1.0 | 0=none, 0.5=NG, 1.0=SNG |
| 4 | weapon_gl | 0/1 | |
| 5 | weapon_rl | 0/1 | |
| 6 | weapon_lg | 0/1 | |
| 7 | ammo_shells | /100 | |
| 8 | ammo_nails | /200 | |
| 9 | ammo_rockets | /100 | |
| 10 | ammo_cells | /100 | |
| 11-13 | vel[3] | /2000 | view-frame forward/right/up |

### Embeddings

| Field | Table | Notes |
|-------|-------|-------|
| weapon_id | entity_embed | active weapon subject ID |
| armor_type_id | entity_embed | armor type subject ID, masked when 0 |
| powerup_ids[5] | entity_embed | up to 5 active powerups, sum-pooled, masked when 0 |
| movement_id | movement_embed | 0=grounded .. 4=submerged |

### Model Representation

```
self_token = self_proj(scalars)
           + entity_embed(weapon_id)
           + entity_embed(armor_type_id) * (armor_type_id > 0)
           + sum(entity_embed(p) * (p > 0) for p in powerup_ids)
           + movement_embed(movement_id)
           + kind_embed(SELF)
```

---

## Entity Tokens

Variable-length stream of type-tagged tokens. Four entity types, each with
different scalar and ID dimensions.

### Type Tags

| Tag | Type | IDs | Scalars |
|-----|------|-----|---------|
| 0 | Projectile | 2 (subject, modality) | 8 |
| 1 | Actor | 3 (subject, modality, player_id) | 19 |
| 2 | Item | 2 (subject, modality) | 15 |
| 3 | Mover | 2 (subject, modality) | 14 |

### Projectile Scalars (8)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0-2 | rel[3] | /1000 | view-frame relative position |
| 3 | dist | /1000 | Euclidean distance |
| 4-6 | vel[3] | /2000 | view-frame velocity (encodes trajectory) |
| 7 | recency | [0,1] | time since last observation |

### Actor Scalars (19)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0-2 | half_extents[3] | /1000 | bounding box half-size |
| 3-5 | rel[3] | /1000 | view-frame relative position |
| 6 | dist | /1000 | Euclidean distance |
| 7-9 | vel[3] | /2000 | view-frame velocity |
| 10-12 | path[3] | /1000 | view-frame path direction |
| 13 | path_dist | /1000 | navmesh path distance |
| 14 | eta | /60 | estimated travel time (seconds) |
| 15 | facing | [-1,1] | dot product of their forward vs direction to us |
| 16 | team | 0/1 | 0=enemy, 1=ally |
| 17 | score | [0,1] | frag ranking |
| 18 | recency | [0,1] | time since last observation |

### Item Scalars (15)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0-2 | half_extents[3] | /1000 | bounding box half-size |
| 3-5 | rel[3] | /1000 | view-frame relative position |
| 6 | dist | /1000 | Euclidean distance |
| 7-9 | path[3] | /1000 | view-frame path direction |
| 10 | path_dist | /1000 | navmesh path distance |
| 11 | eta | /60 | estimated travel time |
| 12 | amount | [0,1] | deterministic game value (health/armor/ammo) |
| 13 | regen | /60 | seconds until respawn (0 = available) |
| 14 | recency | [0,1] | time since last observation |

### Mover Scalars (14)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0-2 | half_extents[3] | /1000 | bounding box half-size |
| 3-5 | rel[3] | /1000 | view-frame relative position |
| 6 | dist | /1000 | Euclidean distance |
| 7-9 | path[3] | /1000 | view-frame path direction |
| 10 | path_dist | /1000 | navmesh path distance |
| 11 | eta | /60 | estimated travel time |
| 12 | state | varies | type-dependent state scalar |
| 13 | recency | [0,1] | time since last observation |

### Events

Each entity can carry up to 4 events. Each event is a pair:
`(action_id, source_id)` using the action and entity vocabularies.

### Model Representation

The `Tokenizer._project_entity_scalars` method routes raw scalars through
type-specific linear layers:

```
entity_token = proj_{type}(scalars[:type_sdim])
             + entity_embed(subject_id)
             + modality_embed(modality_id)
             + player_embed(player_id) * is_actor
             + sum(action_embed(evt_action) + entity_embed(evt_source)
                   for valid events)
             + kind_embed(ENTITY)
```

---

## Spatial Tokens

9 view-relative directional sectors probing the local environment.

### Sectors

| Index | Sector |
|-------|--------|
| 0 | FOV_Center |
| 1 | FOV_Left |
| 2 | FOV_Right |
| 3 | Flank_Left |
| 4 | Flank_Right |
| 5 | Rear_Left |
| 6 | Rear_Right |
| 7 | Ground_State |
| 8 | Ceiling_State |

### Scalars (13, from `qnn_spatial_token_t`)

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0-2 | dir[3] | unit | view-relative sector direction |
| 3 | nearest_dist | /1000 | nearest surface distance |
| 4 | mean_dist | /1000 | mean ray distance |
| 5 | openness | [0,1] | how open the space is |
| 6 | solid_frac | [0,1] | BSP solid content fraction |
| 7 | water_frac | [0,1] | BSP water content fraction |
| 8 | slime_frac | [0,1] | BSP slime content fraction |
| 9 | lava_frac | [0,1] | BSP lava content fraction |
| 10 | traversable | 0/1 | can walk |
| 11 | dropoff | 0/1 | cliff edge |
| 12 | clearance | [0,1] | headroom / passability |

### Model Representation

```
spatial_token = spatial_proj(scalars) + kind_embed(SPATIAL)
```

---

## Action History

8 most recent action frames, each with 8 dimensions. Recall actions are not
included in the history.

| Index | Field |
|-------|-------|
| 0-2 | move[3] (forward, strafe, jump-Z) |
| 3-5 | look[3] (yaw, pitch, roll) |
| 6 | fire |
| 7 | switch (normalized) |

When enabled (`action_history_tokens > 0`), the N most recent entries are
projected as individual tokens:

```
action_token = action_proj(history_row)
             + action_pos_embed(position)
             + kind_embed(ACTION)
```

---

## Action Space

12 output values from 9 heads.

### Play Heads (4)

| Head | Type | Dim | Notes |
|------|------|-----|-------|
| move | continuous | 3 | [forward, strafe, jump-Z], bounded [-1, 1] |
| look | continuous | 3 | [yaw, pitch, roll], bounded [-1, 1] |
| fire | discrete | 2 | off/on |
| switch | discrete | 6 | none, SG, NG, GL, RL, LG |

### Recall Heads (4)

| Head | Type | Dim | Notes |
|------|------|-----|-------|
| recall_0..3 | discrete | 65 | no-op + 64 register slots |

---

## Embedding Tables

| Table | Vocab Size | Used By |
|-------|-----------|---------|
| entity_embed | 42 | self (weapon, armor, powerup), entity (subject, event source) |
| action_embed | 20 | entity events only |
| modality_embed | 4 | entity tokens (SIGHT, PROXIMITY, SOUND, MEMORY) |
| player_embed | 33 | actor entities only (0 = non-player) |
| movement_embed | 5 | self only (grounded..submerged) |
| kind_embed | 4 | all tokens (SELF, ENTITY, SPATIAL, ACTION) |

---

## Normalization Constants

| Constant | Value | Domain |
|----------|-------|--------|
| QNN_DIST_SCALE | 1000.0 | position, distance, extents, path_dist |
| QNN_VELOCITY_SCALE | 2000.0 | velocity (Quake units/sec) |
| QNN_TIME_SCALE | 60.0 | time (seconds): eta, regen |

---

## Token Sequence and Budgets

| Budget | Count |
|--------|-------|
| Max entity tokens | 16 |
| Max events per entity | 4 |
| Spatial tokens | 9 |
| Action history | up to 8 tokens (configurable) |

Token sequence: `self(1) + spatial(9) + [action_history(N)] + entities(up to 16)`

Self token readout at position 0 after transformer attention.

---

## Densification

The C worker emits a variable-length entity stream. The Python adapter
(`wire.densify_entity_tokens`) converts this to fixed-size arrays:

- `entity_types`: (16,) int32, -1 for empty slots
- `entity_scalars_raw`: (16, 19) float32, zero-padded to max scalar dim
- `entity_ids`: (16, 3) int32, zero-padded to max ID dim
- `entity_event_actions`: (16, 4) int32
- `entity_event_sources`: (16, 4) int32
- `entity_event_counts`: (16,) uint8

The transformer `Tokenizer._project_entity_scalars` reads `entity_types` to
route each slot through the correct type-specific projection layer, slicing
only the first N scalars for that type.
