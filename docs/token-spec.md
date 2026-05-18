# Token Specification v11

This document defines the token contract between the C worker and the Python
model. It describes the wire format (`obs_buffer_v1`), the Python adapter
output, the transformer input contract, and the model's output heads.

The canonical Python source of truth for obs shapes is `OBS_SCHEMA` in
`qnn/schema.py`. The C wire layout is in `qnn_io.h`.

## Version History

- **v11** (this): entity vocab 42 → 44 (SHOTGUN/SUPER_SHOTGUN and NAILGUN/
  SUPER_NAILGUN split, weapons renumbered into Quake impulse order 3-10),
  `self_scalars` 14 → 16 (per-weapon one-hot ownership), TargetPointer
  supervised pointer head over actor slots, `target_feat` conditioning
  action heads, 8-class weapon head replaces 5-slot switch end-to-end,
  move = 3 ternary categorical axes, recall removed from the wire,
  action_history removed from the wire and Tokenizer end-to-end
  (only the checkpoint-converter migration remembers it).
- **v10** (archived, [token-spec-v10.md](../../docs/archive/token-spec-v10.md)):
  silent version bump during 0.15.0 — move[2]+jump collapsed into move[3],
  jump head removed.
- **v9** (archived, [token-spec-v9.md](../../docs/archive/token-spec-v9.md)):
  nav oracle → token wiring, MENTAL recall channel (later parked in v11).

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
- **Target as intermediary.** The target is not a sampled action; it is an
  internal feature produced by a pointer head and consumed by downstream
  action heads as conditioning signal.

---

## Wire Format (obs_buffer_v1)

Binary obs buffer, 4096 bytes, little-endian. Layout from `engine/common/qnn_io.h`:

| Offset | Field | Type | Shape | Bytes |
|--------|-------|------|-------|-------|
| 0 | self_scalars | float32 | [16] | 64 |
| 64 | self_weapon_id | int32 | [1] | 4 |
| 68 | self_armor_type_id | int32 | [1] | 4 |
| 72 | self_powerup_ids | int32 | [5] | 20 |
| 92 | self_movement_id | int32 | [1] | 4 |
| 96 | spatial_scalars | float32 | [9, 13] | 468 |
| 564 | entity_stream | variable | see below | ~1825 max |

Total max: ~2389 bytes. Buffer oversized to 4096 for safety.

`action_history` is removed end-to-end in v11: no wire region, no entry in
`qnn/wire.py` or `qnn/schema.py`, no Tokenizer support in
`qnn/model/transformer.py`, and no `action_history_tokens` key in BC or PPO
`model.json` templates. The only residue is
`migrate_drop_action_history` in `qnn/utils/checkpoint_converter.py`, which
strips the pre-rip-out Tokenizer weights from old checkpoints. Re-enabling
would require restoring the wire region, the adapter unpack, the Tokenizer
path, and the template knob.

---

## Python Adapter Output

`unpack_obs_buffer()` in `wire.py` parses the wire format and returns a
dict of numpy arrays. The entity stream is densified into fixed-size arrays by
`densify_entity_tokens()`. All keys and shapes are defined in `OBS_SCHEMA`.

| Key | Shape | Dtype | Source |
|-----|-------|-------|--------|
| self_scalars | (16,) | float32 | fixed section |
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

---

## Self Token

16 scalars + 4 embedding lookups.

### Scalars (from `qnn_self_token_t`)

Per-weapon ownership is one-hot: each owned weapon gets its own scalar bit
instead of the old paired 0/0.5/1.0 floats for SG/SSG and NG/SNG.

| Index | Field | Scale | Notes |
|-------|-------|-------|-------|
| 0 | health | /100 | 1.0 = full, >1.0 = megahealth |
| 1 | armor | /100 | effective armor value |
| 2 | weapon_sg | 0/1 | shotgun owned |
| 3 | weapon_ssg | 0/1 | super shotgun owned |
| 4 | weapon_ng | 0/1 | nailgun owned |
| 5 | weapon_sng | 0/1 | super nailgun owned |
| 6 | weapon_gl | 0/1 | grenade launcher owned |
| 7 | weapon_rl | 0/1 | rocket launcher owned |
| 8 | weapon_lg | 0/1 | lightning gun owned |
| 9 | ammo_shells | /100 | |
| 10 | ammo_nails | /200 | |
| 11 | ammo_rockets | /100 | |
| 12 | ammo_cells | /100 | |
| 13-15 | vel[3] | /2000 | view-frame forward/right/up |

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

## Action History (parked)

Not present in the v11 wire format. Tokenizer infrastructure remains in
`qnn/model/transformer.py` and BC + PPO templates set
`action_history_tokens: 0`. Re-enabling requires re-adding a wire region
and bumping the spec version.

When previously enabled, the N most recent action frames were projected as
individual tokens with shape `(8,)`:

| Index | Field |
|-------|-------|
| 0-2 | move[3] (forward, strafe, jump-Z) |
| 3-5 | look[3] (yaw, pitch, roll) |
| 6 | fire |
| 7 | weapon (normalized) |

```
action_token = action_proj(history_row)
             + action_pos_embed(position)
             + kind_embed(ACTION)
```

---

## Transformer Trunk Outputs

`TransformerTrunk.forward` returns four d_model-sized summary tensors from
the post-attention token sequence:

| Output | Shape | Source | Consumer |
|--------|-------|--------|----------|
| `self_readout` | (B, d) | self-token at position 0 | GRU input, head conditioning |
| `entity_outs` | (B, N, d) | post-attention entity slots | TargetPointer pool, look-head base direction |
| `target_feat` | (B, d) | soft-gather `sum_i softmax(logits)[i] * entity_out[i]` over actor slots | head conditioning |
| `target_logits` | (B, 16) | pre-softmax attention scores over actor slots (non-actor slots masked to -1e9) | supervised CE loss vs labeler; not sampled |

### GRU Input

```
gru_input = self_readout
```

GRU runs over the sequence of `self_readout` tensors (no extra projection,
no actor-token pooling). Hidden size 64 (see `gru_hidden` in model config).

### TargetPointer

Supervised pointer head over actor slots; see `qnn/model/target.py`.

- Query: `linear(self_readout)`, d_model dim.
- Scores: `(entity_outs * query).sum(-1)` masked to actor slots.
- `target_logits`: raw scores with non-actor slots set to -1e9.
- `target_feat`: `sum_i softmax(target_logits)[i] * entity_outs[i]`; zeroed
  when no actors present.

Trained directly by cross-entropy against BC labels (see
`qnn/bc/target_labeler.py`).  CE default `ignore_index=-100` skips unlabeled
ticks.

### Fused Head Features

Motor heads (move, look, fire) read a concat of GRU output, target_feat,
and a weapon-context embedding:

```
features = cat(gru_out, target_feat, weapon_context)
```

The weapon head reads gru_out (optional via `weapon_use_gru`), self_readout,
and target_feat:

```
weapon_features = cat(gru_out, self_readout, target_feat)   # or omit gru_out
```

When the trunk-level GRU is disabled, motor heads see
`cat(self_readout, target_feat, weapon_context)` instead. Each head is a
Linear (optionally with a ReLU bottleneck per `head_bottleneck_dim`).

---

## Action Space

Four sampled heads. Engine consumes all four per tick. `target`
exists as a supervised auxiliary label and internal attention mechanism;
it is not a sampled action.

| Head | Type | Dim | Notes |
|------|------|-----|-------|
| move | 3 categorical axes × 3 classes | 9 logits | axes [fb, lr, ud] × classes {neg, none, pos}; engine decodes to float[3] in {-1, 0, +1} |
| look | continuous | 3 | view-frame next-frame forward direction, bounded [-1, 1] |
| fire | binary | 1 logit | sigmoid; 0 = not firing, 1 = firing |
| weapon | discrete | 8 | direct Quake impulse class 0..7 → engine impulse byte 1..8 (axe..lightning) |

Per-axis class encoding for `move`: 0 = neg, 1 = none, 2 = pos. The engine
receives `float[3]` after decoding (`class - 1`), so the wire-side struct
is unchanged.

### Engine Action Struct

The Python bridge packs actions into a 32-byte `qnn_action_t` struct:

```c
struct qnn_action_t {
    float move[3];        // 12 bytes — decoded categorical class - 1
    float look[3];        // 12 bytes
    int   fire;           //  4 bytes
    int   weapon;         //  4 bytes — raw Quake impulse byte (0 = no impulse, 1..8 = axe..lightning)
};
```

### Auxiliary Target Label

`actions["target"]`: (T,) int64, values in `{0..15, -100}`. Derived
offline by `qnn.bc.target_labeler.label_enemy_target` from `fire`, `look`,
and entity tokens. Algorithm (adaptive-cone Schmitt-trigger, sticky-by-PID):

1. For each fire tick, decide acquire vs sticky-keep:
   - If current sticky pid is in obs and `cos(look, rel)` is above the
     **release cone** `clamp(atan(416/d), 5°, 45°)`, sticky-keeps.
   - Otherwise admit any actor with cos above the per-actor **acquire cone**
     `clamp(atan(208/d), 5°, 30°)`; pick the cos-argmax as the new sticky pid.
   - If sticky pid disappeared from obs between the previous fire and this
     one (transient stream loss), release sticky before deciding.
2. Group consecutive same-pid fire attributions into engagements.
3. Extend each engagement backward to the previous engagement's end and
   forward until the pid leaves obs (or the next engagement starts).

The acquire/release asymmetry (K=2 transverse multiplier, capped at 30°/45°)
is a Schmitt trigger that suppresses co-angular A↔B oscillation during
two-enemy engagements. The same state machine runs at inference time inside
`engine/common/qnn_oracle.c` so labeler and engine pid agree on ~97.5% of
labeled SIGHT-only frames (~89% on the full 4-pool stream — see
[target_labeler_engine_alignment.md](target_labeler_engine_alignment.md)).

Unlabeled ticks carry `-100`; `F.cross_entropy` ignores them by default.

---

## Embedding Tables

| Table | Vocab Size | Used By |
|-------|-----------|---------|
| entity_embed | 44 | self (weapon, armor, powerup), entity (subject, event source) |
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
| QNN_TIME_SCALE | 60.0 | time (seconds): eta, regen, recency |

---

## Token Sequence and Budgets

| Budget | Count |
|--------|-------|
| Max entity tokens | 16 |
| Max events per entity | 4 |
| Spatial tokens | 9 |
| Action history | up to 8 tokens (currently 0) |

Token sequence: `self(1) + spatial(9) + [action_history(N)] + entities(up to 16)`.

Self readout at position 0 after transformer attention.  Entity slot indices
into `transformed` start at `1 + SPATIAL_TOKEN_COUNT + action_history_tokens`.

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

---

## Parked Features

Infrastructure retained but not active in v11:

- **action_history**: removed from the v11 wire region; Tokenizer support
  intact in `qnn/model/transformer.py`, templates set
  `action_history_tokens: 0`. Re-enabling requires restoring the wire region
  and bumping the spec version.
- **recall**: removed from the action struct, the engine event path, and
  the wire format in 0.17. `qnn_store[].mem` field is still present in the
  C-side oracle store as a dormant revival hook.
