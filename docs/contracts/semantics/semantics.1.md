# Semantics contract `semantics.1` — current (v11 → HEAD)

The meaning behind the wire tensors: normalization scales + vocabulary id
mappings. Shared by every contract in the load set ([`wire.7`](../wire/wire.7.md),
[`wire.8`](../wire/wire.8.md), [`wire.9`](../wire/wire.9.md)) — it has **not moved
since token-spec v11** (the v11 entity-vocab 42→44 bump + weapon impulse
renumber is the last semantics break; everything before is `semantics.0*`, see
[`pre-v11.md`](pre-v11.md)).

Stamped as `semantics_contract=semantics.1`. A mismatch is a **silent**
failure (tensors load, values mean the wrong thing), so the bin refuses on a
`semantics_sig` mismatch. The fingerprint is computed by `_semantics_sig()` in
`tools/export_onnx.py` over exactly the items below; current value
**`de5f93f393503793`** (recompute from the function — do not treat as frozen).

**Source of truth:** `src/qnn/engine_norm.py` (scales, masks, `Field.scale`/
`Field.transform`, `ITEM_AMOUNT_*`) and `src/qnn/vocab.py` (vocab tables). See
also [`src/docs/vocab.md`](../../vocab.md).

## Scalar normalization constants (`engine_norm.py`)
| const | value | const | value |
|-------|-------|-------|-------|
| `MAX_HEALTH` | 100 | `MAX_VELOCITY` (per-axis clamp) | 2000 |
| `MAX_ARMOR_EFFECT` | 160 | `TIME_SCALE` | 60.0 |
| `MAX_SHELLS` | 100 | `DIST_SCALE` | 1000.0 |
| `MAX_NAILS` | 200 | `MAX_ROCKETS` | 100 |
| `MAX_CELLS` | 100 | | |

C mirrors (cross-checked): `QNN_DIST_SCALE=1000`, `QNN_VELOCITY_SCALE=2000`,
`QNN_TIME_SCALE=60` in `qnn_io.h`.

## `cl.items` bit layout & movement ids
- Masks: `ITEMS_WEAPON_MASK`, `ITEMS_ARMOR_MASK`, `ITEMS_POWERUP_MASK`,
  `ITEMS_MEANINGFUL` — define which `self_items` bits decode to weapon ownership
  (0–6,12), armor type (13–15), powerups (19–22).
- Movement: `GROUND=0, AIR=1, WATER_LOW=2, WATER_MID=3, WATER_HIGH=4`.

## Per-field (scale, transform)
The `Field.scale` / `Field.transform` of every field in `SELF_FIELDS`,
`SPATIAL_FIELDS`, `ENTITY_COMMON_FIELDS`, and the four per-type tables
(`PROJECTILE/ACTOR/ITEM/MOVER_FIELDS`) — the dequant semantics. (Enumerated in
`engine_norm.py`; included in the hash field-by-field.) Plus the resolved
`ITEM_AMOUNT_MULT` / `ITEM_AMOUNT_CONST` arrays (per-pickup amount normalization).

## Vocabularies (`vocab.py`)
| table | size / value |
|-------|--------------|
| `ENTITY_IDS` / `ENTITY_VOCAB_SIZE` | 44 (NONE=0 … TRAIN=43); weapons contiguous **AXE=3 … THUNDERBOLT=10 in impulse order** |
| `ACTION_IDS` / `ACTION_VOCAB_SIZE` | 20 |
| `MODALITY_IDS` / `MODALITY_VOCAB_SIZE` | 4 (SIGHT/PROXIMITY/SOUND/MEMORY) |
| `SPATIAL_SECTOR_IDS` | 9 (defines the 9 spatial slots' order) |
| `MAX_PLAYER_INDICES` | 32 |
| token tags | `PROJECTILE=0, ACTOR=1, ITEM=2, MOVER=3` |
| per-type scalar dims | PROJ 8, ACTOR 19, ITEM 15, MOVER 14 |
| capacities | `MAX_ENTITY_EVENTS=4`, `MAX_TOKEN_OBJECTS=16`, `SPATIAL_TOKEN_COUNT=9` |
| weapon impulse | 1=AXE … 8=LG; weapon-head class = impulse−1 (0..7), 8 classes. `self_weapon_id` is ENTITY_IDS-encoded with `self_weapon_id_to_impulse(x)=max(0,x−2)` |

## Temporal alignment — the timeliness clause (normative)

The scales and vocab above pin what a tensor's *values* mean; this clause pins
what its *tick* means. It is part of the semantics contract because violating
it is exactly this axis's failure mode — well-formed tensors, silently wrong
meaning — but it is **behavioral, not tensor-valued**, so it is NOT captured by
`semantics_sig` and cannot be checked at load. It is enforced per frontend by
the conformance tests below.

**Clause:** `obs(t)` reflects the agent's **own commands through t−1**, and the
SELF-state fields (`vel`, origin-derived spatial features, grounded/movement
id) are **server-time-aligned** — the alignment of the training data
(server-side demo collect) and of the in-process eval bridge
(`qnn_trainer_main.c`: the step's action is applied in the same `Host_Frame`
that produces the response obs). A frontend that delivers self-state lagging
this alignment feeds the model obs from a semantics it was never trained
under.

**Conformance layers.** Every protocol frontend must honor the clause by
whatever means its protocol affords; the model is never asked to tolerate
transport.

| Frontend | Transport lag | Conformance layer |
|----------|---------------|-------------------|
| eval bridge / trainer (in-process server) | none | trivially conformant |
| NQ live client (`nq_client`) | +1 tick structural (no cmd acks) | `qnn_predict.c` — own-cmd replay through mirrored horizontal physics; online lag vote (EMA over cmd offsets) because NQ cannot ack |
| QW live client (future) | RTT, acked | native QW prediction (cmd-sequence acks + shared pmove); read predicted state into the snapshot |
| other modern protocols (FTE/DP, future) | varies | per-protocol; native prediction where available, `qnn_predict`-style replay where not |

**Certification (run before trusting any new frontend):**
1. **Own-cmd response at k=0** — paired `QNN_CLIENT_ACTION_LOG` +
   `QNN_CLIENT_ENGINE_LOG` (+ `QNN_CLIENT_PREDICT_LOG` where applicable)
   capture; cross-correlate commanded lr against view-relative side-velocity
   deltas in the obs the model actually sees. Conformant: response peaks at
   k=0 (raw NQ fails at k=1).
2. **Stream human-band under the standard decode stack** — lr/fb
   switch-rate, dwell median, and dwell-age hazard curves within the human
   band; the a24e watermark must be a
   behavioral no-op on a conformant frontend.
3. **Physics-floor canary** — where a replay layer runs, its lag-vote EMA
   floor across ALL candidates stays low; a high floor fingerprints
   server-side movement-cvar mismatch, not transport.

A deliberate decision to ship a non-conformant frontend (e.g. prediction
disabled via `QNN_CLIENT_PREDICT=0`) is a *known semantics deviation* and must
ride with its compensating decode (the stamped a24e switch-back watermark) —
never silently.

## Bump rule
Bump `SEMANTICS_CONTRACT_ID` (→ `semantics.2`) on **any** change to a scale
constant or a vocab id mapping, even when the wire (`wire_sig`) is byte-identical.
A renumbered vocab or a rescaled field changes what the same bytes mean.
The timeliness clause follows the same spirit: a frontend may not change the
effective tick alignment of self-state without it being treated as a semantics
deviation.
