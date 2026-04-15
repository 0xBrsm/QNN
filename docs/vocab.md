# Semantic Vocabulary

Shared ID tables used by the C worker (`qnn_vocab.h`) and Python model
(`vocab.py`). One entity table, one action table, one modality table. All
embedding lookups index into these.

## Entity IDs (42)

Unified subject/source table. Every entity token, event subject, and event
source indexes into this single vocabulary.

| ID | Name | Category |
|----|------|----------|
| 0 | NONE | sentinel |
| 1 | PLAYER | actor |
| 2 | WEAPON | source (generic pickup) |
| 3 | AXE | weapon |
| 4 | SHOTGUN | weapon |
| 5 | NAILGUN | weapon |
| 6 | GRENADE_LAUNCHER | weapon |
| 7 | ROCKET_LAUNCHER | weapon |
| 8 | THUNDERBOLT | weapon |
| 9 | AMMO | source (generic pickup) |
| 10 | SHELLS | ammo |
| 11 | NAILS | ammo |
| 12 | ROCKETS | ammo |
| 13 | CELLS | ammo |
| 14 | BACKPACK | ammo (dynamic drop) |
| 15 | ARMOR | source (generic pickup) |
| 16 | ARMOR_GREEN | armor |
| 17 | ARMOR_YELLOW | armor |
| 18 | ARMOR_RED | armor |
| 19 | HEALTH | health |
| 20 | MEGAHEALTH | health |
| 21 | POWERUP | source (generic) |
| 22 | QUAD | powerup |
| 23 | PENT | powerup |
| 24 | RING | powerup |
| 25 | SUIT | powerup |
| 26 | PROJECTILE_NAIL | projectile |
| 27 | PROJECTILE_GRENADE | projectile |
| 28 | PROJECTILE_ROCKET | projectile |
| 29 | LIGHTNING_BEAM | projectile |
| 30 | GROUND | environment |
| 31 | WATER | environment |
| 32 | SLIME | environment |
| 33 | LAVA | environment |
| 34 | GIB | source (death type) |
| 35 | BUTTON | mover |
| 36 | PLATFORM | mover |
| 37 | TELEPORTER | mover |
| 38 | DOOR | mover |
| 39 | KEYED | source (locked door) |
| 40 | SECRET | source (secret door) |
| 41 | TRAIN | mover |

## Action IDs (20)

Event actions. Each event triple is (subject, action, source).

| ID | Name | Used for |
|----|------|----------|
| 0 | NONE | sentinel |
| 1 | FIRE | weapon discharge |
| 2 | JUMP | player jump |
| 3 | LAND | player landing |
| 4 | PICKUP | item collected |
| 5 | ENTER | entered liquid |
| 6 | BREATH | surfaced from water |
| 7 | EXIT | left liquid |
| 8 | PAIN | damage taken |
| 9 | DEATH | killed |
| 10 | CONNECT | joined match |
| 11 | DISCONNECT | left match |
| 12 | RESPAWN | item reappeared |
| 13 | ACTIVE | powerup humming |
| 14 | ENDING | powerup expiring |
| 15 | BOUNCE | grenade bounce |
| 16 | TELEPORT | teleporter used |
| 17 | MOVE | mover activated |
| 18 | ACTIVATE | button/door use |
| 19 | REJECT | locked door |

## Modality IDs (4)

Observation channel that produced the entity token. Lower ID = higher priority
when the same entity is observed through multiple channels.

| ID | Name | Condition |
|----|------|-----------|
| 0 | SIGHT | in FOV (60 half-angle) |
| 1 | PROXIMITY | in PVS, outside FOV |
| 2 | SOUND | sound event, omnidirectional |
| 3 | MEMORY | decaying, last-known state |

Per-modality recency thresholds: SIGHT 2.0s, PROXIMITY 0.1s, SOUND 0.1s,
MEMORY 1.0s.

## Spatial Sector IDs (9)

View-relative directional sectors for spatial tokens.

| ID | Name |
|----|------|
| 0 | FOV_Center |
| 1 | FOV_Left |
| 2 | FOV_Right |
| 3 | Flank_Left |
| 4 | Flank_Right |
| 5 | Rear_Left |
| 6 | Rear_Right |
| 7 | Ground_State |
| 8 | Ceiling_State |

## Token Types (wire format)

Type tag byte in the entity stream. Each type has different scalar and ID
dimensions.

| Tag | Type | Scalars | IDs | ID fields |
|-----|------|---------|-----|-----------|
| 0 | Projectile | 8 | 2 | subject, modality |
| 1 | Actor | 19 | 3 | subject, modality, player |
| 2 | Item | 15 | 2 | subject, modality |
| 3 | Mover | 14 | 2 | subject, modality |

Up to 16 entity tokens per tick. Up to 4 events per token, each an
(action, source) pair.

## Normalization Constants

Resource caps used for scalar normalization throughout the pipeline.

| Resource | Cap | Notes |
|----------|-----|-------|
| Health | 100 | mega decays to this |
| Armor | 160 | max effective: red 200 * 0.8 |
| Shells | 100 | |
| Nails | 200 | |
| Rockets | 100 | |
| Cells | 100 | |

## Event Mapping

Events are detected from game sounds. Each sound maps to a (subject, action,
source) triple using the IDs above.

### Player Combat

| Sound | Action | Source |
|-------|--------|--------|
| weapons/ax1.wav | FIRE | AXE |
| weapons/guncock.wav | FIRE | SHOTGUN |
| weapons/shotgn2.wav | FIRE | SHOTGUN |
| weapons/rocket1i.wav | FIRE | NAILGUN |
| weapons/spike2.wav | FIRE | NAILGUN |
| weapons/grenade.wav | FIRE | GRENADE_LAUNCHER |
| weapons/sgun1.wav | FIRE | ROCKET_LAUNCHER |
| weapons/lstart.wav | FIRE | THUNDERBOLT |
| weapons/lhit.wav | FIRE | THUNDERBOLT |
| player/pain1-6.wav | PAIN | NONE |
| player/axhit1.wav | PAIN | AXE |
| player/drown1-2.wav | PAIN | WATER |
| player/lburn1-2.wav | PAIN | LAVA |
| player/death1-5.wav | DEATH | NONE |
| player/h2odeath.wav | DEATH | WATER |
| player/gib.wav | DEATH | GIB |
| player/udeath.wav | DEATH | GIB |

### Player Movement

| Sound | Action | Source |
|-------|--------|--------|
| player/plyrjmp8.wav | JUMP | NONE |
| player/land.wav | LAND | NONE |
| player/land2.wav | LAND | NONE |
| player/h2ojump.wav | LAND | WATER |
| player/inh2o.wav | ENTER | WATER |
| misc/outwater.wav | EXIT | WATER |
| player/inlava.wav | ENTER | LAVA |
| player/slimbrn2.wav | ENTER | SLIME |
| player/gasp1-2.wav | BREATH | WATER |

### Player Pickups

| Sound | Action | Source |
|-------|--------|--------|
| items/r_item1.wav | PICKUP | HEALTH |
| items/health1.wav | PICKUP | HEALTH |
| items/r_item2.wav | PICKUP | MEGAHEALTH |
| items/armor1.wav | PICKUP | ARMOR |
| weapons/pkup.wav | PICKUP | WEAPON |
| weapons/lock4.wav | PICKUP | AMMO |
| items/damage.wav | PICKUP | QUAD |
| items/protect.wav | PICKUP | PENT |
| items/inv1.wav | PICKUP | RING |
| items/suit.wav | PICKUP | SUIT |

### Powerup State

| Sound | Action | Source |
|-------|--------|--------|
| items/damage3.wav | ACTIVE | QUAD |
| items/protect3.wav | ACTIVE | PENT |
| items/inv3.wav | ACTIVE | RING |
| items/damage2.wav | ENDING | QUAD |
| items/protect2.wav | ENDING | PENT |
| items/inv2.wav | ENDING | RING |
| items/suit2.wav | ENDING | SUIT |

### Connection

| Sound | Action | Source |
|-------|--------|--------|
| (server message) | CONNECT | NONE |
| player/tornoff2.wav | DISCONNECT | NONE |

### Items

| Sound | Action | Source |
|-------|--------|--------|
| items/itembk2.wav | RESPAWN | NONE |

### Projectiles

| Sound | Action | Source |
|-------|--------|--------|
| weapons/bounce.wav | BOUNCE | NONE |

### Movers

| Sound | Action | Source |
|-------|--------|--------|
| doors/drclos4.wav | MOVE | NONE |
| doors/doormv1.wav | MOVE | NONE |
| doors/hydro1-2.wav | MOVE | NONE |
| doors/stndr1-2.wav | MOVE | NONE |
| doors/ddoor1-2.wav | MOVE | NONE |
| doors/latch2.wav | MOVE | SECRET |
| doors/winch2.wav | MOVE | SECRET |
| doors/airdoor1-2.wav | MOVE | SECRET |
| doors/basesec1-2.wav | MOVE | SECRET |
| doors/meduse.wav | ACTIVATE | KEYED |
| doors/runeuse.wav | ACTIVATE | KEYED |
| doors/baseuse.wav | ACTIVATE | KEYED |
| doors/medtry.wav | REJECT | KEYED |
| doors/runetry.wav | REJECT | KEYED |
| doors/basetry.wav | REJECT | KEYED |
| plats/plat1-2.wav | MOVE | NONE |
| plats/medplat1-2.wav | MOVE | NONE |
| plats/train1-2.wav | MOVE | NONE |
| buttons/airbut1.wav | ACTIVATE | NONE |
| buttons/switch21.wav | ACTIVATE | NONE |
| buttons/switch02.wav | ACTIVATE | NONE |
| buttons/switch04.wav | ACTIVATE | NONE |
| misc/r_tele1-5.wav | TELEPORT | NONE |
