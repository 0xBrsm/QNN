# Event Vocabulary v2

Events are triples: **subject, action, source**. Subject is the entity the event belongs to. Action is what happened. Source is what caused it or what's involved (NONE when not applicable). No magnitude scalar. All three are IDs from a shared vocab.

## Player Events

Events attached to actor (player) entities.

### Combat

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| weapons/ax1.wav | PLAYER | FIRE | AXE | axe swing |
| weapons/guncock.wav | PLAYER | FIRE | SHOTGUN | |
| weapons/shotgn2.wav | PLAYER | FIRE | SHOTGUN | double-barrel variant |
| weapons/rocket1i.wav | PLAYER | FIRE | NAILGUN | |
| weapons/spike2.wav | PLAYER | FIRE | NAILGUN | super nailgun variant |
| weapons/grenade.wav | PLAYER | FIRE | GRENADE_LAUNCHER | |
| weapons/sgun1.wav | PLAYER | FIRE | ROCKET_LAUNCHER | |
| weapons/lstart.wav | PLAYER | FIRE | THUNDERBOLT | initial activation |
| weapons/lhit.wav | PLAYER | FIRE | THUNDERBOLT | sustained fire (every 0.6s) |
| player/pain1-6.wav | PLAYER | PAIN | NONE | generic damage taken |
| player/axhit1.wav | PLAYER | PAIN | AXE | axe damage (plays on victim) |
| player/drown1-2.wav | PLAYER | PAIN | DROWN | drowning damage |
| player/lburn1-2.wav | PLAYER | PAIN | LAVA | lava damage |
| player/death1-5.wav | PLAYER | DEATH | NONE | killed |
| player/h2odeath.wav | PLAYER | DEATH | DROWN | drowned to death |
| player/gib.wav | PLAYER | DEATH | GIB | gibbed |
| player/udeath.wav | PLAYER | DEATH | GIB | gibbed |

### Movement

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| player/plyrjmp8.wav | PLAYER | JUMP | NONE | |
| player/land.wav | PLAYER | LAND | NONE | |
| player/land2.wav | PLAYER | LAND | NONE | hard landing |
| player/h2ojump.wav | PLAYER | LAND | WATER | water landing |
| player/inh2o.wav | PLAYER | ENTER | WATER | |
| misc/outwater.wav | PLAYER | EXIT | WATER | |
| player/inlava.wav | PLAYER | ENTER | LAVA | |
| player/slimbrn2.wav | PLAYER | ENTER | SLIME | |
| player/gasp1.wav | PLAYER | BREATH | WATER | surfacing |
| player/gasp2.wav | PLAYER | BREATH | WATER | surfacing |

### Pickups

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| items/r_item1.wav | PLAYER | PICKUP | HEALTH | |
| items/health1.wav | PLAYER | PICKUP | HEALTH | |
| items/r_item2.wav | PLAYER | PICKUP | MEGAHEALTH | |
| items/armor1.wav | PLAYER | PICKUP | ARMOR | generic — can't distinguish tier |
| weapons/pkup.wav | PLAYER | PICKUP | WEAPON | weapon pickup |
| weapons/lock4.wav | PLAYER | PICKUP | AMMO | ammo pickup |
| items/damage.wav | PLAYER | PICKUP | QUAD | |
| items/protect.wav | PLAYER | PICKUP | PENT | |
| items/inv1.wav | PLAYER | PICKUP | RING | |
| items/suit.wav | PLAYER | PICKUP | SUIT | |

### Powerup State

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| items/damage3.wav | PLAYER | ACTIVE | QUAD | quad humming |
| items/protect3.wav | PLAYER | ACTIVE | PENT | pent humming |
| items/inv3.wav | PLAYER | ACTIVE | RING | ring humming |
| items/damage2.wav | PLAYER | ENDING | QUAD | quad expiring (3s left) |
| items/protect2.wav | PLAYER | ENDING | PENT | pent expiring |
| items/inv2.wav | PLAYER | ENDING | RING | ring expiring |
| items/suit2.wav | PLAYER | ENDING | SUIT | suit expiring |

### Connection

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| player/tornoff2.wav | PLAYER | DISCONNECT | NONE | always disconnect, never gib |

## Item Events

Events attached to item entities in the object store.

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| items/itembk2.wav | (item's subject) | RESPAWN | NONE | item reappeared |

Note: itembk2.wav plays for all item types. Subject comes from the item entity it's attached to.

## Projectile Events

Events attached to projectile entities.

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| weapons/bounce.wav | PROJECTILE_GRENADE | BOUNCE | NONE | grenade bouncing |

## Mover Events

Events attached to mover entities (matched by spatial proximity).

### Doors

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| doors/drclos4.wav | DOOR | MOVE | NONE | |
| doors/doormv1.wav | DOOR | MOVE | NONE | |
| doors/hydro1-2.wav | DOOR | MOVE | NONE | |
| doors/stndr1-2.wav | DOOR | MOVE | NONE | |
| doors/ddoor1-2.wav | DOOR | MOVE | NONE | |
| doors/latch2.wav | DOOR | MOVE | SECRET | secret door |
| doors/winch2.wav | DOOR | MOVE | SECRET | secret door |
| doors/airdoor1-2.wav | DOOR | MOVE | SECRET | secret door |
| doors/basesec1-2.wav | DOOR | MOVE | SECRET | secret door |
| doors/meduse.wav | DOOR | ACTIVATE | KEYED | key door opened |
| doors/runeuse.wav | DOOR | ACTIVATE | KEYED | key door opened |
| doors/baseuse.wav | DOOR | ACTIVATE | KEYED | key door opened |
| doors/medtry.wav | DOOR | REJECT | KEYED | key door locked |
| doors/runetry.wav | DOOR | REJECT | KEYED | key door locked |
| doors/basetry.wav | DOOR | REJECT | KEYED | key door locked |

### Platforms

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| plats/plat1-2.wav | PLATFORM | MOVE | NONE | |
| plats/medplat1-2.wav | PLATFORM | MOVE | NONE | |

### Trains

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| plats/train1-2.wav | TRAIN | MOVE | NONE | |

### Buttons

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| buttons/airbut1.wav | BUTTON | ACTIVATE | NONE | |
| buttons/switch21.wav | BUTTON | ACTIVATE | NONE | |
| buttons/switch02.wav | BUTTON | ACTIVATE | NONE | |
| buttons/switch04.wav | BUTTON | ACTIVATE | NONE | |

### Teleporters

| Sound | Subject | Action | Source | Notes |
|-------|---------|--------|--------|-------|
| misc/r_tele1-5.wav | TELEPORTER | TELEPORT | NONE | |

## Vocab Summary

### Subjects (used in events)

PLAYER, HEALTH, MEGAHEALTH, QUAD, PENT, RING, SUIT, PROJECTILE_GRENADE, DOOR, PLATFORM, TRAIN, BUTTON, TELEPORTER

### Actions

FIRE, PAIN, DEATH, JUMP, LAND, ENTER, EXIT, BREATH, PICKUP, ACTIVE, ENDING, DISCONNECT, RESPAWN, BOUNCE, MOVE, ACTIVATE, REJECT, TELEPORT

### Sources

NONE, GIB, AXE, SHOTGUN, NAILGUN, GRENADE_LAUNCHER, ROCKET_LAUNCHER, THUNDERBOLT, HEALTH, MEGAHEALTH, ARMOR, WEAPON, AMMO, QUAD, PENT, RING, SUIT, DROWN, WATER, LAVA, SLIME, SECRET, KEYED

Note: Source shares IDs with entity subject_id vocab (weapons, powerups). Additional source-only IDs: ARMOR, WEAPON, AMMO, DROWN, WATER, LAVA, SLIME, SECRET, KEYED.

## Changes from v1

- Subject is always the entity owner (PLAYER for player events, not the weapon)
- Qualifier renamed to Source — what caused/involved in the event
- No magnitude scalar — GIB is its own action, weapon variants implicit in class
- Weapon fire: subject PLAYER, source is the weapon (was: subject was the weapon)
- Powerup active/ending: subject PLAYER, source is the powerup (was: subject was the powerup)
- Pickups: subject PLAYER, source is category (HEALTH, ARMOR, WEAPON, AMMO, powerup)
- axhit1.wav: PLAYER/PAIN/AXE (was: AXE/IMPACT/FLESH)
- lhit.wav: PLAYER/FIRE/THUNDERBOLT (was: LIGHTNING_BEAM/IMPACT/NONE)
- gib/udeath: PLAYER/DEATH/GIB (was: PLAYER/DEATH/NONE with magnitude 1.0)
- GIB added as source (PLAYER/DEATH/GIB)
- IMPACT action dropped
- FLESH, WORLD, INVISIBLE qualifiers dropped
- WARNING renamed to ENDING
- New source IDs: ARMOR, WEAPON, AMMO, GIB

## Open Questions

- **Vector summing vs concatenation**: How should multiple embed IDs (subject, action, source; entity subject + modality + events) be combined into token representations? Summing loses information, concatenation is expensive. Need to discuss tradeoffs.
