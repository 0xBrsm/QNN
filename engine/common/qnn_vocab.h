/*
 * qnn_vocab.h — Shared semantic vocabulary for the QNN token protocol.
 *
 * Subject, action, qualifier, modality, and spatial sector IDs used across
 * state classification, oracle entity store, self/spatial tokens, and obs
 * buffer packing.  Must match Python src/quake_ai/vocab.py.
 */

#ifndef QNN_VOCAB_H
#define QNN_VOCAB_H

/* ── Modality (priority order: lower = higher priority) ────────── */

#define QNN_MODALITY_SIGHT     0
#define QNN_MODALITY_PROXIMITY 1
#define QNN_MODALITY_SOUND     2
#define QNN_MODALITY_MEMORY    3
#define QNN_MODALITY_VOCAB_SIZE 4

/* ── Subject IDs ───────────────────────────────────────────────── */
/* Weapons occupy 3..10 in Quake impulse order so the embed rows for
 * shotgun/SSG and nailgun/SNG sit next to each other; everything else
 * follows in original order with a +2 shift to make room. */

#define QNN_SUBJECT_NONE               0
#define QNN_SUBJECT_PLAYER             1
#define QNN_SOURCE_WEAPON              2
#define QNN_SUBJECT_AXE                3
#define QNN_SUBJECT_SHOTGUN            4
#define QNN_SUBJECT_SUPER_SHOTGUN      5
#define QNN_SUBJECT_NAILGUN            6
#define QNN_SUBJECT_SUPER_NAILGUN      7
#define QNN_SUBJECT_GRENADE_LAUNCHER   8
#define QNN_SUBJECT_ROCKET_LAUNCHER    9
#define QNN_SUBJECT_THUNDERBOLT       10
#define QNN_SOURCE_AMMO               11
#define QNN_SUBJECT_SHELLS            12
#define QNN_SUBJECT_NAILS             13
#define QNN_SUBJECT_ROCKETS           14
#define QNN_SUBJECT_CELLS             15
#define QNN_SUBJECT_BACKPACK          16
#define QNN_SOURCE_ARMOR              17
#define QNN_SUBJECT_ARMOR_GREEN       18
#define QNN_SUBJECT_ARMOR_YELLOW      19
#define QNN_SUBJECT_ARMOR_RED         20
#define QNN_SUBJECT_HEALTH            21
#define QNN_SUBJECT_MEGAHEALTH        22
#define QNN_SUBJECT_POWERUP           23
#define QNN_SUBJECT_QUAD              24
#define QNN_SUBJECT_PENT              25
#define QNN_SUBJECT_RING              26
#define QNN_SUBJECT_SUIT              27
#define QNN_SUBJECT_PROJECTILE_NAIL   28
#define QNN_SUBJECT_PROJECTILE_GRENADE 29
#define QNN_SUBJECT_PROJECTILE_ROCKET 30
#define QNN_SUBJECT_LIGHTNING_BEAM    31
#define QNN_SOURCE_GROUND             32
#define QNN_SOURCE_WATER              33
#define QNN_SOURCE_SLIME              34
#define QNN_SOURCE_LAVA               35
#define QNN_SOURCE_GIB                36
#define QNN_SUBJECT_BUTTON            37
#define QNN_SUBJECT_PLATFORM          38
#define QNN_SUBJECT_TELEPORTER        39
#define QNN_SUBJECT_DOOR              40
#define QNN_SOURCE_KEYED              41
#define QNN_SOURCE_SECRET             42
#define QNN_SUBJECT_TRAIN             43

/* ── Action IDs ────────────────────────────────────────────────── */

#define QNN_ACTION_NONE         0
#define QNN_ACTION_FIRE         1
#define QNN_ACTION_JUMP         2
#define QNN_ACTION_LAND         3
#define QNN_ACTION_PICKUP       4
#define QNN_ACTION_ENTER        5
#define QNN_ACTION_BREATH       6
#define QNN_ACTION_EXIT         7
#define QNN_ACTION_PAIN         8
#define QNN_ACTION_DEATH        9
#define QNN_ACTION_CONNECT     10
#define QNN_ACTION_DISCONNECT  11
#define QNN_ACTION_RESPAWN     12
#define QNN_ACTION_ACTIVE      13
#define QNN_ACTION_ENDING      14
#define QNN_ACTION_BOUNCE      15
#define QNN_ACTION_TELEPORT    16
#define QNN_ACTION_MOVE        17
#define QNN_ACTION_ACTIVATE    18
#define QNN_ACTION_REJECT      19
#define QNN_ACTION_COUNT       20

#define QNN_SOURCE_NONE    0  /* alias for QNN_SUBJECT_NONE */
#define QNN_ENTITY_VOCAB_SIZE 44  /* total entries in shared subject/source table */

/* ── Spatial sector IDs ────────────────────────────────────────── */

#define QNN_SPATIAL_FOV_CENTER   0
#define QNN_SPATIAL_FOV_LEFT     1
#define QNN_SPATIAL_FOV_RIGHT    2
#define QNN_SPATIAL_FLANK_LEFT   3
#define QNN_SPATIAL_FLANK_RIGHT  4
#define QNN_SPATIAL_REAR_LEFT    5
#define QNN_SPATIAL_REAR_RIGHT   6
#define QNN_SPATIAL_GROUND       7
#define QNN_SPATIAL_CEILING      8

/* ── Tuning constants ──────────────────────────────────────────── */

#define QNN_FOV_HALF_DEG               60.0f
#define QNN_RECENCY_MAX_EVENT             0.1f

/* Per-modality recency thresholds (seconds) */
#define QNN_RECENCY_MAX_SIGHT      2.0f
#define QNN_RECENCY_MAX_PROXIMITY  0.1f
#define QNN_RECENCY_MAX_SOUND      0.1f
#define QNN_RECENCY_MAX_MEMORY     1.0f

static inline float QNN_RecencyMaxForModality(int modality)
{
	switch (modality)
	{
	case QNN_MODALITY_SIGHT:     return QNN_RECENCY_MAX_SIGHT;
	case QNN_MODALITY_PROXIMITY: return QNN_RECENCY_MAX_PROXIMITY;
	case QNN_MODALITY_SOUND:     return QNN_RECENCY_MAX_SOUND;
	case QNN_MODALITY_MEMORY:    return QNN_RECENCY_MAX_MEMORY;
	default:                     return 0.0f;
	}
}

#define QNN_ITEM_PVS_MATCH_SQ         (16.0f * 16.0f)
#define QNN_ITEM_PICKUP_MATCH_SQ      (64.0f * 64.0f)
#define QNN_ITEM_PICKUP_PLAYER_SQ     (96.0f * 96.0f)
#define QNN_ITEM_RESPAWN_MATCH_SQ     (16.0f * 16.0f)
#define QNN_STATIC_SOUND_MATCH_SQ    (128.0f * 128.0f)
#define QNN_PROJECTILE_SOUND_MATCH_SQ (96.0f * 96.0f)
#define QNN_MAX_MATCH_CANDIDATES      4096

#define QNN_NAIL_STREAM_DOT_THRESHOLD  0.97f
#define QNN_MAX_NAIL_STREAMS           16

/* Game resource caps — used for normalization everywhere. */
#define QNN_MAX_HEALTH     100.0f   /* normal health cap (mega decays to this) */
#define QNN_MAX_ARMOR      160.0f   /* max effective armor HP: red 200 * 0.8 */
#define QNN_MAX_SHELLS     100.0f
#define QNN_MAX_NAILS      200.0f
#define QNN_MAX_ROCKETS    100.0f
#define QNN_MAX_CELLS      100.0f

#define QNN_EF_DIMLIGHT 8  /* EF_DIMLIGHT from quake engine — powerup glow */

/* ── Inline helpers ────────────────────────────────────────────── */

static inline int QNN_SubjectIsItem(int subject_id)
{
	/* Static map items: ammo, armor, health, powerups — the contiguous
	 * SHELLS..SUIT block.  Backpacks (dynamic drops) are intentionally
	 * excluded. */
	return subject_id >= QNN_SUBJECT_SHELLS && subject_id <= QNN_SUBJECT_SUIT;
}

#endif /* QNN_VOCAB_H */
