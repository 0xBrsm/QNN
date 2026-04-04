/*
 * qnn_vocab.h — Shared semantic vocabulary for the QNN token protocol.
 *
 * Subject, action, qualifier, modality, and spatial sector IDs used across
 * state classification, oracle entity store, self/spatial tokens, and obs
 * buffer packing.  Must match Python src/quake_ai/vocab.py.
 */

#ifndef QNN_VOCAB_H
#define QNN_VOCAB_H

/* ── Modality priority (lower = higher priority) ───────────────── */

#define QNN_MODALITY_NONE      0
#define QNN_MODALITY_SIGHT     1
#define QNN_MODALITY_PROXIMITY 2
#define QNN_MODALITY_SOUND     3
#define QNN_MODALITY_MEMORY    4

/* ── Subject IDs ───────────────────────────────────────────────── */

#define QNN_SUBJECT_NONE               0
#define QNN_SUBJECT_PLAYER             1
#define QNN_SUBJECT_BACKPACK           2
#define QNN_SUBJECT_AXE                3
#define QNN_SUBJECT_SHOTGUN            4
#define QNN_SUBJECT_NAILGUN            5
#define QNN_SUBJECT_GRENADE_LAUNCHER   6
#define QNN_SUBJECT_ROCKET_LAUNCHER    7
#define QNN_SUBJECT_THUNDERBOLT        8
#define QNN_SUBJECT_SHELLS             9
#define QNN_SUBJECT_NAILS             10
#define QNN_SUBJECT_ROCKETS           11
#define QNN_SUBJECT_CELLS             12
#define QNN_SUBJECT_ARMOR_GREEN       13
#define QNN_SUBJECT_ARMOR_YELLOW      14
#define QNN_SUBJECT_ARMOR_RED         15
#define QNN_SUBJECT_HEALTH            16
#define QNN_SUBJECT_MEGAHEALTH        17
#define QNN_SUBJECT_QUAD              18
#define QNN_SUBJECT_PENT              19
#define QNN_SUBJECT_RING              20
#define QNN_SUBJECT_SUIT              21
#define QNN_SUBJECT_POWERUP           22
#define QNN_SUBJECT_PROJECTILE_NAIL   23
#define QNN_SUBJECT_PROJECTILE_GRENADE 24
#define QNN_SUBJECT_PROJECTILE_ROCKET 25
#define QNN_SUBJECT_LIGHTNING_BEAM    26
#define QNN_SUBJECT_TELEPORTER        27
#define QNN_SUBJECT_DOOR              28
#define QNN_SUBJECT_PLATFORM          29
#define QNN_SUBJECT_TRAIN             30
#define QNN_SUBJECT_BUTTON            31
#define QNN_SUBJECT_COUNT             32

/* ── Action IDs ────────────────────────────────────────────────── */

#define QNN_ACTION_NONE      0
#define QNN_ACTION_FIRE      1
#define QNN_ACTION_IMPACT    2
#define QNN_ACTION_BOUNCE    3
#define QNN_ACTION_PICKUP    4
#define QNN_ACTION_RESPAWN   5
#define QNN_ACTION_PAIN      6
#define QNN_ACTION_DEATH     7
#define QNN_ACTION_WARNING   8
#define QNN_ACTION_ACTIVE    9
#define QNN_ACTION_JUMP     10
#define QNN_ACTION_LAND     11
#define QNN_ACTION_ENTER    12
#define QNN_ACTION_EXIT     13
#define QNN_ACTION_TELEPORT 14
#define QNN_ACTION_MOVE     15
#define QNN_ACTION_ACTIVATE 16
#define QNN_ACTION_REJECT   17
#define QNN_ACTION_BREATH      18
#define QNN_ACTION_DISCONNECT  19
#define QNN_ACTION_CONNECT     20

/* ── Qualifier IDs ─────────────────────────────────────────────── */

#define QNN_QUAL_NONE      0
#define QNN_QUAL_DROWN     1
#define QNN_QUAL_WATER     2
#define QNN_QUAL_LAVA      3
#define QNN_QUAL_SLIME     4
#define QNN_QUAL_FLESH     5
#define QNN_QUAL_WORLD     6
#define QNN_QUAL_KEYED     7
#define QNN_QUAL_SECRET    8
#define QNN_QUAL_INVISIBLE 9

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
#define QNN_RECENCY_DECAY_S             0.1f
#define QNN_RECENCY_DECAY_PLAYER_S      2.0f

#define QNN_ITEM_PVS_MATCH_SQ         (16.0f * 16.0f)
#define QNN_ITEM_PICKUP_MATCH_SQ      (64.0f * 64.0f)
#define QNN_ITEM_PICKUP_PLAYER_SQ     (96.0f * 96.0f)
#define QNN_ITEM_RESPAWN_MATCH_SQ     (16.0f * 16.0f)
#define QNN_STATIC_SOUND_MATCH_SQ    (128.0f * 128.0f)
#define QNN_PROJECTILE_SOUND_MATCH_SQ (96.0f * 96.0f)
#define QNN_MAX_MATCH_CANDIDATES      4096

#define QNN_NAIL_STREAM_DOT_THRESHOLD  0.97f
#define QNN_MAX_NAIL_STREAMS           16

#define QNN_SELF_HEALTH_CAP    100.0f
#define QNN_SELF_ARMOR_CAP     200.0f
#define QNN_SELF_SHELLS_CAP    100.0f
#define QNN_SELF_NAILS_CAP     200.0f
#define QNN_SELF_ROCKETS_CAP   100.0f
#define QNN_SELF_CELLS_CAP     100.0f

#define QNN_EF_DIMLIGHT 8  /* EF_DIMLIGHT from quake engine — powerup glow */

/* ── Inline helpers ────────────────────────────────────────────── */

static inline int QNN_SubjectIsItem(int subject_id)
{
	/* Ammo (9-12), armor (13-15), health (16-17), powerups (18-21).
	   Backpack (2) is a dynamic drop, not a static item. */
	return subject_id >= QNN_SUBJECT_SHELLS && subject_id <= QNN_SUBJECT_SUIT;
}

#endif /* QNN_VOCAB_H */
