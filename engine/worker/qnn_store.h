/*
 * qnn_store.h — Unified entity store.
 *
 * One struct, one array, indexed by entity_num like the engine's
 * edict array.  Every entry has all fields; type tag says which
 * are valid.  BSP-only entities (trigger_teleport, trigger_push)
 * go in overflow slots after MAX_EDICTS.
 */

#ifndef QNN_STORE_H
#define QNN_STORE_H

#include "qnn.h"
#include "qnn_vocab.h"

/* ── Entity types ─────────────────────────────────────────────── */

#define QNN_ENT_NONE        0
#define QNN_ENT_ITEM        1
#define QNN_ENT_MOVER       2
#define QNN_ENT_ACTOR       3
#define QNN_ENT_PROJECTILE  4
#define QNN_ENT_BACKPACK    5
#define QNN_ENT_TELEPORTER  6
#define QNN_ENT_PUSH        7

/* ── Item definition table ────────────────────────────────────── */

typedef struct {
	const char *classname;
	int spawnflags_mask;
	int spawnflags_value;
	int subject_id;
	int amount;
	float regen_time;
} qnn_item_def_t;

extern const qnn_item_def_t qnn_item_defs[];
extern const int qnn_item_def_count;

/* ── Mover defaults table ─────────────────────────────────────── */

typedef struct {
	int subject_id;
	float default_speed;
	float default_wait;
} qnn_mover_def_t;

extern const qnn_mover_def_t qnn_mover_defs[];
extern const int qnn_mover_def_count;

/* ── Unified entity entry ─────────────────────────────────────── */

#define QNN_STORE_OVERFLOW 128

typedef struct {
	int type;              /* QNN_ENT_* */
	int subject_id;
	int qualifier_id;
	int entity_num;

	/* World state */
	vec3_t origin;
	vec3_t velocity;
	vec3_t angles;

	/* Observation timestamps (cl.mtime[0] when last observed, 0 = never) */
	float pvs;             /* last in PVS transport */
	float vis;             /* last visible (PVS + FOV for actors, == pvs for others) */
	float snd;             /* last heard via sound */
	float mem;             /* last recalled by model */

	/* Item */
	int amount;
	float regen;
	float regen_time;

	/* Mover */
	float speed;
	float wait;
	float state;
	vec3_t baseline_origin;

	/* Actor */
	int colormap;
	int effects;
	int weapon_subject_id;
	int powerup_subject_id;
	float powerup_warning_elapsed;

	/* Teleporter */
	vec3_t destination;

	/* Push */
	float push_speed;
	vec3_t push_direction;
} qnn_entity_t;

/* ── Store globals ────────────────────────────────────────────── */

extern qnn_entity_t qnn_store[MAX_EDICTS + QNN_STORE_OVERFLOW];
extern int qnn_store_overflow_count; /* BSP-only entries after MAX_EDICTS */

/* ── Store API ────────────────────────────────────────────────── */

void QNN_StoreInit(const qnn_map_state_t *map_state);
void QNN_StoreUpdate(const qnn_snapshot_t *snapshot, float emit_dt);

/* Total store capacity (MAX_EDICTS + full overflow pool) */
static inline int QNN_StoreCapacity(void)
{
	return MAX_EDICTS + QNN_STORE_OVERFLOW;
}

/* Debug dump */
void QNN_StoreDumpTick(FILE *out, int tick, float server_time);
void QNN_StoreDumpSounds(FILE *out, int tick, const qnn_snapshot_t *snapshot);

#endif /* QNN_STORE_H */
