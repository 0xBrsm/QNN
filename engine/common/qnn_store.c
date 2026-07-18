/*
 * qnn_store.c — Unified entity store.
 *
 * One array indexed by entity_num, like the engine's edict array.
 * Static entities (items, movers) populated at init from baselines + BSP.
 * Dynamic entities (players, projectiles, backpacks) created at runtime.
 * BSP-only entities (teleporters, push triggers) in overflow slots.
 */

#include "qnn_store.h"
#include "qnn_object.h"
#include "qnn_map.h"
#include "qnn_context.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

/* ══════════════════════════════════════════════════════════════════
 * Item / mover definition tables
 * ══════════════════════════════════════════════════════════════════ */

const qnn_item_def_t qnn_item_defs[] = {
	{"item_health",    2, 2, QNN_SUBJECT_MEGAHEALTH,      100, 120.0f},
	{"item_health",    1, 1, QNN_SUBJECT_HEALTH,            15,  20.0f},
	{"item_health",    0, 0, QNN_SUBJECT_HEALTH,            25,  20.0f},
	{"item_health_rotten", 0, 0, QNN_SUBJECT_HEALTH,        15,  20.0f},
	{"item_health_mega",   0, 0, QNN_SUBJECT_MEGAHEALTH,   100, 120.0f},
	{"item_armor1",    0, 0, QNN_SUBJECT_ARMOR_GREEN,      100,  20.0f},
	{"item_armor2",    0, 0, QNN_SUBJECT_ARMOR_YELLOW,     150,  20.0f},
	{"item_armorInv",  0, 0, QNN_SUBJECT_ARMOR_RED,        200,  20.0f},
	{"item_shells",    1, 1, QNN_SUBJECT_SHELLS,            40,  30.0f},
	{"item_shells",    0, 0, QNN_SUBJECT_SHELLS,            20,  30.0f},
	{"item_spikes",    1, 1, QNN_SUBJECT_NAILS,             50,  30.0f},
	{"item_spikes",    0, 0, QNN_SUBJECT_NAILS,             25,  30.0f},
	{"item_rockets",   1, 1, QNN_SUBJECT_ROCKETS,           10,  30.0f},
	{"item_rockets",   0, 0, QNN_SUBJECT_ROCKETS,            5,  30.0f},
	{"item_cells",     1, 1, QNN_SUBJECT_CELLS,             12,  30.0f},
	{"item_cells",     0, 0, QNN_SUBJECT_CELLS,              6,  30.0f},
	{"weapon_supershotgun",    0, 0, QNN_SUBJECT_SUPER_SHOTGUN,     5, 30.0f},
	{"weapon_nailgun",         0, 0, QNN_SUBJECT_NAILGUN,          30, 30.0f},
	{"weapon_supernailgun",    0, 0, QNN_SUBJECT_SUPER_NAILGUN,    30, 30.0f},
	{"weapon_grenadelauncher", 0, 0, QNN_SUBJECT_GRENADE_LAUNCHER,  5, 30.0f},
	{"weapon_rocketlauncher",  0, 0, QNN_SUBJECT_ROCKET_LAUNCHER,   5, 30.0f},
	{"weapon_lightning",       0, 0, QNN_SUBJECT_THUNDERBOLT,       15, 30.0f},
	{"item_artifact_super_damage",       0, 0, QNN_SUBJECT_QUAD,  1,  60.0f},
	{"item_artifact_invulnerability",    0, 0, QNN_SUBJECT_PENT,  1, 300.0f},
	{"item_artifact_invisibility",       0, 0, QNN_SUBJECT_RING,  1, 300.0f},
	{"item_artifact_envirosuit",         0, 0, QNN_SUBJECT_SUIT,  1,  60.0f},
};
const int qnn_item_def_count = (int)(sizeof(qnn_item_defs) / sizeof(qnn_item_defs[0]));

const qnn_mover_def_t qnn_mover_defs[] = {
	{QNN_SUBJECT_DOOR,     100.0f,  3.0f},
	{QNN_SUBJECT_PLATFORM, 150.0f,  3.0f},
	{QNN_SUBJECT_TRAIN,    100.0f,  0.0f},
	{QNN_SUBJECT_BUTTON,    40.0f,  1.0f},
};
const int qnn_mover_def_count = (int)(sizeof(qnn_mover_defs) / sizeof(qnn_mover_defs[0]));

/* ══════════════════════════════════════════════════════════════════
 * Global state
 * ══════════════════════════════════════════════════════════════════ */

qnn_entity_t qnn_store[MAX_EDICTS + QNN_STORE_OVERFLOW];
int qnn_store_overflow_count;

/* Track previous presence for projectile/backpack deletion */
static qboolean qnn_ephemeral_present[MAX_EDICTS];

/* Track scoreboard for connect/disconnect */
static qboolean qnn_player_present[MAX_EDICTS];

/* Primary observation source by entity type. Keep this centralized so
 * token qualification, metrics, and reward all key off the same source. */
#define QNN_PRIMARY_OBS_ACTOR      QNN_PRIMARY_OBS_VIS
#define QNN_PRIMARY_OBS_ITEM       QNN_PRIMARY_OBS_PVS
#define QNN_PRIMARY_OBS_MOVER      QNN_PRIMARY_OBS_PVS
#define QNN_PRIMARY_OBS_PROJECTILE QNN_PRIMARY_OBS_PVS
#define QNN_PRIMARY_OBS_BACKPACK   QNN_PRIMARY_OBS_PVS
#define QNN_PRIMARY_OBS_TELEPORTER QNN_PRIMARY_OBS_PVS
#define QNN_PRIMARY_OBS_PUSH       QNN_PRIMARY_OBS_PVS

/* ══════════════════════════════════════════════════════════════════
 * Helpers
 * ══════════════════════════════════════════════════════════════════ */

static const char *QNN_RawProp(const qnn_raw_entity_t *r, const char *key)
{
	int i;
	for (i = 0; i < r->property_count; ++i)
		if (!strcmp(r->properties[i].key, key))
			return r->properties[i].value;
	return NULL;
}

static float QNN_RawPropFloat(const qnn_raw_entity_t *r, const char *key, float fb)
{
	const char *v = QNN_RawProp(r, key);
	return (v && v[0]) ? (float)atof(v) : fb;
}

static const qnn_item_def_t *QNN_LookupItem(const char *classname, int spawnflags)
{
	int i;
	for (i = 0; i < qnn_item_def_count; ++i)
	{
		if (strcasecmp(qnn_item_defs[i].classname, classname) != 0)
			continue;
		if (qnn_item_defs[i].spawnflags_mask == 0
			|| (spawnflags & qnn_item_defs[i].spawnflags_mask) == qnn_item_defs[i].spawnflags_value)
			return &qnn_item_defs[i];
	}
	return NULL;
}

static void QNN_LookupMoverDef(int subject_id, float *speed, float *wait)
{
	int i;
	for (i = 0; i < qnn_mover_def_count; ++i)
	{
		if (qnn_mover_defs[i].subject_id == subject_id)
		{
			*speed = qnn_mover_defs[i].default_speed;
			*wait = qnn_mover_defs[i].default_wait;
			return;
		}
	}
	*speed = 100.0f;
	*wait = 0.0f;
}

/* ── Mover classname → subject mapping for StoreInit ─────────── */

typedef struct {
	const char *classname;
	int prefix_len;
	int subject_id;
	qboolean read_wait;   /* true = read "wait" from BSP props */
} qnn_mover_init_t;

static const qnn_mover_init_t qnn_mover_init[] = {
	{"func_door",   9,  QNN_SUBJECT_DOOR,     true},
	{"func_plat",   9,  QNN_SUBJECT_PLATFORM,  false},
	{"func_train",  10, QNN_SUBJECT_TRAIN,     false},
	{"func_button", 11, QNN_SUBJECT_BUTTON,    true},
};
static const int qnn_mover_init_count = sizeof(qnn_mover_init) / sizeof(qnn_mover_init[0]);

int QNN_PrimaryObservationSourceForType(int entity_type)
{
	switch (entity_type)
	{
	case QNN_ENT_ACTOR:      return QNN_PRIMARY_OBS_ACTOR;
	case QNN_ENT_ITEM:       return QNN_PRIMARY_OBS_ITEM;
	case QNN_ENT_MOVER:      return QNN_PRIMARY_OBS_MOVER;
	case QNN_ENT_PROJECTILE: return QNN_PRIMARY_OBS_PROJECTILE;
	case QNN_ENT_BACKPACK:   return QNN_PRIMARY_OBS_BACKPACK;
	case QNN_ENT_TELEPORTER: return QNN_PRIMARY_OBS_TELEPORTER;
	case QNN_ENT_PUSH:       return QNN_PRIMARY_OBS_PUSH;
	default:                 return QNN_PRIMARY_OBS_PVS;
	}
}

float QNN_PrimaryObservationTimestamp(const qnn_entity_t *entity)
{
	if (entity == NULL)
		return 0.0f;
	return (QNN_PrimaryObservationSourceForType(entity->type) == QNN_PRIMARY_OBS_VIS)
		? entity->vis
		: entity->pvs;
}

int QNN_PrimaryObservationModalityId(const qnn_entity_t *entity)
{
	if (entity == NULL)
		return QNN_MODALITY_PROXIMITY;
	return (QNN_PrimaryObservationSourceForType(entity->type) == QNN_PRIMARY_OBS_VIS)
		? QNN_MODALITY_SIGHT
		: QNN_MODALITY_PROXIMITY;
}

qboolean QNN_PrimaryObservationIsCurrent(const qnn_entity_t *entity, float now)
{
	return (QNN_PrimaryObservationTimestamp(entity) == now) ? true : false;
}

void QNN_LookupEntityBounds(int entity_num, float *out_half, vec3_t out_center_adjust)
{
	entity_t *entity;

	out_half[0] = out_half[1] = out_half[2] = 0.0f;
	out_center_adjust[0] = out_center_adjust[1] = out_center_adjust[2] = 0.0f;
	if (entity_num <= 0 || entity_num >= MAX_EDICTS)
		return;
	entity = &cl_entities[entity_num];
	if (entity->model == NULL)
		return;
	{
		vec3_t bmins, bmaxs;
		VectorCopy(entity->model->mins, bmins);
		VectorCopy(entity->model->maxs, bmaxs);
		out_center_adjust[0] = (bmins[0] + bmaxs[0]) * 0.5f;
		out_center_adjust[1] = (bmins[1] + bmaxs[1]) * 0.5f;
		out_center_adjust[2] = (bmins[2] + bmaxs[2]) * 0.5f;
		out_half[0] = (bmaxs[0] - bmins[0]) * 0.5f;
		out_half[1] = (bmaxs[1] - bmins[1]) * 0.5f;
		out_half[2] = (bmaxs[2] - bmins[2]) * 0.5f;
	}
}

void QNN_EntityAnchorFromModel(int entity_num, const vec3_t raw_origin, vec3_t out_anchor, float *out_half)
{
	float local_half[3];
	vec3_t center_adjust;

	QNN_LookupEntityBounds(entity_num, local_half, center_adjust);
	VectorCopy(raw_origin, out_anchor);
	out_anchor[0] += center_adjust[0];
	out_anchor[1] += center_adjust[1];
	out_anchor[2] += center_adjust[2];
	if (out_half != NULL)
	{
		out_half[0] = local_half[0];
		out_half[1] = local_half[1];
		out_half[2] = local_half[2];
	}
}

/* Stamp PVS + visibility timestamps. */
static void QNN_StampPvs(qnn_entity_t *e, float now, qboolean in_fov)
{
	e->pvs = now;
	if (in_fov)
		e->vis = now;
}

/* Only ACTORS teleport (slipgates, respawns); projectiles/items/movers
 * obey physics and never do, so they keep their finite-diff velocity
 * unclamped (slot-reuse / first-sight garbage is handled by valid_prev,
 * not a magnitude limit). The engine clamps live player velocity to
 * sv_maxvelocity (~2000 u/s) PER AXIS, so any per-component speed beyond
 * that is a teleport — which jumps a whole map segment in one native
 * frame (0.05-0.073 s = tens of thousands of u/s). A per-axis threshold
 * just above the cap rejects teleports while preserving fast real motion
 * (a diagonal/vertical burst can reach ~2000 on each axis, ~3464
 * magnitude — a single magnitude limit would clip it). */
#define QNN_ACTOR_TELEPORT_LIMIT 2100.0f

/* Compute velocity from positional delta over one emit interval.
 * `valid_prev` must be false on first-sight or when this store slot's
 * previous origin belonged to a different entity (edict-slot reuse) —
 * finite-differencing across an identity change yields a cross-map
 * delta that is not real movement.  In that case velocity is zeroed
 * (no prior position to difference against) rather than fabricated. */
static void QNN_ComputeStoreVelocity(qnn_entity_t *e, const vec3_t new_origin,
	float emit_dt, qboolean valid_prev)
{
	vec3_t delta;
	if (!valid_prev)
	{
		/* No trustworthy prior sample (first sight / slot reuse) — velocity
		 * is unknown, not zero-because-stationary, but zero is the honest
		 * "no info" emission. */
		e->velocity[0] = 0.0f;
		e->velocity[1] = 0.0f;
		e->velocity[2] = 0.0f;
		return;
	}
	VectorSubtract(new_origin, e->origin, delta);
	VectorScale(delta, 1.0f / emit_dt, e->velocity);
	/* Teleport rejection — actors only, per axis (see QNN_ACTOR_TELEPORT_LIMIT).
	 * Non-actors are never clamped: their finite-diff is real velocity. A
	 * teleported player's post-teleport velocity is unknown, so zero it. */
	if (e->type == QNN_ENT_ACTOR
		&& (fabsf(e->velocity[0]) > QNN_ACTOR_TELEPORT_LIMIT
		||  fabsf(e->velocity[1]) > QNN_ACTOR_TELEPORT_LIMIT
		||  fabsf(e->velocity[2]) > QNN_ACTOR_TELEPORT_LIMIT))
	{
		e->velocity[0] = 0.0f;
		e->velocity[1] = 0.0f;
		e->velocity[2] = 0.0f;
	}
}

static qboolean QNN_IsEphemeral(int subject_id)
{
	return (subject_id == QNN_SUBJECT_PROJECTILE_NAIL
		|| subject_id == QNN_SUBJECT_PROJECTILE_GRENADE
		|| subject_id == QNN_SUBJECT_PROJECTILE_ROCKET
		|| subject_id == QNN_SUBJECT_LIGHTNING_BEAM
		|| subject_id == QNN_SUBJECT_BACKPACK);
}

/* ══════════════════════════════════════════════════════════════════
 * Init
 * ══════════════════════════════════════════════════════════════════ */

void QNN_StoreInit(const qnn_map_state_t *map_state)
{
	qnn_raw_entity_t *baseline_raw;
	qnn_raw_entity_t *bsp_raw;
	int baseline_count, bsp_count;
	int i;

	memset(qnn_store, 0, sizeof(qnn_store));
	memset(qnn_ephemeral_present, 0, sizeof(qnn_ephemeral_present));
	memset(qnn_player_present, 0, sizeof(qnn_player_present));
	qnn_store_overflow_count = 0;

	baseline_raw = (qnn_raw_entity_t *)calloc(QNN_MAX_RAW_ENTITIES, sizeof(*baseline_raw));
	bsp_raw = (qnn_raw_entity_t *)calloc(QNN_MAX_RAW_ENTITIES, sizeof(*bsp_raw));
	if (!baseline_raw || !bsp_raw)
	{
		free(baseline_raw);
		free(bsp_raw);
		return;
	}

	baseline_count = QNN_MapBuildFromBaselines(baseline_raw, QNN_MAX_RAW_ENTITIES);
	bsp_count = QNN_MapParseEntities(bsp_raw, QNN_MAX_RAW_ENTITIES);

	/* Pass 1: baselines → items and movers at their entity_num */
	for (i = 0; i < baseline_count; ++i)
	{
		qnn_raw_entity_t *r = &baseline_raw[i];
		qnn_entity_t *e;
		const qnn_item_def_t *item;

		if (r->entity_num <= 0 || r->entity_num >= MAX_EDICTS)
			continue;
		e = &qnn_store[r->entity_num];

		item = QNN_LookupItem(r->classname, r->spawnflags);
		if (item)
		{

			e->type = QNN_ENT_ITEM;
			e->subject_id = item->subject_id;
			e->entity_num = r->entity_num;
			VectorCopy(r->origin, e->origin);
			e->amount = item->amount;
			e->regen = 0.0f;
			e->regen_time = item->regen_time;
			continue;
		}

		{
			int mi;
			for (mi = 0; mi < qnn_mover_init_count; ++mi)
			{
				const qnn_mover_init_t *m = &qnn_mover_init[mi];
				if (!strncasecmp(r->classname, m->classname, m->prefix_len))
				{
					float ds, dw;
					QNN_LookupMoverDef(m->subject_id, &ds, &dw);
					e->type = QNN_ENT_MOVER;
					e->subject_id = m->subject_id;
					e->entity_num = r->entity_num;
					VectorCopy(r->origin, e->origin);
					VectorCopy(r->origin, e->baseline_origin);
					e->speed = QNN_RawPropFloat(r, "speed", ds);
					e->wait = m->read_wait ? QNN_RawPropFloat(r, "wait", dw) : dw;
					break;
				}
			}
			if (mi < qnn_mover_init_count)
				continue;
		}
	}

	/* Pass 2: BSP → movers missing from baselines + teleporters + push */
	for (i = 0; i < bsp_count; ++i)
	{
		qnn_raw_entity_t *r = &bsp_raw[i];

		/* Movers from BSP (may overlap with baselines) */
		{
			int mi;
			qboolean is_mover = false;
			for (mi = 0; mi < qnn_mover_init_count; ++mi)
			{
				if (!strncasecmp(r->classname, qnn_mover_init[mi].classname, qnn_mover_init[mi].prefix_len))
				{ is_mover = true; break; }
			}
			if (is_mover)
				continue;
		}

		/* Teleporters → overflow */
		if (!strcasecmp(r->classname, "trigger_teleport"))
		{
			int idx = MAX_EDICTS + qnn_store_overflow_count;
			if (idx < MAX_EDICTS + QNN_STORE_OVERFLOW)
			{
				qnn_entity_t *e = &qnn_store[idx];
				const char *target;
	
				e->type = QNN_ENT_TELEPORTER;
				e->subject_id = QNN_SUBJECT_TELEPORTER;
				if (r->has_origin) VectorCopy(r->origin, e->origin);
				target = QNN_RawProp(r, "target");
				if (target)
				{
					int j;
					for (j = 0; j < bsp_count; ++j)
					{
						const char *tn = QNN_RawProp(&bsp_raw[j], "targetname");
						if (tn && !strcmp(tn, target) && bsp_raw[j].has_origin)
						{
							VectorCopy(bsp_raw[j].origin, e->destination);
							break;
						}
					}
				}
				qnn_store_overflow_count++;
			}
			continue;
		}

		/* Push triggers → overflow */
		if (!strcasecmp(r->classname, "trigger_push"))
		{
			int idx = MAX_EDICTS + qnn_store_overflow_count;
			if (idx < MAX_EDICTS + QNN_STORE_OVERFLOW)
			{
				qnn_entity_t *e = &qnn_store[idx];
				float angle;

				e->type = QNN_ENT_PUSH;
				e->subject_id = QNN_SUBJECT_NONE;
				if (r->has_origin) VectorCopy(r->origin, e->origin);
				e->push_speed = QNN_RawPropFloat(r, "speed", 1000.0f);

				/* Quake movedir from "angle" key:
				 * -1 = up, -2 = down, else yaw in degrees. */
				angle = QNN_RawPropFloat(r, "angle", 0.0f);
				if (angle == -1.0f)
				{
					e->push_direction[0] = 0; e->push_direction[1] = 0; e->push_direction[2] = 1;
				}
				else if (angle == -2.0f)
				{
					e->push_direction[0] = 0; e->push_direction[1] = 0; e->push_direction[2] = -1;
				}
				else
				{
					float rad = angle * ((float)M_PI / 180.0f);
					e->push_direction[0] = cosf(rad);
					e->push_direction[1] = sinf(rad);
					e->push_direction[2] = 0;
				}

				/* Brush model index for trigger volume bounds. */
				e->push_model_index = 0;
				if (r->has_model && r->model_name[0] == '*')
				{
					int mi = atoi(r->model_name + 1);
					if (mi > 0 && mi < MAX_MODELS)
						e->push_model_index = mi;
				}
				qnn_store_overflow_count++;
			}
			continue;
		}
	}

	free(baseline_raw);
	free(bsp_raw);
}

/* ══════════════════════════════════════════════════════════════════
 * Update
 * ══════════════════════════════════════════════════════════════════ */

void QNN_StoreUpdate(const qnn_snapshot_t *snapshot, float emit_dt)
{
	qnn_entity_update_t entity_updates[QNN_MAX_ENTITY_UPDATES];
	qnn_pvs_item_t pvs_items[QNN_MAX_PVS_ITEMS];
	int entity_count, pvs_count;
	int i;
	float now = (float)cl.mtime[0];
	qboolean ephemeral_seen[MAX_EDICTS];

	/* Map-change guard. cl.mtime is wiped by CL_ClearState during
	 * svc_serverdata (new map within a demo), so the new segment's
	 * `now` can be lower than stamps already in qnn_store from the
	 * prior segment. QualifyEntity's `age = now - newest` then goes
	 * negative, blowing up downstream exp(-recency/tau) in the target
	 * labeler. When we detect cl.mtime moving backwards, wipe the
	 * stamp fields so stale entries don't poison the age calc. */
	{
		static float prev_now = 0.0f;
		if (now + 0.001f < prev_now)
		{
			int n_store = (int)(sizeof(qnn_store) / sizeof(qnn_store[0]));
			for (i = 0; i < n_store; ++i)
			{
				qnn_store[i].pvs = 0.0f;
				qnn_store[i].vis = 0.0f;
				qnn_store[i].snd = 0.0f;
				qnn_store[i].mem = 0.0f;
			}
		}
		prev_now = now;
	}

	if (emit_dt < 0.001f)
		emit_dt = 1.0f / 20.0f;
	memset(ephemeral_seen, 0, sizeof(ephemeral_seen));

	/* ---- Scoreboard connect/disconnect cleanup ----
	 * A name-bearing scoreboard slot is necessary but not sufficient to
	 * promote the slot to an ACTOR entity: spectators (QW) and non-player
	 * edicts (NQ) also have names.  Defer to the per-game predicate. */
	if (cl.scores != NULL)
	{
		for (i = 1; i <= cl.maxclients && i < MAX_EDICTS; ++i)
		{
			qboolean is_live_player = QNN_IsLivePlayerSlot(i);
			if (qnn_player_present[i] && !is_live_player)
				memset(&qnn_store[i], 0, sizeof(qnn_store[i]));
			else if (!qnn_player_present[i] && is_live_player)
			{
				memset(&qnn_store[i], 0, sizeof(qnn_store[i]));
				qnn_store[i].type = QNN_ENT_ACTOR;
				qnn_store[i].subject_id = QNN_SUBJECT_PLAYER;
				qnn_store[i].entity_num = i;
				qnn_store[i].colormap = cl.scores[i - 1].colors;
			}
			qnn_player_present[i] = is_live_player;
		}
	}

	/* ---- Transport ---- */
	entity_count = QNN_EntityClassifyKnown(snapshot, entity_updates, QNN_MAX_ENTITY_UPDATES,
		pvs_items, QNN_MAX_PVS_ITEMS, &pvs_count);

	/* Static entities from PVS (items + movers) */
	for (i = 0; i < pvs_count; ++i)
	{
		qnn_entity_t *e = &qnn_store[pvs_items[i].entity_num];
		if (e->type == QNN_ENT_ITEM)
		{
			VectorCopy(pvs_items[i].origin, e->origin);
			e->regen = 0.0f;
			QNN_StampPvs(e, now, pvs_items[i].in_fov);
		}
		else if (e->type == QNN_ENT_MOVER)
		{
			VectorCopy(pvs_items[i].origin, e->origin);
			QNN_StampPvs(e, now, pvs_items[i].in_fov);
		}
	}

	/* Entity updates (players, projectiles, backpacks, brush movers) */
	for (i = 0; i < entity_count; ++i)
	{
		qnn_entity_update_t *eu = &entity_updates[i];
		qnn_entity_t *e;

		if (eu->entity_num <= 0 || eu->entity_num >= MAX_EDICTS)
			continue;
		e = &qnn_store[eu->entity_num];

		/* Brush movers: update existing mover entry */
		if (eu->is_brush)
		{
			if (e->type == QNN_ENT_MOVER)
			{
				VectorCopy(eu->origin, e->origin);
				QNN_StampPvs(e, now, eu->in_fov);
				{
					float disp_sq = QNN_DistSq(e->origin, e->baseline_origin);
					e->state = (disp_sq > 1.0f) ? 1.0f : 0.0f;
				}
			}
			continue;
		}

		/* Players */
		if (eu->subject_id == QNN_SUBJECT_PLAYER)
		{
			if (eu->entity_num > cl.maxclients)
				continue; /* bodyque */
			if (eu->entity_num == cl.viewentity)
				continue;


			e->type = QNN_ENT_ACTOR;
			e->subject_id = QNN_SUBJECT_PLAYER;
			e->qualifier_id = eu->qualifier_id;
			e->entity_num = eu->entity_num;

			/* Velocity: prefer the wire-authoritative player velocity
			 * (eu->velocity = playerstate->velocity, the server's own
			 * value carried in svc_playerinfo).  QWD/MVD player packets
			 * arrive slower (~10-13 Hz) than the emit rate (20+ Hz), so
			 * an origin finite-difference reads zero on every in-between
			 * emit where the packet — and hence the cached origin — has
			 * not advanced, flickering a moving player to standstill
			 * (~10pp of all actor observations).  The wire velocity does
			 * not suffer the cadence gap: it is the player's real speed
			 * at packet time, and is exactly zero for a genuinely parked
			 * player (PF_VELOCITY suppressed).  Fall back to the origin
			 * finite-difference only when the wire carries no velocity
			 * (e.g. a frame the server omitted it but the origin moved). */
			if (eu->velocity[0] != 0.0f || eu->velocity[1] != 0.0f
				|| eu->velocity[2] != 0.0f)
			{
				VectorCopy(eu->velocity, e->velocity);
			}
			else if (e->pvs > 0.0f || e->snd > 0.0f || e->mem > 0.0f)
			{
				QNN_ComputeStoreVelocity(e, eu->origin, emit_dt, true);
			}
			else
			{
				VectorCopy(eu->velocity, e->velocity);
			}
			VectorCopy(eu->origin, e->origin);
			VectorCopy(eu->angles, e->angles);
			QNN_StampPvs(e, now, eu->in_fov);

			if (cl.scores != NULL && (eu->entity_num - 1) >= 0
				&& (eu->entity_num - 1) < cl.maxclients)
				e->colormap = cl.scores[eu->entity_num - 1].colors;

			e->effects = eu->effects;

			/* Dimlight powerup detection */
			if (eu->effects & QNN_EF_DIMLIGHT)
			{
				if (e->powerup_subject_id == 0)
					e->powerup_subject_id = QNN_SUBJECT_POWERUP;
			}
			else if (e->powerup_subject_id == QNN_SUBJECT_POWERUP
				|| e->powerup_subject_id == QNN_SUBJECT_QUAD
				|| e->powerup_subject_id == QNN_SUBJECT_PENT)
			{
				e->powerup_subject_id = 0;
			}
			continue;
		}

		/* Ephemeral: projectiles + backpacks */
		if (QNN_IsEphemeral(eu->subject_id))
		{
			int new_type = (eu->subject_id == QNN_SUBJECT_BACKPACK)
				? QNN_ENT_BACKPACK : QNN_ENT_PROJECTILE;

			/* Velocity is only meaningful when e->origin holds THIS
			 * same projectile's position from the previous emit.  QW
			 * recycles edict slots, so a freshly-spawned rocket can land
			 * in a slot whose stored origin belonged to the rocket (or
			 * backpack) that lived there before — differencing across
			 * that identity change produces a spurious cross-map delta.
			 * Require: same store identity (type + subject) AND observed
			 * within the previous emit interval (pvs stamp not stale). */
			qboolean valid_prev = (e->type == new_type)
				&& (e->subject_id == eu->subject_id)
				&& (e->pvs > 0.0f)
				&& (now - e->pvs <= 1.5f * emit_dt);

			e->type = new_type;
			e->subject_id = eu->subject_id;
			e->entity_num = eu->entity_num;

			/* Difference the anchored origin against the anchored origin
			 * stored last emit (e->origin), not the raw msg_origins[0]:
			 * mixing the model-anchored previous position with the raw
			 * current one injected a constant per-model bias into the
			 * delta. */
			QNN_ComputeStoreVelocity(e, eu->origin, emit_dt, valid_prev);
			VectorCopy(eu->origin, e->origin);
			QNN_StampPvs(e, now, eu->in_fov);

			ephemeral_seen[eu->entity_num] = true;
			continue;
		}
	}

	/* Keep ephemeral presence tracking for transport disappearance */
	for (i = 1; i < MAX_EDICTS; ++i)
		qnn_ephemeral_present[i] = ephemeral_seen[i];

	/* ---- Item regen countdown ---- */
	for (i = 1; i < MAX_EDICTS; ++i)
	{
		qnn_entity_t *e = &qnn_store[i];
		if (e->type == QNN_ENT_ITEM && e->regen > 0.0f)
		{
			e->regen -= emit_dt;
			if (e->regen < 0.0f)
				e->regen = 0.0f;
		}
	}

	/* ---- Powerup warning timers ---- */
	for (i = 1; i < MAX_EDICTS; ++i)
	{
		qnn_entity_t *e = &qnn_store[i];
		if (e->type == QNN_ENT_ACTOR && e->powerup_warning_elapsed > 0.0f)
		{
			e->powerup_warning_elapsed += emit_dt;
			if (e->powerup_warning_elapsed >= 3.0f)
			{
				e->powerup_subject_id = 0;
				e->powerup_warning_elapsed = 0.0f;
			}
		}
	}

	/* Sound event processing moved to qnn_event.c QNN_EventProcessTick.
	   The event system calls the classifier once and handles both store
	   state changes and event atom creation from the same records. */
}

/* ══════════════════════════════════════════════════════════════════
 * Debug dump
 * ══════════════════════════════════════════════════════════════════ */

static const char *QNN_EntTypeName(int type)
{
	switch (type)
	{
	case QNN_ENT_ITEM:       return "item";
	case QNN_ENT_MOVER:      return "mover";
	case QNN_ENT_ACTOR:      return "actor";
	case QNN_ENT_PROJECTILE: return "projectile";
	case QNN_ENT_BACKPACK:   return "backpack";
	case QNN_ENT_TELEPORTER: return "teleporter";
	case QNN_ENT_PUSH:       return "push";
	default:                 return "unknown";
	}
}

void QNN_StoreDumpTick(FILE *out, int tick, float server_time)
{
	int i, first;
	int size = QNN_StoreCapacity();

	fprintf(out, "{\"tick\":%d,\"time\":%.3f,\"entities\":[", tick, server_time);
	first = 1;
	for (i = 0; i < size; ++i)
	{
		qnn_entity_t *e = &qnn_store[i];
		if (e->type == QNN_ENT_NONE)
			continue;
		if (e->pvs <= 0 && e->snd <= 0 && e->mem <= 0)
			continue;
		if (!first) fprintf(out, ",");
		first = 0;
		fprintf(out, "{\"type\":\"%s\",\"subject\":%d,\"ent\":%d,"
			"\"origin\":[%.1f,%.1f,%.1f],\"pvs\":%d,\"snd\":%d",
			QNN_EntTypeName(e->type), e->subject_id, e->entity_num,
			e->origin[0], e->origin[1], e->origin[2],
			e->pvs > 0 ? 1 : 0, e->snd > 0 ? 1 : 0);
		if (e->type == QNN_ENT_ITEM)
			fprintf(out, ",\"amount\":%d,\"regen\":%.1f,\"regen_time\":%.0f",
				e->amount, e->regen, e->regen_time);
		else if (e->type == QNN_ENT_MOVER)
			fprintf(out, ",\"speed\":%.0f,\"wait\":%.1f,\"state\":%.0f",
				e->speed, e->wait, e->state);
		else if (e->type == QNN_ENT_ACTOR)
			fprintf(out, ",\"colormap\":%d,\"effects\":%d,\"vel\":[%.0f,%.0f,%.0f]",
				e->colormap, e->effects, e->velocity[0], e->velocity[1], e->velocity[2]);
		else if (e->type == QNN_ENT_PROJECTILE || e->type == QNN_ENT_BACKPACK)
			fprintf(out, ",\"vel\":[%.0f,%.0f,%.0f]",
				e->velocity[0], e->velocity[1], e->velocity[2]);
		else if (e->type == QNN_ENT_TELEPORTER)
			fprintf(out, ",\"dest\":[%.1f,%.1f,%.1f]",
				e->destination[0], e->destination[1], e->destination[2]);
		fprintf(out, "}");
	}
	fprintf(out, "]}\n");
}

void QNN_StoreDumpSounds(FILE *out, int tick, const qnn_snapshot_t *snapshot)
{
	int i;
	if (snapshot->sound_count == 0)
		return;
	for (i = 0; i < snapshot->sound_count; ++i)
	{
		const qnn_sound_event_t *s = &snapshot->sounds[i];
		fprintf(out, "{\"tick\":%d,\"ent\":%d,\"snd\":\"%s\",\"origin\":[%.0f,%.0f,%.0f]}\n",
			tick, s->entity_num, s->name,
			s->origin[0], s->origin[1], s->origin[2]);
	}
}

/* ── Mover occlusion cache (shared by QNN_TraceLine, both engines) ─────
 *
 * Solid brush-submodel movers (func_door / func_plat / func_train /
 * func_button — model name "*N") are excluded from world hull 0, so the
 * world-only PM_/SV_RecursiveHullCheck that QNN_TraceLine runs sees
 * straight through them.  That makes the bot's spatial rays and enemy
 * LOS X-ray through closed doors / moving platforms.
 *
 * QNN_TraceLine now clips against each solid mover at its live origin.
 * To keep that off the per-ray hot path, the mover set is enumerated
 * ONCE per observation frame here (keyed on cl.time — movers only move
 * as cl.time advances) and reused by every ray of that frame.  The
 * per-variant qnn_sys.c pulls the cached model+origin and runs the
 * engine's own hull-clip against each.
 *
 * Enumeration mirrors QNN_BuildMoverRefs (qnn_collect_helpers.c) but
 * lives here because qnn_store.c is linked into every worker — including
 * nq_client, which does NOT link qnn_collect_helpers.c. */
#define QNN_TRACE_MAX_MOVERS QNN_MAX_PHYS_MOVERS

static double   qnn_trace_mover_time = -1.0;
static int      qnn_trace_mover_valid = 0;
static int      qnn_trace_mover_count = 0;
static model_t *qnn_trace_mover_models[QNN_TRACE_MAX_MOVERS];
static vec3_t   qnn_trace_mover_origins[QNN_TRACE_MAX_MOVERS];

void QNN_StoreRegisterContext(void)
{
	QNN_ContextRegister(qnn_store, sizeof(qnn_store));
	QNN_ContextRegister(&qnn_store_overflow_count, sizeof(qnn_store_overflow_count));
	QNN_ContextRegister(qnn_ephemeral_present, sizeof(qnn_ephemeral_present));
	QNN_ContextRegister(qnn_player_present, sizeof(qnn_player_present));
	QNN_ContextRegister(&qnn_trace_mover_time, sizeof(qnn_trace_mover_time));
	QNN_ContextRegister(&qnn_trace_mover_valid, sizeof(qnn_trace_mover_valid));
	QNN_ContextRegister(&qnn_trace_mover_count, sizeof(qnn_trace_mover_count));
	QNN_ContextRegister(qnn_trace_mover_models, sizeof(qnn_trace_mover_models));
	QNN_ContextRegister(qnn_trace_mover_origins, sizeof(qnn_trace_mover_origins));
}

static void QNN_TraceGatherMovers(void)
{
	int i;

	qnn_trace_mover_count = 0;
	for (i = 1; i < MAX_EDICTS && qnn_trace_mover_count < QNN_TRACE_MAX_MOVERS; i++)
	{
		entity_t *ent;
		model_t *m;

		/* Only entities the store has classified as solid movers.  This
		 * excludes non-occluding brush models (func_illusionary) and
		 * trigger fields, matching QNN_BuildMoverRefs. */
		if (qnn_store[i].type != QNN_ENT_MOVER)
			continue;
		ent = &cl_entities[i];
		m = ent->model;
		if (m == NULL || m->name[0] != '*')
			continue;
		/* Needs a usable clip tree for the recursive hull check. */
		if (m->hulls[0].firstclipnode > m->hulls[0].lastclipnode)
			continue;
		qnn_trace_mover_models[qnn_trace_mover_count] = m;
		VectorCopy(ent->origin, qnn_trace_mover_origins[qnn_trace_mover_count]);
		qnn_trace_mover_count++;
	}
}

/* Refresh the per-frame mover cache and return the current count.  Called
 * at the top of each QNN_TraceLine; rebuilds at most once per cl.time. */
int QNN_TraceMoverCacheRefresh(void)
{
	if (cl.worldmodel == NULL)
	{
		qnn_trace_mover_count = 0;
		qnn_trace_mover_valid = 0;
		qnn_trace_mover_time = -1.0;
		return 0;
	}
	if (!qnn_trace_mover_valid || (double)cl.time != qnn_trace_mover_time)
	{
		QNN_TraceGatherMovers();
		qnn_trace_mover_time = (double)cl.time;
		qnn_trace_mover_valid = 1;
	}
	return qnn_trace_mover_count;
}

model_t *QNN_TraceMoverModel(int index)
{
	if (index < 0 || index >= qnn_trace_mover_count)
		return NULL;
	return qnn_trace_mover_models[index];
}

float *QNN_TraceMoverOrigin(int index)
{
	static vec3_t zero = {0.0f, 0.0f, 0.0f};
	if (index < 0 || index >= qnn_trace_mover_count)
		return zero;
	return qnn_trace_mover_origins[index];
}
