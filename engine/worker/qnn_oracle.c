/*
 * qnn_oracle.c — Token emission for the semantic oracle.
 *
 * Reads from the unified store, builds per-type tokens (projectile,
 * actor, item, mover), and outputs a variable-length tagged token
 * stream for the model.
 */

#include "qnn_object.h"
#include "qnn_store.h"
#include "qnn_io.h"
#include "qnn_route.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern FILE *qnn_sound_dump;

/* Returns the best qualifying modality, or QNN_MODALITY_NONE if stale.
   Each modality's own timestamp is checked against its own threshold.
   Priority: sight/proximity (pvs) > sound > memory.
   Sets *out_age to the age of the qualifying observation. */
static int QNN_QualifyEntity(const qnn_entity_t *e, float now, float *out_age)
{
	/* For actors, "nearby" means visible (PVS + FOV).
	   For everything else, PVS alone suffices. */
	float near = (e->type == QNN_ENT_ACTOR) ? e->vis : e->pvs;
	float newest = near;
	int modality = QNN_MODALITY_NONE;
	float age, threshold;

	if (e->snd > newest) newest = e->snd;
	if (e->mem > newest) newest = e->mem;
	if (newest <= 0) return QNN_MODALITY_NONE;

	/* Determine modality from newest observation.
	   Later checks overwrite earlier: mem < snd < near. */
	if (e->mem == newest) modality = QNN_MODALITY_MEMORY;
	if (e->snd == newest) modality = QNN_MODALITY_SOUND;
	if (near == newest && near > 0)
		modality = (e->type == QNN_ENT_ACTOR)
			? QNN_MODALITY_SIGHT : QNN_MODALITY_PROXIMITY;

	if (modality == QNN_MODALITY_NONE)
		return QNN_MODALITY_NONE;

	/* Check age against modality threshold */
	age = now - newest;
	threshold = QNN_RecencyMaxForModality(modality);
	if (age > threshold)
	{
		/* Actor fallback: stay in candidate pool for the full sight
		   window if we've seen them recently, even if the most recent
		   observation (sound/memory) has expired. */
		if (e->type == QNN_ENT_ACTOR && e->vis > 0
			&& (now - e->vis) <= QNN_RECENCY_MAX_SIGHT)
		{
			age = now - e->vis;
		}
		else
			return QNN_MODALITY_NONE;
	}

	*out_age = age;
	return modality;
}

/* ── Candidate types ──────────────────────────────────────────── */

#define QNN_CAND_PROJECTILE 0
#define QNN_CAND_ACTOR      1
#define QNN_CAND_ITEM       2
#define QNN_CAND_MOVER      3

typedef struct {
	int type;
	int modality;
	int store_index;
	float recency;
	void *entry;
} qnn_candidate_t;

static int QNN_CandidateCompare(const void *a, const void *b)
{
	const qnn_candidate_t *ca = (const qnn_candidate_t *)a;
	const qnn_candidate_t *cb = (const qnn_candidate_t *)b;
	if (ca->type != cb->type)
		return ca->type - cb->type;
	if (ca->recency < cb->recency) return -1;
	if (ca->recency > cb->recency) return 1;
	return 0;
}

/* ── Half extents from cl_entities model ──────────────────────── */

static void QNN_LookupHalfExtents(int entity_num, float *out_half, vec3_t origin_adjust)
{
	entity_t *entity;
	out_half[0] = out_half[1] = out_half[2] = 0.0f;
	origin_adjust[0] = origin_adjust[1] = origin_adjust[2] = 0.0f;
	if (entity_num <= 0 || entity_num >= MAX_EDICTS)
		return;
	entity = &cl_entities[entity_num];
	if (entity->model == NULL)
		return;
	{
		vec3_t bmins, bmaxs;
		VectorCopy(entity->model->mins, bmins);
		VectorCopy(entity->model->maxs, bmaxs);
		origin_adjust[0] = (bmins[0] + bmaxs[0]) * 0.5f;
		origin_adjust[1] = (bmins[1] + bmaxs[1]) * 0.5f;
		origin_adjust[2] = (bmins[2] + bmaxs[2]) * 0.5f;
		out_half[0] = (bmaxs[0] - bmins[0]) * 0.5f;
		out_half[1] = (bmaxs[1] - bmins[1]) * 0.5f;
		out_half[2] = (bmaxs[2] - bmins[2]) * 0.5f;
	}
}

/* ── Fill events into token event array ───────────────────────── */

static int QNN_FillEvents(qnn_token_event_t *events, int max_events, int store_index)
{
	int slot = 0;
	if (store_index >= 0 && store_index < QNN_EVENT_HEAD_CAPACITY)
	{
		int ei = qnn_event_head[store_index];
		while (ei >= 0 && slot < max_events)
		{
			if (qnn_semantic_events[ei].active)
			{
				events[slot].action_id = qnn_semantic_events[ei].action_id;
				events[slot].source_id = qnn_semantic_events[ei].source_id;
				if (qnn_sound_dump)
					fprintf(qnn_sound_dump, "EMIT\t%.3f\tent=%d\tact=%d\tsrc=%d\n",
						(float)cl.mtime[0], store_index,
						events[slot].action_id, events[slot].source_id);
				slot++;
			}
			ei = qnn_semantic_events[ei].next_for_owner;
		}
	}
	return slot;
}

/* ── Compute relative frame + velocity ────────────────────────── */

static float QNN_ComputeRel(const qnn_snapshot_t *snapshot, const vec3_t origin, float *out_rel)
{
	vec3_t delta, rel;
	float dist;
	VectorSubtract(origin, snapshot->player_origin, delta);
	dist = QNN_VecLength(delta);
	QNN_RelativeFrame(snapshot->player_view_angles, delta, rel);
	out_rel[0] = rel[0] / QNN_DIST_SCALE;
	out_rel[1] = rel[1] / QNN_DIST_SCALE;
	out_rel[2] = rel[2] / QNN_DIST_SCALE;
	return dist / QNN_DIST_SCALE;
}

static void QNN_ComputeVel(const qnn_snapshot_t *snapshot, const vec3_t velocity, float *out_vel)
{
	vec3_t vel_view;
	QNN_RelativeFrame(snapshot->player_view_angles, velocity, vel_view);
	out_vel[0] = QNN_Normalize(vel_view[0], QNN_VELOCITY_SCALE);
	out_vel[1] = QNN_Normalize(vel_view[1], QNN_VELOCITY_SCALE);
	out_vel[2] = QNN_Normalize(vel_view[2], QNN_VELOCITY_SCALE);
}

static float QNN_ComputePath(const qnn_snapshot_t *snapshot, const vec3_t path_world, float *out_path)
{
	vec3_t path_rel;
	float dist = QNN_VecLength(path_world);
	QNN_RelativeFrame(snapshot->player_view_angles, path_world, path_rel);
	out_path[0] = path_rel[0] / QNN_DIST_SCALE;
	out_path[1] = path_rel[1] / QNN_DIST_SCALE;
	out_path[2] = path_rel[2] / QNN_DIST_SCALE;
	return dist / QNN_DIST_SCALE;
}

static void QNN_NormalizeHalfExtents(const float *raw, float *out)
{
	out[0] = raw[0] / QNN_DIST_SCALE;
	out[1] = raw[1] / QNN_DIST_SCALE;
	out[2] = raw[2] / QNN_DIST_SCALE;
}

/* ── Item amount normalization ────────────────────────────────── */

static float QNN_NormalizeItemAmount(int subject_id, int amount)
{
	switch (subject_id)
	{
	case QNN_SUBJECT_HEALTH:
	case QNN_SUBJECT_MEGAHEALTH:
		return QNN_Normalize((float)amount, QNN_MAX_HEALTH);
	case QNN_SUBJECT_ARMOR_GREEN:
		return ((float)amount * 0.3f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_ARMOR_YELLOW:
		return ((float)amount * 0.6f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_ARMOR_RED:
		return ((float)amount * 0.8f) / QNN_MAX_ARMOR;
	case QNN_SUBJECT_SHELLS:
		return QNN_Normalize((float)amount, QNN_MAX_SHELLS);
	case QNN_SUBJECT_NAILS:
		return QNN_Normalize((float)amount, QNN_MAX_NAILS);
	case QNN_SUBJECT_ROCKETS:
		return QNN_Normalize((float)amount, QNN_MAX_ROCKETS);
	case QNN_SUBJECT_CELLS:
		return QNN_Normalize((float)amount, QNN_MAX_CELLS);
	case QNN_SUBJECT_QUAD:
	case QNN_SUBJECT_PENT:
	case QNN_SUBJECT_RING:
	case QNN_SUBJECT_SUIT:
		return 1.0f;
	case QNN_SUBJECT_SHOTGUN:
		return QNN_Normalize(5.0f, QNN_MAX_SHELLS);
	case QNN_SUBJECT_NAILGUN:
		return QNN_Normalize(30.0f, QNN_MAX_NAILS);
	case QNN_SUBJECT_GRENADE_LAUNCHER:
		return QNN_Normalize(5.0f, QNN_MAX_ROCKETS);
	case QNN_SUBJECT_ROCKET_LAUNCHER:
		return QNN_Normalize(5.0f, QNN_MAX_ROCKETS);
	case QNN_SUBJECT_THUNDERBOLT:
		return QNN_Normalize(15.0f, QNN_MAX_CELLS);
	default:
		return 0.0f;
	}
}

/* ── Nail stream aggregation ──────────────────────────────────── */

static void QNN_AggregateNailCandidates(
	qnn_candidate_t *candidates, int *count,
	const vec3_t player_origin)
{
	int nail_indices[QNN_MAX_TOKEN_OBJECTS];
	qboolean absorbed[QNN_MAX_TOKEN_OBJECTS];
	int nail_count = 0;
	int n = *count;
	int i, j;

	memset(absorbed, 0, sizeof(absorbed));
	for (i = 0; i < n; ++i)
	{
		if (candidates[i].type == QNN_CAND_PROJECTILE)
		{
			qnn_entity_t *p = (qnn_entity_t *)candidates[i].entry;
			if (p->subject_id == QNN_SUBJECT_PROJECTILE_NAIL)
				nail_indices[nail_count++] = i;
		}
	}
	if (nail_count <= 1)
		return;

	for (i = 0; i < nail_count; ++i)
	{
		int ni = nail_indices[i];
		qnn_entity_t *pi;
		float leader_speed;
		vec3_t leader_vel;
		float leader_dsq;
		int leader_idx;

		if (absorbed[ni]) continue;
		pi = (qnn_entity_t *)candidates[ni].entry;
		leader_speed = QNN_VecLength(pi->velocity);
		if (leader_speed < 1.0f) continue;
		leader_vel[0] = pi->velocity[0] / leader_speed;
		leader_vel[1] = pi->velocity[1] / leader_speed;
		leader_vel[2] = pi->velocity[2] / leader_speed;
		leader_idx = ni;
		{ vec3_t d; VectorSubtract(pi->origin, player_origin, d); leader_dsq = DotProduct(d, d); }
		absorbed[ni] = true;

		for (j = i + 1; j < nail_count; ++j)
		{
			int nj = nail_indices[j];
			qnn_entity_t *pj;
			float speed, dot;
			if (absorbed[nj]) continue;
			pj = (qnn_entity_t *)candidates[nj].entry;
			speed = QNN_VecLength(pj->velocity);
			if (speed < 1.0f) continue;
			dot = (pj->velocity[0] * leader_vel[0]
				+ pj->velocity[1] * leader_vel[1]
				+ pj->velocity[2] * leader_vel[2]) / speed;
			if (dot < QNN_NAIL_STREAM_DOT_THRESHOLD) continue;
			absorbed[nj] = true;
			{ vec3_t d; float dsq; VectorSubtract(pj->origin, player_origin, d); dsq = DotProduct(d, d);
			  if (dsq < leader_dsq) { leader_dsq = dsq; leader_idx = nj; } }
		}
		absorbed[leader_idx] = false;
	}

	j = 0;
	for (i = 0; i < n; ++i)
		if (!absorbed[i])
			candidates[j++] = candidates[i];
	*count = j;
}

/* ── Token emission ───────────────────────────────────────────── */

int QNN_OracleEmitTokens(
	qnn_tagged_token_t *out_tokens, int max_tokens,
	const qnn_snapshot_t *snapshot,
	const qnn_map_state_t *map_state,
	int *out_player_cluster_id)
{
	qnn_candidate_t candidates[MAX_EDICTS + QNN_STORE_OVERFLOW];
	int candidate_count = 0;
	int token_count = 0;
	int i;
	const qnn_route_runtime_t *route;
	int player_area_id;

	*out_player_cluster_id = 0;

	/* ---- Collect candidates ---- */
	{
		int store_size = QNN_StoreCapacity();
		float now = (float)cl.mtime[0];
		for (i = 0; i < store_size; ++i)
		{
			qnn_entity_t *e = &qnn_store[i];
			int cand_type, modality;
			float age;

			modality = QNN_QualifyEntity(e, now, &age);
			if (modality == QNN_MODALITY_NONE)
				continue;
			switch (e->type)
			{
			case QNN_ENT_PROJECTILE:
			case QNN_ENT_BACKPACK:   cand_type = (e->type == QNN_ENT_BACKPACK) ? QNN_CAND_ITEM : QNN_CAND_PROJECTILE; break;
			case QNN_ENT_ACTOR:      cand_type = QNN_CAND_ACTOR; break;
			case QNN_ENT_ITEM:       cand_type = QNN_CAND_ITEM; break;
			case QNN_ENT_MOVER:      cand_type = QNN_CAND_MOVER; break;
			default: continue;
			}
			candidates[candidate_count].type = cand_type;
			candidates[candidate_count].modality = modality;
			candidates[candidate_count].store_index = i;
			candidates[candidate_count].recency = age;
			candidates[candidate_count].entry = e;
			candidate_count++;
		}
	}

	if (candidate_count > 1)
		qsort(candidates, (size_t)candidate_count, sizeof(candidates[0]), QNN_CandidateCompare);

	QNN_AggregateNailCandidates(candidates, &candidate_count, snapshot->player_origin);

	if (candidate_count > max_tokens)
		candidate_count = max_tokens;

	/* ---- Resolve nav oracle ---- */
	route = map_state->route;
	player_area_id = -1;
	if (route)
	{
		qnn_route_area_result_t player_area;
		char area_err[128];
		if (QNN_RouteFindArea(route, snapshot->player_origin, &player_area, area_err, sizeof(area_err))
			&& player_area.found)
		{
			player_area_id = player_area.area_id;
			*out_player_cluster_id = player_area.cluster_id;
		}
	}

	/* ---- Build tokens ---- */
	for (i = 0; i < candidate_count && token_count < max_tokens; ++i)
	{
		qnn_candidate_t *cand = &candidates[i];
		qnn_entity_t *e = (qnn_entity_t *)cand->entry;
		qnn_tagged_token_t *out = &out_tokens[token_count];
		vec3_t origin;
		float half_ext[3];
		vec3_t origin_adj;
		int entity_num = e->entity_num;
		int obj_area_id = -1;
		int path_idx;

		VectorCopy(e->origin, origin);
		half_ext[0] = half_ext[1] = half_ext[2] = 0.0f;

		/* Half extents + bbox center (not for projectiles) */
		if (cand->type != QNN_CAND_PROJECTILE && entity_num > 0)
		{
			QNN_LookupHalfExtents(entity_num, half_ext, origin_adj);
			origin[0] += origin_adj[0];
			origin[1] += origin_adj[1];
			origin[2] += origin_adj[2];
		}

		/* Route */
		if (route && player_area_id >= 0)
		{
			qnn_route_area_result_t obj_area;
			char obj_err[128];
			if (QNN_RouteFindArea(route, origin, &obj_area, obj_err, sizeof(obj_err))
				&& obj_area.found)
				obj_area_id = obj_area.area_id;
		}

		/* Path alternatives — emit duplicate tokens with different path/eta */
		{
		int same_cluster = (obj_area_id >= 0 && player_area_id >= 0
			&& route != NULL); /* simplified — let route logic handle same-cluster */

		for (path_idx = 0; path_idx < QNN_MAX_ROUTE_PATHS && token_count < max_tokens; ++path_idx)
		{
			float path_world[3] = {0, 0, 0};
			float eta = 0.0f;
			int used_path = 0;

			if (cand->type != QNN_CAND_PROJECTILE && route && player_area_id >= 0 && obj_area_id >= 0)
			{
				float route_cost;
				char path_err[128];
				if (QNN_RoutePathPositionNth(route, player_area_id, obj_area_id,
					snapshot->player_origin, origin,
					path_idx, path_world, &route_cost, path_err, sizeof(path_err)))
				{
					eta = route_cost;
					used_path = 1;
				}
				else if (path_idx > 0)
					break;
			}

			if (!used_path && path_idx > 0)
				break;

			memset(out, 0, sizeof(*out));

			switch (cand->type)
			{
			case QNN_CAND_PROJECTILE:
				{
					qnn_projectile_token_t *tok = &out->projectile;
					out->type = QNN_TOKEN_PROJECTILE;
					tok->subject_id = e->subject_id;
					tok->modality_id = cand->modality;
					tok->dist = QNN_ComputeRel(snapshot, origin, tok->rel);
					QNN_ComputeVel(snapshot, e->velocity, tok->vel);
					tok->recency = cand->recency / QNN_TIME_SCALE;
					tok->event_count = QNN_FillEvents(tok->events, QNN_MAX_ENTITY_EVENTS, cand->store_index);
				}
				break;

			case QNN_CAND_ACTOR:
				{
					qnn_actor_token_t *tok = &out->actor;
					int max_frags;
					out->type = QNN_TOKEN_ACTOR;
					tok->subject_id = QNN_SUBJECT_PLAYER;
					tok->modality_id = cand->modality;
					tok->player_id = e->entity_num;
					QNN_NormalizeHalfExtents(half_ext, tok->half_extents);
					tok->dist = QNN_ComputeRel(snapshot, origin, tok->rel);
					QNN_ComputeVel(snapshot, e->velocity, tok->vel);
					if (used_path)
						tok->path_dist = QNN_ComputePath(snapshot, path_world, tok->path);
					else
					{
						tok->path[0] = tok->rel[0];
						tok->path[1] = tok->rel[1];
						tok->path[2] = tok->rel[2];
						tok->path_dist = tok->dist;
					}
					tok->eta = eta / QNN_TIME_SCALE;

					/* Facing */
					{
						vec3_t their_forward, to_us;
						float dist, dot;
						QNN_ForwardFromAngles(e->angles, their_forward);
						VectorSubtract(snapshot->player_origin, origin, to_us);
						dist = QNN_VecLength(to_us);
						if (dist > 1.0f)
						{
							to_us[0] /= dist; to_us[1] /= dist; to_us[2] /= dist;
							dot = DotProduct(their_forward, to_us);
							tok->facing = QNN_Clamp((1.0f - dot) * 0.5f, 0.0f, 1.0f);
						}
					}

					tok->team = QNN_IsSameTeam(e->entity_num);

					/* Score */
					max_frags = 0;
					if (cl.scores != NULL)
					{
						int si;
						for (si = 0; si < cl.maxclients; ++si)
							if (cl.scores[si].frags > max_frags)
								max_frags = cl.scores[si].frags;
					}
					if (max_frags != 0 && cl.scores != NULL
						&& (e->entity_num - 1) >= 0 && (e->entity_num - 1) < cl.maxclients)
						tok->score = (float)cl.scores[e->entity_num - 1].frags / (float)max_frags;
					else
						tok->score = 0.0f;

					tok->recency = cand->recency / QNN_TIME_SCALE;
					tok->event_count = QNN_FillEvents(tok->events, QNN_MAX_ENTITY_EVENTS, cand->store_index);
				}
				break;

			case QNN_CAND_ITEM:
				{
					qnn_item_token_t *tok = &out->item;
					out->type = QNN_TOKEN_ITEM;
					tok->subject_id = e->subject_id;
					tok->modality_id = cand->modality;
					QNN_NormalizeHalfExtents(half_ext, tok->half_extents);
					tok->dist = QNN_ComputeRel(snapshot, origin, tok->rel);
					if (used_path)
						tok->path_dist = QNN_ComputePath(snapshot, path_world, tok->path);
					else
					{
						tok->path[0] = tok->rel[0];
						tok->path[1] = tok->rel[1];
						tok->path[2] = tok->rel[2];
						tok->path_dist = tok->dist;
					}
					tok->eta = eta / QNN_TIME_SCALE;
					tok->amount = QNN_NormalizeItemAmount(e->subject_id, e->amount);
					tok->regen = e->regen / QNN_TIME_SCALE;
					tok->recency = cand->recency / QNN_TIME_SCALE;
					tok->event_count = QNN_FillEvents(tok->events, QNN_MAX_ENTITY_EVENTS, cand->store_index);
				}
				break;

			case QNN_CAND_MOVER:
				{
					qnn_mover_token_t *tok = &out->mover;
					out->type = QNN_TOKEN_MOVER;
					tok->subject_id = e->subject_id;
					tok->modality_id = cand->modality;
					QNN_NormalizeHalfExtents(half_ext, tok->half_extents);
					tok->dist = QNN_ComputeRel(snapshot, origin, tok->rel);
					if (used_path)
						tok->path_dist = QNN_ComputePath(snapshot, path_world, tok->path);
					else
					{
						tok->path[0] = tok->rel[0];
						tok->path[1] = tok->rel[1];
						tok->path[2] = tok->rel[2];
						tok->path_dist = tok->dist;
					}
					tok->eta = eta / QNN_TIME_SCALE;
					tok->state = e->state;
					tok->recency = cand->recency / QNN_TIME_SCALE;
					tok->event_count = QNN_FillEvents(tok->events, QNN_MAX_ENTITY_EVENTS, cand->store_index);
				}
				break;
			}

			token_count++;
		}
		}
	}

	/* Update recall mapping */
	for (i = 0; i < candidate_count && i < QNN_MAX_TOKEN_OBJECTS; ++i)
		qnn_prev_object_indices[i] = candidates[i].store_index;
	qnn_prev_object_count = candidate_count < QNN_MAX_TOKEN_OBJECTS ? candidate_count : QNN_MAX_TOKEN_OBJECTS;

	return token_count;
}
