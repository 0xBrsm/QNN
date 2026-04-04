/*
 * qnn_oracle.c — Token emission for the semantic oracle.
 *
 * Reads directly from the three stores (object, actor, projectile).
 * Computes all token metadata at emission time: relative frame,
 * half_extents, recency, modality, route/cluster, events.
 * Never mutates stored world state.
 */

#include "qnn_object.h"
#include "qnn_store.h"
#include "qnn_io.h"
#include "qnn_route.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Candidate types ──────────────────────────────────────────── */

#define QNN_CAND_PROJECTILE 0  /* highest priority */
#define QNN_CAND_ACTOR      1
#define QNN_CAND_ITEM       2
#define QNN_CAND_MOVER      3  /* lowest priority */

typedef struct {
	int type;          /* QNN_CAND_* */
	int store_index;   /* unified index for event atom lookup */
	float recency;     /* seconds since last observation */
	void *entry;       /* pointer into store array */
} qnn_candidate_t;

/* ── Candidate comparison ─────────────────────────────────────── */

static int QNN_CandidateCompare(const void *a, const void *b)
{
	const qnn_candidate_t *ca = (const qnn_candidate_t *)a;
	const qnn_candidate_t *cb = (const qnn_candidate_t *)b;

	/* Priority: lower type = higher priority */
	if (ca->type != cb->type)
		return ca->type - cb->type;
	/* Within same type: lower recency (more recent) first */
	if (ca->recency < cb->recency)
		return -1;
	if (ca->recency > cb->recency)
		return 1;
	return 0;
}

/* ── Half extents from cl_entities model ──────────────────────── */

static void QNN_LookupHalfExtents(int entity_num, float *out_half, vec3_t origin_adjust)
{
	entity_t *entity;
	out_half[0] = 0.0f;
	out_half[1] = 0.0f;
	out_half[2] = 0.0f;
	origin_adjust[0] = 0.0f;
	origin_adjust[1] = 0.0f;
	origin_adjust[2] = 0.0f;

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

/* ── Fill common token fields ─────────────────────────────────── */

static void QNN_FillTokenSpatial(
	qnn_entity_token_t *tok,
	const qnn_snapshot_t *snapshot,
	const vec3_t origin,
	const vec3_t velocity,
	const float *half_extents,
	const vec3_t rel,
	float world_dist)
{
	tok->rel[0] = rel[0] / QNN_DIST_SCALE;
	tok->rel[1] = rel[1] / QNN_DIST_SCALE;
	tok->rel[2] = rel[2] / QNN_DIST_SCALE;
	tok->distance = world_dist / QNN_DIST_SCALE;

	/* Velocity in view frame */
	{
		vec3_t vel_view;
		QNN_RelativeFrame(snapshot->player_view_angles, velocity, vel_view);
		tok->vel[0] = QNN_Normalize(vel_view[0], QNN_VELOCITY_SCALE);
		tok->vel[1] = QNN_Normalize(vel_view[1], QNN_VELOCITY_SCALE);
		tok->vel[2] = QNN_Normalize(vel_view[2], QNN_VELOCITY_SCALE);
	}

	/* Look angles to target */
	{
		float rx = rel[0], ry = rel[1], rz = rel[2];
		float horiz_dist = sqrtf(rx * rx + ry * ry);
		float yaw_deg = 0.0f, pitch_deg = 0.0f;
		int yaw_counts, pitch_counts;

		if (horiz_dist > 1.0f)
		{
			yaw_deg = atan2f(ry, rx) * (180.0f / (float)M_PI);
			pitch_deg = atan2f(-rz, horiz_dist) * (180.0f / (float)M_PI);
		}
		yaw_counts = (int)roundf(-yaw_deg / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
		pitch_counts = (int)roundf(pitch_deg / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
		tok->rel_yaw = QNN_LookAxisFromMouseCount(yaw_counts);
		tok->rel_pitch = QNN_LookAxisFromMouseCount(pitch_counts);
	}

	tok->half_extents[0] = half_extents[0] / QNN_DIST_SCALE;
	tok->half_extents[1] = half_extents[1] / QNN_DIST_SCALE;
	tok->half_extents[2] = half_extents[2] / QNN_DIST_SCALE;
}

/* ── Fill per-entity events ───────────────────────────────────── */

static void QNN_FillTokenEvents(qnn_entity_token_t *tok, int store_index)
{
	int slot = 0;
	if (store_index >= 0 && store_index < QNN_EVENT_HEAD_CAPACITY)
	{
		int ei = qnn_event_head[store_index];
		while (ei >= 0 && slot < QNN_MAX_ENTITY_EVENTS)
		{
			if (qnn_semantic_events[ei].active)
			{
				tok->event_subject[slot] = qnn_semantic_events[ei].subject_id;
				tok->event_action[slot] = qnn_semantic_events[ei].action_id;
				tok->event_qualifier[slot] = qnn_semantic_events[ei].qualifier_id;
				tok->event_recency[slot] = QNN_Clamp(qnn_semantic_events[ei].recency, 0.0f, 1.0f);
				slot++;
			}
			ei = qnn_semantic_events[ei].next_for_owner;
		}
	}
	tok->event_count = slot;
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

	/* Group nails by velocity direction */
	for (i = 0; i < nail_count; ++i)
	{
		int ni = nail_indices[i];
		qnn_entity_t *pi;
		float leader_speed;
		vec3_t leader_vel;
		float leader_dsq;
		int leader_idx;

		if (absorbed[ni])
			continue;

		pi = (qnn_entity_t *)candidates[ni].entry;
		leader_speed = QNN_VecLength(pi->velocity);
		if (leader_speed < 1.0f)
			continue;

		leader_vel[0] = pi->velocity[0] / leader_speed;
		leader_vel[1] = pi->velocity[1] / leader_speed;
		leader_vel[2] = pi->velocity[2] / leader_speed;
		leader_idx = ni;
		{
			vec3_t d;
			VectorSubtract(pi->origin, player_origin, d);
			leader_dsq = DotProduct(d, d);
		}
		absorbed[ni] = true;

		for (j = i + 1; j < nail_count; ++j)
		{
			int nj = nail_indices[j];
			qnn_entity_t *pj;
			float speed, dot;

			if (absorbed[nj])
				continue;
			pj = (qnn_entity_t *)candidates[nj].entry;
			speed = QNN_VecLength(pj->velocity);
			if (speed < 1.0f)
				continue;
			dot = (pj->velocity[0] * leader_vel[0]
				+ pj->velocity[1] * leader_vel[1]
				+ pj->velocity[2] * leader_vel[2]) / speed;
			if (dot < QNN_NAIL_STREAM_DOT_THRESHOLD)
				continue;
			absorbed[nj] = true;

			/* Pick closest to player as leader */
			{
				vec3_t d;
				float dsq;
				VectorSubtract(pj->origin, player_origin, d);
				dsq = DotProduct(d, d);
				if (dsq < leader_dsq)
				{
					leader_dsq = dsq;
					leader_idx = nj;
				}
			}
		}
		absorbed[leader_idx] = false; /* keep the leader */
	}

	/* Compact: remove absorbed candidates */
	j = 0;
	for (i = 0; i < n; ++i)
	{
		if (!absorbed[i])
			candidates[j++] = candidates[i];
	}
	*count = j;
}

/* ── Token emission ───────────────────────────────────────────── */

int QNN_OracleEmitTokens(
	qnn_entity_token_t *out_tokens, int max_tokens,
	const qnn_snapshot_t *snapshot,
	const qnn_map_state_t *map_state,
	int *out_player_cluster_id)
{
	qnn_candidate_t candidates[MAX_EDICTS + 512];
	int candidate_count = 0;
	int token_count = 0;
	int i, j;
	const qnn_route_runtime_t *route;
	int player_area_id;
	float now = (float)cl.mtime[0];

	*out_player_cluster_id = 0;

	/* ---- Collect candidates from unified store ---- */

	{
		int store_size = QNN_StoreSize();
		for (i = 0; i < store_size; ++i)
		{
			qnn_entity_t *e = &qnn_store[i];
			float last_seen;
			int cand_type;

			if (!e->active)
				continue;

			/* Map entity type to candidate priority class */
			switch (e->type)
			{
			case QNN_ENT_PROJECTILE: cand_type = QNN_CAND_PROJECTILE; break;
			case QNN_ENT_BACKPACK:   cand_type = QNN_CAND_PROJECTILE; break;
			case QNN_ENT_ACTOR:      cand_type = QNN_CAND_ACTOR; break;
			case QNN_ENT_ITEM:       cand_type = QNN_CAND_ITEM; break;
			case QNN_ENT_MOVER:      cand_type = QNN_CAND_MOVER; break;
			default: continue; /* teleporters, push triggers — not emitted */
			}

			last_seen = e->pvs > e->snd ? e->pvs : e->snd;
			if (e->mem > last_seen) last_seen = e->mem;
			if (last_seen <= 0.0f)
				continue;

			candidates[candidate_count].type = cand_type;
			candidates[candidate_count].store_index = i;
			candidates[candidate_count].recency = now - last_seen;
			candidates[candidate_count].entry = e;
			candidate_count++;
		}
	}

	/* Sort by priority then recency */
	if (candidate_count > 1)
		qsort(candidates, (size_t)candidate_count, sizeof(candidates[0]), QNN_CandidateCompare);

	/* Nail stream aggregation before trim — collapse streams so
	   they don't consume multiple slots in the token limit. */
	QNN_AggregateNailCandidates(candidates, &candidate_count, snapshot->player_origin);

	/* Trim to max tokens */
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
		qnn_entity_token_t *tok;
		vec3_t origin, velocity, delta, rel;
		float half_ext[3];
		vec3_t origin_adj;
		float world_dist;
		int entity_num = 0;
		int obj_area_id = -1;
		int obj_cluster_id = 0;
		float obj_route_cost = 0.0f;
		int obj_route_cluster_ids[QNN_MAX_ROUTE_CLUSTERS];
		int obj_route_cluster_count = 0;
		int path_idx;

		/* Extract origin, velocity, entity_num by candidate type */
		VectorCopy(vec3_origin, velocity);
		half_ext[0] = half_ext[1] = half_ext[2] = 0.0f;
		VectorCopy(vec3_origin, origin_adj);

		switch (cand->type)
		{
		case QNN_CAND_PROJECTILE:
			{
				qnn_entity_t *p = (qnn_entity_t *)cand->entry;
				VectorCopy(p->origin, origin);
				VectorCopy(p->velocity, velocity);
				entity_num = p->entity_num;
			}
			break;
		case QNN_CAND_ACTOR:
			{
				qnn_entity_t *a = (qnn_entity_t *)cand->entry;
				VectorCopy(a->origin, origin);
				VectorCopy(a->velocity, velocity);
				entity_num = a->entity_num;
			}
			break;
		case QNN_CAND_ITEM:
		case QNN_CAND_MOVER:
			{
				qnn_entity_t *e = (qnn_entity_t *)cand->entry;
				VectorCopy(e->origin, origin);
				entity_num = e->entity_num;
			}
			break;
		}

		/* Half extents from model (actors/projectiles with entity_num) */
		if (entity_num > 0)
		{
			QNN_LookupHalfExtents(entity_num, half_ext, origin_adj);
			origin[0] += origin_adj[0];
			origin[1] += origin_adj[1];
			origin[2] += origin_adj[2];
		}

		/* Actors/projectiles: must be in FOV or heard to emit.
		   PVS-but-not-FOV entities are tracked but not emitted.
		   Backpacks are exempt — they're pickups, not threats. */
		if (cand->type == QNN_CAND_ACTOR || cand->type == QNN_CAND_PROJECTILE)
		{
			qboolean is_backpack = (cand->type == QNN_CAND_PROJECTILE
				&& ((qnn_entity_t *)cand->entry)->subject_id == QNN_SUBJECT_BACKPACK);
			if (!is_backpack)
			{
				qboolean in_fov = QNN_InFov(snapshot->player_origin, snapshot->player_view_angles, origin);
				if (!in_fov)
				{
					/* Projectiles have no sound while in flight — skip entirely */
					if (cand->type == QNN_CAND_PROJECTILE)
						continue;
					/* Actors: emit only if heard recently */
					{
						qnn_entity_t *a = (qnn_entity_t *)cand->entry;
						if (!(a->snd > 0 && (now - a->snd) < QNN_RECENCY_DECAY_PLAYER_S))
							continue;
					}
				}
			}
		}

		/* Route/cluster */
		if (route && player_area_id >= 0)
		{
			qnn_route_area_result_t obj_area;
			char obj_err[128];
			if (QNN_RouteFindArea(route, origin, &obj_area, obj_err, sizeof(obj_err))
				&& obj_area.found)
			{
				obj_cluster_id = obj_area.cluster_id;
				obj_area_id = obj_area.area_id;
			}
		}

		VectorSubtract(origin, snapshot->player_origin, delta);
		world_dist = QNN_VecLength(delta);

		if (route && player_area_id >= 0 && obj_area_id >= 0)
		{
			QNN_RouteGetClusters(route, player_area_id, obj_area_id,
				obj_route_cluster_ids, QNN_MAX_ROUTE_CLUSTERS,
				&obj_route_cluster_count);
		}

		/* Route path alternatives */
		{
		int same_cluster = (obj_cluster_id == *out_player_cluster_id && *out_player_cluster_id >= 0);

		for (path_idx = 0; path_idx < QNN_MAX_ROUTE_PATHS && token_count < max_tokens; ++path_idx)
		{
			int used_path = 0;

			tok = &out_tokens[token_count];
			memset(tok, 0, sizeof(*tok));

			if (!same_cluster && route && player_area_id >= 0 && obj_area_id >= 0)
			{
				float path_rel[3];
				float route_cost;
				char path_err[128];

				if (QNN_RoutePathPositionNth(route, player_area_id, obj_area_id,
					snapshot->player_origin, origin,
					path_idx, path_rel, &route_cost, path_err, sizeof(path_err)))
				{
					QNN_RelativeFrame(snapshot->player_view_angles, path_rel, rel);
					obj_route_cost = route_cost;
					used_path = 1;
				}
				else if (path_idx > 0)
					break;
			}

			if (!used_path)
			{
				if (path_idx > 0) break;
				QNN_RelativeFrame(snapshot->player_view_angles, delta, rel);
			}

			/* Common spatial */
			QNN_FillTokenSpatial(tok, snapshot, origin, velocity, half_ext, rel, world_dist);

			/* Route */
			tok->cluster_id = obj_cluster_id;
			tok->route_cost = obj_route_cost / QNN_TIME_SCALE;
			tok->route_cluster_count = obj_route_cluster_count;
			for (j = 0; j < QNN_MAX_ROUTE_CLUSTERS; ++j)
				tok->route_cluster_ids[j] = (j < obj_route_cluster_count) ? obj_route_cluster_ids[j] : -1;

			/* Recency + confidence */
			{
				float decay_s = (cand->type == QNN_CAND_ACTOR) ? QNN_RECENCY_DECAY_PLAYER_S : QNN_RECENCY_DECAY_S;
				tok->recency = QNN_Clamp(1.0f - (cand->recency / decay_s), 0.0f, 1.0f);
			}
			tok->confidence = 1.0f;

			/* Modality from timestamps.
			   Players/projectiles: SIGHT or SOUND.
			   Items/movers: PROXIMITY or SOUND. */
			switch (cand->type)
			{
			case QNN_CAND_PROJECTILE:
				{
					qnn_entity_t *p = (qnn_entity_t *)cand->entry;
					if (p->subject_id == QNN_SUBJECT_BACKPACK)
						tok->modality_id = QNN_MODALITY_PROXIMITY;
					else
						tok->modality_id = QNN_MODALITY_SIGHT;
				}
				break;
			case QNN_CAND_ACTOR:
				{
					qnn_entity_t *a = (qnn_entity_t *)cand->entry;
					float recall_age = (a->mem > 0) ? now - a->mem : 999.0f;
					if (recall_age < 0.1f)
						tok->modality_id = QNN_MODALITY_MEMORY;
					else
					{
						qboolean in_fov = QNN_InFov(snapshot->player_origin,
							snapshot->player_view_angles, origin);
						tok->modality_id = in_fov ? QNN_MODALITY_SIGHT : QNN_MODALITY_SOUND;
					}
				}
				break;
			case QNN_CAND_ITEM:
			case QNN_CAND_MOVER:
				{
					qnn_entity_t *e = (qnn_entity_t *)cand->entry;
					float pvs_age = (e->pvs > 0) ? now - e->pvs : 999.0f;
					float snd_age = (e->snd > 0) ? now - e->snd : 999.0f;
					float recall_age = (e->mem > 0) ? now - e->mem : 999.0f;
					if (recall_age < 0.1f)
						tok->modality_id = QNN_MODALITY_MEMORY;
					else if (pvs_age < 0.1f)
						tok->modality_id = QNN_MODALITY_PROXIMITY;
					else if (snd_age < 0.1f)
						tok->modality_id = QNN_MODALITY_SOUND;
					else if (pvs_age < snd_age)
						tok->modality_id = QNN_MODALITY_PROXIMITY;
					else
						tok->modality_id = QNN_MODALITY_SOUND;
				}
				break;
			}

			/* Type-specific fields */
			switch (cand->type)
			{
			case QNN_CAND_PROJECTILE:
				{
					qnn_entity_t *p = (qnn_entity_t *)cand->entry;
					tok->subject_id = p->subject_id;
					tok->magnitude = 0.0f;
					tok->state = 0.0f;
				}
				break;
			case QNN_CAND_ACTOR:
				{
					qnn_entity_t *a = (qnn_entity_t *)cand->entry;
					tok->subject_id = QNN_SUBJECT_PLAYER;
					tok->qualifier_id = a->qualifier_id;
					tok->player_id = a->entity_num;
					tok->weapon_subject_id = a->weapon_subject_id;
					tok->powerup_subject_id = a->powerup_subject_id;
					tok->magnitude = QNN_FragFraction(
						(cl.scores != NULL && (a->entity_num - 1) >= 0 && (a->entity_num - 1) < cl.maxclients)
						? cl.scores[a->entity_num - 1].frags : 0);
					tok->state = QNN_IsSameTeam(a->entity_num);
				}
				break;
			case QNN_CAND_ITEM:
				{
					qnn_entity_t *e = (qnn_entity_t *)cand->entry;
					tok->subject_id = e->subject_id;
					if (e->regen > 0.0f)
					{
						float remaining = e->regen;
						if (remaining < 0.0f) remaining = 0.0f;
						tok->state = remaining / QNN_TIME_SCALE;
					}
					else
					{
						tok->state = 0.0f;
					}
					/* Per-type normalization matching old QNN_ClassifyItemSubject logic */
					switch (e->subject_id)
					{
					case QNN_SUBJECT_HEALTH:
					case QNN_SUBJECT_MEGAHEALTH:
						tok->magnitude = QNN_Normalize((float)e->amount, QNN_SELF_HEALTH_CAP);
						break;
					case QNN_SUBJECT_ARMOR_GREEN:
						tok->magnitude = ((float)e->amount * 0.3f) / QNN_SELF_HEALTH_CAP;
						break;
					case QNN_SUBJECT_ARMOR_YELLOW:
						tok->magnitude = ((float)e->amount * 0.6f) / QNN_SELF_HEALTH_CAP;
						break;
					case QNN_SUBJECT_ARMOR_RED:
						tok->magnitude = ((float)e->amount * 0.8f) / QNN_SELF_HEALTH_CAP;
						break;
					case QNN_SUBJECT_SHELLS:
						tok->magnitude = QNN_Normalize((float)e->amount, QNN_SELF_SHELLS_CAP);
						break;
					case QNN_SUBJECT_NAILS:
						tok->magnitude = QNN_Normalize((float)e->amount, QNN_SELF_NAILS_CAP);
						break;
					case QNN_SUBJECT_ROCKETS:
						tok->magnitude = QNN_Normalize((float)e->amount, QNN_SELF_ROCKETS_CAP);
						break;
					case QNN_SUBJECT_CELLS:
						tok->magnitude = QNN_Normalize((float)e->amount, QNN_SELF_CELLS_CAP);
						break;
					case QNN_SUBJECT_QUAD:
					case QNN_SUBJECT_PENT:
					case QNN_SUBJECT_RING:
					case QNN_SUBJECT_SUIT:
						tok->magnitude = 1.0f;
						break;
					default:
						tok->magnitude = 0.0f;
						break;
					}
				}
				break;
			case QNN_CAND_MOVER:
				{
					qnn_entity_t *e = (qnn_entity_t *)cand->entry;
					tok->subject_id = e->subject_id;
					tok->state = e->state;
					tok->magnitude = 0.0f;
				}
				break;
			}

			/* Events */
			QNN_FillTokenEvents(tok, cand->store_index);

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
