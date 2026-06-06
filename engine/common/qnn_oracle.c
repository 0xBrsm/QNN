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

/* Determine the entity's best qualifying modality.  Returns true and
 * sets *out_modality + *out_age if any modality threshold is met.
 * Returns false if all observations are stale or absent.
 * Priority: sight/proximity (pvs) > sound > memory. */
static qboolean QNN_QualifyEntity(const qnn_entity_t *e, float now,
	int *out_modality, float *out_age)
{
	float near = QNN_PrimaryObservationTimestamp(e);
	float newest = near;
	int modality = QNN_PrimaryObservationModalityId(e);
	int primary_modality = modality;
	float age, threshold;

	if (e->snd > newest) newest = e->snd;
	if (e->mem > newest) newest = e->mem;
	if (newest <= 0) return false;

	/* Determine modality from newest observation.
	   Later checks overwrite earlier: mem < snd < near.
	   Always sets modality since newest > 0 implies at least one match. */
	modality = QNN_MODALITY_MEMORY;
	if (e->snd == newest) modality = QNN_MODALITY_SOUND;
	if (near == newest) modality = primary_modality;

	/* Check age against modality threshold */
	age = now - newest;
	threshold = QNN_RecencyMaxForModality(modality);
	if (age > threshold)
	{
		/* Fall back to the entity's primary observation channel if it is
		   still within that channel's recency window. */
		if (near > 0.0f
			&& (now - near) <= QNN_RecencyMaxForModality(primary_modality))
		{
			modality = primary_modality;
			age = now - near;
		}
		else
			return false;
	}

	*out_modality = modality;
	*out_age = age;
	return true;
}

/* ── Candidate types ──────────────────────────────────────────── */

#define QNN_CAND_PROJECTILE 0
#define QNN_CAND_ACTOR      1
#define QNN_CAND_ITEM       2
#define QNN_CAND_MOVER      3

/* Pool: one per candidate type so the per-pool sub-sort is local.
 * Order across pools (lower = earlier in token stream):
 *   0 = ACTOR        (slot 0 lives here — the engagement target)
 *   1 = PROJECTILE   (incoming threats)
 *   2 = ITEM         (pickups)
 *   3 = MOVER        (doors / platforms)
 */
#define QNN_POOL_ACTOR      0
#define QNN_POOL_PROJECTILE 1
#define QNN_POOL_ITEM       2
#define QNN_POOL_MOVER      3

typedef struct {
	int   type;
	int   modality;
	int   store_index;
	float recency;
	void *entry;
	int   pool;        /* see QNN_POOL_* above */
	int   entity_num;  /* engine edict number; secondary sort key. */
} qnn_candidate_t;

/* Token ordering: pool then engine edict number.
 *
 * Pool grouping (actor → projectile → item → mover) keeps the slot layout
 * structurally predictable so projectiles can never be truncated out by
 * item pressure (scripts/analysis/token_slot_pressure.py confirms actors
 * + projectiles never exceed 16 in 31M training frames). Within each pool,
 * edict number is the secondary key — deterministic, permutation-agnostic,
 * no threat / recency / engagement priors baked in. The model learns slot
 * routing from per-token features. */
static int QNN_CandidateCompare(const void *a, const void *b)
{
	const qnn_candidate_t *ca = (const qnn_candidate_t *)a;
	const qnn_candidate_t *cb = (const qnn_candidate_t *)b;
	if (ca->pool != cb->pool)
		return ca->pool - cb->pool;
	return ca->entity_num - cb->entity_num;
}

/* QNN_OracleResetState previously cleared the sticky-engagement state
 * machine that pinned the engaged enemy PID to slot 0.  That machine was
 * removed when the candidate sort was simplified to pool-then-edict; the
 * function survives as a no-op so existing callers (qnn_io.c) stay valid
 * without a coupled header change. */
void QNN_OracleResetState(void)
{
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

/* ── Compute relative frame + velocity ──────────────────────────
 *
 * Native-width policy: oracle emits raw Quake units / raw seconds.
 * Scaling (DIST_SCALE / VELOCITY_SCALE / TIME_SCALE) moves to the
 * model-side dequantizers; the wire packer quantizes to i16/u16/f16
 * native widths in qnn_io.c. */

static float QNN_ComputeRel(const qnn_snapshot_t *snapshot, const vec3_t origin, float *out_rel)
{
	vec3_t delta, rel;
	float dist;
	VectorSubtract(origin, snapshot->player_origin, delta);
	dist = QNN_VecLength(delta);
	QNN_RelativeFrame(snapshot->player_view_angles, delta, rel);
	out_rel[0] = rel[0];
	out_rel[1] = rel[1];
	out_rel[2] = rel[2];
	return dist;
}

static void QNN_ComputeVel(const qnn_snapshot_t *snapshot, const vec3_t velocity, float *out_vel)
{
	vec3_t vel_view;
	QNN_RelativeFrame(snapshot->player_view_angles, velocity, vel_view);
	out_vel[0] = vel_view[0];
	out_vel[1] = vel_view[1];
	out_vel[2] = vel_view[2];
}

static float QNN_ComputePath(const qnn_snapshot_t *snapshot, const vec3_t path_world, float *out_path)
{
	vec3_t path_rel;
	float dist = QNN_VecLength(path_world);
	QNN_RelativeFrame(snapshot->player_view_angles, path_world, path_rel);
	out_path[0] = path_rel[0];
	out_path[1] = path_rel[1];
	out_path[2] = path_rel[2];
	return dist;
}

static void QNN_NormalizeHalfExtents(const float *raw, float *out)
{
	/* Raw Quake units — wire u8 quantization saturates at 255. */
	out[0] = raw[0];
	out[1] = raw[1];
	out[2] = raw[2];
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

			if (!QNN_QualifyEntity(e, now, &modality, &age))
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
			/* Clamp recency to non-negative. age = cl.mtime[0] - newest_obs
			 * can go negative across map-change-within-demo boundaries
			 * (cl.mtime resets while qnn_store retains prior-segment
			 * timestamps). Negative recency in the obs blows up the
			 * target labeler's exp(-recency/tau). */
			candidates[candidate_count].recency = (age < 0.0f) ? 0.0f : age;
			candidates[candidate_count].entry = e;
			candidates[candidate_count].entity_num = e->entity_num;

			switch (cand_type)
			{
			case QNN_CAND_ACTOR:      candidates[candidate_count].pool = QNN_POOL_ACTOR;      break;
			case QNN_CAND_PROJECTILE: candidates[candidate_count].pool = QNN_POOL_PROJECTILE; break;
			case QNN_CAND_ITEM:       candidates[candidate_count].pool = QNN_POOL_ITEM;       break;
			case QNN_CAND_MOVER:      candidates[candidate_count].pool = QNN_POOL_MOVER;      break;
			default:                  candidates[candidate_count].pool = QNN_POOL_MOVER;      break;
			}

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
		qnn_tagged_token_t *out;
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
			QNN_LookupEntityBounds(entity_num, half_ext, origin_adj);
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

			/* Each path iteration emits to its own slot. */
			out = &out_tokens[token_count];

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
					tok->recency = cand->recency;
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
					tok->eta = eta;

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

#ifdef QNN_QW_BUILD
					if (getenv("QNN_DUMP_TOKENS") != NULL)
					{
						static int frame_idx = 0;
						static int dumped_frames = 0;
						/* dump 4 frames spread out: 50, 200, 500, 1000 */
						int targets[4] = {50, 200, 500, 1000};
						int hit = 0, ti;
						for (ti = 0; ti < 4; ++ti)
							if (frame_idx == targets[ti]) hit = 1;
						if (hit)
						{
							const char *name = "?";
							if (e->entity_num > 0 && e->entity_num <= MAX_CLIENTS)
								name = cl.players[e->entity_num - 1].name;
							fprintf(stderr, "[tok-dump] frame=%d slot=%d entnum=%d name=\"%s\" team=%.1f dist=%.0f rec=%.2f\n",
								frame_idx, token_count, e->entity_num, name,
								tok->team, tok->dist, cand->recency);
						}
						if (token_count == 0)
							++frame_idx;
						(void)dumped_frames;
					}
#endif

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

					tok->recency = cand->recency;
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
					tok->eta = eta;
					/* Raw engine pickup amount.  Per-subject
					 * normalization happens model-side via
					 * qnn.engine_norm.ITEM_AMOUNT_MULT/CONST. */
					tok->amount = (float)e->amount;
					tok->regen = e->regen;
					tok->recency = cand->recency;
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
					tok->eta = eta;
					tok->state = e->state;
					tok->recency = cand->recency;
					tok->event_count = QNN_FillEvents(tok->events, QNN_MAX_ENTITY_EVENTS, cand->store_index);
				}
				break;
			}

			token_count++;
		}
		}
	}

	return token_count;
}
