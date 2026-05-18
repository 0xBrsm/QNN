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
	int type;
	int modality;
	int store_index;
	float recency;
	void *entry;
	int pool;        /* see QNN_POOL_* above */
	int team;        /* 0 = enemy / unknown, 1 = teammate. Only meaningful for
	                  * actors; projectiles default to enemy, items irrelevant. */
	float sort_key;  /* lower = more important: distance × (1 - alpha·dot) for
	                  * threat pools, distance for non-threat pools. */
	int entity_num;  /* engine entity id; used as PID for sticky tracking. */
	float cos_aim;   /* dot(player_fwd, toward_actor) for actors; used by the
	                  * cos-aware tiebreak at engagement start. 0 for non-actors. */
	float dist;      /* raw distance (Quake units) from player to candidate.
	                  * Used by the adaptive acquire/release cones. 0 for
	                  * non-actors (cone tests only applied to actors). */
	int sticky;      /* 1 = this candidate is the current engagement target;
	                  * it overrides the normal sort to claim slot 0. */
} qnn_candidate_t;

/* Threat-axis weighting for actor/projectile pools.
 * Projectiles: dot = velocity_unit · toward_player  (is it flying at me?)
 * Actors:      dot = player_fwd · toward_actor      (is it in my crosshair?)
 * Lower sort_key = higher priority (slot 0). */
#define QNN_THREAT_ALPHA_PROJ  0.4f   /* projectile velocity-facing weight */
#define QNN_THREAT_ALPHA_ACTOR 1.0f   /* player-facing weight for actors */

/* Fire-anchored sticky transfer threshold (legacy 30° cone, kept for
 * reference). The opt3 design replaces this with the adaptive cones below. */
#define QNN_COS_FIRE_THRESHOLD   0.866f

/* Adaptive acquire/release cones (opt3). Both expressed as a transverse
 * offset in Quake units: the cone admits enemies within X units perpendicular
 * of the aim line, capped at an angular maximum at close range and floored
 * at 5° at extreme range. Release cone is twice the acquire offset
 * (K=2 Schmitt-trigger ratio), capped at 45° instead of 30°. */
#define QNN_ACQUIRE_TRANSVERSE_U  208.0f
#define QNN_RELEASE_TRANSVERSE_U  416.0f
#define QNN_ACQUIRE_CAP_COS       0.8660254f   /* cos(30°) */
#define QNN_RELEASE_CAP_COS       0.7071068f   /* cos(45°) */
#define QNN_CONE_FLOOR_COS        0.9961947f   /* cos(5°) */

static inline float QNN_OracleAcquireConeCos(float dist)
{
	float d = (dist < 1e-3f) ? 1e-3f : dist;
	float c = cosf(atanf(QNN_ACQUIRE_TRANSVERSE_U / d));
	if (c < QNN_ACQUIRE_CAP_COS) return QNN_ACQUIRE_CAP_COS;
	if (c > QNN_CONE_FLOOR_COS) return QNN_CONE_FLOOR_COS;
	return c;
}

static inline float QNN_OracleReleaseConeCos(float dist)
{
	float d = (dist < 1e-3f) ? 1e-3f : dist;
	float c = cosf(atanf(QNN_RELEASE_TRANSVERSE_U / d));
	if (c < QNN_RELEASE_CAP_COS) return QNN_RELEASE_CAP_COS;
	if (c > QNN_CONE_FLOOR_COS) return QNN_CONE_FLOOR_COS;
	return c;
}

static int QNN_CandidateCompare(const void *a, const void *b)
{
	const qnn_candidate_t *ca = (const qnn_candidate_t *)a;
	const qnn_candidate_t *cb = (const qnn_candidate_t *)b;

	/* Sticky always wins slot 0. Only actor candidates can be sticky (state
	 * machine only sets the flag on enemy actors with PVS modality), so this
	 * effectively promotes the engaged enemy PID to slot 0. */
	if (ca->sticky != cb->sticky)
		return ca->sticky ? -1 : 1;

	/* Pool: actor → projectile → item → mover. */
	if (ca->pool != cb->pool)
		return ca->pool - cb->pool;
	/* Team: enemies (0) before teammates (1). */
	if (ca->team != cb->team)
		return ca->team - cb->team;
	/* Recency: lower age value = more recently observed → emit first. */
	if (ca->recency < cb->recency) return -1;
	if (ca->recency > cb->recency) return 1;
	/* Threat-modulated distance (actor / projectile pools) or pure distance. */
	if (ca->sort_key < cb->sort_key) return -1;
	if (ca->sort_key > cb->sort_key) return 1;
	return 0;
}

/* ── Sticky-by-PID engagement state ────────────────────────────────
 *
 * The engine tracks one "current engagement target" via PID, persisted across
 * frames. The sticky PID claims slot 0 whenever it is visible, regardless of
 * recency churn or new actors entering PVS. Audit data showed the labeler's
 * within-engagement label is 99.0% stable per PID; sticky makes the engine's
 * slot ordering reflect that stability.
 *
 * Transitions:
 *   ACQUIRE  : sticky empty AND demonstrator fires AND a visible enemy actor
 *              has cos_aim > QNN_COS_FIRE_THRESHOLD → set sticky to that PID.
 *              Also acquired implicitly at engagement start through the
 *              cos-aware tiebreak (the actor that wins slot 0 gets sticky).
 *   TRANSFER : demonstrator fires AND a visible enemy actor (cos_aim above
 *              threshold) has a different PID than sticky → switch sticky.
 *              Catches ~84% of mid-engagement target switches per the audit.
 *   RELEASE  : sticky PID is not present in the candidate set this tick
 *              (token expired from the entity pool — recency exceeded the
 *              SIGHT cap at 2s / 40 frames). No separate timeout.
 *
 * Reset: cleared by QNN_OracleResetState() on episode boundaries
 * (called from QNN_IOUpdate when reset_flag is set).
 */
typedef struct {
	int active;       /* 0 = no sticky; 1 = sticky PID is set */
	int pid;          /* entity_num of the current engagement target */
} qnn_engagement_state_t;

static qnn_engagement_state_t qnn_engagement_state;

void QNN_OracleResetState(void)
{
	qnn_engagement_state.active = 0;
	qnn_engagement_state.pid = 0;
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
	case QNN_SUBJECT_SUPER_SHOTGUN:
		return QNN_Normalize(5.0f, QNN_MAX_SHELLS);
	case QNN_SUBJECT_NAILGUN:
	case QNN_SUBJECT_SUPER_NAILGUN:
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
		vec3_t player_fwd, _pfwd_right, _pfwd_up;
		AngleVectors(snapshot->player_view_angles, player_fwd, _pfwd_right, _pfwd_up);
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
			candidates[candidate_count].recency = age;
			candidates[candidate_count].entry = e;
			candidates[candidate_count].entity_num = e->entity_num;
			candidates[candidate_count].sticky = 0;
			candidates[candidate_count].cos_aim = 0.0f;
			candidates[candidate_count].dist = 0.0f;

			/* Pool: one slot per type. Actor leads, then projectile, then
			 * item, then mover. (Previously actor+projectile shared pool 0.) */
			switch (cand_type)
			{
			case QNN_CAND_ACTOR:      candidates[candidate_count].pool = QNN_POOL_ACTOR;      break;
			case QNN_CAND_PROJECTILE: candidates[candidate_count].pool = QNN_POOL_PROJECTILE; break;
			case QNN_CAND_ITEM:       candidates[candidate_count].pool = QNN_POOL_ITEM;       break;
			case QNN_CAND_MOVER:      candidates[candidate_count].pool = QNN_POOL_MOVER;      break;
			default:                  candidates[candidate_count].pool = QNN_POOL_MOVER;      break;
			}

			/* Team: only actors carry a team. Projectiles default to enemy
			 * (we don't track the firer); items/movers are neutral so the
			 * field never participates in their sort path. */
			if (cand_type == QNN_CAND_ACTOR)
				candidates[candidate_count].team = (QNN_IsSameTeam(e->entity_num) > 0.5f) ? 1 : 0;
			else
				candidates[candidate_count].team = 0;

			/* Precompute sort_key (distance, optionally threat-modulated). */
			{
				vec3_t to_us;
				float dist;
				VectorSubtract(snapshot->player_origin, e->origin, to_us);
				dist = QNN_VecLength(to_us);
				candidates[candidate_count].dist = dist;
				if (cand_type == QNN_CAND_PROJECTILE && dist > 1e-3f)
				{
					/* Projectile: dot = velocity_unit · toward_player.
					 * Stationary projectile has no threat axis → pure distance. */
					float vlen = QNN_VecLength(e->velocity);
					float dot;
					if (vlen > 1e-3f)
						dot = (e->velocity[0] * to_us[0] + e->velocity[1] * to_us[1]
						       + e->velocity[2] * to_us[2]) / (vlen * dist);
					else
						dot = 0.0f;
					candidates[candidate_count].sort_key = dist * (1.0f - QNN_THREAT_ALPHA_PROJ * dot);
				}
				else if (cand_type == QNN_CAND_ACTOR && dist > 1e-3f)
				{
					/* Actor: dot = player_fwd · toward_actor.
					 * Entities more in the player's crosshair sort first. */
					float dot = -(player_fwd[0] * to_us[0] + player_fwd[1] * to_us[1]
					              + player_fwd[2] * to_us[2]) / dist;
					candidates[candidate_count].cos_aim = dot;
					candidates[candidate_count].sort_key = dist * (1.0f - QNN_THREAT_ALPHA_ACTOR * dot);
				}
				else
				{
					candidates[candidate_count].sort_key = dist;
				}
			}

			candidate_count++;
		}
	}

	/* ---- Sticky engagement state machine (opt3) ----
	 *
	 * Schmitt-trigger hysteresis with per-candidate adaptive cones:
	 *   acquire_cone(d) = clamp(atan(208/d), 5°, 30°)
	 *   release_cone(d) = clamp(atan(416/d), 5°, 45°)
	 *
	 * Behaviour per tick:
	 *   - RELEASE (always): if sticky PID is not in the visible candidate
	 *     set this tick → clear sticky.
	 *   - On fire press:
	 *       (a) sticky-keep test: if sticky cos_aim >= release_cone(dist),
	 *           keep sticky regardless of higher-cos in-cone candidates.
	 *       (b) otherwise: cone-argmax over actors passing their own
	 *           acquire_cone(dist) threshold. If found, acquire/transfer.
	 *   - Between fires, sticky persists as long as it remains in stream.
	 */
	{
		int sticky_idx = -1;
		int fire_target_idx = -1;
		float fire_target_cos = -2.0f;
		int j;

		/* Locate the sticky candidate (if any) and the highest-cos visible
		 * enemy actor that passes its own adaptive acquire cone. Restricted
		 * to PVS modalities (SIGHT / PROXIMITY) so sticky's in-stream check
		 * aligns with the BC labeler's. */
		for (j = 0; j < candidate_count; ++j)
		{
			qnn_candidate_t *c = &candidates[j];
			if (c->pool != QNN_POOL_ACTOR || c->team != 0)
				continue;
			if (c->modality != QNN_MODALITY_SIGHT
			    && c->modality != QNN_MODALITY_PROXIMITY)
				continue;
			if (qnn_engagement_state.active
			    && c->entity_num == qnn_engagement_state.pid)
				sticky_idx = j;
			if (c->cos_aim >= QNN_OracleAcquireConeCos(c->dist)
			    && c->cos_aim > fire_target_cos)
			{
				fire_target_cos = c->cos_aim;
				fire_target_idx = j;
			}
		}

		/* Release if the sticky PID has aged out of the candidate set. */
		if (qnn_engagement_state.active && sticky_idx < 0)
			qnn_engagement_state.active = 0;

		/* Fire-anchored sticky-keep / acquire / transfer. */
		if (snapshot->action_label.fire)
		{
			int sticky_kept = 0;
			if (sticky_idx >= 0)
			{
				float rel_thr = QNN_OracleReleaseConeCos(candidates[sticky_idx].dist);
				if (candidates[sticky_idx].cos_aim >= rel_thr)
					sticky_kept = 1;
			}
			if (!sticky_kept)
			{
				/* Sticky failed release-cone check (or was never set): fall
				 * through to acquire/transfer. */
				if (sticky_idx >= 0)
				{
					/* Sticky existed but is out of release cone → release. */
					qnn_engagement_state.active = 0;
					sticky_idx = -1;
				}
				if (fire_target_idx >= 0)
				{
					qnn_engagement_state.active = 1;
					qnn_engagement_state.pid = candidates[fire_target_idx].entity_num;
					sticky_idx = fire_target_idx;
				}
			}
		}

		/* Mark the sticky candidate so the comparator promotes it. */
		if (sticky_idx >= 0)
			candidates[sticky_idx].sticky = 1;
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

	return token_count;
}
