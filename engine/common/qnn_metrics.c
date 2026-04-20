/*
 * qnn_metrics.c — Raw per-tick game metrics.
 *
 * Pure measurement: tracking cosine (geometry), metrics struct definition,
 * and binary write. Damage/frag/pickup data is populated by qnn_reward.c
 * which owns the QC builtin callbacks.
 */

#include "qnn.h"
#include "qnn_metrics.h"
#include "qnn_store.h"

#include <math.h>
#include <string.h>

/* ── Per-entity tracking cosine ──────────────────────────────────── */

void QNN_ComputeTracking(qnn_metrics_t *out, const qnn_snapshot_t *snapshot)
{
	vec3_t forward;
	float now = (float)cl.mtime[0];
	int entity_num;

	QNN_ForwardFromAngles(snapshot->player_view_angles, forward);

	for (entity_num = 1; entity_num <= cl.maxclients && entity_num < MAX_EDICTS; ++entity_num)
	{
		const qnn_entity_t *ent = &qnn_store[entity_num];
		float dx, dy, dz, dist_sq, inv_dist, cos_val;
		int slot, j;

		if (entity_num == cl.viewentity)
			continue;
		if (ent->type != QNN_ENT_ACTOR)
			continue;
		if (!QNN_PrimaryObservationIsCurrent(ent, now))
			continue;

		dx = ent->origin[0] - snapshot->player_origin[0];
		dy = ent->origin[1] - snapshot->player_origin[1];
		dz = ent->origin[2] - snapshot->player_origin[2];
		dist_sq = dx * dx + dy * dy + dz * dz;
		if (dist_sq < 1e-8f)
			continue;

		inv_dist = 1.0f / sqrtf(dist_sq);
		cos_val = (forward[0] * dx + forward[1] * dy + forward[2] * dz) * inv_dist;

		/* Find or create entity slot */
		slot = -1;
		for (j = 0; j < out->entity_count; ++j)
		{
			if (out->entities[j].entity_num == entity_num)
			{
				slot = j;
				break;
			}
		}
		if (slot < 0 && out->entity_count < QNN_MAX_METRIC_ENTITIES)
		{
			slot = out->entity_count++;
			memset(&out->entities[slot], 0, sizeof(out->entities[slot]));
			out->entities[slot].entity_num = entity_num;
		}
		if (slot >= 0)
			out->entities[slot].tracking_cos = cos_val;
	}
}

/* ── Effective HP (shared utility) ───────────────────────────────── */

float QNN_EffectiveHP(float health, float armor, float armor_type)
{
	float armor_first, health_first;

	if (health < 1.0f)
		health = 1.0f;
	if (armor <= 0.0f || armor_type <= 0.0f)
		return health;
	armor_first = health + armor;
	health_first = health / (1.0f - armor_type);
	return armor_first < health_first ? armor_first : health_first;
}

/* ── Entity slot lookup ──────────────────────────────────────────── */

int QNN_MetricsFindOrAddEntity(qnn_metrics_t *m, int entity_num)
{
	int j;
	for (j = 0; j < m->entity_count; ++j)
	{
		if (m->entities[j].entity_num == entity_num)
			return j;
	}
	if (m->entity_count < QNN_MAX_METRIC_ENTITIES)
	{
		int slot = m->entity_count++;
		memset(&m->entities[slot], 0, sizeof(m->entities[slot]));
		m->entities[slot].entity_num = entity_num;
		return slot;
	}
	return -1;
}

/* ── Binary write ────────────────────────────────────────────────── */

void QNN_WriteMetrics(FILE *out, const qnn_metrics_t *metrics, int tick, int steps, int flags)
{
	QNN_WriteI32LE(out, (int32_t)tick);
	QNN_WriteI32LE(out, (int32_t)steps);
	QNN_WriteI32LE(out, (int32_t)flags);
	fwrite(metrics, 1, sizeof(qnn_metrics_t), out);
}
