/*
 * qnn_metrics.h — Raw per-tick game metrics.
 *
 * Metrics are pure measurements of what happened. No reward shaping,
 * no aggregation, no policy decisions. Consumers (qnn_reward.c, Python eval)
 * read the struct and compute whatever derived values they need.
 *
 * Per-entity metrics are keyed by entity_num so consumers can filter
 * or aggregate across targets without 1v1 assumptions.
 */

#ifndef QNN_METRICS_H
#define QNN_METRICS_H

#include <stdio.h>

#define QNN_MAX_METRIC_ENTITIES 16

typedef struct
{
	int	entity_num;		/* Quake entity index */
	float	tracking_cos;		/* cosine from player aim to this entity */
	float	damage_dealt;		/* damage dealt TO this entity this tick */
	float	damage_taken;		/* damage taken FROM this entity this tick */
	float	ehp_before;		/* entity effective HP before this tick */
	float	ehp_after;		/* entity effective HP after this tick */
} qnn_entity_metrics_t;

typedef struct
{
	/* Per-entity */
	qnn_entity_metrics_t	entities[QNN_MAX_METRIC_ENTITIES];
	int			entity_count;

	/* Player-global (not per-entity) */
	float	health_before;
	float	health_after;
	float	armor_before;
	float	armor_after;
	float	armor_type;
	int	hit_count;
	int	shots_fired;
	int	frag_gain;
	int	frag_loss;
	int	pickup_health;
	int	pickup_armor;
	int	pickup_ammo;
	int	pickup_weapon;
	int	damage_flags;		/* QNN_DAMAGE_FLAG_* bits */
} qnn_metrics_t;

/* Damage flag bits */
#define QNN_DAMAGE_FLAG_SPLASH  0x0004

/* Requires qnn.h to be included first for qnn_snapshot_t. */

/* Compute tracking cosine for all visible enemies.
   Populates entities[] with per-entity tracking_cos.
   Call this first, then QNN_FillMetricsFromRecords for damage data. */
void QNN_ComputeTracking(qnn_metrics_t *out, const qnn_snapshot_t *snapshot);

/* Effective HP calculation (used by both metrics and reward). */
float QNN_EffectiveHP(float health, float armor, float armor_type);

/* Find or create an entity slot in the metrics struct. Returns slot index or -1. */
int QNN_MetricsFindOrAddEntity(qnn_metrics_t *m, int entity_num);

/* Fill damage, frag, pickup fields from accumulated records.
   Defined in qnn_reward.c which owns the PF_ builtin callbacks. */
void QNN_FillMetricsFromRecords(qnn_metrics_t *out, const qnn_snapshot_t *snapshot);

/* Write metrics to stream as fixed-size binary record. */
void QNN_WriteMetrics(FILE *out, const qnn_metrics_t *metrics, int tick, int steps, int flags);

#endif /* QNN_METRICS_H */
