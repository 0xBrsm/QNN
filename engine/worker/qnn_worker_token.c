#include "qnn_worker.h"
#include "qnn_nav_oracle.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define QNN_TOKEN_MAGIC "QTOK"
#define QNN_TOKEN_VERSION 6

#define QNN_TOKEN_FLAG_RESET      0x0001
#define QNN_TOKEN_FLAG_DONE       0x0002
#define QNN_TOKEN_FLAG_HAS_ACTION 0x0004

/* Modality priority: lower number = higher priority.
   VISUAL beats AUDITORY beats MENTAL.  qnn_ensure_dynamic_object
   only upgrades modality, never downgrades. */
#define QNN_MODALITY_NONE 0
#define QNN_MODALITY_VISUAL 1
#define QNN_MODALITY_AUDITORY 2
#define QNN_MODALITY_MENTAL 3

/* player_id: 0 = non-player, 1..32 = player slot */

#define QNN_SUBJECT_NONE 0
#define QNN_SUBJECT_PLAYER 1
#define QNN_SUBJECT_BACKPACK 2
#define QNN_SUBJECT_AXE 3
#define QNN_SUBJECT_SHOTGUN 4
#define QNN_SUBJECT_NAILGUN 5
#define QNN_SUBJECT_GRENADE_LAUNCHER 6
#define QNN_SUBJECT_ROCKET_LAUNCHER 7
#define QNN_SUBJECT_THUNDERBOLT 8
#define QNN_SUBJECT_SHELLS 9
#define QNN_SUBJECT_NAILS 10
#define QNN_SUBJECT_ROCKETS 11
#define QNN_SUBJECT_CELLS 12
#define QNN_SUBJECT_HEALTH 13
#define QNN_SUBJECT_MEGAHEALTH 14
#define QNN_SUBJECT_ARMOR_GREEN 15
#define QNN_SUBJECT_ARMOR_YELLOW 16
#define QNN_SUBJECT_ARMOR_RED 17
#define QNN_SUBJECT_QUAD 18
#define QNN_SUBJECT_PENT 19
#define QNN_SUBJECT_RING 20
#define QNN_SUBJECT_SUIT 21
#define QNN_SUBJECT_PROJECTILE_NAIL 22
#define QNN_SUBJECT_PROJECTILE_GRENADE 23
#define QNN_SUBJECT_PROJECTILE_ROCKET 24
#define QNN_SUBJECT_LIGHTNING_BEAM 25
#define QNN_SUBJECT_TELEPORTER 26
#define QNN_SUBJECT_DOOR 27
#define QNN_SUBJECT_PLATFORM 28
#define QNN_SUBJECT_TRAIN 29
#define QNN_SUBJECT_BUTTON 30

#define QNN_ACTION_NONE 0
#define QNN_ACTION_FIRE 1
#define QNN_ACTION_IMPACT 2
#define QNN_ACTION_BOUNCE 3
#define QNN_ACTION_PICKUP 4
#define QNN_ACTION_RESPAWN 5
#define QNN_ACTION_PAIN 6
#define QNN_ACTION_DEATH 7
#define QNN_ACTION_WARNING 8
#define QNN_ACTION_ACTIVE 9
#define QNN_ACTION_JUMP 10
#define QNN_ACTION_LAND 11
#define QNN_ACTION_ENTER 12
#define QNN_ACTION_EXIT 13
#define QNN_ACTION_TELEPORT 14
#define QNN_ACTION_MOVE 15
#define QNN_ACTION_ACTIVATE 16
#define QNN_ACTION_REJECT 17
#define QNN_ACTION_BREATH 18

#define QNN_QUAL_NONE 0
#define QNN_QUAL_DROWN 1
#define QNN_QUAL_WATER 2
#define QNN_QUAL_LAVA 3
#define QNN_QUAL_SLIME 4
#define QNN_QUAL_FLESH 5
#define QNN_QUAL_WORLD 6
#define QNN_QUAL_KEYED 7
#define QNN_QUAL_SECRET 8
#define QNN_QUAL_INVISIBLE 9

#define QNN_SPATIAL_FOV_CENTER 0
#define QNN_SPATIAL_FOV_LEFT 1
#define QNN_SPATIAL_FOV_RIGHT 2
#define QNN_SPATIAL_FLANK_LEFT 3
#define QNN_SPATIAL_FLANK_RIGHT 4
#define QNN_SPATIAL_REAR_LEFT 5
#define QNN_SPATIAL_REAR_RIGHT 6
#define QNN_SPATIAL_GROUND 7
#define QNN_SPATIAL_CEILING 8

#define QNN_FOV_HALF_DEG 60.0f
#define QNN_RECENCY_DECAY_S 4.0f

#define QNN_ITEM_PVS_MATCH_SQ (16.0f * 16.0f)
#define QNN_ITEM_PICKUP_MATCH_SQ (64.0f * 64.0f)
#define QNN_ITEM_RESPAWN_MATCH_SQ (16.0f * 16.0f)
#define QNN_STATIC_SOUND_MATCH_SQ (128.0f * 128.0f)
#define QNN_PROJECTILE_SOUND_MATCH_SQ (96.0f * 96.0f)
#define QNN_MAX_MATCH_CANDIDATES 4096

#define QNN_NAIL_STREAM_DOT_THRESHOLD 0.97f
#define QNN_MAX_NAIL_STREAMS 16

#define QNN_SELF_SCALAR_COUNT 23
#define QNN_SELF_ID_COUNT 3
#define QNN_OBJECT_SCALAR_COUNT 8
#define QNN_OBJECT_ID_COUNT 5
#define QNN_MAX_ROUTE_CLUSTERS 8

#define QNN_SELF_HEALTH_CAP 250.0f
#define QNN_SELF_ARMOR_CAP 200.0f
#define QNN_SELF_ARMOR_TYPE_CAP 0.8f
#define QNN_SELF_SHELLS_CAP 100.0f
#define QNN_SELF_NAILS_CAP 200.0f
#define QNN_SELF_ROCKETS_CAP 100.0f
#define QNN_SELF_CELLS_CAP 100.0f
#define QNN_SELF_VELOCITY_CAP 2000.0f
#define QNN_OBJECT_REL_SCALE 1024.0f
#define QNN_OBJECT_ROUTE_COST_SCALE 30.0f
#define QNN_SPATIAL_DIST_SCALE 1024.0f

#define QNN_VISUAL_FIRE_SHOT_FIRST 113
#define QNN_VISUAL_FIRE_SHOT_LAST 118
#define QNN_VISUAL_FIRE_NAIL_FIRST 103
#define QNN_VISUAL_FIRE_NAIL_LAST 104
#define QNN_VISUAL_FIRE_LIGHT_FIRST 105
#define QNN_VISUAL_FIRE_LIGHT_LAST 106

typedef struct
{
	const char *name;
	int category;
	int subject_id;
	int action_id;
	int qualifier_id;
	float magnitude;
} qnn_sound_rule_t;

typedef struct
{
	float dist_sq;
	int ent_idx;
	int obj_idx;
} qnn_match_candidate_t;

typedef struct
{
	qboolean active;
	qboolean is_static;
	qboolean is_item;
	int static_index;
	int entity_num;
	int subject_id;
	int qualifier_id;
	int modality_id;
	int player_id;
	int region_id;
	vec3_t origin;
	vec3_t velocity;
	vec3_t angles;
	float recency;
	float confidence;
	float magnitude;
	float state;
	float respawn_s;
	float pickup_elapsed;
	int cluster_id;
	float route_cost;
	int route_cluster_ids[QNN_MAX_ROUTE_CLUSTERS];
	int route_cluster_count;
	qboolean pvs_seen;
	qboolean surfaced_this_tick;
	float half_extents[3];
} qnn_semantic_object_t;

typedef struct
{
	qboolean active;
	int owner_index;   /* index into qnn_semantic_objects[] */
	int subject_id;
	int action_id;
	int qualifier_id;
	int modality_id;
	float recency;
	float confidence;
	float magnitude;
} qnn_semantic_event_atom_t;

typedef struct
{
	int sector_id;
	float nearest_dist;
	float mean_dist;
	float openness;
	float solid_frac;
	float water_frac;
	float slime_frac;
	float lava_frac;
	float traversable;
	float dropoff;
	float clearance;
} qnn_spatial_token_t;

static qnn_semantic_object_t *qnn_semantic_objects;
static int qnn_semantic_object_capacity;
static int qnn_semantic_static_count;
static qnn_semantic_event_atom_t qnn_semantic_events[QNN_WORKER_MAX_EVENT_ATOMS];

/* Maps emitted token slot (0-based) to semantic object array index.
 * Used by recall to resolve which object the model wants to attend to. */
static int qnn_prev_object_indices[QNN_WORKER_MAX_TOKEN_OBJECTS];
static int qnn_prev_object_count = 0;

/* Emission priority: dynamic objects first, then static. Recency tiebreak. */
static int qnn_object_row_compare(const void *a, const void *b)
{
	const qnn_semantic_object_t *oa = *(const qnn_semantic_object_t *const *)a;
	const qnn_semantic_object_t *ob = *(const qnn_semantic_object_t *const *)b;

	if (oa->is_static != ob->is_static)
		return oa->is_static ? 1 : -1;
	if (oa->recency > ob->recency)
		return -1;
	if (oa->recency < ob->recency)
		return 1;
	return 0;
}

static int qnn_object_index(const qnn_semantic_object_t *obj)
{
	return (int)(obj - qnn_semantic_objects);
}

static qboolean qnn_action_has_signal(const qnn_worker_action_t *action)
{
	if (action->move[0] != 0.0f || action->move[1] != 0.0f)
		return true;
	if (action->look[0] != 0.0f || action->look[1] != 0.0f)
		return true;
	if (action->fire || action->jump || action->switch_slot)
		return true;
	if (action->recall[0] || action->recall[1] || action->recall[2] || action->recall[3])
		return true;
	return false;
}

extern qboolean SV_RecursiveHullCheck(hull_t *hull, int num, float p1f, float p2f, vec3_t p1, vec3_t p2, trace_t *trace);

static const qnn_sound_rule_t qnn_player_sound_rules[] = {
	{"player/h2odeath.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_DEATH, QNN_QUAL_DROWN, 0.0f},
	{"player/plyrjmp8.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_JUMP, QNN_QUAL_NONE, 0.0f},
	{"player/land.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_NONE, 0.0f},
	{"player/land2.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_NONE, 1.0f},
	{"player/h2ojump.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_WATER, 0.0f},
	{"misc/h2ohit1.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_LAND, QNN_QUAL_WATER, 0.0f},
	{"player/gasp1.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_BREATH, QNN_QUAL_WATER, 0.0f},
	{"player/gasp2.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_BREATH, QNN_QUAL_WATER, 0.0f},
	{"player/inh2o.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_WATER, 0.0f},
	{"misc/outwater.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_EXIT, QNN_QUAL_WATER, 0.0f},
	{"player/inlava.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_LAVA, 0.0f},
	{"player/slimbrn2.wav", 0, QNN_SUBJECT_PLAYER, QNN_ACTION_ENTER, QNN_QUAL_SLIME, 0.0f},
	{"player/axhit1.wav", 0, QNN_SUBJECT_AXE, QNN_ACTION_IMPACT, QNN_QUAL_FLESH, 0.0f},
	{"player/axhit2.wav", 0, QNN_SUBJECT_AXE, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{NULL, 0, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_weapon_sound_rules[] = {
	{"weapons/ax1.wav", 0, QNN_SUBJECT_AXE, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/guncock.wav", 0, QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/shotgn2.wav", 0, QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 1.0f},
	{"weapons/rocket1i.wav", 0, QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/spike2.wav", 0, QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, 1.0f},
	{"weapons/grenade.wav", 0, QNN_SUBJECT_GRENADE_LAUNCHER, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/sgun1.wav", 0, QNN_SUBJECT_ROCKET_LAUNCHER, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{"weapons/lstart.wav", 0, QNN_SUBJECT_THUNDERBOLT, QNN_ACTION_FIRE, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_projectile_sound_rules[] = {
	{"weapons/bounce.wav", 0, QNN_SUBJECT_PROJECTILE_GRENADE, QNN_ACTION_BOUNCE, QNN_QUAL_WORLD, 0.0f},
	{"weapons/tink1.wav", 0, QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric1.wav", 0, QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric2.wav", 0, QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/ric3.wav", 0, QNN_SUBJECT_PROJECTILE_NAIL, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/r_exp3.wav", 0, QNN_SUBJECT_PROJECTILE_ROCKET, QNN_ACTION_IMPACT, QNN_QUAL_WORLD, 0.0f},
	{"weapons/lhit.wav", 0, QNN_SUBJECT_LIGHTNING_BEAM, QNN_ACTION_IMPACT, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0, 0.0f},
};

static const qnn_sound_rule_t qnn_static_sound_rules[] = {
	{"misc/r_tele1.wav", 0, QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele2.wav", 0, QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele3.wav", 0, QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele4.wav", 0, QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"misc/r_tele5.wav", 0, QNN_SUBJECT_TELEPORTER, QNN_ACTION_TELEPORT, QNN_QUAL_NONE, 0.0f},
	{"doors/medtry.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/runetry.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/basetry.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_REJECT, QNN_QUAL_KEYED, 0.0f},
	{"doors/meduse.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/runeuse.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/baseuse.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_ACTIVATE, QNN_QUAL_KEYED, 0.0f},
	{"doors/drclos4.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/doormv1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/hydro1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/hydro2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/stndr1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/stndr2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/ddoor1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/ddoor2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"doors/latch2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/winch2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/airdoor1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/airdoor2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/basesec1.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"doors/basesec2.wav", 0, QNN_SUBJECT_DOOR, QNN_ACTION_MOVE, QNN_QUAL_SECRET, 0.0f},
	{"plats/plat1.wav", 0, QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/plat2.wav", 0, QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/medplat1.wav", 0, QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/medplat2.wav", 0, QNN_SUBJECT_PLATFORM, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/train1.wav", 0, QNN_SUBJECT_TRAIN, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"plats/train2.wav", 0, QNN_SUBJECT_TRAIN, QNN_ACTION_MOVE, QNN_QUAL_NONE, 0.0f},
	{"buttons/airbut1.wav", 0, QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch21.wav", 0, QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch02.wav", 0, QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{"buttons/switch04.wav", 0, QNN_SUBJECT_BUTTON, QNN_ACTION_ACTIVATE, QNN_QUAL_NONE, 0.0f},
	{NULL, 0, 0, 0, 0, 0.0f},
};

static int qnn_match_candidate_compare(const void *lhs_ptr, const void *rhs_ptr)
{
	const qnn_match_candidate_t *lhs;
	const qnn_match_candidate_t *rhs;

	lhs = (const qnn_match_candidate_t *)lhs_ptr;
	rhs = (const qnn_match_candidate_t *)rhs_ptr;
	if (lhs->dist_sq < rhs->dist_sq)
		return -1;
	if (lhs->dist_sq > rhs->dist_sq)
		return 1;
	return 0;
}

static void qnn_token_write_u16_le(FILE *out, uint16_t value)
{
	unsigned char bytes[2];

	bytes[0] = (unsigned char)(value & 0xff);
	bytes[1] = (unsigned char)((value >> 8) & 0xff);
	fwrite(bytes, 1, sizeof(bytes), out);
}

static void qnn_token_write_u32_le(FILE *out, uint32_t value)
{
	unsigned char bytes[4];

	bytes[0] = (unsigned char)(value & 0xff);
	bytes[1] = (unsigned char)((value >> 8) & 0xff);
	bytes[2] = (unsigned char)((value >> 16) & 0xff);
	bytes[3] = (unsigned char)((value >> 24) & 0xff);
	fwrite(bytes, 1, sizeof(bytes), out);
}

static void qnn_token_write_i32_le(FILE *out, int32_t value)
{
	qnn_token_write_u32_le(out, (uint32_t)value);
}

static void qnn_token_write_f32_le(FILE *out, float value)
{
	union
	{
		float f;
		uint32_t u;
	} bits;

	bits.f = value;
	qnn_token_write_u32_le(out, bits.u);
}

static float qnn_clampf(float value, float low, float high)
{
	if (value < low)
		return low;
	if (value > high)
		return high;
	return value;
}

static float qnn_normalize(float value, float scale)
{
	if (scale <= 0.0f)
		return 0.0f;
	return value / scale;
}

static float qnn_angle_sin_deg(float degrees)
{
	return (float)sin((double)(degrees * (float)M_PI / 180.0f));
}

static float qnn_angle_cos_deg(float degrees)
{
	return (float)cos((double)(degrees * (float)M_PI / 180.0f));
}

static float qnn_health_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_HEALTH_CAP);
}

static float qnn_armor_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_ARMOR_CAP);
}

static float qnn_shells_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_SHELLS_CAP);
}

static float qnn_nails_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_NAILS_CAP);
}

static float qnn_rockets_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_ROCKETS_CAP);
}

static float qnn_cells_magnitude(float amount)
{
	return qnn_normalize(amount, QNN_SELF_CELLS_CAP);
}

static int qnn_self_weapon_embed_id(int weapon_id)
{
	return qnn_weapon_class_from_id(weapon_id);
}

static int qnn_self_weapon_super(int weapon_id)
{
	return (weapon_id == 3 || weapon_id == 5) ? 1 : 0;
}

static int qnn_self_movement_id(qboolean grounded, int waterlevel)
{
	switch (waterlevel)
	{
	case 1:
		return 2;
	case 2:
		return 3;
	case 3:
		return 4;
	default:
		return grounded ? 0 : 1;
	}
}

static void qnn_write_self_token(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick_hz, int player_cluster_id)
{
	float scalars[QNN_SELF_SCALAR_COUNT];
	int ids[QNN_SELF_ID_COUNT];
	int i;

	scalars[0] = qnn_normalize((float)snapshot->health, QNN_SELF_HEALTH_CAP);
	scalars[1] = qnn_normalize((float)snapshot->armor, QNN_SELF_ARMOR_CAP);
	scalars[2] = qnn_normalize(snapshot->armor_type, QNN_SELF_ARMOR_TYPE_CAP);
	scalars[3] = (snapshot->weapons_owned & IT_SHOTGUN) ? 1.0f : 0.0f;
	scalars[4] = (snapshot->weapons_owned & IT_SUPER_SHOTGUN) ? 1.0f : 0.0f;
	scalars[5] = (snapshot->weapons_owned & IT_NAILGUN) ? 1.0f : 0.0f;
	scalars[6] = (snapshot->weapons_owned & IT_SUPER_NAILGUN) ? 1.0f : 0.0f;
	scalars[7] = (snapshot->weapons_owned & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
	scalars[8] = (snapshot->weapons_owned & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
	scalars[9] = (snapshot->weapons_owned & IT_LIGHTNING) ? 1.0f : 0.0f;
	scalars[10] = (float)qnn_self_weapon_super(snapshot->weapon_id);
	scalars[11] = qnn_shells_magnitude((float)snapshot->ammo_shells);
	scalars[12] = qnn_nails_magnitude((float)snapshot->ammo_nails);
	scalars[13] = qnn_rockets_magnitude((float)snapshot->ammo_rockets);
	scalars[14] = qnn_cells_magnitude((float)snapshot->ammo_cells);
	scalars[15] = qnn_normalize(snapshot->player_velocity[0], QNN_SELF_VELOCITY_CAP);
	scalars[16] = qnn_normalize(snapshot->player_velocity[1], QNN_SELF_VELOCITY_CAP);
	scalars[17] = qnn_normalize(snapshot->player_velocity[2], QNN_SELF_VELOCITY_CAP);
	scalars[18] = qnn_angle_sin_deg(snapshot->player_view_angles[1]);
	scalars[19] = qnn_angle_cos_deg(snapshot->player_view_angles[1]);
	scalars[20] = qnn_angle_sin_deg(snapshot->player_view_angles[0]);
	scalars[21] = qnn_angle_cos_deg(snapshot->player_view_angles[0]);
	scalars[22] = tick_hz > 0 ? (1.0f / (float)tick_hz) : 0.0f;

	ids[0] = qnn_self_weapon_embed_id(snapshot->weapon_id);
	ids[1] = qnn_self_movement_id(snapshot->grounded, snapshot->waterlevel);
	ids[2] = player_cluster_id;

	for (i = 0; i < QNN_SELF_SCALAR_COUNT; ++i)
		qnn_token_write_f32_le(out, scalars[i]);
	for (i = 0; i < QNN_SELF_ID_COUNT; ++i)
		qnn_token_write_i32_le(out, ids[i]);
}

static void qnn_write_object_token(FILE *out, const qnn_semantic_object_t *obj, const vec3_t rel, uint16_t local_event_count, uint16_t local_event_base, uint32_t wire_handle)
{
	float scalars[QNN_OBJECT_SCALAR_COUNT];
	uint16_t ids[QNN_OBJECT_ID_COUNT];
	int i;

	ids[0] = (uint16_t)obj->subject_id;
	ids[1] = (uint16_t)obj->qualifier_id;
	ids[2] = (uint16_t)obj->modality_id;
	ids[3] = (uint16_t)obj->player_id;
	ids[4] = (uint16_t)obj->cluster_id;

	scalars[0] = qnn_normalize(rel[0], QNN_OBJECT_REL_SCALE);
	scalars[1] = qnn_normalize(rel[1], QNN_OBJECT_REL_SCALE);
	scalars[2] = qnn_normalize(rel[2], QNN_OBJECT_REL_SCALE);
	scalars[3] = qnn_normalize(obj->route_cost, QNN_OBJECT_ROUTE_COST_SCALE);
	scalars[4] = obj->recency;
	scalars[5] = obj->confidence;
	scalars[6] = obj->magnitude;
	scalars[7] = obj->state;

	qnn_token_write_u32_le(out, wire_handle);
	for (i = 0; i < QNN_OBJECT_ID_COUNT; ++i)
		qnn_token_write_u16_le(out, ids[i]);
	for (i = 0; i < QNN_OBJECT_SCALAR_COUNT; ++i)
		qnn_token_write_f32_le(out, scalars[i]);
	qnn_token_write_u16_le(out, local_event_count);
	qnn_token_write_u16_le(out, local_event_base);
	qnn_token_write_u16_le(out, (uint16_t)obj->route_cluster_count);
	for (i = 0; i < QNN_MAX_ROUTE_CLUSTERS; ++i)
		qnn_token_write_u16_le(out, (uint16_t)(i < obj->route_cluster_count ? obj->route_cluster_ids[i] : 0));
}

static void qnn_write_spatial_token(FILE *out, const qnn_spatial_token_t *token)
{
	qnn_token_write_u16_le(out, (uint16_t)token->sector_id);
	qnn_token_write_u16_le(out, 0);
	qnn_token_write_f32_le(out, qnn_normalize(token->nearest_dist, QNN_SPATIAL_DIST_SCALE));
	qnn_token_write_f32_le(out, qnn_normalize(token->mean_dist, QNN_SPATIAL_DIST_SCALE));
	qnn_token_write_f32_le(out, token->openness);
	qnn_token_write_f32_le(out, token->clearance);
	qnn_token_write_f32_le(out, token->traversable);
	qnn_token_write_f32_le(out, token->dropoff);
	qnn_token_write_f32_le(out, token->solid_frac);
	qnn_token_write_f32_le(out, token->water_frac);
	qnn_token_write_f32_le(out, token->slime_frac);
	qnn_token_write_f32_le(out, token->lava_frac);
}

static float qnn_dist_sq(const vec3_t a, const vec3_t b)
{
	float dx;
	float dy;
	float dz;

	dx = a[0] - b[0];
	dy = a[1] - b[1];
	dz = a[2] - b[2];
	return dx * dx + dy * dy + dz * dz;
}

static float qnn_vec_length(const vec3_t v)
{
	return (float)sqrt((double)(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
}

static const char *qnn_static_property(const qnn_worker_static_object_t *obj, const char *key)
{
	int i;

	for (i = 0; i < obj->property_count; ++i)
	{
		if (!strcmp(obj->properties[i].key, key))
			return obj->properties[i].value;
	}
	return NULL;
}

static int qnn_static_property_int(const qnn_worker_static_object_t *obj, const char *key, int fallback)
{
	const char *value;

	value = qnn_static_property(obj, key);
	if (value == NULL || value[0] == 0)
		return fallback;
	return atoi(value);
}

static qboolean qnn_subject_is_item(int subject_id)
{
	return subject_id >= QNN_SUBJECT_SHELLS && subject_id <= QNN_SUBJECT_SUIT;
}

static float qnn_item_respawn_s(const qnn_worker_static_object_t *obj, int subject_id)
{
	const char *classname;

	classname = obj->classname;
	if (subject_id == QNN_SUBJECT_MEGAHEALTH)
		return 120.0f;
	if (subject_id == QNN_SUBJECT_QUAD || subject_id == QNN_SUBJECT_SUIT)
		return 60.0f;
	if (!strcasecmp(classname, "item_artifact_invulnerability")
		|| !strcasecmp(classname, "item_artifact_invisibility"))
		return 300.0f;
	if (subject_id == QNN_SUBJECT_ARMOR_GREEN
		|| subject_id == QNN_SUBJECT_ARMOR_YELLOW
		|| subject_id == QNN_SUBJECT_ARMOR_RED
		|| subject_id == QNN_SUBJECT_HEALTH)
		return 20.0f;
	return 30.0f;
}

static qboolean qnn_classify_item_subject(const char *classname, int spawnflags, int *subject_id, float *magnitude)
{
	if (!strcasecmp(classname, "item_health"))
	{
		if (spawnflags & 2)
		{
			*subject_id = QNN_SUBJECT_MEGAHEALTH;
			*magnitude = qnn_health_magnitude(100.0f);
			return true;
		}
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = (spawnflags & 1) ? qnn_health_magnitude(15.0f) : qnn_health_magnitude(25.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_health_rotten"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = qnn_health_magnitude(15.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_health_mega"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = qnn_health_magnitude(100.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_armor1"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_GREEN;
		*magnitude = qnn_armor_magnitude(100.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_armor2"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		*magnitude = qnn_armor_magnitude(150.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_armorInv"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_RED;
		*magnitude = qnn_armor_magnitude(200.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_shells"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = (spawnflags & 1) ? qnn_shells_magnitude(40.0f) : qnn_shells_magnitude(20.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_spikes"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = (spawnflags & 1) ? qnn_nails_magnitude(50.0f) : qnn_nails_magnitude(25.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_rockets"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = (spawnflags & 1) ? qnn_rockets_magnitude(10.0f) : qnn_rockets_magnitude(5.0f);
		return true;
	}
	if (!strcasecmp(classname, "item_cells"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = (spawnflags & 1) ? qnn_cells_magnitude(12.0f) : qnn_cells_magnitude(6.0f);
		return true;
	}
	if (!strcasecmp(classname, "weapon_supershotgun"))
	{
		*subject_id = QNN_SUBJECT_SHOTGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_nailgun"))
	{
		*subject_id = QNN_SUBJECT_NAILGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_supernailgun"))
	{
		*subject_id = QNN_SUBJECT_NAILGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_grenadelauncher"))
	{
		*subject_id = QNN_SUBJECT_GRENADE_LAUNCHER;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_rocketlauncher"))
	{
		*subject_id = QNN_SUBJECT_ROCKET_LAUNCHER;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_lightning"))
	{
		*subject_id = QNN_SUBJECT_THUNDERBOLT;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_super_damage"))
	{
		*subject_id = QNN_SUBJECT_QUAD;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_invulnerability"))
	{
		*subject_id = QNN_SUBJECT_PENT;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_invisibility"))
	{
		*subject_id = QNN_SUBJECT_RING;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_envirosuit"))
	{
		*subject_id = QNN_SUBJECT_SUIT;
		*magnitude = 1.0f;
		return true;
	}
	return false;
}

static qboolean qnn_classify_static_subject(const qnn_worker_static_object_t *obj, int *subject_id, int *qualifier_id, float *magnitude, qboolean *is_item, float *respawn_s)
{
	int spawnflags;

	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;
	*is_item = false;
	*respawn_s = 0.0f;

	spawnflags = qnn_static_property_int(obj, "spawnflags", 0);
	if (qnn_classify_item_subject(obj->classname, spawnflags, subject_id, magnitude))
	{
		*is_item = true;
		*respawn_s = qnn_item_respawn_s(obj, *subject_id);
		return true;
	}
	if (!strncasecmp(obj->classname, "func_door", 9))
	{
		*subject_id = QNN_SUBJECT_DOOR;
		return true;
	}
	if (!strncasecmp(obj->classname, "func_plat", 9))
	{
		*subject_id = QNN_SUBJECT_PLATFORM;
		return true;
	}
	if (!strncasecmp(obj->classname, "func_train", 10))
	{
		*subject_id = QNN_SUBJECT_TRAIN;
		return true;
	}
	if (!strncasecmp(obj->classname, "func_button", 11))
	{
		*subject_id = QNN_SUBJECT_BUTTON;
		return true;
	}
	if (!strcasecmp(obj->classname, "trigger_teleport")
		|| !strcasecmp(obj->classname, "info_teleport_destination")
		|| !strcasecmp(obj->classname, "misc_teleporttrain"))
	{
		*subject_id = QNN_SUBJECT_TELEPORTER;
		return true;
	}
	return false;
}

static qboolean qnn_classify_visible_subject(const qnn_worker_visible_entity_t *ent, int *subject_id, int *qualifier_id, float *magnitude)
{
	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;

	if (!strcmp(ent->classname, "player") || !strcmp(ent->model_name, "progs/player.mdl"))
	{
		*subject_id = QNN_SUBJECT_PLAYER;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/eyes.mdl"))
	{
		*subject_id = QNN_SUBJECT_PLAYER;
		*qualifier_id = QNN_QUAL_INVISIBLE;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/backpack.mdl"))
	{
		*subject_id = QNN_SUBJECT_BACKPACK;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/spike.mdl"))
	{
		*subject_id = QNN_SUBJECT_PROJECTILE_NAIL;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/s_spike.mdl"))
	{
		*subject_id = QNN_SUBJECT_PROJECTILE_NAIL;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/grenade.mdl"))
	{
		*subject_id = QNN_SUBJECT_PROJECTILE_GRENADE;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/missile.mdl"))
	{
		*subject_id = QNN_SUBJECT_PROJECTILE_ROCKET;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/bolt2.mdl") || !strcmp(ent->model_name, "progs/bolt3.mdl"))
	{
		*subject_id = QNN_SUBJECT_LIGHTNING_BEAM;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/teleport.mdl"))
	{
		*subject_id = QNN_SUBJECT_TELEPORTER;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_bh10.bsp"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = qnn_health_magnitude(15.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_bh25.bsp"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = qnn_health_magnitude(25.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_bh100.bsp"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = qnn_health_magnitude(100.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "progs/armor.mdl"))
	{
		if (ent->skin <= 0)
		{
			*subject_id = QNN_SUBJECT_ARMOR_GREEN;
			*magnitude = qnn_armor_magnitude(100.0f);
		}
		else if (ent->skin == 1)
		{
			*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
			*magnitude = qnn_armor_magnitude(150.0f);
		}
		else
		{
			*subject_id = QNN_SUBJECT_ARMOR_RED;
			*magnitude = qnn_armor_magnitude(200.0f);
		}
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_shell0.bsp"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = qnn_shells_magnitude(20.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_shell1.bsp"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = qnn_shells_magnitude(40.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_nail0.bsp"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = qnn_nails_magnitude(25.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_nail1.bsp"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = qnn_nails_magnitude(50.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_rock0.bsp"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = qnn_rockets_magnitude(5.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_rock1.bsp"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = qnn_rockets_magnitude(10.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_batt0.bsp"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = qnn_cells_magnitude(6.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_batt1.bsp"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = qnn_cells_magnitude(12.0f);
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_shot.mdl"))
	{
		*subject_id = QNN_SUBJECT_SHOTGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_nail.mdl"))
	{
		*subject_id = QNN_SUBJECT_NAILGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_nail2.mdl"))
	{
		*subject_id = QNN_SUBJECT_NAILGUN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_rock.mdl"))
	{
		*subject_id = QNN_SUBJECT_GRENADE_LAUNCHER;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_rock2.mdl"))
	{
		*subject_id = QNN_SUBJECT_ROCKET_LAUNCHER;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_light.mdl"))
	{
		*subject_id = QNN_SUBJECT_THUNDERBOLT;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/quaddama.mdl"))
	{
		*subject_id = QNN_SUBJECT_QUAD;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/invulner.mdl"))
	{
		*subject_id = QNN_SUBJECT_PENT;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/invisibl.mdl"))
	{
		*subject_id = QNN_SUBJECT_RING;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/suit.mdl"))
	{
		*subject_id = QNN_SUBJECT_SUIT;
		*magnitude = 1.0f;
		return true;
	}
	if (qnn_classify_item_subject(ent->classname, 0, subject_id, magnitude))
		return true;
	return false;
}

static int qnn_sound_pickup_category(const char *name)
{
	if (!strcmp(name, "items/r_item1.wav") || !strcmp(name, "items/health1.wav"))
		return 1;
	if (!strcmp(name, "items/r_item2.wav"))
		return 2;
	if (!strcmp(name, "items/armor1.wav"))
		return 3;
	if (!strcmp(name, "weapons/pkup.wav"))
		return 4;
	if (!strcmp(name, "weapons/lock4.wav"))
		return 5;
	if (!strcmp(name, "items/damage.wav")
		|| !strcmp(name, "items/protect.wav")
		|| !strcmp(name, "items/inv1.wav")
		|| !strcmp(name, "items/suit.wav"))
		return 6;
	return 0;
}

static int qnn_subject_pickup_category(int subject_id)
{
	if (subject_id == QNN_SUBJECT_HEALTH)
		return 1;
	if (subject_id == QNN_SUBJECT_MEGAHEALTH)
		return 2;
	if (subject_id == QNN_SUBJECT_ARMOR_GREEN || subject_id == QNN_SUBJECT_ARMOR_YELLOW || subject_id == QNN_SUBJECT_ARMOR_RED)
		return 3;
	if (subject_id == QNN_SUBJECT_SHOTGUN || subject_id == QNN_SUBJECT_NAILGUN || subject_id == QNN_SUBJECT_GRENADE_LAUNCHER || subject_id == QNN_SUBJECT_ROCKET_LAUNCHER || subject_id == QNN_SUBJECT_THUNDERBOLT)
		return 4;
	if (subject_id == QNN_SUBJECT_SHELLS || subject_id == QNN_SUBJECT_NAILS || subject_id == QNN_SUBJECT_ROCKETS || subject_id == QNN_SUBJECT_CELLS)
		return 5;
	if (subject_id == QNN_SUBJECT_QUAD || subject_id == QNN_SUBJECT_PENT || subject_id == QNN_SUBJECT_RING || subject_id == QNN_SUBJECT_SUIT)
		return 6;
	return 0;
}

static qboolean qnn_in_fov(const vec3_t player_origin, const vec3_t view_angles, const vec3_t target)
{
	vec3_t forward;
	vec3_t right;
	vec3_t up;
	vec3_t delta;
	float dist;
	float dot;
	float cos_half;

	AngleVectors(view_angles, forward, right, up);
	VectorSubtract(target, player_origin, delta);
	dist = qnn_vec_length(delta);
	if (dist < 1.0f)
		return true;
	dot = DotProduct(forward, delta) / dist;
	cos_half = (float)cos((double)(QNN_FOV_HALF_DEG * M_PI / 180.0f));
	return dot >= cos_half;
}

static void qnn_relative_frame(const vec3_t view_angles, const vec3_t world_delta, vec3_t out)
{
	vec3_t forward;
	vec3_t right;
	vec3_t up;

	AngleVectors(view_angles, forward, right, up);
	out[0] = DotProduct(world_delta, forward);
	out[1] = DotProduct(world_delta, right);
	out[2] = DotProduct(world_delta, up);
}

/* Dynamic objects are stored at qnn_semantic_objects[qnn_semantic_static_count + entity_num].
 * This gives O(1) lookup by entity_num with no handle indirection. */
static int qnn_dynamic_index(int entity_num)
{
	if (entity_num <= 0 || entity_num >= MAX_EDICTS)
		return -1;
	return qnn_semantic_static_count + entity_num;
}

static qnn_semantic_object_t *qnn_ensure_dynamic_object(int entity_num, int subject_id, int qualifier_id, const vec3_t origin, const vec3_t angles, int region_id, int modality_id, float confidence, float magnitude)
{
	int idx;
	qnn_semantic_object_t *obj;

	idx = qnn_dynamic_index(entity_num);
	if (idx < 0 || idx >= qnn_semantic_object_capacity)
		return NULL;

	obj = &qnn_semantic_objects[idx];
	if (!obj->active)
	{
		memset(obj, 0, sizeof(*obj));
		obj->active = true;
		obj->entity_num = entity_num;
		obj->is_static = false;
	}

	obj->subject_id = subject_id;
	obj->qualifier_id = qualifier_id;
	/* Priority overwrite: only upgrade modality (lower number = higher priority).
	   Recency always refreshes regardless of modality change. */
	if (modality_id < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
		obj->modality_id = modality_id;
	obj->player_id = (subject_id == QNN_SUBJECT_PLAYER && entity_num > 0) ? entity_num : 0;
	obj->region_id = region_id;
	VectorCopy(origin, obj->origin);
	VectorCopy(angles, obj->angles);
	obj->recency = 1.0f;
	obj->confidence = confidence;
	obj->magnitude = magnitude;
	obj->state = qnn_subject_is_item(subject_id) ? 1.0f : 0.0f;
	obj->surfaced_this_tick = true;
	return obj;
}

static qnn_semantic_object_t *qnn_nearest_static_subject(int subject_id, const vec3_t origin, float max_dist_sq, qboolean require_unavailable, int pickup_category)
{
	qnn_semantic_object_t *best;
	float best_dsq;
	int i;

	best = NULL;
	best_dsq = max_dist_sq;
	for (i = 0; i < qnn_semantic_static_count; ++i)
	{
		qnn_semantic_object_t *obj;
		float dsq;

		obj = &qnn_semantic_objects[i];
		if (!obj->active || !obj->is_static)
			continue;
		if (subject_id > 0)
		{
			if (obj->subject_id != subject_id)
				continue;
		}
		else if (pickup_category > 0)
		{
			if (qnn_subject_pickup_category(obj->subject_id) != pickup_category)
				continue;
		}
		if (require_unavailable && obj->state >= 1.0f)
			continue;
		dsq = qnn_dist_sq(obj->origin, origin);
		if (dsq < best_dsq)
		{
			best = obj;
			best_dsq = dsq;
		}
	}
	return best;
}

static qnn_semantic_object_t *qnn_nearest_dynamic_subject(int subject_id, const vec3_t origin, float max_dist_sq)
{
	qnn_semantic_object_t *best;
	float best_dsq;
	int i;

	best = NULL;
	best_dsq = max_dist_sq;
	for (i = qnn_semantic_static_count; i < qnn_semantic_object_capacity; ++i)
	{
		float dsq;

		if (!qnn_semantic_objects[i].active || qnn_semantic_objects[i].subject_id != subject_id)
			continue;
		dsq = qnn_dist_sq(qnn_semantic_objects[i].origin, origin);
		if (dsq < best_dsq)
		{
			best = &qnn_semantic_objects[i];
			best_dsq = dsq;
		}
	}
	return best;
}

static void qnn_append_event(int owner_index, int subject_id, int action_id, int qualifier_id, int modality_id, float confidence, float magnitude)
{
	int i;
	int free_index;

	free_index = -1;
	for (i = 0; i < QNN_WORKER_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
		{
			if (free_index < 0)
				free_index = i;
			continue;
		}
		if (qnn_semantic_events[i].owner_index == owner_index
			&& qnn_semantic_events[i].subject_id == subject_id
			&& qnn_semantic_events[i].action_id == action_id
			&& qnn_semantic_events[i].qualifier_id == qualifier_id
			&& qnn_semantic_events[i].modality_id == modality_id)
		{
			qnn_semantic_events[i].recency = 1.0f;
			if (confidence > qnn_semantic_events[i].confidence)
				qnn_semantic_events[i].confidence = confidence;
			if (magnitude > qnn_semantic_events[i].magnitude)
				qnn_semantic_events[i].magnitude = magnitude;
			return;
		}
	}
	if (free_index < 0)
		return;
	memset(&qnn_semantic_events[free_index], 0, sizeof(qnn_semantic_events[free_index]));
	qnn_semantic_events[free_index].active = true;
	qnn_semantic_events[free_index].owner_index = owner_index;
	qnn_semantic_events[free_index].subject_id = subject_id;
	qnn_semantic_events[free_index].action_id = action_id;
	qnn_semantic_events[free_index].qualifier_id = qualifier_id;
	qnn_semantic_events[free_index].modality_id = modality_id;
	qnn_semantic_events[free_index].recency = 1.0f;
	qnn_semantic_events[free_index].confidence = confidence;
	qnn_semantic_events[free_index].magnitude = magnitude;
}

static void qnn_decay_store(float dt)
{
	int i;

	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		qnn_semantic_object_t *obj;

		obj = &qnn_semantic_objects[i];
		if (!obj->active)
			continue;
		obj->surfaced_this_tick = false;
		if (obj->recency > 0.0f)
			obj->recency = qnn_clampf(obj->recency - (dt / QNN_RECENCY_DECAY_S), 0.0f, 1.0f);
		if (!obj->is_static && obj->recency <= 0.0f)
			obj->active = false;
	}

	for (i = 0; i < QNN_WORKER_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
			continue;
		qnn_semantic_events[i].recency = qnn_clampf(qnn_semantic_events[i].recency - (dt / QNN_RECENCY_DECAY_S), 0.0f, 1.0f);
		if (qnn_semantic_events[i].recency <= 0.0f)
			qnn_semantic_events[i].active = false;
	}
}

static void qnn_update_items_from_visibility(const qnn_worker_snapshot_t *snapshot)
{
	qnn_match_candidate_t candidates[QNN_MAX_MATCH_CANDIDATES];
	int candidate_count;
	int *item_matched;
	int ent_matched[QNN_WORKER_MAX_VISIBLE];
	int i;
	int j;

	candidate_count = 0;
	memset(ent_matched, 0, sizeof(ent_matched));
	item_matched = (int *)calloc((size_t)(qnn_semantic_static_count > 0 ? qnn_semantic_static_count : 1), sizeof(int));
	if (item_matched == NULL)
		return;

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;

		if (!qnn_classify_visible_subject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;
		if (!qnn_subject_is_item(subject_id))
			continue;
		for (j = 0; j < qnn_semantic_static_count; ++j)
		{
			float dsq;
			qnn_semantic_object_t *obj;

			obj = &qnn_semantic_objects[j];
			if (!obj->active || !obj->is_item || obj->subject_id != subject_id)
				continue;
			dsq = qnn_dist_sq(obj->origin, snapshot->visible[i].origin);
			if (dsq >= QNN_ITEM_PVS_MATCH_SQ || candidate_count >= QNN_MAX_MATCH_CANDIDATES)
				continue;
			candidates[candidate_count].dist_sq = dsq;
			candidates[candidate_count].ent_idx = i;
			candidates[candidate_count].obj_idx = j;
			candidate_count += 1;
		}
	}

	if (candidate_count > 1)
		qsort(candidates, (size_t)candidate_count, sizeof(candidates[0]), qnn_match_candidate_compare);

	for (i = 0; i < candidate_count; ++i)
	{
		int ent_idx;
		int obj_idx;

		ent_idx = candidates[i].ent_idx;
		obj_idx = candidates[i].obj_idx;
		if (ent_matched[ent_idx] || item_matched[obj_idx])
			continue;
		ent_matched[ent_idx] = 1;
		item_matched[obj_idx] = 1;
	}

	for (i = 0; i < qnn_semantic_static_count; ++i)
	{
		qnn_semantic_object_t *obj;

		obj = &qnn_semantic_objects[i];
		if (!obj->active || !obj->is_item)
			continue;
		if (item_matched[i])
		{
			obj->state = 1.0f;
			obj->pickup_elapsed = 0.0f;
			obj->pvs_seen = true;
		}
		else if (obj->pvs_seen && obj->state >= 1.0f)
		{
			obj->state = 0.0f;
			obj->pickup_elapsed = 0.0f;
			obj->pvs_seen = false;
		}
		else
		{
			obj->pvs_seen = false;
		}
	}

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;
		qnn_semantic_object_t *obj;

		if (!qnn_classify_visible_subject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;
		if (!qnn_subject_is_item(subject_id))
			continue;
		if (!qnn_in_fov(snapshot->player_origin, snapshot->player_view_angles, snapshot->visible[i].origin))
			continue;
		obj = qnn_nearest_static_subject(subject_id, snapshot->visible[i].origin, QNN_ITEM_PVS_MATCH_SQ, false, 0);
		if (obj == NULL)
			continue;
		if (QNN_MODALITY_VISUAL < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
			obj->modality_id = QNN_MODALITY_VISUAL;
		obj->recency = 1.0f;
		obj->confidence = 1.0f;
		obj->magnitude = magnitude;
		obj->surfaced_this_tick = true;
	}

	free(item_matched);
}

static void qnn_advance_item_timers(float dt)
{
	int i;

	for (i = 0; i < qnn_semantic_static_count; ++i)
	{
		qnn_semantic_object_t *obj;

		obj = &qnn_semantic_objects[i];
		if (!obj->active || !obj->is_item)
			continue;
		if (obj->state >= 1.0f)
			continue;
		obj->pickup_elapsed += dt;
		if (obj->respawn_s > 0.0f)
			obj->state = qnn_clampf(obj->pickup_elapsed / obj->respawn_s, 0.0f, 1.0f);
	}
}

static void qnn_refresh_static_object(qnn_semantic_object_t *obj, int modality_id, float confidence)
{
	if (modality_id < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
		obj->modality_id = modality_id;
	obj->recency = 1.0f;
	obj->confidence = confidence;
	obj->surfaced_this_tick = true;
}

static void qnn_handle_item_sound(const qnn_worker_sound_event_t *snd, const char *name)
{
	int pickup_category;
	qnn_semantic_object_t *obj;

	if (!strcmp(name, "items/itembk2.wav"))
	{
		obj = qnn_nearest_static_subject(0, snd->origin, QNN_ITEM_RESPAWN_MATCH_SQ, true, 0);
		if (obj == NULL)
			return;
		obj->state = 1.0f;
		obj->pickup_elapsed = 0.0f;
		qnn_refresh_static_object(obj, QNN_MODALITY_AUDITORY, 1.0f);
		qnn_append_event(qnn_object_index(obj), obj->subject_id, QNN_ACTION_RESPAWN, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
		return;
	}

	pickup_category = qnn_sound_pickup_category(name);
	if (pickup_category == 0)
		return;

	if (!strcmp(name, "weapons/lock4.wav"))
	{
		obj = qnn_nearest_dynamic_subject(QNN_SUBJECT_BACKPACK, snd->origin, QNN_ITEM_PICKUP_MATCH_SQ);
		if (obj != NULL)
		{
			if (QNN_MODALITY_AUDITORY < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
				obj->modality_id = QNN_MODALITY_AUDITORY;
			obj->recency = 1.0f;
			obj->confidence = 1.0f;
			obj->surfaced_this_tick = true;
			qnn_append_event(qnn_object_index(obj), QNN_SUBJECT_BACKPACK, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
			return;
		}
	}

	obj = qnn_nearest_static_subject(0, snd->origin, QNN_ITEM_PICKUP_MATCH_SQ, false, pickup_category);
	if (obj == NULL || obj->state < 1.0f)
		return;
	obj->state = 0.0f;
	obj->pickup_elapsed = 0.0f;
	qnn_refresh_static_object(obj, QNN_MODALITY_AUDITORY, 1.0f);
	qnn_append_event(qnn_object_index(obj), obj->subject_id, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
}

static const qnn_sound_rule_t *qnn_find_sound_rule(const qnn_sound_rule_t *rules, const char *name)
{
	int i;

	for (i = 0; rules[i].name != NULL; ++i)
	{
		if (!strcmp(rules[i].name, name))
			return &rules[i];
	}
	return NULL;
}

static void qnn_handle_sound_player(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	obj = qnn_ensure_dynamic_object(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
	if (obj == NULL)
		return;
	qnn_append_event(qnn_object_index(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void qnn_handle_sound_weapon(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	obj = qnn_ensure_dynamic_object(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
	if (obj == NULL)
		return;
	qnn_append_event(qnn_object_index(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void qnn_handle_sound_projectile(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	if (snd->entity_num > 0)
		obj = qnn_ensure_dynamic_object(snd->entity_num, rule->subject_id, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
	else
		obj = qnn_nearest_dynamic_subject(rule->subject_id, snd->origin, QNN_PROJECTILE_SOUND_MATCH_SQ);
	if (obj == NULL)
		return;
	if (snd->entity_num <= 0)
	{
		if (QNN_MODALITY_AUDITORY < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
			obj->modality_id = QNN_MODALITY_AUDITORY;
		obj->recency = 1.0f;
		obj->confidence = 1.0f;
		obj->surfaced_this_tick = true;
	}
	qnn_append_event(qnn_object_index(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void qnn_handle_sound_static(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	qnn_semantic_object_t *obj;

	obj = qnn_nearest_static_subject(rule->subject_id, snd->origin, QNN_STATIC_SOUND_MATCH_SQ, false, 0);
	if (obj == NULL)
		return;
	qnn_refresh_static_object(obj, QNN_MODALITY_AUDITORY, 1.0f);
	qnn_append_event(qnn_object_index(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void qnn_update_from_sounds(const qnn_worker_snapshot_t *snapshot)
{
	int i;

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		char name[QNN_WORKER_MAX_SOUND_NAME];
		const qnn_sound_rule_t *rule;
		int j;

		name[0] = 0;
		for (j = 0; snapshot->sounds[i].name[j] && j < QNN_WORKER_MAX_SOUND_NAME - 1; ++j)
			name[j] = (char)tolower((unsigned char)snapshot->sounds[i].name[j]);
		name[j] = 0;
		if (name[0] == 0)
			continue;

		qnn_handle_item_sound(&snapshot->sounds[i], name);

		rule = qnn_find_sound_rule(qnn_player_sound_rules, name);
		if (rule != NULL)
		{
			qnn_handle_sound_player(&snapshot->sounds[i], rule);
			continue;
		}
		if (!strncmp(name, "player/pain", 11))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 0.0f;
			qnn_handle_sound_player(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/drown", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_DROWN;
			temp.magnitude = 0.0f;
			qnn_handle_sound_player(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/lburn", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_LAVA;
			temp.magnitude = 0.0f;
			qnn_handle_sound_player(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/death", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_DEATH;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 0.0f;
			qnn_handle_sound_player(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strcmp(name, "player/gib.wav") || !strcmp(name, "player/udeath.wav") || !strcmp(name, "player/tornoff2.wav"))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_DEATH;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 1.0f;
			qnn_handle_sound_player(&snapshot->sounds[i], &temp);
			continue;
		}

		rule = qnn_find_sound_rule(qnn_weapon_sound_rules, name);
		if (rule != NULL)
		{
			qnn_handle_sound_weapon(&snapshot->sounds[i], rule);
			continue;
		}
		rule = qnn_find_sound_rule(qnn_projectile_sound_rules, name);
		if (rule != NULL)
		{
			qnn_handle_sound_projectile(&snapshot->sounds[i], rule);
			continue;
		}
		rule = qnn_find_sound_rule(qnn_static_sound_rules, name);
		if (rule != NULL)
			qnn_handle_sound_static(&snapshot->sounds[i], rule);
	}
}

static void qnn_maybe_append_visual_fire(const qnn_worker_visible_entity_t *ent, qnn_semantic_object_t *obj)
{
	if (!(ent->effects & EF_MUZZLEFLASH))
		return;
	if (ent->frame >= QNN_VISUAL_FIRE_SHOT_FIRST && ent->frame <= QNN_VISUAL_FIRE_SHOT_LAST)
	{
		qnn_append_event(qnn_object_index(obj), QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_NAIL_FIRST && ent->frame <= QNN_VISUAL_FIRE_NAIL_LAST)
	{
		qnn_append_event(qnn_object_index(obj), QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_LIGHT_FIRST && ent->frame <= QNN_VISUAL_FIRE_LIGHT_LAST)
	{
		qnn_append_event(qnn_object_index(obj), QNN_SUBJECT_THUNDERBOLT, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
}

static float qnn_frag_fraction(int entity_frags)
{
	int i, max_frags;

	max_frags = 1;
	if (cl.scores != NULL)
	{
		for (i = 0; i < cl.maxclients; ++i)
		{
			if (cl.scores[i].frags > max_frags)
				max_frags = cl.scores[i].frags;
		}
	}
	return qnn_clampf((float)entity_frags / (float)max_frags, 0.0f, 1.0f);
}

static void qnn_update_from_visible_entities(const qnn_worker_snapshot_t *snapshot)
{
	int i;

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;

		if (!qnn_classify_visible_subject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;

		if (qnn_subject_is_item(subject_id))
			continue;
		if (!qnn_in_fov(snapshot->player_origin, snapshot->player_view_angles, snapshot->visible[i].origin))
			continue;

		if (snapshot->visible[i].entity_num > 0)
		{
			qnn_semantic_object_t *obj;

			obj = qnn_ensure_dynamic_object(
				snapshot->visible[i].entity_num,
				subject_id,
				qualifier_id,
				snapshot->visible[i].origin,
				snapshot->visible[i].angles,
				snapshot->visible[i].region_id,
				QNN_MODALITY_VISUAL,
				1.0f,
				magnitude
			);
			if (obj != NULL)
				VectorCopy(snapshot->visible[i].half_extents, obj->half_extents);
			if (obj != NULL && subject_id == QNN_SUBJECT_PROJECTILE_NAIL)
				VectorCopy(snapshot->visible[i].velocity, obj->velocity);
			if (obj != NULL && subject_id == QNN_SUBJECT_PLAYER)
			{
				obj->magnitude = qnn_frag_fraction(snapshot->visible[i].frags);
				obj->state = 0.0f; /* enemy=0, ally=1 in team modes */
				qnn_maybe_append_visual_fire(&snapshot->visible[i], obj);
			}
		}
	}
}

static int qnn_trace_contents(const vec3_t point)
{
	mleaf_t *leaf;

	if (cl.worldmodel == NULL)
		return CONTENTS_EMPTY;
	leaf = Mod_PointInLeaf((float *)point, cl.worldmodel);
	if (leaf == NULL)
		return CONTENTS_EMPTY;
	return leaf->contents;
}

static float qnn_trace_line_distance(const vec3_t start, const vec3_t end, vec3_t impact)
{
	trace_t trace;
	vec3_t delta;

	if (!cl.worldmodel)
	{
		VectorCopy(end, impact);
		VectorSubtract(end, start, delta);
		return qnn_vec_length(delta);
	}
	memset(&trace, 0, sizeof(trace));
	SV_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1, (float *)start, (float *)end, &trace);
	VectorCopy(trace.endpos, impact);
	VectorSubtract(trace.endpos, start, delta);
	return qnn_vec_length(delta);
}

static void qnn_spatial_reset(qnn_spatial_token_t *token, int sector_id)
{
	memset(token, 0, sizeof(*token));
	token->sector_id = sector_id;
}

static void qnn_spatial_finalize(qnn_spatial_token_t *token, int samples, float max_dist)
{
	if (samples <= 0)
		return;
	token->mean_dist /= (float)samples;
	token->openness = qnn_clampf(token->mean_dist / max_dist, 0.0f, 1.0f);
	token->solid_frac /= (float)samples;
	token->water_frac /= (float)samples;
	token->slime_frac /= (float)samples;
	token->lava_frac /= (float)samples;
	token->traversable /= (float)samples;
	token->dropoff /= (float)samples;
	token->clearance /= (float)samples;
}

static void qnn_spatial_sample_ray(qnn_spatial_token_t *token, const vec3_t start, const vec3_t dir, float max_dist)
{
	vec3_t end;
	vec3_t impact;
	vec3_t impact_probe;
	float dist;
	int contents;

	VectorMA(start, max_dist, dir, end);
	dist = qnn_trace_line_distance(start, end, impact);
	token->mean_dist += dist;
	if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
		token->nearest_dist = dist;

	VectorCopy(impact, impact_probe);
	VectorMA(impact_probe, -1.0f, dir, impact_probe);
	contents = qnn_trace_contents(impact_probe);
	if (dist < max_dist - 1.0f)
		token->solid_frac += 1.0f;
	if (contents == CONTENTS_WATER)
		token->water_frac += 1.0f;
	else if (contents == CONTENTS_SLIME)
		token->slime_frac += 1.0f;
	else if (contents == CONTENTS_LAVA)
		token->lava_frac += 1.0f;
}

static void qnn_build_horizontal_spatial(const qnn_worker_snapshot_t *snapshot, qnn_spatial_token_t *token, float center_deg, float span_deg)
{
	int i;
	int samples;
	vec3_t dir;
	vec3_t end;
	vec3_t impact;
	vec3_t down_start;
	vec3_t down_end;
	vec3_t down_impact;
	float yaw_deg;
	float yaw_rad;
	float max_dist;
	float clear_dist;
	float ground_dist;

	samples = 5;
	max_dist = 1024.0f;
	for (i = 0; i < samples; ++i)
	{
		yaw_deg = snapshot->player_view_angles[1] + center_deg + ((float)i - 2.0f) * (span_deg / 4.0f);
		yaw_rad = yaw_deg * (float)M_PI / 180.0f;
		dir[0] = cos((double)yaw_rad);
		dir[1] = sin((double)yaw_rad);
		dir[2] = 0.0f;
		qnn_spatial_sample_ray(token, snapshot->player_origin, dir, max_dist);

		if (i == 2)
		{
			VectorMA(snapshot->player_origin, 64.0f, dir, end);
			clear_dist = qnn_trace_line_distance(snapshot->player_origin, end, impact);
			token->clearance += qnn_clampf(clear_dist / 64.0f, 0.0f, 1.0f);

			VectorCopy(impact, down_start);
			down_start[2] += 24.0f;
			VectorCopy(impact, down_end);
			down_end[2] -= 64.0f;
			ground_dist = qnn_trace_line_distance(down_start, down_end, down_impact);
			token->traversable += (clear_dist > 56.0f && ground_dist <= 40.0f) ? 1.0f : 0.0f;
			token->dropoff += qnn_clampf((ground_dist - 18.0f) / 46.0f, 0.0f, 1.0f);
		}
		else
		{
			token->clearance += 0.0f;
			token->traversable += 0.0f;
			token->dropoff += 0.0f;
		}
	}
	qnn_spatial_finalize(token, samples, max_dist);
}

static void qnn_build_ground_spatial(const qnn_worker_snapshot_t *snapshot, qnn_spatial_token_t *token)
{
	int i;
	int samples;
	vec3_t offsets[5];
	vec3_t start;
	vec3_t end;
	vec3_t impact;
	float max_dist;
	float dist;
	int contents;

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f; offsets[0][1] = 0.0f; offsets[0][2] = 0.0f;
	offsets[1][0] = 16.0f; offsets[1][1] = 0.0f; offsets[1][2] = 0.0f;
	offsets[2][0] = -16.0f; offsets[2][1] = 0.0f; offsets[2][2] = 0.0f;
	offsets[3][0] = 0.0f; offsets[3][1] = 16.0f; offsets[3][2] = 0.0f;
	offsets[4][0] = 0.0f; offsets[4][1] = -16.0f; offsets[4][2] = 0.0f;
	for (i = 0; i < samples; ++i)
	{
		VectorAdd(snapshot->player_origin, offsets[i], start);
		VectorCopy(start, end);
		end[2] -= max_dist;
		dist = qnn_trace_line_distance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = qnn_trace_contents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist <= 24.0f ? 1.0f : 0.0f;
		token->dropoff += qnn_clampf((dist - 18.0f) / 48.0f, 0.0f, 1.0f);
		token->clearance += qnn_clampf(1.0f - (dist / max_dist), 0.0f, 1.0f);
	}
	qnn_spatial_finalize(token, samples, max_dist);
}

static void qnn_build_ceiling_spatial(const qnn_worker_snapshot_t *snapshot, qnn_spatial_token_t *token)
{
	int i;
	int samples;
	vec3_t offsets[5];
	vec3_t start;
	vec3_t end;
	vec3_t impact;
	float max_dist;
	float dist;
	int contents;

	samples = 5;
	max_dist = 128.0f;
	offsets[0][0] = 0.0f; offsets[0][1] = 0.0f; offsets[0][2] = 24.0f;
	offsets[1][0] = 16.0f; offsets[1][1] = 0.0f; offsets[1][2] = 24.0f;
	offsets[2][0] = -16.0f; offsets[2][1] = 0.0f; offsets[2][2] = 24.0f;
	offsets[3][0] = 0.0f; offsets[3][1] = 16.0f; offsets[3][2] = 24.0f;
	offsets[4][0] = 0.0f; offsets[4][1] = -16.0f; offsets[4][2] = 24.0f;
	for (i = 0; i < samples; ++i)
	{
		VectorAdd(snapshot->player_origin, offsets[i], start);
		VectorCopy(start, end);
		end[2] += max_dist;
		dist = qnn_trace_line_distance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = qnn_trace_contents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist >= 56.0f ? 1.0f : 0.0f;
		token->clearance += qnn_clampf(dist / max_dist, 0.0f, 1.0f);
	}
	qnn_spatial_finalize(token, samples, max_dist);
}

static void qnn_build_spatial_tokens(const qnn_worker_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_WORKER_SPATIAL_TOKEN_COUNT])
{
	int i;

	for (i = 0; i < QNN_WORKER_SPATIAL_TOKEN_COUNT; ++i)
		qnn_spatial_reset(&tokens[i], i);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_FOV_CENTER], 0.0f, 40.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_FOV_LEFT], 40.0f, 40.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_FOV_RIGHT], -40.0f, 40.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_FLANK_LEFT], 90.0f, 40.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_FLANK_RIGHT], -90.0f, 40.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_REAR_LEFT], 150.0f, 30.0f);
	qnn_build_horizontal_spatial(snapshot, &tokens[QNN_SPATIAL_REAR_RIGHT], -150.0f, 30.0f);
	qnn_build_ground_spatial(snapshot, &tokens[QNN_SPATIAL_GROUND]);
	qnn_build_ceiling_spatial(snapshot, &tokens[QNN_SPATIAL_CEILING]);
}

void qnn_worker_semantic_reset(const qnn_worker_map_state_t *map_state)
{
	int i;

	if (qnn_semantic_objects != NULL)
	{
		free(qnn_semantic_objects);
		qnn_semantic_objects = NULL;
	}
	qnn_semantic_static_count = map_state->static_object_count;
	qnn_semantic_object_capacity = map_state->static_object_count + MAX_EDICTS;
	qnn_semantic_objects = (qnn_semantic_object_t *)calloc((size_t)qnn_semantic_object_capacity, sizeof(*qnn_semantic_objects));
	memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
	qnn_prev_object_count = 0;
	if (qnn_semantic_objects == NULL)
	{
		qnn_semantic_object_capacity = 0;
		qnn_semantic_static_count = 0;
		return;
	}

	for (i = 0; i < map_state->static_object_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;
		qboolean is_item;
		float respawn_s;

		if (!qnn_classify_static_subject(&map_state->static_objects[i], &subject_id, &qualifier_id, &magnitude, &is_item, &respawn_s))
			continue;
		qnn_semantic_objects[i].active = true;
		qnn_semantic_objects[i].is_static = true;
		qnn_semantic_objects[i].is_item = is_item;
		qnn_semantic_objects[i].static_index = i;
		qnn_semantic_objects[i].entity_num = 0;
		qnn_semantic_objects[i].subject_id = subject_id;
		qnn_semantic_objects[i].qualifier_id = qualifier_id;
		qnn_semantic_objects[i].modality_id = QNN_MODALITY_NONE;
		qnn_semantic_objects[i].player_id = 0;
		qnn_semantic_objects[i].region_id = map_state->static_objects[i].region_id;
		VectorCopy(map_state->static_objects[i].origin, qnn_semantic_objects[i].origin);
		VectorCopy(map_state->static_objects[i].angles, qnn_semantic_objects[i].angles);
		qnn_semantic_objects[i].recency = 0.0f;
		qnn_semantic_objects[i].confidence = 1.0f;
		qnn_semantic_objects[i].magnitude = magnitude;
		qnn_semantic_objects[i].state = 1.0f;
		qnn_semantic_objects[i].respawn_s = respawn_s;
		qnn_semantic_objects[i].pickup_elapsed = 0.0f;
		qnn_semantic_objects[i].pvs_seen = false;
	}
}

void qnn_worker_semantic_update(const qnn_worker_map_state_t *map_state, const qnn_worker_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	(void)map_state;
	if (qnn_semantic_objects == NULL || qnn_semantic_object_capacity <= 0)
		return;
	if (reset_flag)
	{
		int i;

		for (i = 0; i < QNN_WORKER_MAX_EVENT_ATOMS; ++i)
			memset(&qnn_semantic_events[i], 0, sizeof(qnn_semantic_events[i]));
		for (i = qnn_semantic_static_count; i < qnn_semantic_object_capacity; ++i)
			memset(&qnn_semantic_objects[i], 0, sizeof(qnn_semantic_objects[i]));
	}

	qnn_decay_store(dt);
	qnn_update_items_from_visibility(snapshot);
	qnn_advance_item_timers(dt);
	qnn_update_from_visible_entities(snapshot);
	qnn_update_from_sounds(snapshot);

	/* MENTAL recall: refresh recalled objects that aren't already surfaced.
	 * This runs after all perception updates so surfaced_this_tick is set
	 * for visual/auditory objects — recall only activates for stale ones. */
	if (!reset_flag)
	{
		int r;
		for (r = 0; r < 4; ++r)
		{
			int target;
			int target_index;
			qnn_semantic_object_t *src;

			target = snapshot->action_label.recall[r];
			if (target <= 0 || target > qnn_prev_object_count)
				continue;
			target_index = qnn_prev_object_indices[target - 1];
			if (target_index < 0 || target_index >= qnn_semantic_object_capacity)
				continue;
			src = &qnn_semantic_objects[target_index];
			if (!src->active || src->surfaced_this_tick)
				continue;
			src->modality_id = QNN_MODALITY_MENTAL;
			src->recency = 1.0f;
			src->surfaced_this_tick = true;
		}
	}
}

/*
 * Aggregate individual nail projectile tokens into stream tokens.
 * Groups nails by velocity direction (dot > threshold), picks the leading
 * nail (closest to player) as representative, and preserves the canonical
 * projectile magnitude semantics where projectile magnitudes stay at 0.0.
 * Events from absorbed nails are reassigned to the stream leader.
 */
static void qnn_aggregate_nail_streams(
	qnn_semantic_object_t **object_rows,
	int *object_count,
	qnn_semantic_object_t *stream_copies,
	int *stream_copy_count,
	const vec3_t player_origin)
{
	int nail_indices[QNN_WORKER_MAX_TOKEN_OBJECTS];
	qboolean absorbed[QNN_WORKER_MAX_TOKEN_OBJECTS];
	int nail_count;
	int streams;
	int i;
	int j;
	int n;

	n = *object_count;
	nail_count = 0;
	streams = 0;
	memset(absorbed, 0, sizeof(absorbed));

	/* collect indices of nail objects */
	for (i = 0; i < n; ++i)
	{
		if (object_rows[i]->subject_id == QNN_SUBJECT_PROJECTILE_NAIL)
			nail_indices[nail_count++] = i;
	}
	if (nail_count == 0)
	{
		*stream_copy_count = 0;
		return;
	}

	/* Even a single nail is rewritten so stream copies preserve the canonical semantics. */
	if (nail_count == 1)
	{
		stream_copies[0] = *object_rows[nail_indices[0]];
		stream_copies[0].magnitude = 0.0f;
		object_rows[nail_indices[0]] = &stream_copies[0];
		*stream_copy_count = 1;
		return;
	}

	/* group nails by velocity direction */
	for (i = 0; i < nail_count && streams < QNN_MAX_NAIL_STREAMS; ++i)
	{
		int leader_idx;
		float leader_dsq;
		float max_recency;
		vec3_t leader_vel;
		float leader_speed;
		int ni;

		ni = nail_indices[i];
		if (absorbed[ni])
			continue;

		leader_speed = qnn_vec_length(object_rows[ni]->velocity);
		if (leader_speed < 1.0f)
		{
			/* stationary or no velocity data — skip grouping */
			continue;
		}
		leader_vel[0] = object_rows[ni]->velocity[0] / leader_speed;
		leader_vel[1] = object_rows[ni]->velocity[1] / leader_speed;
		leader_vel[2] = object_rows[ni]->velocity[2] / leader_speed;

		/* start stream with this nail as tentative leader */
		leader_idx = ni;
		max_recency = object_rows[ni]->recency;
		{
			vec3_t d;
			VectorSubtract(object_rows[ni]->origin, player_origin, d);
			leader_dsq = DotProduct(d, d);
		}
		absorbed[ni] = true;

		/* find other nails with similar velocity */
		for (j = i + 1; j < nail_count; ++j)
		{
			int nj;
			float speed;
			float dot;
			vec3_t d;
			float dsq;

			nj = nail_indices[j];
			if (absorbed[nj])
				continue;
			speed = qnn_vec_length(object_rows[nj]->velocity);
			if (speed < 1.0f)
				continue;
			dot = (object_rows[nj]->velocity[0] * leader_vel[0]
				+ object_rows[nj]->velocity[1] * leader_vel[1]
				+ object_rows[nj]->velocity[2] * leader_vel[2]) / speed;
			if (dot < QNN_NAIL_STREAM_DOT_THRESHOLD)
				continue;

			absorbed[nj] = true;

			/* pick the closest nail to player as leader */
			VectorSubtract(object_rows[nj]->origin, player_origin, d);
			dsq = DotProduct(d, d);
			if (dsq < leader_dsq)
			{
				leader_dsq = dsq;
				leader_idx = nj;
			}

			if (object_rows[nj]->recency > max_recency)
				max_recency = object_rows[nj]->recency;
		}

		/* capture real leader identity before swapping to stack copy */
		{
			int leader_owner = qnn_object_index(object_rows[leader_idx]);

			/* create a copy of the leader with stream magnitude */
			stream_copies[streams] = *object_rows[leader_idx];
			stream_copies[streams].recency = max_recency;
			stream_copies[streams].magnitude = 0.0f;
			/* reassign the slot to point at our copy */
			object_rows[leader_idx] = &stream_copies[streams];
			absorbed[leader_idx] = false; /* keep the leader */
			streams += 1;

			/* reassign events from absorbed nails to leader */
			for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
			{
				int k;

				if (!qnn_semantic_events[j].active)
					continue;
				for (k = i + 1; k < nail_count; ++k)
				{
					int nk = nail_indices[k];
					if (!absorbed[nk])
						continue;
					if (qnn_semantic_events[j].owner_index == qnn_object_index(object_rows[nk]))
					{
						qnn_semantic_events[j].owner_index = leader_owner;
						break;
					}
				}
			}
		}
	}

	/* Rewrite ungrouped nails (no velocity data) to canonical projectile magnitude. */
	for (i = 0; i < nail_count && streams < QNN_MAX_NAIL_STREAMS; ++i)
	{
		int ni = nail_indices[i];
		if (absorbed[ni] || qnn_vec_length(object_rows[ni]->velocity) >= 1.0f)
			continue;
		stream_copies[streams] = *object_rows[ni];
		stream_copies[streams].magnitude = 0.0f;
		object_rows[ni] = &stream_copies[streams];
		streams += 1;
	}

	/* compact: remove absorbed nails from object_rows */
	j = 0;
	for (i = 0; i < n; ++i)
	{
		if (!absorbed[i])
			object_rows[j++] = object_rows[i];
	}
	*object_count = j;
	*stream_copy_count = streams;
}

void qnn_worker_write_token_step_binary(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag)
{
	/* Collection buffer large enough for all possible eligible objects.
	 * After priority sort, only the top QNN_WORKER_MAX_TOKEN_OBJECTS survive. */
	qnn_semantic_object_t *candidate_rows[MAX_EDICTS + 512];
	qnn_spatial_token_t spatial_tokens[QNN_WORKER_SPATIAL_TOKEN_COUNT];
	qnn_semantic_object_t *object_rows[QNN_WORKER_MAX_TOKEN_OBJECTS];
	qnn_semantic_event_atom_t *event_rows[QNN_WORKER_MAX_EVENT_ATOMS];
	qnn_semantic_object_t stream_copies[QNN_MAX_NAIL_STREAMS];
	uint16_t event_base[QNN_WORKER_MAX_TOKEN_OBJECTS];
	uint16_t event_count[QNN_WORKER_MAX_TOKEN_OBJECTS];
	uint16_t flags;
	int stream_copy_count;
	int candidate_count;
	int object_count;
	int event_total;
	int i;
	int j;
	const qnn_nav_oracle_runtime_t *oracle;
	int player_area_id;
	int player_cluster_id;

	candidate_count = 0;
	object_count = 0;
	event_total = 0;
	flags = 0;
	if (reset_flag)
		flags |= QNN_TOKEN_FLAG_RESET;
	if (snapshot->done)
		flags |= QNN_TOKEN_FLAG_DONE;
	if (qnn_action_has_signal(&snapshot->action_label))
		flags |= QNN_TOKEN_FLAG_HAS_ACTION;

	/* first pass: collect all eligible objects */
	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		int has_event;

		if (!qnn_semantic_objects[i].active)
			continue;
		has_event = 0;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (qnn_semantic_events[j].active && qnn_semantic_events[j].owner_index == i)
			{
				has_event = 1;
				break;
			}
		}
		if (qnn_semantic_objects[i].recency <= 0.0f && !has_event)
			continue;
		if (candidate_count < (int)(sizeof(candidate_rows) / sizeof(candidate_rows[0])))
			candidate_rows[candidate_count++] = &qnn_semantic_objects[i];
	}

	/* priority sort: players first, then projectiles, then everything else by recency */
	if (candidate_count > 1)
		qsort(candidate_rows, (size_t)candidate_count, sizeof(candidate_rows[0]), qnn_object_row_compare);

	/* truncate to emission budget */
	object_count = candidate_count < QNN_WORKER_MAX_TOKEN_OBJECTS ? candidate_count : QNN_WORKER_MAX_TOKEN_OBJECTS;
	for (i = 0; i < object_count; ++i)
		object_rows[i] = candidate_rows[i];

	/* aggregate nail projectiles into stream tokens */
	qnn_aggregate_nail_streams(object_rows, &object_count, stream_copies,
		&stream_copy_count, snapshot->player_origin);

	/* second pass: collect events for surviving objects */
	for (i = 0; i < object_count; ++i)
	{
		int oi = qnn_object_index(object_rows[i]);
		event_base[i] = (uint16_t)event_total;
		event_count[i] = 0;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS && event_total < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (!qnn_semantic_events[j].active || qnn_semantic_events[j].owner_index != oi)
				continue;
			event_rows[event_total] = &qnn_semantic_events[j];
			event_total += 1;
			event_count[i] += 1;
		}
	}

	qnn_build_spatial_tokens(snapshot, spatial_tokens);

	/* Resolve the player's nav area/cluster once for self token and route computation. */
	oracle = qnn_worker_map_state.nav_oracle;
	player_area_id = -1;
	player_cluster_id = 0;

	if (oracle)
	{
		qnn_nav_area_result_t player_area;
		char area_err[128];

		if (qnn_nav_oracle_find_area(oracle, snapshot->player_origin, &player_area, area_err, sizeof(area_err))
			&& player_area.found)
		{
			player_area_id = player_area.area_id;
			player_cluster_id = player_area.cluster_id;
		}
	}

	/* Write header */
	fwrite(QNN_TOKEN_MAGIC, 1, 4, out);
	qnn_token_write_u16_le(out, (uint16_t)QNN_TOKEN_VERSION);
	qnn_token_write_u16_le(out, flags);
	qnn_token_write_u32_le(out, (uint32_t)tick);
	qnn_token_write_u32_le(out, (uint32_t)steps);
	qnn_token_write_i32_le(out, snapshot->current_region_id);
	qnn_token_write_u16_le(out, (uint16_t)object_count);
	qnn_token_write_u16_le(out, (uint16_t)event_total);
	qnn_token_write_u16_le(out, (uint16_t)QNN_WORKER_SPATIAL_TOKEN_COUNT);
	qnn_token_write_u16_le(out, (uint16_t)tick_hz);

	qnn_write_self_token(out, snapshot, tick_hz, player_cluster_id);

	/* Write object tokens — route computed uniformly for all modalities */
	for (i = 0; i < object_count; ++i)
	{
		vec3_t delta;
		vec3_t rel;
		int used_path = 0;

		object_rows[i]->cluster_id = 0;
		object_rows[i]->route_cost = 0.0f;
		object_rows[i]->route_cluster_count = 0;

		if (oracle && player_area_id >= 0)
		{
			qnn_nav_area_result_t obj_area;
			char obj_err[128];

			if (qnn_nav_oracle_find_area(oracle, object_rows[i]->origin, &obj_area, obj_err, sizeof(obj_err))
				&& obj_area.found)
			{
				float path_rel[3];
				float route_cost;
				char path_err[128];

				object_rows[i]->cluster_id = obj_area.cluster_id;

				if (qnn_nav_oracle_path_position(oracle, player_area_id, obj_area.area_id,
					snapshot->player_origin, object_rows[i]->origin,
					path_rel, &route_cost, path_err, sizeof(path_err)))
				{
					qnn_relative_frame(snapshot->player_view_angles, path_rel, rel);
					object_rows[i]->route_cost = route_cost;
					used_path = 1;
				}

				qnn_nav_oracle_route_clusters(oracle, player_area_id, obj_area.area_id,
					object_rows[i]->route_cluster_ids, QNN_MAX_ROUTE_CLUSTERS,
					&object_rows[i]->route_cluster_count);
			}
		}

		if (!used_path)
		{
			VectorSubtract(object_rows[i]->origin, snapshot->player_origin, delta);
			qnn_relative_frame(snapshot->player_view_angles, delta, rel);
		}

		{
			uint32_t wire_handle;
			if (object_rows[i] >= qnn_semantic_objects
				&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
				wire_handle = (uint32_t)qnn_object_index(object_rows[i]);
			else
				wire_handle = (uint32_t)object_rows[i]->entity_num;
			qnn_write_object_token(out, object_rows[i], rel, event_count[i], event_base[i], wire_handle);
		}
	}

	/* Save index mapping for next tick's recall targets.
	 * Stream copies (nail aggregation) are stack-local and not valid recall
	 * targets, so store -1 for those — recall will skip them. */
	for (i = 0; i < object_count && i < QNN_WORKER_MAX_TOKEN_OBJECTS; ++i)
	{
		if (object_rows[i] >= qnn_semantic_objects
			&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
			qnn_prev_object_indices[i] = qnn_object_index(object_rows[i]);
		else
			qnn_prev_object_indices[i] = -1;
	}
	qnn_prev_object_count = object_count;

	for (i = 0; i < event_total; ++i)
	{
		qnn_token_write_u16_le(out, (uint16_t)event_rows[i]->subject_id);
		qnn_token_write_u16_le(out, (uint16_t)event_rows[i]->action_id);
		qnn_token_write_u16_le(out, (uint16_t)event_rows[i]->qualifier_id);
		qnn_token_write_u16_le(out, (uint16_t)event_rows[i]->modality_id);
		qnn_token_write_f32_le(out, event_rows[i]->recency);
		qnn_token_write_f32_le(out, event_rows[i]->confidence);
		qnn_token_write_f32_le(out, event_rows[i]->magnitude);
	}

	for (i = 0; i < QNN_WORKER_SPATIAL_TOKEN_COUNT; ++i)
		qnn_write_spatial_token(out, &spatial_tokens[i]);

	if (qnn_action_has_signal(&snapshot->action_label))
	{
		qnn_token_write_f32_le(out, snapshot->action_label.move[0]);
		qnn_token_write_f32_le(out, snapshot->action_label.move[1]);
		qnn_token_write_f32_le(out, snapshot->action_label.look[0]);
		qnn_token_write_f32_le(out, snapshot->action_label.look[1]);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.fire);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.jump);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.switch_slot);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.recall[0]);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.recall[1]);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.recall[2]);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.recall[3]);
	}
	fflush(out);
}

/* ====================================================================
 * Direct-pack observation buffer writer.
 *
 * Produces the fixed-size obs buffer (15380 bytes).  Layout defined in
 * qnn_obs_buffer.h.  Training extras continue to use the existing QTRN
 * binary writer (not in the hot path).
 *
 * This replaces the QTOK token writer for the obs_buffer_v1 step format.
 * ==================================================================== */

#include "qnn_obs_buffer.h"

/* Action history ring buffer — maintained across steps, reset on episode reset. */
static float qnn_action_history[QNN_OBS_ACTION_HISTORY_LEN][QNN_OBS_ACTION_HISTORY_DIM];
static int qnn_action_history_count = 0;

static void qnn_obs_reset_action_history(void)
{
	memset(qnn_action_history, 0, sizeof(qnn_action_history));
	qnn_action_history_count = 0;
}

static void qnn_obs_push_action(const qnn_worker_action_t *action)
{
	float features[QNN_OBS_ACTION_HISTORY_DIM];
	int i;

	features[0] = action->move[0];
	features[1] = action->move[1];
	features[2] = action->look[0];
	features[3] = action->look[1];
	features[4] = (float)action->fire;
	features[5] = (float)action->jump;
	features[6] = (float)(action->switch_slot < 0 ? 0 : action->switch_slot > QNN_ACTION_SWITCH_SLOTS ? QNN_ACTION_SWITCH_SLOTS : action->switch_slot)
		/ (float)QNN_ACTION_SWITCH_SLOTS;

	/* Shift window if full */
	if (qnn_action_history_count >= QNN_OBS_ACTION_HISTORY_LEN)
	{
		memmove(qnn_action_history[0], qnn_action_history[1],
			(QNN_OBS_ACTION_HISTORY_LEN - 1) * sizeof(qnn_action_history[0]));
		qnn_action_history_count = QNN_OBS_ACTION_HISTORY_LEN - 1;
	}
	for (i = 0; i < QNN_OBS_ACTION_HISTORY_DIM; ++i)
		qnn_action_history[qnn_action_history_count][i] = features[i];
	qnn_action_history_count++;
}

/* Helper: clamp float to [0,1] */
static float qnn_clamp01(float v)
{
	if (v < 0.0f) return 0.0f;
	if (v > 1.0f) return 1.0f;
	return v;
}

/* Helper: write a float32 into a byte buffer at an offset (little-endian) */
static void qnn_buf_write_f32(uint8_t *buf, int offset, float value)
{
	union { float f; uint32_t u; } bits;
	bits.f = value;
	buf[offset + 0] = (uint8_t)(bits.u & 0xffu);
	buf[offset + 1] = (uint8_t)((bits.u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((bits.u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((bits.u >> 24) & 0xffu);
}

static void qnn_buf_write_i32(uint8_t *buf, int offset, int32_t value)
{
	uint32_t u = (uint32_t)value;
	buf[offset + 0] = (uint8_t)(u & 0xffu);
	buf[offset + 1] = (uint8_t)((u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((u >> 24) & 0xffu);
}

static void qnn_buf_write_u32(uint8_t *buf, int offset, uint32_t value)
{
	buf[offset + 0] = (uint8_t)(value & 0xffu);
	buf[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((value >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((value >> 24) & 0xffu);
}

void qnn_worker_write_obs_buffer(FILE *out, const qnn_worker_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag)
{
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];

	qnn_semantic_object_t *candidate_rows[MAX_EDICTS + 512];
	qnn_spatial_token_t spatial_tokens[QNN_WORKER_SPATIAL_TOKEN_COUNT];
	qnn_semantic_object_t *object_rows[QNN_WORKER_MAX_TOKEN_OBJECTS];
	qnn_semantic_event_atom_t *event_rows[QNN_WORKER_MAX_EVENT_ATOMS];
	qnn_semantic_object_t stream_copies[QNN_MAX_NAIL_STREAMS];
	uint16_t ev_base[QNN_WORKER_MAX_TOKEN_OBJECTS];
	uint16_t ev_count[QNN_WORKER_MAX_TOKEN_OBJECTS];
	int stream_copy_count;
	int candidate_count;
	int object_count;
	int event_total;
	int i, j;
	const qnn_nav_oracle_runtime_t *oracle;
	int player_area_id;
	int player_cluster_id;
	qboolean has_action;
	vec3_t object_rel[QNN_WORKER_MAX_TOKEN_OBJECTS];

	/* Zero both buffers */
	memset(obs, 0, sizeof(obs));
	(void)tick; (void)steps; /* used by QTRN writer, not obs buffer */

	if (reset_flag)
		qnn_obs_reset_action_history();

	/* ---- Build phase (same as qnn_worker_write_token_step_binary) ---- */

	candidate_count = 0;
	object_count = 0;
	event_total = 0;

	has_action = qnn_action_has_signal(&snapshot->action_label);

	/* Collect eligible objects */
	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		int has_event = 0;
		if (!qnn_semantic_objects[i].active)
			continue;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (qnn_semantic_events[j].active && qnn_semantic_events[j].owner_index == i)
			{
				has_event = 1;
				break;
			}
		}
		if (qnn_semantic_objects[i].recency <= 0.0f && !has_event)
			continue;
		if (candidate_count < (int)(sizeof(candidate_rows) / sizeof(candidate_rows[0])))
			candidate_rows[candidate_count++] = &qnn_semantic_objects[i];
	}

	if (candidate_count > 1)
		qsort(candidate_rows, (size_t)candidate_count, sizeof(candidate_rows[0]), qnn_object_row_compare);

	object_count = candidate_count < QNN_WORKER_MAX_TOKEN_OBJECTS ? candidate_count : QNN_WORKER_MAX_TOKEN_OBJECTS;
	for (i = 0; i < object_count; ++i)
		object_rows[i] = candidate_rows[i];

	qnn_aggregate_nail_streams(object_rows, &object_count, stream_copies,
		&stream_copy_count, snapshot->player_origin);

	/* Collect events for surviving objects */
	for (i = 0; i < object_count; ++i)
	{
		int oi = qnn_object_index(object_rows[i]);
		ev_base[i] = (uint16_t)event_total;
		ev_count[i] = 0;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS && event_total < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (!qnn_semantic_events[j].active || qnn_semantic_events[j].owner_index != oi)
				continue;
			event_rows[event_total] = &qnn_semantic_events[j];
			event_total++;
			ev_count[i]++;
		}
	}

	qnn_build_spatial_tokens(snapshot, spatial_tokens);

	/* Nav oracle */
	oracle = qnn_worker_map_state.nav_oracle;
	player_area_id = -1;
	player_cluster_id = 0;
	if (oracle)
	{
		qnn_nav_area_result_t player_area;
		char area_err[128];
		if (qnn_nav_oracle_find_area(oracle, snapshot->player_origin, &player_area, area_err, sizeof(area_err))
			&& player_area.found)
		{
			player_area_id = player_area.area_id;
			player_cluster_id = player_area.cluster_id;
		}
	}

	/* Compute per-object relative positions and nav routes */
	for (i = 0; i < object_count; ++i)
	{
		vec3_t delta;
		int used_path = 0;

		object_rows[i]->cluster_id = 0;
		object_rows[i]->route_cost = 0.0f;
		object_rows[i]->route_cluster_count = 0;

		if (oracle && player_area_id >= 0)
		{
			qnn_nav_area_result_t obj_area;
			char obj_err[128];
			if (qnn_nav_oracle_find_area(oracle, object_rows[i]->origin, &obj_area, obj_err, sizeof(obj_err))
				&& obj_area.found)
			{
				float path_rel[3];
				float route_cost;
				char path_err[128];

				object_rows[i]->cluster_id = obj_area.cluster_id;
				if (qnn_nav_oracle_path_position(oracle, player_area_id, obj_area.area_id,
					snapshot->player_origin, object_rows[i]->origin,
					path_rel, &route_cost, path_err, sizeof(path_err)))
				{
					qnn_relative_frame(snapshot->player_view_angles, path_rel, object_rel[i]);
					object_rows[i]->route_cost = route_cost;
					used_path = 1;
				}
				qnn_nav_oracle_route_clusters(oracle, player_area_id, obj_area.area_id,
					object_rows[i]->route_cluster_ids, QNN_MAX_ROUTE_CLUSTERS,
					&object_rows[i]->route_cluster_count);
			}
		}

		if (!used_path)
		{
			VectorSubtract(object_rows[i]->origin, snapshot->player_origin, delta);
			qnn_relative_frame(snapshot->player_view_angles, delta, object_rel[i]);
		}
	}

	/* Update recall mapping */
	for (i = 0; i < object_count && i < QNN_WORKER_MAX_TOKEN_OBJECTS; ++i)
	{
		if (object_rows[i] >= qnn_semantic_objects
			&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
			qnn_prev_object_indices[i] = qnn_object_index(object_rows[i]);
		else
			qnn_prev_object_indices[i] = -1;
	}
	qnn_prev_object_count = object_count;

	/* ---- Pack observation buffer ---- */

	/* Self scalars [23] */
	{
		float scalars[QNN_OBS_SELF_SCALAR_DIM];
		scalars[0] = qnn_normalize((float)snapshot->health, QNN_SELF_HEALTH_CAP);
		scalars[1] = qnn_normalize((float)snapshot->armor, QNN_SELF_ARMOR_CAP);
		scalars[2] = qnn_normalize(snapshot->armor_type, QNN_SELF_ARMOR_TYPE_CAP);
		scalars[3] = (snapshot->weapons_owned & IT_SHOTGUN) ? 1.0f : 0.0f;
		scalars[4] = (snapshot->weapons_owned & IT_SUPER_SHOTGUN) ? 1.0f : 0.0f;
		scalars[5] = (snapshot->weapons_owned & IT_NAILGUN) ? 1.0f : 0.0f;
		scalars[6] = (snapshot->weapons_owned & IT_SUPER_NAILGUN) ? 1.0f : 0.0f;
		scalars[7] = (snapshot->weapons_owned & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
		scalars[8] = (snapshot->weapons_owned & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
		scalars[9] = (snapshot->weapons_owned & IT_LIGHTNING) ? 1.0f : 0.0f;
		scalars[10] = (float)qnn_self_weapon_super(snapshot->weapon_id);
		scalars[11] = qnn_shells_magnitude((float)snapshot->ammo_shells);
		scalars[12] = qnn_nails_magnitude((float)snapshot->ammo_nails);
		scalars[13] = qnn_rockets_magnitude((float)snapshot->ammo_rockets);
		scalars[14] = qnn_cells_magnitude((float)snapshot->ammo_cells);
		scalars[15] = qnn_normalize(snapshot->player_velocity[0], QNN_SELF_VELOCITY_CAP);
		scalars[16] = qnn_normalize(snapshot->player_velocity[1], QNN_SELF_VELOCITY_CAP);
		scalars[17] = qnn_normalize(snapshot->player_velocity[2], QNN_SELF_VELOCITY_CAP);
		scalars[18] = qnn_angle_sin_deg(snapshot->player_view_angles[1]);
		scalars[19] = qnn_angle_cos_deg(snapshot->player_view_angles[1]);
		scalars[20] = qnn_angle_sin_deg(snapshot->player_view_angles[0]);
		scalars[21] = qnn_angle_cos_deg(snapshot->player_view_angles[0]);
		scalars[22] = tick_hz > 0 ? (1.0f / (float)tick_hz) : 0.0f;
		for (i = 0; i < QNN_OBS_SELF_SCALAR_DIM; ++i)
			qnn_buf_write_f32(obs, QNN_OBS_OFF_SELF_SCALARS + i * 4, scalars[i]);
	}

	/* Self IDs */
	qnn_buf_write_i32(obs, QNN_OBS_OFF_SELF_WEAPON_ID, qnn_self_weapon_embed_id(snapshot->weapon_id));
	qnn_buf_write_i32(obs, QNN_OBS_OFF_SELF_MOVEMENT_ID, qnn_self_movement_id(snapshot->grounded, snapshot->waterlevel));
	qnn_buf_write_i32(obs, QNN_OBS_OFF_SELF_CLUSTER_ID, player_cluster_id);

	/* Objects [64, 5] ids + [64, 8] scalars + [64] mask + [64, 8] route_cluster_ids */
	for (i = 0; i < object_count; ++i)
	{
		int ids_off = QNN_OBS_OFF_OBJECT_IDS + i * QNN_OBS_OBJECT_ID_DIM * 4;
		int sc_off = QNN_OBS_OFF_OBJECT_SCALARS + i * QNN_OBS_OBJECT_SCALAR_DIM * 4;
		int rc_off = QNN_OBS_OFF_OBJECT_ROUTE_IDS + i * QNN_OBS_MAX_ROUTE_CLUSTERS * 4;

		qnn_buf_write_i32(obs, ids_off + 0, object_rows[i]->subject_id);
		qnn_buf_write_i32(obs, ids_off + 4, object_rows[i]->qualifier_id);
		qnn_buf_write_i32(obs, ids_off + 8, object_rows[i]->modality_id);
		qnn_buf_write_i32(obs, ids_off + 12, object_rows[i]->player_id);
		qnn_buf_write_i32(obs, ids_off + 16, object_rows[i]->cluster_id);

		qnn_buf_write_f32(obs, sc_off + 0, qnn_normalize(object_rel[i][0], QNN_OBJECT_REL_SCALE));
		qnn_buf_write_f32(obs, sc_off + 4, qnn_normalize(object_rel[i][1], QNN_OBJECT_REL_SCALE));
		qnn_buf_write_f32(obs, sc_off + 8, qnn_normalize(object_rel[i][2], QNN_OBJECT_REL_SCALE));
		qnn_buf_write_f32(obs, sc_off + 12, qnn_normalize(object_rows[i]->route_cost, QNN_OBJECT_ROUTE_COST_SCALE));
		qnn_buf_write_f32(obs, sc_off + 16, object_rows[i]->recency);
		qnn_buf_write_f32(obs, sc_off + 20, object_rows[i]->confidence);
		qnn_buf_write_f32(obs, sc_off + 24, object_rows[i]->magnitude);
		qnn_buf_write_f32(obs, sc_off + 28, object_rows[i]->state);
		qnn_buf_write_f32(obs, sc_off + 32, object_rows[i]->half_extents[0] / QNN_OBJECT_REL_SCALE);
		qnn_buf_write_f32(obs, sc_off + 36, object_rows[i]->half_extents[1] / QNN_OBJECT_REL_SCALE);
		qnn_buf_write_f32(obs, sc_off + 40, object_rows[i]->half_extents[2] / QNN_OBJECT_REL_SCALE);

		/* Target look axis: the look[0]/look[1] values that would center
		   the crosshair on this object in one tick.  Computed from the
		   raw view-frame relative position (object_rel) so the model
		   can learn tracking as: look[0] ≈ target_yaw_axis. */
		{
			float rx = object_rel[i][0];  /* forward */
			float ry = object_rel[i][1];  /* right */
			float rz = object_rel[i][2];  /* up */
			float horiz_dist = sqrtf(rx * rx + ry * ry);
			float yaw_deg = 0.0f, pitch_deg = 0.0f;
			int yaw_counts, pitch_counts;

			if (horiz_dist > 1.0f)
			{
				yaw_deg = atan2f(ry, rx) * (180.0f / M_PI);
				pitch_deg = atan2f(-rz, horiz_dist) * (180.0f / M_PI);
			}
			/* Negate yaw: positive ry = target to the right, but Quake
			   positive yaw mouse count = turn left. */
			yaw_counts = (int)roundf(-yaw_deg / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
			pitch_counts = (int)roundf(pitch_deg / QNN_DEMO_MOUSE_DEGREES_PER_COUNT);
			qnn_buf_write_f32(obs, sc_off + 44, qnn_look_axis_from_mouse_count(yaw_counts));
			qnn_buf_write_f32(obs, sc_off + 48, qnn_look_axis_from_mouse_count(pitch_counts));
		}

		obs[QNN_OBS_OFF_OBJECT_MASK + i] = 1;

		for (j = 0; j < QNN_OBS_MAX_ROUTE_CLUSTERS; ++j)
		{
			int rc_val = (j < object_rows[i]->route_cluster_count) ? object_rows[i]->route_cluster_ids[j] : 0;
			qnn_buf_write_i32(obs, rc_off + j * 4, rc_val);
		}
	}

	/* Events — flattened with owner tracking */
	for (i = 0; i < object_count; ++i)
	{
		int base = (int)ev_base[i];
		int count = (int)ev_count[i];
		for (j = 0; j < count && (base + j) < QNN_OBS_MAX_EVENTS; ++j)
		{
			int ei = base + j;
			int eid_off = QNN_OBS_OFF_EVENT_IDS + ei * QNN_OBS_EVENT_ID_DIM * 4;
			int esc_off = QNN_OBS_OFF_EVENT_SCALARS + ei * QNN_OBS_EVENT_SCALAR_DIM * 4;

			qnn_buf_write_i32(obs, eid_off + 0, event_rows[ei]->subject_id);
			qnn_buf_write_i32(obs, eid_off + 4, event_rows[ei]->action_id);
			qnn_buf_write_i32(obs, eid_off + 8, event_rows[ei]->qualifier_id);
			qnn_buf_write_i32(obs, eid_off + 12, event_rows[ei]->modality_id);

			qnn_buf_write_f32(obs, esc_off + 0, qnn_clamp01(event_rows[ei]->recency));
			qnn_buf_write_f32(obs, esc_off + 4, qnn_clamp01(event_rows[ei]->confidence));
			qnn_buf_write_f32(obs, esc_off + 8, qnn_clamp01(event_rows[ei]->magnitude));

			qnn_buf_write_i32(obs, QNN_OBS_OFF_EVENT_OWNER + ei * 4, i);  /* owner = object slot index */
			obs[QNN_OBS_OFF_EVENT_MASK + ei] = 1;
		}
	}

	/* Spatial [9] ids + [9, 10] scalars */
	for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
	{
		int sid_off = QNN_OBS_OFF_SPATIAL_IDS + i * 4;
		int ssc_off = QNN_OBS_OFF_SPATIAL_SCALARS + i * QNN_OBS_SPATIAL_SCALAR_DIM * 4;

		qnn_buf_write_i32(obs, sid_off, spatial_tokens[i].sector_id);
		qnn_buf_write_f32(obs, ssc_off + 0, qnn_normalize(spatial_tokens[i].nearest_dist, QNN_SPATIAL_DIST_SCALE));
		qnn_buf_write_f32(obs, ssc_off + 4, qnn_normalize(spatial_tokens[i].mean_dist, QNN_SPATIAL_DIST_SCALE));
		qnn_buf_write_f32(obs, ssc_off + 8, spatial_tokens[i].openness);
		qnn_buf_write_f32(obs, ssc_off + 12, spatial_tokens[i].clearance);
		qnn_buf_write_f32(obs, ssc_off + 16, spatial_tokens[i].traversable);
		qnn_buf_write_f32(obs, ssc_off + 20, spatial_tokens[i].dropoff);
		qnn_buf_write_f32(obs, ssc_off + 24, spatial_tokens[i].solid_frac);
		qnn_buf_write_f32(obs, ssc_off + 28, spatial_tokens[i].water_frac);
		qnn_buf_write_f32(obs, ssc_off + 32, spatial_tokens[i].slime_frac);
		qnn_buf_write_f32(obs, ssc_off + 36, spatial_tokens[i].lava_frac);
	}

	/* Action history [8, 7] — pack current history BEFORE updating with this tick's action */
	{
		int ah_off = QNN_OBS_OFF_ACTION_HISTORY;
		int n = qnn_action_history_count < QNN_OBS_ACTION_HISTORY_LEN
			? qnn_action_history_count : QNN_OBS_ACTION_HISTORY_LEN;
		for (i = 0; i < n; ++i)
			for (j = 0; j < QNN_OBS_ACTION_HISTORY_DIM; ++j)
				qnn_buf_write_f32(obs, ah_off + (i * QNN_OBS_ACTION_HISTORY_DIM + j) * 4,
					qnn_action_history[i][j]);
	}

	/* Now update action history for next tick (matches Python: encode first, then push) */
	if (has_action && !reset_flag)
		qnn_obs_push_action(&snapshot->action_label);

	/* Write obs buffer */
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, out);
	fflush(out);
}
