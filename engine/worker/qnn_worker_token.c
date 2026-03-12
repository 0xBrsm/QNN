#include "qnn_worker.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define QNN_TOKEN_MAGIC "QTOK"
#define QNN_TOKEN_VERSION 3

#define QNN_TOKEN_FLAG_RESET      0x0001
#define QNN_TOKEN_FLAG_DONE       0x0002
#define QNN_TOKEN_FLAG_HAS_ACTION 0x0004

#define QNN_MODALITY_NONE 0
#define QNN_MODALITY_VISUAL 1
#define QNN_MODALITY_AUDITORY 2
#define QNN_MODALITY_SPATIAL 3
#define QNN_MODALITY_MENTAL 4

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
#define QNN_OBJECT_RECENCY_S 4.0f
#define QNN_EVENT_RECENCY_S 0.5f
#define QNN_DYNAMIC_CONFIDENCE_S 6.0f
#define QNN_PLAYER_DECAY_S 4.0f
#define QNN_PROJECTILE_DECAY_S 1.0f
#define QNN_GENERIC_DYNAMIC_DECAY_S 2.0f

#define QNN_ITEM_PVS_MATCH_SQ (16.0f * 16.0f)
#define QNN_ITEM_PICKUP_MATCH_SQ (64.0f * 64.0f)
#define QNN_ITEM_RESPAWN_MATCH_SQ (16.0f * 16.0f)
#define QNN_STATIC_SOUND_MATCH_SQ (128.0f * 128.0f)
#define QNN_PROJECTILE_SOUND_MATCH_SQ (96.0f * 96.0f)
#define QNN_MAX_MATCH_CANDIDATES 4096

#define QNN_NAIL_STREAM_DOT_THRESHOLD 0.97f
#define QNN_NAIL_STREAM_DENSITY_DIVISOR 8.0f
#define QNN_NAIL_STREAM_SUPER_WEIGHT 2.0f
#define QNN_MAX_NAIL_STREAMS 16

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
	uint32_t handle;
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
	float decay_s;
	float respawn_s;
	float pickup_elapsed;
	qboolean pvs_seen;
	qboolean surfaced_this_tick;
} qnn_semantic_object_t;

typedef struct
{
	qboolean active;
	uint32_t object_handle;
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
static uint32_t qnn_semantic_next_handle = 0x90000000u;
static qnn_semantic_event_atom_t qnn_semantic_events[QNN_WORKER_MAX_EVENT_ATOMS];

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
			*magnitude = 1.0f;
			return true;
		}
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = (spawnflags & 1) ? 0.0f : 1.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_health_rotten"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_health_mega"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_armor1"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_GREEN;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_armor2"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_armorInv"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_RED;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_shells"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = (spawnflags & 1) ? 1.0f : 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_spikes"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = (spawnflags & 1) ? 1.0f : 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_rockets"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = (spawnflags & 1) ? 1.0f : 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_cells"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = (spawnflags & 1) ? 1.0f : 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "weapon_supershotgun"))
	{
		*subject_id = QNN_SUBJECT_SHOTGUN;
		*magnitude = 1.0f;
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
		*magnitude = 1.0f;
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
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_invulnerability"))
	{
		*subject_id = QNN_SUBJECT_PENT;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_invisibility"))
	{
		*subject_id = QNN_SUBJECT_RING;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcasecmp(classname, "item_artifact_envirosuit"))
	{
		*subject_id = QNN_SUBJECT_SUIT;
		*magnitude = 0.0f;
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
		*magnitude = 1.0f;
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
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_bh25.bsp"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_bh100.bsp"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/armor.mdl"))
	{
		if (ent->skin <= 0)
			*subject_id = QNN_SUBJECT_ARMOR_GREEN;
		else if (ent->skin == 1)
			*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		else
			*subject_id = QNN_SUBJECT_ARMOR_RED;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_shell0.bsp"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_shell1.bsp"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_nail0.bsp"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_nail1.bsp"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_rock0.bsp"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_rock1.bsp"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_batt0.bsp"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = 0.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "maps/b_batt1.bsp"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_shot.mdl"))
	{
		*subject_id = QNN_SUBJECT_SHOTGUN;
		*magnitude = 1.0f;
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
		*magnitude = 1.0f;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_rock.mdl"))
	{
		*subject_id = QNN_SUBJECT_GRENADE_LAUNCHER;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_rock2.mdl"))
	{
		*subject_id = QNN_SUBJECT_ROCKET_LAUNCHER;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/g_light.mdl"))
	{
		*subject_id = QNN_SUBJECT_THUNDERBOLT;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/quaddama.mdl"))
	{
		*subject_id = QNN_SUBJECT_QUAD;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/invulner.mdl"))
	{
		*subject_id = QNN_SUBJECT_PENT;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/invisibl.mdl"))
	{
		*subject_id = QNN_SUBJECT_RING;
		return true;
	}
	if (!strcmp(ent->model_name, "progs/suit.mdl"))
	{
		*subject_id = QNN_SUBJECT_SUIT;
		return true;
	}
	if (qnn_classify_item_subject(ent->classname, 0, subject_id, magnitude))
		return true;
	return false;
}

static float qnn_dynamic_decay_s(int subject_id)
{
	if (subject_id == QNN_SUBJECT_PLAYER)
		return QNN_PLAYER_DECAY_S;
	if (subject_id == QNN_SUBJECT_PROJECTILE_NAIL
		|| subject_id == QNN_SUBJECT_PROJECTILE_GRENADE
		|| subject_id == QNN_SUBJECT_PROJECTILE_ROCKET
		|| subject_id == QNN_SUBJECT_LIGHTNING_BEAM)
		return QNN_PROJECTILE_DECAY_S;
	return QNN_GENERIC_DYNAMIC_DECAY_S;
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

static qnn_semantic_object_t *qnn_object_by_handle(uint32_t handle)
{
	int i;

	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		if (qnn_semantic_objects[i].active && qnn_semantic_objects[i].handle == handle)
			return &qnn_semantic_objects[i];
	}
	return NULL;
}

static qnn_semantic_object_t *qnn_ensure_dynamic_object(int entity_num, int subject_id, int qualifier_id, const vec3_t origin, const vec3_t angles, int region_id, int modality_id, float confidence, float magnitude)
{
	qnn_semantic_object_t *obj;
	uint32_t handle;
	int i;

	handle = entity_num > 0 ? (0x80000000u | (uint32_t)entity_num) : qnn_semantic_next_handle++;
	obj = qnn_object_by_handle(handle);
	if (obj == NULL)
	{
		for (i = qnn_semantic_static_count; i < qnn_semantic_object_capacity; ++i)
		{
			if (!qnn_semantic_objects[i].active)
			{
				obj = &qnn_semantic_objects[i];
				memset(obj, 0, sizeof(*obj));
				obj->active = true;
				obj->handle = handle;
				obj->entity_num = entity_num;
				obj->is_static = false;
				break;
			}
		}
	}
	if (obj == NULL)
		return NULL;

	obj->subject_id = subject_id;
	obj->qualifier_id = qualifier_id;
	obj->modality_id = modality_id;
	obj->player_id = (subject_id == QNN_SUBJECT_PLAYER && entity_num > 0) ? entity_num : 0;
	obj->region_id = region_id;
	VectorCopy(origin, obj->origin);
	VectorCopy(angles, obj->angles);
	obj->recency = 1.0f;
	obj->confidence = confidence;
	obj->magnitude = magnitude;
	obj->state = qnn_subject_is_item(subject_id) ? 1.0f : 0.0f;
	obj->decay_s = qnn_dynamic_decay_s(subject_id);
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

static void qnn_append_event(uint32_t object_handle, int subject_id, int action_id, int qualifier_id, int modality_id, float confidence, float magnitude)
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
		if (qnn_semantic_events[i].object_handle == object_handle
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
	qnn_semantic_events[free_index].object_handle = object_handle;
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
			obj->recency = qnn_clampf(obj->recency - (dt / QNN_OBJECT_RECENCY_S), 0.0f, 1.0f);
		if (!obj->is_static && obj->confidence > 0.0f)
			obj->confidence = qnn_clampf(obj->confidence - (dt / QNN_DYNAMIC_CONFIDENCE_S), 0.0f, 1.0f);
		if (!obj->is_static && obj->recency <= 0.0f && obj->confidence <= 0.0f)
			obj->active = false;
	}

	for (i = 0; i < QNN_WORKER_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
			continue;
		qnn_semantic_events[i].recency = qnn_clampf(qnn_semantic_events[i].recency - (dt / QNN_EVENT_RECENCY_S), 0.0f, 1.0f);
		if (qnn_semantic_events[i].confidence > 0.0f)
			qnn_semantic_events[i].confidence = qnn_clampf(qnn_semantic_events[i].confidence - (dt / QNN_DYNAMIC_CONFIDENCE_S), 0.0f, 1.0f);
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
		qnn_append_event(obj->handle, obj->subject_id, QNN_ACTION_RESPAWN, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
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
			obj->modality_id = QNN_MODALITY_AUDITORY;
			obj->recency = 1.0f;
			obj->confidence = snd->entity_num > 0 ? 0.9f : 0.7f;
			qnn_append_event(obj->handle, QNN_SUBJECT_BACKPACK, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, obj->confidence, 0.0f);
			return;
		}
	}

	obj = qnn_nearest_static_subject(0, snd->origin, QNN_ITEM_PICKUP_MATCH_SQ, false, pickup_category);
	if (obj == NULL || obj->state < 1.0f)
		return;
	obj->state = 0.0f;
	obj->pickup_elapsed = 0.0f;
	qnn_refresh_static_object(obj, QNN_MODALITY_AUDITORY, 1.0f);
	qnn_append_event(obj->handle, obj->subject_id, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
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
	float confidence;

	VectorCopy(vec3_origin, angles);
	confidence = snd->entity_num > 0 ? 0.9f : 0.6f;
	obj = qnn_ensure_dynamic_object(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, confidence, 0.0f);
	if (obj == NULL)
		return;
	qnn_append_event(obj->handle, rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, confidence, rule->magnitude);
}

static void qnn_handle_sound_weapon(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;
	float confidence;

	VectorCopy(vec3_origin, angles);
	confidence = snd->entity_num > 0 ? 0.9f : 0.6f;
	obj = qnn_ensure_dynamic_object(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, confidence, 0.0f);
	if (obj == NULL)
		return;
	qnn_append_event(obj->handle, rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, confidence, rule->magnitude);
}

static void qnn_handle_sound_projectile(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;
	float confidence;

	VectorCopy(vec3_origin, angles);
	confidence = snd->entity_num > 0 ? 0.85f : 0.6f;
	if (snd->entity_num > 0)
		obj = qnn_ensure_dynamic_object(snd->entity_num, rule->subject_id, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, confidence, rule->magnitude);
	else
		obj = qnn_nearest_dynamic_subject(rule->subject_id, snd->origin, QNN_PROJECTILE_SOUND_MATCH_SQ);
	if (obj == NULL)
		return;
	if (snd->entity_num <= 0)
	{
		obj->modality_id = QNN_MODALITY_AUDITORY;
		obj->recency = 1.0f;
		if (confidence > obj->confidence)
			obj->confidence = confidence;
	}
	qnn_append_event(obj->handle, rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, confidence, rule->magnitude);
}

static void qnn_handle_sound_static(const qnn_worker_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	qnn_semantic_object_t *obj;

	obj = qnn_nearest_static_subject(rule->subject_id, snd->origin, QNN_STATIC_SOUND_MATCH_SQ, false, 0);
	if (obj == NULL)
		return;
	qnn_refresh_static_object(obj, QNN_MODALITY_AUDITORY, 1.0f);
	qnn_append_event(obj->handle, rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
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
		qnn_append_event(obj->handle, QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_NAIL_FIRST && ent->frame <= QNN_VISUAL_FIRE_NAIL_LAST)
	{
		qnn_append_event(obj->handle, QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_LIGHT_FIRST && ent->frame <= QNN_VISUAL_FIRE_LIGHT_LAST)
	{
		qnn_append_event(obj->handle, QNN_SUBJECT_THUNDERBOLT, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
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
	qnn_semantic_object_capacity = map_state->static_object_count + QNN_WORKER_MAX_DYNAMIC_OBJECTS;
	qnn_semantic_objects = (qnn_semantic_object_t *)calloc((size_t)qnn_semantic_object_capacity, sizeof(*qnn_semantic_objects));
	memset(qnn_semantic_events, 0, sizeof(qnn_semantic_events));
	qnn_semantic_next_handle = 0x90000000u;
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
		qnn_semantic_objects[i].handle = (uint32_t)(i + 1);
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
		qnn_semantic_objects[i].decay_s = 0.0f;
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
}

/*
 * Aggregate individual nail projectile tokens into stream tokens.
 * Groups nails by velocity direction (dot > threshold), picks the leading
 * nail (closest to player) as representative, and sets magnitude to
 * threat-weighted density: super nails count 2x, clamped to [0,1].
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

	/* even a single nail gets density-based magnitude for consistency */
	if (nail_count == 1)
	{
		float w = (object_rows[nail_indices[0]]->magnitude >= 0.5f)
			? QNN_NAIL_STREAM_SUPER_WEIGHT : 1.0f;
		stream_copies[0] = *object_rows[nail_indices[0]];
		stream_copies[0].magnitude = qnn_clampf(
			w / QNN_NAIL_STREAM_DENSITY_DIVISOR, 0.0f, 1.0f);
		object_rows[nail_indices[0]] = &stream_copies[0];
		*stream_copy_count = 1;
		return;
	}

	/* group nails by velocity direction */
	for (i = 0; i < nail_count && streams < QNN_MAX_NAIL_STREAMS; ++i)
	{
		int leader_idx;
		float leader_dsq;
		float weighted;
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
		/* super nails (magnitude 1.0) count as 2, regular as 1 */
		weighted = (object_rows[ni]->magnitude >= 0.5f)
			? QNN_NAIL_STREAM_SUPER_WEIGHT : 1.0f;
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
			weighted += (object_rows[nj]->magnitude >= 0.5f)
				? QNN_NAIL_STREAM_SUPER_WEIGHT : 1.0f;

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

		/* create a copy of the leader with stream magnitude */
		stream_copies[streams] = *object_rows[leader_idx];
		stream_copies[streams].recency = max_recency;
		stream_copies[streams].magnitude = qnn_clampf(
			weighted / QNN_NAIL_STREAM_DENSITY_DIVISOR, 0.0f, 1.0f);
		/* reassign the slot to point at our copy */
		object_rows[leader_idx] = &stream_copies[streams];
		absorbed[leader_idx] = false; /* keep the leader */
		streams += 1;

		/* reassign events from absorbed nails to leader handle */
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
				if (qnn_semantic_events[j].object_handle == object_rows[nk]->handle)
				{
					qnn_semantic_events[j].object_handle = stream_copies[streams - 1].handle;
					break;
				}
			}
		}
	}

	/* rewrite ungrouped nails (no velocity data) to density magnitude */
	for (i = 0; i < nail_count && streams < QNN_MAX_NAIL_STREAMS; ++i)
	{
		float w;
		int ni = nail_indices[i];
		if (absorbed[ni] || qnn_vec_length(object_rows[ni]->velocity) >= 1.0f)
			continue;
		w = (object_rows[ni]->magnitude >= 0.5f)
			? QNN_NAIL_STREAM_SUPER_WEIGHT : 1.0f;
		stream_copies[streams] = *object_rows[ni];
		stream_copies[streams].magnitude = qnn_clampf(
			w / QNN_NAIL_STREAM_DENSITY_DIVISOR, 0.0f, 1.0f);
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
	qnn_spatial_token_t spatial_tokens[QNN_WORKER_SPATIAL_TOKEN_COUNT];
	qnn_semantic_object_t *object_rows[QNN_WORKER_MAX_TOKEN_OBJECTS];
	qnn_semantic_event_atom_t *event_rows[QNN_WORKER_MAX_EVENT_ATOMS];
	qnn_semantic_object_t stream_copies[QNN_MAX_NAIL_STREAMS];
	uint16_t event_base[QNN_WORKER_MAX_TOKEN_OBJECTS];
	uint16_t event_count[QNN_WORKER_MAX_TOKEN_OBJECTS];
	uint16_t flags;
	int stream_copy_count;
	int object_count;
	int event_total;
	int i;
	int j;

	object_count = 0;
	event_total = 0;
	flags = 0;
	if (reset_flag)
		flags |= QNN_TOKEN_FLAG_RESET;
	if (snapshot->done)
		flags |= QNN_TOKEN_FLAG_DONE;
	if (snapshot->action_label.move || snapshot->action_label.strafe
		|| snapshot->action_label.look_yaw || snapshot->action_label.look_pitch
		|| snapshot->action_label.fire || snapshot->action_label.jump
		|| snapshot->action_label.weapon)
		flags |= QNN_TOKEN_FLAG_HAS_ACTION;

	/* first pass: collect eligible objects */
	for (i = 0; i < qnn_semantic_object_capacity && object_count < QNN_WORKER_MAX_TOKEN_OBJECTS; ++i)
	{
		int has_event;

		if (!qnn_semantic_objects[i].active)
			continue;
		has_event = 0;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (qnn_semantic_events[j].active && qnn_semantic_events[j].object_handle == qnn_semantic_objects[i].handle)
			{
				has_event = 1;
				break;
			}
		}
		if (qnn_semantic_objects[i].recency <= 0.0f && !has_event)
			continue;
		object_rows[object_count] = &qnn_semantic_objects[i];
		object_count += 1;
	}

	/* aggregate nail projectiles into stream tokens */
	qnn_aggregate_nail_streams(object_rows, &object_count, stream_copies,
		&stream_copy_count, snapshot->player_origin);

	/* second pass: collect events for surviving objects */
	for (i = 0; i < object_count; ++i)
	{
		event_base[i] = (uint16_t)event_total;
		event_count[i] = 0;
		for (j = 0; j < QNN_WORKER_MAX_EVENT_ATOMS && event_total < QNN_WORKER_MAX_EVENT_ATOMS; ++j)
		{
			if (!qnn_semantic_events[j].active || qnn_semantic_events[j].object_handle != object_rows[i]->handle)
				continue;
			event_rows[event_total] = &qnn_semantic_events[j];
			event_total += 1;
			event_count[i] += 1;
		}
	}

	qnn_build_spatial_tokens(snapshot, spatial_tokens);

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

	for (i = 0; i < 3; ++i)
		qnn_token_write_f32_le(out, snapshot->player_origin[i]);
	for (i = 0; i < 3; ++i)
		qnn_token_write_f32_le(out, snapshot->player_velocity[i]);
	for (i = 0; i < 3; ++i)
		qnn_token_write_f32_le(out, snapshot->player_view_angles[i]);
	qnn_token_write_i32_le(out, snapshot->health);
	qnn_token_write_i32_le(out, snapshot->armor);
	qnn_token_write_f32_le(out, snapshot->armor_type);
	qnn_token_write_i32_le(out, snapshot->ammo_shells);
	qnn_token_write_i32_le(out, snapshot->ammo_nails);
	qnn_token_write_i32_le(out, snapshot->ammo_rockets);
	qnn_token_write_i32_le(out, snapshot->ammo_cells);
	qnn_token_write_i32_le(out, snapshot->weapon_id);
	qnn_token_write_i32_le(out, snapshot->weapons_owned);
	qnn_token_write_i32_le(out, snapshot->grounded ? 1 : 0);
	qnn_token_write_i32_le(out, snapshot->waterlevel);
	qnn_token_write_i32_le(out, snapshot->current_region_id);

	for (i = 0; i < object_count; ++i)
	{
		vec3_t delta;
		vec3_t rel;

		VectorSubtract(object_rows[i]->origin, snapshot->player_origin, delta);
		qnn_relative_frame(snapshot->player_view_angles, delta, rel);
		qnn_token_write_u32_le(out, object_rows[i]->handle);
		qnn_token_write_u16_le(out, (uint16_t)object_rows[i]->subject_id);
		qnn_token_write_u16_le(out, (uint16_t)object_rows[i]->qualifier_id);
		qnn_token_write_u16_le(out, (uint16_t)object_rows[i]->modality_id);
		qnn_token_write_u16_le(out, (uint16_t)object_rows[i]->player_id);
		qnn_token_write_f32_le(out, rel[0]);
		qnn_token_write_f32_le(out, rel[1]);
		qnn_token_write_f32_le(out, rel[2]);
		qnn_token_write_f32_le(out, object_rows[i]->recency);
		qnn_token_write_f32_le(out, object_rows[i]->confidence);
		qnn_token_write_f32_le(out, object_rows[i]->magnitude);
		qnn_token_write_f32_le(out, object_rows[i]->state);
		qnn_token_write_u16_le(out, event_count[i]);
		qnn_token_write_u16_le(out, event_base[i]);
	}

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
	{
		qnn_token_write_u16_le(out, (uint16_t)spatial_tokens[i].sector_id);
		qnn_token_write_u16_le(out, 0);
		qnn_token_write_f32_le(out, spatial_tokens[i].nearest_dist);
		qnn_token_write_f32_le(out, spatial_tokens[i].mean_dist);
		qnn_token_write_f32_le(out, spatial_tokens[i].openness);
		qnn_token_write_f32_le(out, spatial_tokens[i].solid_frac);
		qnn_token_write_f32_le(out, spatial_tokens[i].water_frac);
		qnn_token_write_f32_le(out, spatial_tokens[i].slime_frac);
		qnn_token_write_f32_le(out, spatial_tokens[i].lava_frac);
		qnn_token_write_f32_le(out, spatial_tokens[i].traversable);
		qnn_token_write_f32_le(out, spatial_tokens[i].dropoff);
		qnn_token_write_f32_le(out, spatial_tokens[i].clearance);
	}

	if (snapshot->action_label.move || snapshot->action_label.strafe
		|| snapshot->action_label.look_yaw || snapshot->action_label.look_pitch
		|| snapshot->action_label.fire || snapshot->action_label.jump
		|| snapshot->action_label.weapon)
	{
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.move);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.strafe);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.look_yaw);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.look_pitch);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.fire);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.jump);
		qnn_token_write_u16_le(out, (uint16_t)snapshot->action_label.weapon);
	}
	fflush(out);
}
