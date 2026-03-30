#include "qnn.h"
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
   VISUAL beats AUDITORY beats MENTAL.  QNN_EnsureDynamicObject
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
	int next_for_owner; /* linked list: next event with same owner, -1 = end */
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
static qnn_semantic_event_atom_t qnn_semantic_events[QNN_MAX_EVENT_ATOMS];

/* Per-object event linked list heads. Index into qnn_semantic_events[], -1 = none.
   Sized to MAX_EDICTS + 1024 (covers static_count + entity slots). */
#define QNN_EVENT_HEAD_CAPACITY (MAX_EDICTS + 1024)
static int qnn_event_head[QNN_EVENT_HEAD_CAPACITY];

/* Maps emitted token slot (0-based) to semantic object array index.
 * Used by recall to resolve which object the model wants to attend to. */
static int qnn_prev_object_indices[QNN_MAX_TOKEN_OBJECTS];
static int qnn_prev_object_count = 0;

/* Emission priority: dynamic objects first, then static. Recency tiebreak. */
static int QNN_ObjectRowCompare(const void *a, const void *b)
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

static int QNN_ObjectIndex(const qnn_semantic_object_t *obj)
{
	return (int)(obj - qnn_semantic_objects);
}

static qboolean QNN_ActionHasSignal(const qnn_action_t *action)
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

static int QNN_MatchCandidateCompare(const void *lhs_ptr, const void *rhs_ptr)
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

/* Write helpers, clamp, normalize, angle, and magnitude functions
   replaced by macros/public functions in qnn.h and qnn_sys.c. */

static int QNN_SelfWeaponEmbedId(int weapon_id)
{
	return qnn_weapon_class_from_id(weapon_id);
}

static int QNN_SelfWeaponSuper(int weapon_id)
{
	return (weapon_id == 3 || weapon_id == 5) ? 1 : 0;
}

static int QNN_SelfMovementId(qboolean grounded, int waterlevel)
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

static void QNN_WriteSelfToken(FILE *out, const qnn_snapshot_t *snapshot, int tick_hz, int player_cluster_id)
{
	float scalars[QNN_SELF_SCALAR_COUNT];
	int ids[QNN_SELF_ID_COUNT];
	int i;

	scalars[0] = QNN_Normalize((float)snapshot->health, QNN_SELF_HEALTH_CAP);
	scalars[1] = QNN_Normalize((float)snapshot->armor, QNN_SELF_ARMOR_CAP);
	scalars[2] = QNN_Normalize(snapshot->armor_type, QNN_SELF_ARMOR_TYPE_CAP);
	scalars[3] = (snapshot->weapons_owned & IT_SHOTGUN) ? 1.0f : 0.0f;
	scalars[4] = (snapshot->weapons_owned & IT_SUPER_SHOTGUN) ? 1.0f : 0.0f;
	scalars[5] = (snapshot->weapons_owned & IT_NAILGUN) ? 1.0f : 0.0f;
	scalars[6] = (snapshot->weapons_owned & IT_SUPER_NAILGUN) ? 1.0f : 0.0f;
	scalars[7] = (snapshot->weapons_owned & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
	scalars[8] = (snapshot->weapons_owned & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
	scalars[9] = (snapshot->weapons_owned & IT_LIGHTNING) ? 1.0f : 0.0f;
	scalars[10] = (float)QNN_SelfWeaponSuper(snapshot->weapon_id);
	scalars[11] = QNN_Normalize((float)snapshot->ammo_shells, QNN_SELF_SHELLS_CAP);
	scalars[12] = QNN_Normalize((float)snapshot->ammo_nails, QNN_SELF_NAILS_CAP);
	scalars[13] = QNN_Normalize((float)snapshot->ammo_rockets, QNN_SELF_ROCKETS_CAP);
	scalars[14] = QNN_Normalize((float)snapshot->ammo_cells, QNN_SELF_CELLS_CAP);
	scalars[15] = QNN_Normalize(snapshot->player_velocity[0], QNN_SELF_VELOCITY_CAP);
	scalars[16] = QNN_Normalize(snapshot->player_velocity[1], QNN_SELF_VELOCITY_CAP);
	scalars[17] = QNN_Normalize(snapshot->player_velocity[2], QNN_SELF_VELOCITY_CAP);
	scalars[18] = QNN_AngleSinDeg(snapshot->player_view_angles[1]);
	scalars[19] = QNN_AngleCosDeg(snapshot->player_view_angles[1]);
	scalars[20] = QNN_AngleSinDeg(snapshot->player_view_angles[0]);
	scalars[21] = QNN_AngleCosDeg(snapshot->player_view_angles[0]);
	scalars[22] = tick_hz > 0 ? (1.0f / (float)tick_hz) : 0.0f;

	ids[0] = QNN_SelfWeaponEmbedId(snapshot->weapon_id);
	ids[1] = QNN_SelfMovementId(snapshot->grounded, snapshot->waterlevel);
	ids[2] = player_cluster_id;

	for (i = 0; i < QNN_SELF_SCALAR_COUNT; ++i)
		QNN_WriteF32LE(out, scalars[i]);
	for (i = 0; i < QNN_SELF_ID_COUNT; ++i)
		QNN_WriteI32LE(out, ids[i]);
}

static void QNN_WriteObjectToken(FILE *out, const qnn_semantic_object_t *obj, const vec3_t rel, uint16_t local_event_count, uint16_t local_event_base, uint32_t wire_handle)
{
	float scalars[QNN_OBJECT_SCALAR_COUNT];
	uint16_t ids[QNN_OBJECT_ID_COUNT];
	int i;

	ids[0] = (uint16_t)obj->subject_id;
	ids[1] = (uint16_t)obj->qualifier_id;
	ids[2] = (uint16_t)obj->modality_id;
	ids[3] = (uint16_t)obj->player_id;
	ids[4] = (uint16_t)obj->cluster_id;

	scalars[0] = QNN_Normalize(rel[0], QNN_OBJECT_REL_SCALE);
	scalars[1] = QNN_Normalize(rel[1], QNN_OBJECT_REL_SCALE);
	scalars[2] = QNN_Normalize(rel[2], QNN_OBJECT_REL_SCALE);
	scalars[3] = QNN_Normalize(obj->route_cost, QNN_OBJECT_ROUTE_COST_SCALE);
	scalars[4] = obj->recency;
	scalars[5] = obj->confidence;
	scalars[6] = obj->magnitude;
	scalars[7] = obj->state;

	QNN_WriteU32LE(out, wire_handle);
	for (i = 0; i < QNN_OBJECT_ID_COUNT; ++i)
		QNN_WriteU16LE(out, ids[i]);
	for (i = 0; i < QNN_OBJECT_SCALAR_COUNT; ++i)
		QNN_WriteF32LE(out, scalars[i]);
	QNN_WriteU16LE(out, local_event_count);
	QNN_WriteU16LE(out, local_event_base);
	QNN_WriteU16LE(out, (uint16_t)obj->route_cluster_count);
	for (i = 0; i < QNN_MAX_ROUTE_CLUSTERS; ++i)
		QNN_WriteU16LE(out, (uint16_t)(i < obj->route_cluster_count ? obj->route_cluster_ids[i] : 0));
}

static void QNN_WriteSpatialToken(FILE *out, const qnn_spatial_token_t *token)
{
	QNN_WriteU16LE(out, (uint16_t)token->sector_id);
	QNN_WriteU16LE(out, 0);
	QNN_WriteF32LE(out, QNN_Normalize(token->nearest_dist, QNN_SPATIAL_DIST_SCALE));
	QNN_WriteF32LE(out, QNN_Normalize(token->mean_dist, QNN_SPATIAL_DIST_SCALE));
	QNN_WriteF32LE(out, token->openness);
	QNN_WriteF32LE(out, token->clearance);
	QNN_WriteF32LE(out, token->traversable);
	QNN_WriteF32LE(out, token->dropoff);
	QNN_WriteF32LE(out, token->solid_frac);
	QNN_WriteF32LE(out, token->water_frac);
	QNN_WriteF32LE(out, token->slime_frac);
	QNN_WriteF32LE(out, token->lava_frac);
}

/* QNN_DistSq is a macro in qnn.h */

static const char *QNN_StaticProperty(const qnn_static_object_t *obj, const char *key)
{
	int i;

	for (i = 0; i < obj->property_count; ++i)
	{
		if (!strcmp(obj->properties[i].key, key))
			return obj->properties[i].value;
	}
	return NULL;
}

static int QNN_StaticPropertyInt(const qnn_static_object_t *obj, const char *key, int fallback)
{
	const char *value;

	value = QNN_StaticProperty(obj, key);
	if (value == NULL || value[0] == 0)
		return fallback;
	return atoi(value);
}

static qboolean QNN_SubjectIsItem(int subject_id)
{
	return subject_id >= QNN_SUBJECT_SHELLS && subject_id <= QNN_SUBJECT_SUIT;
}

static float QNN_ItemRespawnS(const qnn_static_object_t *obj, int subject_id)
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

static qboolean QNN_ClassifyItemSubject(const char *classname, int spawnflags, int *subject_id, float *magnitude)
{
	if (!strcasecmp(classname, "item_health"))
	{
		if (spawnflags & 2)
		{
			*subject_id = QNN_SUBJECT_MEGAHEALTH;
			*magnitude = QNN_Normalize(100.0f, QNN_SELF_HEALTH_CAP);
			return true;
		}
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(15.0f, QNN_SELF_HEALTH_CAP) : QNN_Normalize(25.0f, QNN_SELF_HEALTH_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_health_rotten"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = QNN_Normalize(15.0f, QNN_SELF_HEALTH_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_health_mega"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = QNN_Normalize(100.0f, QNN_SELF_HEALTH_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_armor1"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_GREEN;
		*magnitude = QNN_Normalize(100.0f, QNN_SELF_ARMOR_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_armor2"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		*magnitude = QNN_Normalize(150.0f, QNN_SELF_ARMOR_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_armorInv"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_RED;
		*magnitude = QNN_Normalize(200.0f, QNN_SELF_ARMOR_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_shells"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(40.0f, QNN_SELF_SHELLS_CAP) : QNN_Normalize(20.0f, QNN_SELF_SHELLS_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_spikes"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(50.0f, QNN_SELF_NAILS_CAP) : QNN_Normalize(25.0f, QNN_SELF_NAILS_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_rockets"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(10.0f, QNN_SELF_ROCKETS_CAP) : QNN_Normalize(5.0f, QNN_SELF_ROCKETS_CAP);
		return true;
	}
	if (!strcasecmp(classname, "item_cells"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(12.0f, QNN_SELF_CELLS_CAP) : QNN_Normalize(6.0f, QNN_SELF_CELLS_CAP);
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

static qboolean QNN_ClassifyStaticSubject(const qnn_static_object_t *obj, int *subject_id, int *qualifier_id, float *magnitude, qboolean *is_item, float *respawn_s)
{
	int spawnflags;

	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;
	*is_item = false;
	*respawn_s = 0.0f;

	spawnflags = QNN_StaticPropertyInt(obj, "spawnflags", 0);
	if (QNN_ClassifyItemSubject(obj->classname, spawnflags, subject_id, magnitude))
	{
		*is_item = true;
		*respawn_s = QNN_ItemRespawnS(obj, *subject_id);
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

/* Model name → subject classification lookup table.
   mag_value/mag_scale: magnitude = mag_value / mag_scale (0/0 = no magnitude). */
typedef struct {
	const char *model;
	int subject_id;
	int qualifier_id;
	float mag_value;
	float mag_scale;
} qnn_model_classify_t;

static const qnn_model_classify_t qnn_model_table[] = {
	{"progs/player.mdl",    QNN_SUBJECT_PLAYER,             QNN_QUAL_NONE,      0, 0},
	{"progs/eyes.mdl",      QNN_SUBJECT_PLAYER,             QNN_QUAL_INVISIBLE, 0, 0},
	{"progs/backpack.mdl",  QNN_SUBJECT_BACKPACK,           QNN_QUAL_NONE,      0, 0},
	{"progs/spike.mdl",     QNN_SUBJECT_PROJECTILE_NAIL,    QNN_QUAL_NONE,      0, 0},
	{"progs/s_spike.mdl",   QNN_SUBJECT_PROJECTILE_NAIL,    QNN_QUAL_NONE,      0, 0},
	{"progs/grenade.mdl",   QNN_SUBJECT_PROJECTILE_GRENADE, QNN_QUAL_NONE,      0, 0},
	{"progs/missile.mdl",   QNN_SUBJECT_PROJECTILE_ROCKET,  QNN_QUAL_NONE,      0, 0},
	{"progs/bolt2.mdl",     QNN_SUBJECT_LIGHTNING_BEAM,     QNN_QUAL_NONE,      0, 0},
	{"progs/bolt3.mdl",     QNN_SUBJECT_LIGHTNING_BEAM,     QNN_QUAL_NONE,      0, 0},
	{"progs/teleport.mdl",  QNN_SUBJECT_TELEPORTER,         QNN_QUAL_NONE,      0, 0},
	{"maps/b_bh10.bsp",     QNN_SUBJECT_HEALTH,             QNN_QUAL_NONE,      15,  QNN_SELF_HEALTH_CAP},
	{"maps/b_bh25.bsp",     QNN_SUBJECT_HEALTH,             QNN_QUAL_NONE,      25,  QNN_SELF_HEALTH_CAP},
	{"maps/b_bh100.bsp",    QNN_SUBJECT_MEGAHEALTH,         QNN_QUAL_NONE,      100, QNN_SELF_HEALTH_CAP},
	{"maps/b_shell0.bsp",   QNN_SUBJECT_SHELLS,             QNN_QUAL_NONE,      20,  QNN_SELF_SHELLS_CAP},
	{"maps/b_shell1.bsp",   QNN_SUBJECT_SHELLS,             QNN_QUAL_NONE,      40,  QNN_SELF_SHELLS_CAP},
	{"maps/b_nail0.bsp",    QNN_SUBJECT_NAILS,              QNN_QUAL_NONE,      25,  QNN_SELF_NAILS_CAP},
	{"maps/b_nail1.bsp",    QNN_SUBJECT_NAILS,              QNN_QUAL_NONE,      50,  QNN_SELF_NAILS_CAP},
	{"maps/b_rock0.bsp",    QNN_SUBJECT_ROCKETS,            QNN_QUAL_NONE,      5,   QNN_SELF_ROCKETS_CAP},
	{"maps/b_rock1.bsp",    QNN_SUBJECT_ROCKETS,            QNN_QUAL_NONE,      10,  QNN_SELF_ROCKETS_CAP},
	{"maps/b_batt0.bsp",    QNN_SUBJECT_CELLS,              QNN_QUAL_NONE,      6,   QNN_SELF_CELLS_CAP},
	{"maps/b_batt1.bsp",    QNN_SUBJECT_CELLS,              QNN_QUAL_NONE,      12,  QNN_SELF_CELLS_CAP},
	{"progs/g_shot.mdl",    QNN_SUBJECT_SHOTGUN,            QNN_QUAL_NONE,      0, 0},
	{"progs/g_nail.mdl",    QNN_SUBJECT_NAILGUN,            QNN_QUAL_NONE,      0, 0},
	{"progs/g_nail2.mdl",   QNN_SUBJECT_NAILGUN,            QNN_QUAL_NONE,      0, 0},
	{"progs/g_rock.mdl",    QNN_SUBJECT_GRENADE_LAUNCHER,   QNN_QUAL_NONE,      0, 0},
	{"progs/g_rock2.mdl",   QNN_SUBJECT_ROCKET_LAUNCHER,    QNN_QUAL_NONE,      0, 0},
	{"progs/g_light.mdl",   QNN_SUBJECT_THUNDERBOLT,        QNN_QUAL_NONE,      0, 0},
	{"progs/quaddama.mdl",  QNN_SUBJECT_QUAD,               QNN_QUAL_NONE,      1, 1},
	{"progs/invulner.mdl",  QNN_SUBJECT_PENT,               QNN_QUAL_NONE,      1, 1},
	{"progs/invisibl.mdl",  QNN_SUBJECT_RING,               QNN_QUAL_NONE,      1, 1},
	{"progs/suit.mdl",      QNN_SUBJECT_SUIT,               QNN_QUAL_NONE,      1, 1},
	{NULL, 0, 0, 0, 0}
};

static qboolean QNN_ClassifyVisibleSubject(const qnn_visible_entity_t *ent, int *subject_id, int *qualifier_id, float *magnitude)
{
	const qnn_model_classify_t *row;

	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;

	/* Check classname first (players may have non-standard models) */
	if (!strcmp(ent->classname, "player"))
	{
		*subject_id = QNN_SUBJECT_PLAYER;
		return true;
	}

	/* Table lookup by model name */
	for (row = qnn_model_table; row->model; ++row)
	{
		if (!strcmp(ent->model_name, row->model))
		{
			*subject_id = row->subject_id;
			*qualifier_id = row->qualifier_id;
			*magnitude = row->mag_scale > 0 ? QNN_Normalize(row->mag_value, row->mag_scale) : row->mag_value;
			return true;
		}
	}

	/* Armor is special: subject depends on skin */
	if (!strcmp(ent->model_name, "progs/armor.mdl"))
	{
		if (ent->skin <= 0)      { *subject_id = QNN_SUBJECT_ARMOR_GREEN;  *magnitude = QNN_Normalize(100.0f, QNN_SELF_ARMOR_CAP); }
		else if (ent->skin == 1) { *subject_id = QNN_SUBJECT_ARMOR_YELLOW; *magnitude = QNN_Normalize(150.0f, QNN_SELF_ARMOR_CAP); }
		else                     { *subject_id = QNN_SUBJECT_ARMOR_RED;    *magnitude = QNN_Normalize(200.0f, QNN_SELF_ARMOR_CAP); }
		return true;
	}

	/* Fallback to classname-based item classification */
	if (QNN_ClassifyItemSubject(ent->classname, 0, subject_id, magnitude))
		return true;
	return false;
}

static int QNN_SoundPickupCategory(const char *name)
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

static int QNN_SubjectPickupCategory(int subject_id)
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

static qboolean QNN_InFov(const vec3_t player_origin, const vec3_t view_angles, const vec3_t target)
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
	dist = QNN_VecLength(delta);
	if (dist < 1.0f)
		return true;
	dot = DotProduct(forward, delta) / dist;
	cos_half = (float)cos((double)(QNN_FOV_HALF_DEG * M_PI / 180.0f));
	return dot >= cos_half;
}

static void QNN_RelativeFrame(const vec3_t view_angles, const vec3_t world_delta, vec3_t out)
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
static int QNN_DynamicIndex(int entity_num)
{
	if (entity_num <= 0 || entity_num >= MAX_EDICTS)
		return -1;
	return qnn_semantic_static_count + entity_num;
}

static qnn_semantic_object_t *QNN_EnsureDynamicObject(int entity_num, int subject_id, int qualifier_id, const vec3_t origin, const vec3_t angles, int region_id, int modality_id, float confidence, float magnitude)
{
	int idx;
	qnn_semantic_object_t *obj;

	idx = QNN_DynamicIndex(entity_num);
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
	obj->state = QNN_SubjectIsItem(subject_id) ? 1.0f : 0.0f;
	obj->surfaced_this_tick = true;
	return obj;
}

static qnn_semantic_object_t *QNN_NearestStaticSubject(int subject_id, const vec3_t origin, float max_dist_sq, qboolean require_unavailable, int pickup_category)
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
			if (QNN_SubjectPickupCategory(obj->subject_id) != pickup_category)
				continue;
		}
		if (require_unavailable && obj->state >= 1.0f)
			continue;
		dsq = QNN_DistSq(obj->origin, origin);
		if (dsq < best_dsq)
		{
			best = obj;
			best_dsq = dsq;
		}
	}
	return best;
}

static qnn_semantic_object_t *QNN_NearestDynamicSubject(int subject_id, const vec3_t origin, float max_dist_sq)
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
		dsq = QNN_DistSq(qnn_semantic_objects[i].origin, origin);
		if (dsq < best_dsq)
		{
			best = &qnn_semantic_objects[i];
			best_dsq = dsq;
		}
	}
	return best;
}

static void QNN_AppendEvent(int owner_index, int subject_id, int action_id, int qualifier_id, int modality_id, float confidence, float magnitude)
{
	int i;
	int free_index;

	free_index = -1;
	for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
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

	/* Link into per-owner chain */
	qnn_semantic_events[free_index].next_for_owner = -1;
	if (owner_index >= 0 && owner_index < QNN_EVENT_HEAD_CAPACITY)
	{
		qnn_semantic_events[free_index].next_for_owner = qnn_event_head[owner_index];
		qnn_event_head[owner_index] = free_index;
	}
}

static void QNN_DecayStore(float dt)
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
			obj->recency = QNN_Clamp(obj->recency - (dt / QNN_RECENCY_DECAY_S), 0.0f, 1.0f);
		if (!obj->is_static && obj->recency <= 0.0f)
			obj->active = false;
	}

	for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
	{
		if (!qnn_semantic_events[i].active)
			continue;
		qnn_semantic_events[i].recency = QNN_Clamp(qnn_semantic_events[i].recency - (dt / QNN_RECENCY_DECAY_S), 0.0f, 1.0f);
		if (qnn_semantic_events[i].recency <= 0.0f)
			qnn_semantic_events[i].active = false;
	}

	/* Rebuild per-owner linked lists after deactivations. O(events). */
	{
		int cap = qnn_semantic_object_capacity < QNN_EVENT_HEAD_CAPACITY
			? qnn_semantic_object_capacity : QNN_EVENT_HEAD_CAPACITY;
		memset(qnn_event_head, -1, (size_t)cap * sizeof(int));
		for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
		{
			int oi;
			if (!qnn_semantic_events[i].active)
				continue;
			oi = qnn_semantic_events[i].owner_index;
			if (oi >= 0 && oi < QNN_EVENT_HEAD_CAPACITY)
			{
				qnn_semantic_events[i].next_for_owner = qnn_event_head[oi];
				qnn_event_head[oi] = i;
			}
		}
	}
}

static void QNN_UpdateItemsFromVisibility(const qnn_snapshot_t *snapshot)
{
	qnn_match_candidate_t candidates[QNN_MAX_MATCH_CANDIDATES];
	int candidate_count;
	int *item_matched;
	int ent_matched[QNN_MAX_VISIBLE];
	int i;
	int j;

	candidate_count = 0;
	memset(ent_matched, 0, sizeof(ent_matched));
	/* Static array avoids per-step malloc. 1024 covers any reasonable map. */
	{
		static int item_matched_buf[1024];
		int match_count = qnn_semantic_static_count > 0 ? qnn_semantic_static_count : 1;
		if (match_count > 1024) match_count = 1024;
		item_matched = item_matched_buf;
		memset(item_matched, 0, (size_t)match_count * sizeof(int));
	}

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;

		if (!QNN_ClassifyVisibleSubject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;
		if (!QNN_SubjectIsItem(subject_id))
			continue;
		for (j = 0; j < qnn_semantic_static_count; ++j)
		{
			float dsq;
			qnn_semantic_object_t *obj;

			obj = &qnn_semantic_objects[j];
			if (!obj->active || !obj->is_item || obj->subject_id != subject_id)
				continue;
			dsq = QNN_DistSq(obj->origin, snapshot->visible[i].origin);
			if (dsq >= QNN_ITEM_PVS_MATCH_SQ || candidate_count >= QNN_MAX_MATCH_CANDIDATES)
				continue;
			candidates[candidate_count].dist_sq = dsq;
			candidates[candidate_count].ent_idx = i;
			candidates[candidate_count].obj_idx = j;
			candidate_count += 1;
		}
	}

	if (candidate_count > 1)
		qsort(candidates, (size_t)candidate_count, sizeof(candidates[0]), QNN_MatchCandidateCompare);

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

		if (!QNN_ClassifyVisibleSubject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;
		if (!QNN_SubjectIsItem(subject_id))
			continue;
		if (!QNN_InFov(snapshot->player_origin, snapshot->player_view_angles, snapshot->visible[i].origin))
			continue;
		obj = QNN_NearestStaticSubject(subject_id, snapshot->visible[i].origin, QNN_ITEM_PVS_MATCH_SQ, false, 0);
		if (obj == NULL)
			continue;
		if (QNN_MODALITY_VISUAL < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
			obj->modality_id = QNN_MODALITY_VISUAL;
		obj->recency = 1.0f;
		obj->confidence = 1.0f;
		obj->magnitude = magnitude;
		obj->surfaced_this_tick = true;
	}

	/* item_matched is static — no free needed */
}

static void QNN_AdvanceItemTimers(float dt)
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
			obj->state = QNN_Clamp(obj->pickup_elapsed / obj->respawn_s, 0.0f, 1.0f);
	}
}

static void QNN_RefreshStaticObject(qnn_semantic_object_t *obj, int modality_id, float confidence)
{
	if (modality_id < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
		obj->modality_id = modality_id;
	obj->recency = 1.0f;
	obj->confidence = confidence;
	obj->surfaced_this_tick = true;
}

static void QNN_HandleItemSound(const qnn_sound_event_t *snd, const char *name)
{
	int pickup_category;
	qnn_semantic_object_t *obj;

	if (!strcmp(name, "items/itembk2.wav"))
	{
		obj = QNN_NearestStaticSubject(0, snd->origin, QNN_ITEM_RESPAWN_MATCH_SQ, true, 0);
		if (obj == NULL)
			return;
		obj->state = 1.0f;
		obj->pickup_elapsed = 0.0f;
		QNN_RefreshStaticObject(obj, QNN_MODALITY_AUDITORY, 1.0f);
		QNN_AppendEvent(QNN_ObjectIndex(obj), obj->subject_id, QNN_ACTION_RESPAWN, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
		return;
	}

	pickup_category = QNN_SoundPickupCategory(name);
	if (pickup_category == 0)
		return;

	if (!strcmp(name, "weapons/lock4.wav"))
	{
		obj = QNN_NearestDynamicSubject(QNN_SUBJECT_BACKPACK, snd->origin, QNN_ITEM_PICKUP_MATCH_SQ);
		if (obj != NULL)
		{
			if (QNN_MODALITY_AUDITORY < obj->modality_id || obj->modality_id == QNN_MODALITY_NONE)
				obj->modality_id = QNN_MODALITY_AUDITORY;
			obj->recency = 1.0f;
			obj->confidence = 1.0f;
			obj->surfaced_this_tick = true;
			QNN_AppendEvent(QNN_ObjectIndex(obj), QNN_SUBJECT_BACKPACK, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
			return;
		}
	}

	obj = QNN_NearestStaticSubject(0, snd->origin, QNN_ITEM_PICKUP_MATCH_SQ, false, pickup_category);
	if (obj == NULL || obj->state < 1.0f)
		return;
	obj->state = 0.0f;
	obj->pickup_elapsed = 0.0f;
	QNN_RefreshStaticObject(obj, QNN_MODALITY_AUDITORY, 1.0f);
	QNN_AppendEvent(QNN_ObjectIndex(obj), obj->subject_id, QNN_ACTION_PICKUP, QNN_QUAL_NONE, QNN_MODALITY_AUDITORY, 1.0f, obj->magnitude);
}

static const qnn_sound_rule_t *QNN_FindSoundRule(const qnn_sound_rule_t *rules, const char *name)
{
	int i;

	for (i = 0; rules[i].name != NULL; ++i)
	{
		if (!strcmp(rules[i].name, name))
			return &rules[i];
	}
	return NULL;
}

static void QNN_HandleSoundPlayer(const qnn_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	obj = QNN_EnsureDynamicObject(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
	if (obj == NULL)
		return;
	QNN_AppendEvent(QNN_ObjectIndex(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void QNN_HandleSoundWeapon(const qnn_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	obj = QNN_EnsureDynamicObject(snd->entity_num, QNN_SUBJECT_PLAYER, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, 0.0f);
	if (obj == NULL)
		return;
	QNN_AppendEvent(QNN_ObjectIndex(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void QNN_HandleSoundProjectile(const qnn_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	vec3_t angles;
	qnn_semantic_object_t *obj;

	VectorCopy(vec3_origin, angles);
	if (snd->entity_num > 0)
		obj = QNN_EnsureDynamicObject(snd->entity_num, rule->subject_id, QNN_QUAL_NONE, snd->origin, angles, -1, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
	else
		obj = QNN_NearestDynamicSubject(rule->subject_id, snd->origin, QNN_PROJECTILE_SOUND_MATCH_SQ);
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
	QNN_AppendEvent(QNN_ObjectIndex(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void QNN_HandleSoundStatic(const qnn_sound_event_t *snd, const qnn_sound_rule_t *rule)
{
	qnn_semantic_object_t *obj;

	obj = QNN_NearestStaticSubject(rule->subject_id, snd->origin, QNN_STATIC_SOUND_MATCH_SQ, false, 0);
	if (obj == NULL)
		return;
	QNN_RefreshStaticObject(obj, QNN_MODALITY_AUDITORY, 1.0f);
	QNN_AppendEvent(QNN_ObjectIndex(obj), rule->subject_id, rule->action_id, rule->qualifier_id, QNN_MODALITY_AUDITORY, 1.0f, rule->magnitude);
}

static void QNN_UpdateFromSounds(const qnn_snapshot_t *snapshot)
{
	int i;

	for (i = 0; i < snapshot->sound_count; ++i)
	{
		char name[QNN_MAX_SOUND_NAME];
		const qnn_sound_rule_t *rule;
		int j;

		name[0] = 0;
		for (j = 0; snapshot->sounds[i].name[j] && j < QNN_MAX_SOUND_NAME - 1; ++j)
			name[j] = (char)tolower((unsigned char)snapshot->sounds[i].name[j]);
		name[j] = 0;
		if (name[0] == 0)
			continue;

		QNN_HandleItemSound(&snapshot->sounds[i], name);

		rule = QNN_FindSoundRule(qnn_player_sound_rules, name);
		if (rule != NULL)
		{
			QNN_HandleSoundPlayer(&snapshot->sounds[i], rule);
			continue;
		}
		if (!strncmp(name, "player/pain", 11))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 0.0f;
			QNN_HandleSoundPlayer(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/drown", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_DROWN;
			temp.magnitude = 0.0f;
			QNN_HandleSoundPlayer(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/lburn", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_PAIN;
			temp.qualifier_id = QNN_QUAL_LAVA;
			temp.magnitude = 0.0f;
			QNN_HandleSoundPlayer(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strncmp(name, "player/death", 12))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_DEATH;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 0.0f;
			QNN_HandleSoundPlayer(&snapshot->sounds[i], &temp);
			continue;
		}
		if (!strcmp(name, "player/gib.wav") || !strcmp(name, "player/udeath.wav") || !strcmp(name, "player/tornoff2.wav"))
		{
			qnn_sound_rule_t temp;
			temp.subject_id = QNN_SUBJECT_PLAYER;
			temp.action_id = QNN_ACTION_DEATH;
			temp.qualifier_id = QNN_QUAL_NONE;
			temp.magnitude = 1.0f;
			QNN_HandleSoundPlayer(&snapshot->sounds[i], &temp);
			continue;
		}

		rule = QNN_FindSoundRule(qnn_weapon_sound_rules, name);
		if (rule != NULL)
		{
			QNN_HandleSoundWeapon(&snapshot->sounds[i], rule);
			continue;
		}
		rule = QNN_FindSoundRule(qnn_projectile_sound_rules, name);
		if (rule != NULL)
		{
			QNN_HandleSoundProjectile(&snapshot->sounds[i], rule);
			continue;
		}
		rule = QNN_FindSoundRule(qnn_static_sound_rules, name);
		if (rule != NULL)
			QNN_HandleSoundStatic(&snapshot->sounds[i], rule);
	}
}

static void QNN_MaybeAppendVisualFire(const qnn_visible_entity_t *ent, qnn_semantic_object_t *obj)
{
	if (!(ent->effects & EF_MUZZLEFLASH))
		return;
	if (ent->frame >= QNN_VISUAL_FIRE_SHOT_FIRST && ent->frame <= QNN_VISUAL_FIRE_SHOT_LAST)
	{
		QNN_AppendEvent(QNN_ObjectIndex(obj), QNN_SUBJECT_SHOTGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_NAIL_FIRST && ent->frame <= QNN_VISUAL_FIRE_NAIL_LAST)
	{
		QNN_AppendEvent(QNN_ObjectIndex(obj), QNN_SUBJECT_NAILGUN, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
	if (ent->frame >= QNN_VISUAL_FIRE_LIGHT_FIRST && ent->frame <= QNN_VISUAL_FIRE_LIGHT_LAST)
	{
		QNN_AppendEvent(QNN_ObjectIndex(obj), QNN_SUBJECT_THUNDERBOLT, QNN_ACTION_FIRE, QNN_QUAL_NONE, QNN_MODALITY_VISUAL, 1.0f, 0.0f);
		return;
	}
}

static float QNN_FragFraction(int entity_frags)
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
	return QNN_Clamp((float)entity_frags / (float)max_frags, 0.0f, 1.0f);
}

static void QNN_UpdateFromVisibleEntities(const qnn_snapshot_t *snapshot)
{
	int i;

	for (i = 0; i < snapshot->visible_count; ++i)
	{
		int subject_id;
		int qualifier_id;
		float magnitude;

		if (!QNN_ClassifyVisibleSubject(&snapshot->visible[i], &subject_id, &qualifier_id, &magnitude))
			continue;

		if (QNN_SubjectIsItem(subject_id))
			continue;
		if (!QNN_InFov(snapshot->player_origin, snapshot->player_view_angles, snapshot->visible[i].origin))
			continue;

		if (snapshot->visible[i].entity_num > 0)
		{
			qnn_semantic_object_t *obj;

			obj = QNN_EnsureDynamicObject(
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
				obj->magnitude = QNN_FragFraction(snapshot->visible[i].frags);
				obj->state = 0.0f; /* enemy=0, ally=1 in team modes */
				QNN_MaybeAppendVisualFire(&snapshot->visible[i], obj);
			}
		}
	}
}

static int QNN_TraceContents(const vec3_t point)
{
	mleaf_t *leaf;

	if (cl.worldmodel == NULL)
		return CONTENTS_EMPTY;
	leaf = Mod_PointInLeaf((float *)point, cl.worldmodel);
	if (leaf == NULL)
		return CONTENTS_EMPTY;
	return leaf->contents;
}

static float QNN_TraceLineDistance(const vec3_t start, const vec3_t end, vec3_t impact)
{
	trace_t trace;
	vec3_t delta;

	if (!cl.worldmodel)
	{
		VectorCopy(end, impact);
		VectorSubtract(end, start, delta);
		return QNN_VecLength(delta);
	}
	memset(&trace, 0, sizeof(trace));
	SV_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1, (float *)start, (float *)end, &trace);
	VectorCopy(trace.endpos, impact);
	VectorSubtract(trace.endpos, start, delta);
	return QNN_VecLength(delta);
}

static void QNN_SpatialReset(qnn_spatial_token_t *token, int sector_id)
{
	memset(token, 0, sizeof(*token));
	token->sector_id = sector_id;
}

static void QNN_SpatialFinalize(qnn_spatial_token_t *token, int samples, float max_dist)
{
	if (samples <= 0)
		return;
	token->mean_dist /= (float)samples;
	token->openness = QNN_Clamp(token->mean_dist / max_dist, 0.0f, 1.0f);
	token->solid_frac /= (float)samples;
	token->water_frac /= (float)samples;
	token->slime_frac /= (float)samples;
	token->lava_frac /= (float)samples;
	token->traversable /= (float)samples;
	token->dropoff /= (float)samples;
	token->clearance /= (float)samples;
}

static void QNN_SpatialSampleRay(qnn_spatial_token_t *token, const vec3_t start, const vec3_t dir, float max_dist)
{
	vec3_t end;
	vec3_t impact;
	vec3_t impact_probe;
	float dist;
	int contents;

	VectorMA(start, max_dist, dir, end);
	dist = QNN_TraceLineDistance(start, end, impact);
	token->mean_dist += dist;
	if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
		token->nearest_dist = dist;

	VectorCopy(impact, impact_probe);
	VectorMA(impact_probe, -1.0f, dir, impact_probe);
	contents = QNN_TraceContents(impact_probe);
	if (dist < max_dist - 1.0f)
		token->solid_frac += 1.0f;
	if (contents == CONTENTS_WATER)
		token->water_frac += 1.0f;
	else if (contents == CONTENTS_SLIME)
		token->slime_frac += 1.0f;
	else if (contents == CONTENTS_LAVA)
		token->lava_frac += 1.0f;
}

static void QNN_BuildHorizontalSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token, float center_deg, float span_deg)
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
		QNN_SpatialSampleRay(token, snapshot->player_origin, dir, max_dist);

		if (i == 2)
		{
			VectorMA(snapshot->player_origin, 64.0f, dir, end);
			clear_dist = QNN_TraceLineDistance(snapshot->player_origin, end, impact);
			token->clearance += QNN_Clamp(clear_dist / 64.0f, 0.0f, 1.0f);

			VectorCopy(impact, down_start);
			down_start[2] += 24.0f;
			VectorCopy(impact, down_end);
			down_end[2] -= 64.0f;
			ground_dist = QNN_TraceLineDistance(down_start, down_end, down_impact);
			token->traversable += (clear_dist > 56.0f && ground_dist <= 40.0f) ? 1.0f : 0.0f;
			token->dropoff += QNN_Clamp((ground_dist - 18.0f) / 46.0f, 0.0f, 1.0f);
		}
		else
		{
			token->clearance += 0.0f;
			token->traversable += 0.0f;
			token->dropoff += 0.0f;
		}
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

static void QNN_BuildGroundSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token)
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
		dist = QNN_TraceLineDistance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = QNN_TraceContents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist <= 24.0f ? 1.0f : 0.0f;
		token->dropoff += QNN_Clamp((dist - 18.0f) / 48.0f, 0.0f, 1.0f);
		token->clearance += QNN_Clamp(1.0f - (dist / max_dist), 0.0f, 1.0f);
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

static void QNN_BuildCeilingSpatial(const qnn_snapshot_t *snapshot, qnn_spatial_token_t *token)
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
		dist = QNN_TraceLineDistance(start, end, impact);
		token->mean_dist += dist;
		if (token->nearest_dist == 0.0f || dist < token->nearest_dist)
			token->nearest_dist = dist;
		contents = QNN_TraceContents(impact);
		if (dist < max_dist - 1.0f)
			token->solid_frac += 1.0f;
		if (contents == CONTENTS_WATER)
			token->water_frac += 1.0f;
		else if (contents == CONTENTS_SLIME)
			token->slime_frac += 1.0f;
		else if (contents == CONTENTS_LAVA)
			token->lava_frac += 1.0f;
		token->traversable += dist >= 56.0f ? 1.0f : 0.0f;
		token->clearance += QNN_Clamp(dist / max_dist, 0.0f, 1.0f);
	}
	QNN_SpatialFinalize(token, samples, max_dist);
}

static void QNN_BuildSpatialTokens(const qnn_snapshot_t *snapshot, qnn_spatial_token_t tokens[QNN_SPATIAL_TOKEN_COUNT])
{
	int i;

	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i)
		QNN_SpatialReset(&tokens[i], i);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_CENTER], 0.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_LEFT], 40.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FOV_RIGHT], -40.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FLANK_LEFT], 90.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_FLANK_RIGHT], -90.0f, 40.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_REAR_LEFT], 150.0f, 30.0f);
	QNN_BuildHorizontalSpatial(snapshot, &tokens[QNN_SPATIAL_REAR_RIGHT], -150.0f, 30.0f);
	QNN_BuildGroundSpatial(snapshot, &tokens[QNN_SPATIAL_GROUND]);
	QNN_BuildCeilingSpatial(snapshot, &tokens[QNN_SPATIAL_CEILING]);
}

void QNN_SemanticReset(const qnn_map_state_t *map_state)
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
	memset(qnn_event_head, -1, sizeof(qnn_event_head));
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

		if (!QNN_ClassifyStaticSubject(&map_state->static_objects[i], &subject_id, &qualifier_id, &magnitude, &is_item, &respawn_s))
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

void QNN_SemanticUpdate(const qnn_map_state_t *map_state, const qnn_snapshot_t *snapshot, float dt, qboolean reset_flag)
{
	(void)map_state;
	if (qnn_semantic_objects == NULL || qnn_semantic_object_capacity <= 0)
		return;
	if (reset_flag)
	{
		int i;

		for (i = 0; i < QNN_MAX_EVENT_ATOMS; ++i)
			memset(&qnn_semantic_events[i], 0, sizeof(qnn_semantic_events[i]));
		for (i = qnn_semantic_static_count; i < qnn_semantic_object_capacity; ++i)
			memset(&qnn_semantic_objects[i], 0, sizeof(qnn_semantic_objects[i]));
	}

	QNN_DecayStore(dt);
	QNN_UpdateItemsFromVisibility(snapshot);
	QNN_AdvanceItemTimers(dt);
	QNN_UpdateFromVisibleEntities(snapshot);
	QNN_UpdateFromSounds(snapshot);

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
static void QNN_AggregateNailStreams(
	qnn_semantic_object_t **object_rows,
	int *object_count,
	qnn_semantic_object_t *stream_copies,
	int *stream_copy_count,
	const vec3_t player_origin)
{
	int nail_indices[QNN_MAX_TOKEN_OBJECTS];
	qboolean absorbed[QNN_MAX_TOKEN_OBJECTS];
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

		leader_speed = QNN_VecLength(object_rows[ni]->velocity);
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
			speed = QNN_VecLength(object_rows[nj]->velocity);
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
			int leader_owner = QNN_ObjectIndex(object_rows[leader_idx]);

			/* create a copy of the leader with stream magnitude */
			stream_copies[streams] = *object_rows[leader_idx];
			stream_copies[streams].recency = max_recency;
			stream_copies[streams].magnitude = 0.0f;
			/* reassign the slot to point at our copy */
			object_rows[leader_idx] = &stream_copies[streams];
			absorbed[leader_idx] = false; /* keep the leader */
			streams += 1;

			/* reassign events from absorbed nails to leader */
			for (j = 0; j < QNN_MAX_EVENT_ATOMS; ++j)
			{
				int k;

				if (!qnn_semantic_events[j].active)
					continue;
				for (k = i + 1; k < nail_count; ++k)
				{
					int nk = nail_indices[k];
					if (!absorbed[nk])
						continue;
					if (qnn_semantic_events[j].owner_index == QNN_ObjectIndex(object_rows[nk]))
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
		if (absorbed[ni] || QNN_VecLength(object_rows[ni]->velocity) >= 1.0f)
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

void QNN_WriteTokenStepBinary(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag)
{
	/* Collection buffer large enough for all possible eligible objects.
	 * After priority sort, only the top QNN_MAX_TOKEN_OBJECTS survive. */
	qnn_semantic_object_t *candidate_rows[MAX_EDICTS + 512];
	qnn_spatial_token_t spatial_tokens[QNN_SPATIAL_TOKEN_COUNT];
	qnn_semantic_object_t *object_rows[QNN_MAX_TOKEN_OBJECTS];
	qnn_semantic_event_atom_t *event_rows[QNN_MAX_EVENT_ATOMS];
	qnn_semantic_object_t stream_copies[QNN_MAX_NAIL_STREAMS];
	uint16_t event_base[QNN_MAX_TOKEN_OBJECTS];
	uint16_t event_count[QNN_MAX_TOKEN_OBJECTS];
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
	if (QNN_ActionHasSignal(&snapshot->action_label))
		flags |= QNN_TOKEN_FLAG_HAS_ACTION;

	/* first pass: collect all eligible objects */
	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		int has_event;

		if (!qnn_semantic_objects[i].active)
			continue;
		has_event = 0;
		for (j = 0; j < QNN_MAX_EVENT_ATOMS; ++j)
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
		qsort(candidate_rows, (size_t)candidate_count, sizeof(candidate_rows[0]), QNN_ObjectRowCompare);

	/* truncate to emission budget */
	object_count = candidate_count < QNN_MAX_TOKEN_OBJECTS ? candidate_count : QNN_MAX_TOKEN_OBJECTS;
	for (i = 0; i < object_count; ++i)
		object_rows[i] = candidate_rows[i];

	/* aggregate nail projectiles into stream tokens */
	QNN_AggregateNailStreams(object_rows, &object_count, stream_copies,
		&stream_copy_count, snapshot->player_origin);

	/* second pass: collect events for surviving objects */
	for (i = 0; i < object_count; ++i)
	{
		int oi = QNN_ObjectIndex(object_rows[i]);
		event_base[i] = (uint16_t)event_total;
		event_count[i] = 0;
		for (j = 0; j < QNN_MAX_EVENT_ATOMS && event_total < QNN_MAX_EVENT_ATOMS; ++j)
		{
			if (!qnn_semantic_events[j].active || qnn_semantic_events[j].owner_index != oi)
				continue;
			event_rows[event_total] = &qnn_semantic_events[j];
			event_total += 1;
			event_count[i] += 1;
		}
	}

	QNN_BuildSpatialTokens(snapshot, spatial_tokens);

	/* Resolve the player's nav area/cluster once for self token and route computation. */
	oracle = qnn_map_state.nav_oracle;
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
	QNN_WriteU16LE(out, (uint16_t)QNN_TOKEN_VERSION);
	QNN_WriteU16LE(out, flags);
	QNN_WriteU32LE(out, (uint32_t)tick);
	QNN_WriteU32LE(out, (uint32_t)steps);
	QNN_WriteI32LE(out, snapshot->current_region_id);
	QNN_WriteU16LE(out, (uint16_t)object_count);
	QNN_WriteU16LE(out, (uint16_t)event_total);
	QNN_WriteU16LE(out, (uint16_t)QNN_SPATIAL_TOKEN_COUNT);
	QNN_WriteU16LE(out, (uint16_t)tick_hz);

	QNN_WriteSelfToken(out, snapshot, tick_hz, player_cluster_id);

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
					QNN_RelativeFrame(snapshot->player_view_angles, path_rel, rel);
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
			QNN_RelativeFrame(snapshot->player_view_angles, delta, rel);
		}

		{
			uint32_t wire_handle;
			if (object_rows[i] >= qnn_semantic_objects
				&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
				wire_handle = (uint32_t)QNN_ObjectIndex(object_rows[i]);
			else
				wire_handle = (uint32_t)object_rows[i]->entity_num;
			QNN_WriteObjectToken(out, object_rows[i], rel, event_count[i], event_base[i], wire_handle);
		}
	}

	/* Save index mapping for next tick's recall targets.
	 * Stream copies (nail aggregation) are stack-local and not valid recall
	 * targets, so store -1 for those — recall will skip them. */
	for (i = 0; i < object_count && i < QNN_MAX_TOKEN_OBJECTS; ++i)
	{
		if (object_rows[i] >= qnn_semantic_objects
			&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
			qnn_prev_object_indices[i] = QNN_ObjectIndex(object_rows[i]);
		else
			qnn_prev_object_indices[i] = -1;
	}
	qnn_prev_object_count = object_count;

	for (i = 0; i < event_total; ++i)
	{
		QNN_WriteU16LE(out, (uint16_t)event_rows[i]->subject_id);
		QNN_WriteU16LE(out, (uint16_t)event_rows[i]->action_id);
		QNN_WriteU16LE(out, (uint16_t)event_rows[i]->qualifier_id);
		QNN_WriteU16LE(out, (uint16_t)event_rows[i]->modality_id);
		QNN_WriteF32LE(out, event_rows[i]->recency);
		QNN_WriteF32LE(out, event_rows[i]->confidence);
		QNN_WriteF32LE(out, event_rows[i]->magnitude);
	}

	for (i = 0; i < QNN_SPATIAL_TOKEN_COUNT; ++i)
		QNN_WriteSpatialToken(out, &spatial_tokens[i]);

	if (QNN_ActionHasSignal(&snapshot->action_label))
	{
		QNN_WriteF32LE(out, snapshot->action_label.move[0]);
		QNN_WriteF32LE(out, snapshot->action_label.move[1]);
		QNN_WriteF32LE(out, snapshot->action_label.look[0]);
		QNN_WriteF32LE(out, snapshot->action_label.look[1]);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.fire);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.jump);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.switch_slot);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.recall[0]);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.recall[1]);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.recall[2]);
		QNN_WriteU16LE(out, (uint16_t)snapshot->action_label.recall[3]);
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

static void QNN_ObsResetActionHistory(void)
{
	memset(qnn_action_history, 0, sizeof(qnn_action_history));
	qnn_action_history_count = 0;
}

static void QNN_ObsPushAction(const qnn_action_t *action)
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

/* QNN_Clamp is a macro in qnn.h */

/* Helper: write a float32 into a byte buffer at an offset (little-endian) */
static void QNN_BufWriteF32(uint8_t *buf, int offset, float value)
{
	union { float f; uint32_t u; } bits;
	bits.f = value;
	buf[offset + 0] = (uint8_t)(bits.u & 0xffu);
	buf[offset + 1] = (uint8_t)((bits.u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((bits.u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((bits.u >> 24) & 0xffu);
}

static void QNN_BufWriteI32(uint8_t *buf, int offset, int32_t value)
{
	uint32_t u = (uint32_t)value;
	buf[offset + 0] = (uint8_t)(u & 0xffu);
	buf[offset + 1] = (uint8_t)((u >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((u >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((u >> 24) & 0xffu);
}

static void QNN_BufWriteU32(uint8_t *buf, int offset, uint32_t value)
{
	buf[offset + 0] = (uint8_t)(value & 0xffu);
	buf[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
	buf[offset + 2] = (uint8_t)((value >> 16) & 0xffu);
	buf[offset + 3] = (uint8_t)((value >> 24) & 0xffu);
}

void QNN_PackObsBuffer(uint8_t *obs, const qnn_snapshot_t *snapshot, int tick_hz, qboolean reset_flag)
{

	qnn_semantic_object_t *candidate_rows[MAX_EDICTS + 512];
	qnn_spatial_token_t spatial_tokens[QNN_SPATIAL_TOKEN_COUNT];
	qnn_semantic_object_t *object_rows[QNN_MAX_TOKEN_OBJECTS];
	qnn_semantic_event_atom_t *event_rows[QNN_MAX_EVENT_ATOMS];
	qnn_semantic_object_t stream_copies[QNN_MAX_NAIL_STREAMS];
	uint16_t ev_base[QNN_MAX_TOKEN_OBJECTS];
	uint16_t ev_count[QNN_MAX_TOKEN_OBJECTS];
	int stream_copy_count;
	int candidate_count;
	int object_count;
	int event_total;
	int i, j;
	const qnn_nav_oracle_runtime_t *oracle;
	int player_area_id;
	int player_cluster_id;
	qboolean has_action;
	vec3_t object_rel[QNN_MAX_TOKEN_OBJECTS];

	/* Zero both buffers */
	memset(obs, 0, QNN_OBS_BUFFER_SIZE);
	/* obs buffer doesn't use tick/steps — those go in the framing header */

	if (reset_flag)
		QNN_ObsResetActionHistory();

	/* ---- Build phase (same as QNN_WriteTokenStepBinary) ---- */

	candidate_count = 0;
	object_count = 0;
	event_total = 0;

	has_action = QNN_ActionHasSignal(&snapshot->action_label);

	/* Collect eligible objects — use per-owner event heads for O(1) has_event check */
	for (i = 0; i < qnn_semantic_object_capacity; ++i)
	{
		int has_event;
		if (!qnn_semantic_objects[i].active)
			continue;
		has_event = (i < QNN_EVENT_HEAD_CAPACITY && qnn_event_head[i] >= 0) ? 1 : 0;
		if (qnn_semantic_objects[i].recency <= 0.0f && !has_event)
			continue;
		if (candidate_count < (int)(sizeof(candidate_rows) / sizeof(candidate_rows[0])))
			candidate_rows[candidate_count++] = &qnn_semantic_objects[i];
	}

	if (candidate_count > 1)
		qsort(candidate_rows, (size_t)candidate_count, sizeof(candidate_rows[0]), QNN_ObjectRowCompare);

	object_count = candidate_count < QNN_MAX_TOKEN_OBJECTS ? candidate_count : QNN_MAX_TOKEN_OBJECTS;
	for (i = 0; i < object_count; ++i)
		object_rows[i] = candidate_rows[i];

	QNN_AggregateNailStreams(object_rows, &object_count, stream_copies,
		&stream_copy_count, snapshot->player_origin);

	/* Collect events for surviving objects — walk per-owner linked lists */
	for (i = 0; i < object_count; ++i)
	{
		int oi = QNN_ObjectIndex(object_rows[i]);
		ev_base[i] = (uint16_t)event_total;
		ev_count[i] = 0;
		if (oi >= 0 && oi < QNN_EVENT_HEAD_CAPACITY)
		{
			int ei = qnn_event_head[oi];
			while (ei >= 0 && event_total < QNN_MAX_EVENT_ATOMS)
			{
				if (qnn_semantic_events[ei].active)
				{
					event_rows[event_total++] = &qnn_semantic_events[ei];
					ev_count[i]++;
				}
				ei = qnn_semantic_events[ei].next_for_owner;
			}
		}
	}

	QNN_BuildSpatialTokens(snapshot, spatial_tokens);

	/* Nav oracle */
	oracle = qnn_map_state.nav_oracle;
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
					QNN_RelativeFrame(snapshot->player_view_angles, path_rel, object_rel[i]);
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
			QNN_RelativeFrame(snapshot->player_view_angles, delta, object_rel[i]);
		}
	}

	/* Update recall mapping */
	for (i = 0; i < object_count && i < QNN_MAX_TOKEN_OBJECTS; ++i)
	{
		if (object_rows[i] >= qnn_semantic_objects
			&& object_rows[i] < qnn_semantic_objects + qnn_semantic_object_capacity)
			qnn_prev_object_indices[i] = QNN_ObjectIndex(object_rows[i]);
		else
			qnn_prev_object_indices[i] = -1;
	}
	qnn_prev_object_count = object_count;

	/* ---- Pack observation buffer ---- */

	/* Self scalars [23] */
	{
		float scalars[QNN_OBS_SELF_SCALAR_DIM];
		scalars[0] = QNN_Normalize((float)snapshot->health, QNN_SELF_HEALTH_CAP);
		scalars[1] = QNN_Normalize((float)snapshot->armor, QNN_SELF_ARMOR_CAP);
		scalars[2] = QNN_Normalize(snapshot->armor_type, QNN_SELF_ARMOR_TYPE_CAP);
		scalars[3] = (snapshot->weapons_owned & IT_SHOTGUN) ? 1.0f : 0.0f;
		scalars[4] = (snapshot->weapons_owned & IT_SUPER_SHOTGUN) ? 1.0f : 0.0f;
		scalars[5] = (snapshot->weapons_owned & IT_NAILGUN) ? 1.0f : 0.0f;
		scalars[6] = (snapshot->weapons_owned & IT_SUPER_NAILGUN) ? 1.0f : 0.0f;
		scalars[7] = (snapshot->weapons_owned & IT_GRENADE_LAUNCHER) ? 1.0f : 0.0f;
		scalars[8] = (snapshot->weapons_owned & IT_ROCKET_LAUNCHER) ? 1.0f : 0.0f;
		scalars[9] = (snapshot->weapons_owned & IT_LIGHTNING) ? 1.0f : 0.0f;
		scalars[10] = (float)QNN_SelfWeaponSuper(snapshot->weapon_id);
		scalars[11] = QNN_Normalize((float)snapshot->ammo_shells, QNN_SELF_SHELLS_CAP);
		scalars[12] = QNN_Normalize((float)snapshot->ammo_nails, QNN_SELF_NAILS_CAP);
		scalars[13] = QNN_Normalize((float)snapshot->ammo_rockets, QNN_SELF_ROCKETS_CAP);
		scalars[14] = QNN_Normalize((float)snapshot->ammo_cells, QNN_SELF_CELLS_CAP);
		scalars[15] = QNN_Normalize(snapshot->player_velocity[0], QNN_SELF_VELOCITY_CAP);
		scalars[16] = QNN_Normalize(snapshot->player_velocity[1], QNN_SELF_VELOCITY_CAP);
		scalars[17] = QNN_Normalize(snapshot->player_velocity[2], QNN_SELF_VELOCITY_CAP);
		scalars[18] = QNN_AngleSinDeg(snapshot->player_view_angles[1]);
		scalars[19] = QNN_AngleCosDeg(snapshot->player_view_angles[1]);
		scalars[20] = QNN_AngleSinDeg(snapshot->player_view_angles[0]);
		scalars[21] = QNN_AngleCosDeg(snapshot->player_view_angles[0]);
		scalars[22] = tick_hz > 0 ? (1.0f / (float)tick_hz) : 0.0f;
		for (i = 0; i < QNN_OBS_SELF_SCALAR_DIM; ++i)
			QNN_BufWriteF32(obs, QNN_OBS_OFF_SELF_SCALARS + i * 4, scalars[i]);
	}

	/* Self IDs */
	QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_WEAPON_ID, QNN_SelfWeaponEmbedId(snapshot->weapon_id));
	QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_MOVEMENT_ID, QNN_SelfMovementId(snapshot->grounded, snapshot->waterlevel));
	QNN_BufWriteI32(obs, QNN_OBS_OFF_SELF_CLUSTER_ID, player_cluster_id);

	/* Objects [64, 5] ids + [64, 8] scalars + [64] mask + [64, 8] route_cluster_ids */
	for (i = 0; i < object_count; ++i)
	{
		int ids_off = QNN_OBS_OFF_OBJECT_IDS + i * QNN_OBS_OBJECT_ID_DIM * 4;
		int sc_off = QNN_OBS_OFF_OBJECT_SCALARS + i * QNN_OBS_OBJECT_SCALAR_DIM * 4;
		int rc_off = QNN_OBS_OFF_OBJECT_ROUTE_IDS + i * QNN_OBS_MAX_ROUTE_CLUSTERS * 4;

		QNN_BufWriteI32(obs, ids_off + 0, object_rows[i]->subject_id);
		QNN_BufWriteI32(obs, ids_off + 4, object_rows[i]->qualifier_id);
		QNN_BufWriteI32(obs, ids_off + 8, object_rows[i]->modality_id);
		QNN_BufWriteI32(obs, ids_off + 12, object_rows[i]->player_id);
		QNN_BufWriteI32(obs, ids_off + 16, object_rows[i]->cluster_id);

		QNN_BufWriteF32(obs, sc_off + 0, QNN_Normalize(object_rel[i][0], QNN_OBJECT_REL_SCALE));
		QNN_BufWriteF32(obs, sc_off + 4, QNN_Normalize(object_rel[i][1], QNN_OBJECT_REL_SCALE));
		QNN_BufWriteF32(obs, sc_off + 8, QNN_Normalize(object_rel[i][2], QNN_OBJECT_REL_SCALE));
		QNN_BufWriteF32(obs, sc_off + 12, QNN_Normalize(object_rows[i]->route_cost, QNN_OBJECT_ROUTE_COST_SCALE));
		QNN_BufWriteF32(obs, sc_off + 16, object_rows[i]->recency);
		QNN_BufWriteF32(obs, sc_off + 20, object_rows[i]->confidence);
		QNN_BufWriteF32(obs, sc_off + 24, object_rows[i]->magnitude);
		QNN_BufWriteF32(obs, sc_off + 28, object_rows[i]->state);
		QNN_BufWriteF32(obs, sc_off + 32, object_rows[i]->half_extents[0] / QNN_OBJECT_REL_SCALE);
		QNN_BufWriteF32(obs, sc_off + 36, object_rows[i]->half_extents[1] / QNN_OBJECT_REL_SCALE);
		QNN_BufWriteF32(obs, sc_off + 40, object_rows[i]->half_extents[2] / QNN_OBJECT_REL_SCALE);

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
			QNN_BufWriteF32(obs, sc_off + 44, QNN_LookAxisFromMouseCount(yaw_counts));
			QNN_BufWriteF32(obs, sc_off + 48, QNN_LookAxisFromMouseCount(pitch_counts));
		}

		obs[QNN_OBS_OFF_OBJECT_MASK + i] = 1;

		for (j = 0; j < QNN_OBS_MAX_ROUTE_CLUSTERS; ++j)
		{
			int rc_val = (j < object_rows[i]->route_cluster_count) ? object_rows[i]->route_cluster_ids[j] : 0;
			QNN_BufWriteI32(obs, rc_off + j * 4, rc_val);
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

			QNN_BufWriteI32(obs, eid_off + 0, event_rows[ei]->subject_id);
			QNN_BufWriteI32(obs, eid_off + 4, event_rows[ei]->action_id);
			QNN_BufWriteI32(obs, eid_off + 8, event_rows[ei]->qualifier_id);
			QNN_BufWriteI32(obs, eid_off + 12, event_rows[ei]->modality_id);

			QNN_BufWriteF32(obs, esc_off + 0, QNN_Clamp(event_rows[ei]->recency, 0.0f, 1.0f));
			QNN_BufWriteF32(obs, esc_off + 4, QNN_Clamp(event_rows[ei]->confidence, 0.0f, 1.0f));
			QNN_BufWriteF32(obs, esc_off + 8, QNN_Clamp(event_rows[ei]->magnitude, 0.0f, 1.0f));

			QNN_BufWriteI32(obs, QNN_OBS_OFF_EVENT_OWNER + ei * 4, i);  /* owner = object slot index */
			obs[QNN_OBS_OFF_EVENT_MASK + ei] = 1;
		}
	}

	/* Spatial [9] ids + [9, 10] scalars */
	for (i = 0; i < QNN_OBS_SPATIAL_COUNT; ++i)
	{
		int sid_off = QNN_OBS_OFF_SPATIAL_IDS + i * 4;
		int ssc_off = QNN_OBS_OFF_SPATIAL_SCALARS + i * QNN_OBS_SPATIAL_SCALAR_DIM * 4;

		QNN_BufWriteI32(obs, sid_off, spatial_tokens[i].sector_id);
		QNN_BufWriteF32(obs, ssc_off + 0, QNN_Normalize(spatial_tokens[i].nearest_dist, QNN_SPATIAL_DIST_SCALE));
		QNN_BufWriteF32(obs, ssc_off + 4, QNN_Normalize(spatial_tokens[i].mean_dist, QNN_SPATIAL_DIST_SCALE));
		QNN_BufWriteF32(obs, ssc_off + 8, spatial_tokens[i].openness);
		QNN_BufWriteF32(obs, ssc_off + 12, spatial_tokens[i].clearance);
		QNN_BufWriteF32(obs, ssc_off + 16, spatial_tokens[i].traversable);
		QNN_BufWriteF32(obs, ssc_off + 20, spatial_tokens[i].dropoff);
		QNN_BufWriteF32(obs, ssc_off + 24, spatial_tokens[i].solid_frac);
		QNN_BufWriteF32(obs, ssc_off + 28, spatial_tokens[i].water_frac);
		QNN_BufWriteF32(obs, ssc_off + 32, spatial_tokens[i].slime_frac);
		QNN_BufWriteF32(obs, ssc_off + 36, spatial_tokens[i].lava_frac);
	}

	/* Action history [8, 7] — pack current history BEFORE updating with this tick's action */
	{
		int ah_off = QNN_OBS_OFF_ACTION_HISTORY;
		int n = qnn_action_history_count < QNN_OBS_ACTION_HISTORY_LEN
			? qnn_action_history_count : QNN_OBS_ACTION_HISTORY_LEN;
		for (i = 0; i < n; ++i)
			for (j = 0; j < QNN_OBS_ACTION_HISTORY_DIM; ++j)
				QNN_BufWriteF32(obs, ah_off + (i * QNN_OBS_ACTION_HISTORY_DIM + j) * 4,
					qnn_action_history[i][j]);
	}

	/* Now update action history for next tick (matches Python: encode first, then push) */
	if (has_action && !reset_flag)
		QNN_ObsPushAction(&snapshot->action_label);

}

void QNN_WriteObsBuffer(FILE *out, const qnn_snapshot_t *snapshot, int tick, int steps, int tick_hz, qboolean reset_flag)
{
	static uint8_t obs[QNN_OBS_BUFFER_SIZE];
	(void)tick; (void)steps;
	QNN_PackObsBuffer(obs, snapshot, tick_hz, reset_flag);
	fwrite(obs, 1, QNN_OBS_BUFFER_SIZE, out);
	fflush(out);
}
