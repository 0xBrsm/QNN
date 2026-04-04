/*
 * qnn_entity.c — Visual perception: entity classification, PVS item
 * matching, team detection.
 *
 * Split from qnn_object.c — contains model/classname classification,
 * FOV checks, item-from-visibility matching, known-entity updates,
 * and team/frag helpers.
 */

#include "qnn_object.h"
#include "qnn_map.h"
#include "qnn_io.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ══════════════════════════════════════════════════════════════════
 * Model name -> subject classification lookup table
 * ══════════════════════════════════════════════════════════════════ */

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

/* ══════════════════════════════════════════════════════════════════
 * Static helpers
 * ══════════════════════════════════════════════════════════════════ */

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

/* ══════════════════════════════════════════════════════════════════
 * Classification functions
 * ══════════════════════════════════════════════════════════════════ */

static float QNN_ItemRespawnSFromClassname(const char *classname, int subject_id)
{
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

float QNN_ItemRespawnS(const qnn_static_object_t *obj, int subject_id)
{
	return QNN_ItemRespawnSFromClassname(obj->classname, subject_id);
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
		*magnitude = (100.0f * 0.3f) / QNN_SELF_HEALTH_CAP; /* effective armor, same scale as self token */
		return true;
	}
	if (!strcasecmp(classname, "item_armor2"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		*magnitude = (150.0f * 0.6f) / QNN_SELF_HEALTH_CAP; /* effective armor */
		return true;
	}
	if (!strcasecmp(classname, "item_armorInv"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_RED;
		*magnitude = (200.0f * 0.8f) / QNN_SELF_HEALTH_CAP; /* effective armor */
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

qboolean QNN_ClassifyStaticSubject(const qnn_static_object_t *obj, int *subject_id, int *qualifier_id, float *magnitude, qboolean *is_item, float *respawn_s)
{
	int spawnflags;

	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;
	*is_item = false;
	*respawn_s = 0.0f;

	spawnflags = QNN_ObjectStaticPropertyInt(obj, "spawnflags", 0);
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

/* Classify a client entity by model name and skin.  Client-only — no
   server edict data needed. */
qboolean QNN_ClassifyByModel(const char *model_name, int skin, int *subject_id, int *qualifier_id, float *magnitude)
{
	const qnn_model_classify_t *row;

	*subject_id = QNN_SUBJECT_NONE;
	*qualifier_id = QNN_QUAL_NONE;
	*magnitude = 0.0f;

	if (model_name == NULL || model_name[0] == '\0')
		return false;

	/* Table lookup by model name (covers player, projectiles, items, etc.) */
	for (row = qnn_model_table; row->model; ++row)
	{
		if (!strcmp(model_name, row->model))
		{
			*subject_id = row->subject_id;
			*qualifier_id = row->qualifier_id;
			*magnitude = row->mag_scale > 0 ? QNN_Normalize(row->mag_value, row->mag_scale) : row->mag_value;
			return true;
		}
	}

	/* Armor: subject depends on skin */
	if (!strcmp(model_name, "progs/armor.mdl"))
	{
		if (skin <= 0)      { *subject_id = QNN_SUBJECT_ARMOR_GREEN;  *magnitude = (100.0f * 0.3f) / QNN_SELF_HEALTH_CAP; }
		else if (skin == 1) { *subject_id = QNN_SUBJECT_ARMOR_YELLOW; *magnitude = (150.0f * 0.6f) / QNN_SELF_HEALTH_CAP; }
		else                { *subject_id = QNN_SUBJECT_ARMOR_RED;    *magnitude = (200.0f * 0.8f) / QNN_SELF_HEALTH_CAP; }
		return true;
	}

	return false;
}

/* Legacy wrapper — still used by qnn_object.h declaration */
qboolean QNN_ClassifyKnownSubject(const qnn_known_entity_t *ent, int *subject_id, int *qualifier_id, float *magnitude)
{
	return QNN_ClassifyByModel(ent->model_name, ent->skin, subject_id, qualifier_id, magnitude);
}

int QNN_SubjectPickupCategory(int subject_id)
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

#define QNN_VIEW_OFS_Z 22.0f

qboolean QNN_InFov(const vec3_t player_origin, const vec3_t view_angles, const vec3_t target)
{
	vec3_t forward;
	vec3_t right;
	vec3_t up;
	vec3_t delta;
	vec3_t eye;
	float dist;
	float dot;
	float cos_half;
	trace_t trace;

	AngleVectors(view_angles, forward, right, up);
	VectorSubtract(target, player_origin, delta);
	dist = QNN_VecLength(delta);
	if (dist < 1.0f)
		return true;
	dot = DotProduct(forward, delta) / dist;
	cos_half = 0.5f; /* cos(60°) */
	if (dot < cos_half)
		return false;

	if (!cl.worldmodel)
		return true;

	VectorCopy(player_origin, eye);
	eye[2] += QNN_VIEW_OFS_Z;
	memset(&trace, 0, sizeof(trace));
	SV_RecursiveHullCheck(cl.worldmodel->hulls, 0, 0, 1, (float *)eye, (float *)target, &trace);
	return trace.fraction >= 1.0f;
}

/* ══════════════════════════════════════════════════════════════════
 * Team detection and frag helpers
 * ══════════════════════════════════════════════════════════════════ */

/* Team detection: same pants color (bottom 4 bits of colors) = same team.
   Works for teamplay (teammates share pants color) and FFA (all different = all enemies).
   Matches engine logic: ent->v.team = (colors & 15) + 1  (host_cmd.c). */
/* Cached pants colors for all players.  Latched once when cl.scores first
   has nonzero entries, so that mid-demo level transitions (which zero
   cl.scores via CL_ClearState) don't flip the team signal. */
#define QNN_MAX_CACHED_PLAYERS 32
static int qnn_cached_pants[QNN_MAX_CACHED_PLAYERS];
static int qnn_cached_pants_count = 0;
static int qnn_self_pants_cached = -1;

static void QNN_LatchTeamColors(void)
{
	int i, self_slot;

	if (cl.scores == NULL || cl.viewentity <= 0)
		return;
	self_slot = cl.viewentity - 1;
	if (self_slot < 0 || self_slot >= cl.maxclients)
		return;
	/* Wait until the self slot has a nonzero color — scores allocated
	   but not yet populated means all zeros (indistinguishable). */
	if (cl.scores[self_slot].colors == 0)
		return;
	qnn_self_pants_cached = cl.scores[self_slot].colors & 15;
	qnn_cached_pants_count = cl.maxclients < QNN_MAX_CACHED_PLAYERS ? cl.maxclients : QNN_MAX_CACHED_PLAYERS;
	for (i = 0; i < qnn_cached_pants_count; ++i)
		qnn_cached_pants[i] = cl.scores[i].colors & 15;
}

float QNN_IsSameTeam(int entity_num)
{
	int other_slot;

	if (qnn_self_pants_cached < 0)
		QNN_LatchTeamColors();
	if (qnn_self_pants_cached < 0)
		return 0.0f;

	other_slot = entity_num - 1;
	if (other_slot < 0 || other_slot >= qnn_cached_pants_count)
		return 0.0f;
	return (qnn_self_pants_cached == qnn_cached_pants[other_slot]) ? 1.0f : 0.0f;
}

float QNN_FragFraction(int entity_frags)
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

void QNN_EntityResetTeamCache(void)
{
	qnn_self_pants_cached = -1;
	qnn_cached_pants_count = 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Category / classify helpers (moved from qnn_map.cpp)
 * ══════════════════════════════════════════════════════════════════ */

int QNN_CategoryOrder(const char *category)
{
	if (!strcmp(category, "spawn"))
		return 0;
	if (!strcmp(category, "goal"))
		return 1;
	if (!strcmp(category, "item"))
		return 2;
	if (!strcmp(category, "trigger"))
		return 3;
	if (!strcmp(category, "door"))
		return 4;
	if (!strcmp(category, "lift"))
		return 5;
	if (!strcmp(category, "mover"))
		return 6;
	if (!strcmp(category, "monster"))
		return 7;
	return 8;
}

const char *QNN_Classify(const char *classname)
{
	if (!strcasecmp(classname, "info_player_start")
		|| !strcasecmp(classname, "info_player_coop")
		|| !strcasecmp(classname, "info_player_deathmatch"))
		return "spawn";
	if (!strcasecmp(classname, "trigger_changelevel"))
		return "goal";
	if (!strncasecmp(classname, "item_", 5))
		return "item";
	if (!strncasecmp(classname, "trigger_", 8))
		return "trigger";
	if (!strncasecmp(classname, "func_door", 9))
		return "door";
	if (!strncasecmp(classname, "func_plat", 9)
		|| !strncasecmp(classname, "func_train", 10)
		|| !strncasecmp(classname, "func_button", 11))
		return "lift";
	if (!strncasecmp(classname, "func_", 5))
		return "mover";
	if (!strncasecmp(classname, "monster_", 8))
		return "monster";
	return "misc";
}

/* ══════════════════════════════════════════════════════════════════
 * Static entity classification (map load)
 *
 * Classifies raw entities (parsed by qnn_map.c) into semantic
 * qnn_static_entity_t records with subject_id, magnitude, respawn_s.
 * ══════════════════════════════════════════════════════════════════ */

int QNN_EntityClassifyStatic(const qnn_raw_entity_t *raw, int raw_count,
	qnn_static_entity_t *out, int max)
{
	int i;
	int count;

	count = 0;

	for (i = 0; i < raw_count && count < max; ++i)
	{
		const qnn_raw_entity_t *r = &raw[i];
		int subject_id, qualifier_id;
		float magnitude;
		qboolean is_item;
		float respawn_s;
		const char *category;

		/* Classify subject */
		subject_id = QNN_SUBJECT_NONE;
		qualifier_id = QNN_QUAL_NONE;
		magnitude = 0.0f;
		is_item = false;
		respawn_s = 0.0f;

		if (QNN_ClassifyItemSubject(r->classname, r->spawnflags, &subject_id, &magnitude))
		{
			is_item = true;
			respawn_s = QNN_ItemRespawnSFromClassname(r->classname, subject_id);
		}
		else if (!strncasecmp(r->classname, "func_door", 9))
			subject_id = QNN_SUBJECT_DOOR;
		else if (!strncasecmp(r->classname, "func_plat", 9))
			subject_id = QNN_SUBJECT_PLATFORM;
		else if (!strncasecmp(r->classname, "func_train", 10))
			subject_id = QNN_SUBJECT_TRAIN;
		else if (!strncasecmp(r->classname, "func_button", 11))
			subject_id = QNN_SUBJECT_BUTTON;
		else if (!strcasecmp(r->classname, "trigger_teleport")
			|| !strcasecmp(r->classname, "info_teleport_destination")
			|| !strcasecmp(r->classname, "misc_teleporttrain"))
			subject_id = QNN_SUBJECT_TELEPORTER;
		else if (!strcasecmp(r->classname, "trigger_push"))
			subject_id = QNN_SUBJECT_NONE; /* no subject, but route needs it */

		/* Assign category */
		category = QNN_Classify(r->classname);

		/* Populate output */
		memset(&out[count], 0, sizeof(out[count]));
		out[count].entity_num = r->entity_num;
		out[count].subject_id = subject_id;
		out[count].qualifier_id = qualifier_id;
		out[count].magnitude = magnitude;
		out[count].is_item = is_item;
		out[count].respawn_s = respawn_s;
		VectorCopy(r->origin, out[count].origin);
		VectorCopy(r->angles, out[count].angles);
		strncpy(out[count].classname, r->classname, sizeof(out[count].classname) - 1);
		strncpy(out[count].category, category, sizeof(out[count].category) - 1);
		out[count].property_count = r->property_count;
		if (r->property_count > 0)
			memcpy(out[count].properties, r->properties, (size_t)r->property_count * sizeof(qnn_property_t));
		count++;
	}
	return count;
}

/* ══════════════════════════════════════════════════════════════════
 * Runtime entity collector (per tick)
 *
 * Scans ALL live cl_entities with entity->model != NULL, regardless
 * of FOV.  FOV is computed and stored as a helper for emission
 * (modality selection), but does NOT filter non-item entities out.
 * Brush *N entities are flagged so object.c can match them back to
 * static world objects by entity_num.
 * ══════════════════════════════════════════════════════════════════ */

int QNN_EntityClassifyKnown(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int max_entities,
	qnn_pvs_item_t *out_pvs, int max_pvs, int *out_pvs_count)
{
	int entity_count = 0;
	int pvs_count = 0;
	int entity_num;
	float server_dt = (float)(cl.mtime[0] - cl.mtime[1]);
	if (server_dt < 0.001f || server_dt > 0.5f)
		server_dt = 1.0f / 20.0f; /* fallback if mtime is stale or bogus */

	for (entity_num = 1; entity_num < cl.num_entities; ++entity_num)
	{
		entity_t *entity;
		const char *model_name;
		int subject_id, qualifier_id;
		float magnitude;
		qboolean is_item, in_fov, is_brush;

		if (entity_num == cl.viewentity)
			continue;

		entity = &cl_entities[entity_num];
		if (entity->model == NULL)
			continue;

		model_name = entity->model->name;
		is_brush = (model_name[0] == '*') ? true : false;

		/* Brush models (*N) are movers — emit as entity updates so
		   object.c can refresh static world objects from live transport.
		   They don't classify through the model table. */
		if (is_brush)
		{
			subject_id = QNN_SUBJECT_NONE;
			qualifier_id = QNN_QUAL_NONE;
			magnitude = 0.0f;
		}
		else if (!QNN_ClassifyByModel(model_name, entity->skinnum, &subject_id, &qualifier_id, &magnitude))
			continue;

		/* Bodyque corpses use progs/player.mdl but are allocated above
		   the client range.  Skip them — they're static decoration. */
		if (subject_id == QNN_SUBJECT_PLAYER && entity_num > cl.maxclients)
			continue;

		is_item = (!is_brush && QNN_SubjectIsItem(subject_id));
		in_fov = QNN_InFov(snapshot->player_origin, snapshot->player_view_angles, entity->origin);

		if (is_item)
		{
			if (pvs_count < max_pvs)
			{
				out_pvs[pvs_count].entity_num = entity_num;
				out_pvs[pvs_count].subject_id = subject_id;
				VectorCopy(entity->origin, out_pvs[pvs_count].origin);
				out_pvs[pvs_count].magnitude = magnitude;
				out_pvs[pvs_count].in_fov = in_fov;
				pvs_count++;
			}
			continue;
		}

		/* Non-item entities: collect regardless of FOV.
		   FOV is stored for emission modality selection. */
		if (entity_num <= 0)
			continue;

		if (entity_count < max_entities)
		{
			qnn_entity_update_t *eu = &out_entities[entity_count];
			vec3_t delta;

			eu->entity_num = entity_num;
			eu->subject_id = subject_id;
			eu->qualifier_id = qualifier_id;
			eu->magnitude = magnitude;
			eu->is_brush = is_brush;
			VectorCopy(entity->origin, eu->origin);

			/* Velocity from message origin delta (client-side).
			   Clamp large deltas (teleports, respawns) to zero —
			   anything above 500u/frame is not real movement. */
			VectorSubtract(entity->msg_origins[0], entity->msg_origins[1], delta);
			if (DotProduct(delta, delta) > 500.0f * 500.0f)
			{
				eu->velocity[0] = 0.0f;
				eu->velocity[1] = 0.0f;
				eu->velocity[2] = 0.0f;
			}
			else
				VectorScale(delta, 1.0f / server_dt, eu->velocity);

			VectorCopy(entity->angles, eu->angles);
			eu->effects = entity->effects;

			/* Bounds from model (client-side) */
			{
				vec3_t bmins = {0,0,0}, bmaxs = {0,0,0};
				if (entity->model != NULL)
				{
					VectorCopy(entity->model->mins, bmins);
					VectorCopy(entity->model->maxs, bmaxs);
				}
				eu->origin[0] += (bmins[0] + bmaxs[0]) * 0.5f;
				eu->origin[1] += (bmins[1] + bmaxs[1]) * 0.5f;
				eu->origin[2] += (bmins[2] + bmaxs[2]) * 0.5f;
				eu->half_extents[0] = (bmaxs[0] - bmins[0]) * 0.5f;
				eu->half_extents[1] = (bmaxs[1] - bmins[1]) * 0.5f;
				eu->half_extents[2] = (bmaxs[2] - bmins[2]) * 0.5f;
			}

			/* Frags from client scoreboard (client-side) */
			eu->frags = 0;
			if (subject_id == QNN_SUBJECT_PLAYER && cl.scores != NULL
				&& (entity_num - 1) >= 0 && (entity_num - 1) < cl.maxclients)
				eu->frags = cl.scores[entity_num - 1].frags;

			eu->health = 0; /* not available client-side */
			eu->is_item = false;
			eu->in_fov = in_fov;
			entity_count++;
		}
	}

	*out_pvs_count = pvs_count;
	return entity_count;
}
