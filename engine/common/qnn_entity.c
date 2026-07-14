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
#include "qnn_store.h"

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
	{"progs/player.mdl",    QNN_SUBJECT_PLAYER,             0,      0, 0},
	{"progs/eyes.mdl",      QNN_SUBJECT_PLAYER,             0, 0, 0},
	{"progs/backpack.mdl",  QNN_SUBJECT_BACKPACK,           0,      0, 0},
	{"progs/spike.mdl",     QNN_SUBJECT_PROJECTILE_NAIL,    0,      0, 0},
	{"progs/s_spike.mdl",   QNN_SUBJECT_PROJECTILE_NAIL,    0,      0, 0},
	{"progs/grenade.mdl",   QNN_SUBJECT_PROJECTILE_GRENADE, 0,      0, 0},
	{"progs/missile.mdl",   QNN_SUBJECT_PROJECTILE_ROCKET,  0,      0, 0},
	{"progs/bolt2.mdl",     QNN_SUBJECT_LIGHTNING_BEAM,     0,      0, 0},
	{"progs/bolt3.mdl",     QNN_SUBJECT_LIGHTNING_BEAM,     0,      0, 0},
	{"progs/teleport.mdl",  QNN_SUBJECT_TELEPORTER,         0,      0, 0},
	{"maps/b_bh10.bsp",     QNN_SUBJECT_HEALTH,             0,      15,  QNN_MAX_HEALTH},
	{"maps/b_bh25.bsp",     QNN_SUBJECT_HEALTH,             0,      25,  QNN_MAX_HEALTH},
	{"maps/b_bh100.bsp",    QNN_SUBJECT_MEGAHEALTH,         0,      100, QNN_MAX_HEALTH},
	{"maps/b_shell0.bsp",   QNN_SUBJECT_SHELLS,             0,      20,  QNN_MAX_SHELLS},
	{"maps/b_shell1.bsp",   QNN_SUBJECT_SHELLS,             0,      40,  QNN_MAX_SHELLS},
	{"maps/b_nail0.bsp",    QNN_SUBJECT_NAILS,              0,      25,  QNN_MAX_NAILS},
	{"maps/b_nail1.bsp",    QNN_SUBJECT_NAILS,              0,      50,  QNN_MAX_NAILS},
	{"maps/b_rock0.bsp",    QNN_SUBJECT_ROCKETS,            0,      5,   QNN_MAX_ROCKETS},
	{"maps/b_rock1.bsp",    QNN_SUBJECT_ROCKETS,            0,      10,  QNN_MAX_ROCKETS},
	{"maps/b_batt0.bsp",    QNN_SUBJECT_CELLS,              0,      6,   QNN_MAX_CELLS},
	{"maps/b_batt1.bsp",    QNN_SUBJECT_CELLS,              0,      12,  QNN_MAX_CELLS},
	{"progs/g_shot.mdl",    QNN_SUBJECT_SUPER_SHOTGUN,      0,      0, 0},
	{"progs/g_nail.mdl",    QNN_SUBJECT_NAILGUN,            0,      0, 0},
	{"progs/g_nail2.mdl",   QNN_SUBJECT_SUPER_NAILGUN,      0,      0, 0},
	{"progs/g_rock.mdl",    QNN_SUBJECT_GRENADE_LAUNCHER,   0,      0, 0},
	{"progs/g_rock2.mdl",   QNN_SUBJECT_ROCKET_LAUNCHER,    0,      0, 0},
	{"progs/g_light.mdl",   QNN_SUBJECT_THUNDERBOLT,        0,      0, 0},
	{"progs/quaddama.mdl",  QNN_SUBJECT_QUAD,               0,      1, 1},
	{"progs/invulner.mdl",  QNN_SUBJECT_PENT,               0,      1, 1},
	{"progs/invisibl.mdl",  QNN_SUBJECT_RING,               0,      1, 1},
	{"progs/suit.mdl",      QNN_SUBJECT_SUIT,               0,      1, 1},
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
			*magnitude = QNN_Normalize(100.0f, QNN_MAX_HEALTH);
			return true;
		}
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(15.0f, QNN_MAX_HEALTH) : QNN_Normalize(25.0f, QNN_MAX_HEALTH);
		return true;
	}
	if (!strcasecmp(classname, "item_health_rotten"))
	{
		*subject_id = QNN_SUBJECT_HEALTH;
		*magnitude = QNN_Normalize(15.0f, QNN_MAX_HEALTH);
		return true;
	}
	if (!strcasecmp(classname, "item_health_mega"))
	{
		*subject_id = QNN_SUBJECT_MEGAHEALTH;
		*magnitude = QNN_Normalize(100.0f, QNN_MAX_HEALTH);
		return true;
	}
	if (!strcasecmp(classname, "item_armor1"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_GREEN;
		*magnitude = (100.0f * 0.3f) / QNN_MAX_ARMOR;
		return true;
	}
	if (!strcasecmp(classname, "item_armor2"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_YELLOW;
		*magnitude = (150.0f * 0.6f) / QNN_MAX_ARMOR;
		return true;
	}
	if (!strcasecmp(classname, "item_armorInv"))
	{
		*subject_id = QNN_SUBJECT_ARMOR_RED;
		*magnitude = (200.0f * 0.8f) / QNN_MAX_ARMOR;
		return true;
	}
	if (!strcasecmp(classname, "item_shells"))
	{
		*subject_id = QNN_SUBJECT_SHELLS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(40.0f, QNN_MAX_SHELLS) : QNN_Normalize(20.0f, QNN_MAX_SHELLS);
		return true;
	}
	if (!strcasecmp(classname, "item_spikes"))
	{
		*subject_id = QNN_SUBJECT_NAILS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(50.0f, QNN_MAX_NAILS) : QNN_Normalize(25.0f, QNN_MAX_NAILS);
		return true;
	}
	if (!strcasecmp(classname, "item_rockets"))
	{
		*subject_id = QNN_SUBJECT_ROCKETS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(10.0f, QNN_MAX_ROCKETS) : QNN_Normalize(5.0f, QNN_MAX_ROCKETS);
		return true;
	}
	if (!strcasecmp(classname, "item_cells"))
	{
		*subject_id = QNN_SUBJECT_CELLS;
		*magnitude = (spawnflags & 1) ? QNN_Normalize(12.0f, QNN_MAX_CELLS) : QNN_Normalize(6.0f, QNN_MAX_CELLS);
		return true;
	}
	if (!strcasecmp(classname, "weapon_supershotgun"))
	{
		*subject_id = QNN_SUBJECT_SUPER_SHOTGUN;
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
		*subject_id = QNN_SUBJECT_SUPER_NAILGUN;
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
	*qualifier_id = 0;
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
	*qualifier_id = 0;
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
		if (skin <= 0)      { *subject_id = QNN_SUBJECT_ARMOR_GREEN;  *magnitude = (100.0f * 0.3f) / QNN_MAX_ARMOR; }
		else if (skin == 1) { *subject_id = QNN_SUBJECT_ARMOR_YELLOW; *magnitude = (150.0f * 0.6f) / QNN_MAX_ARMOR; }
		else                { *subject_id = QNN_SUBJECT_ARMOR_RED;    *magnitude = (200.0f * 0.8f) / QNN_MAX_ARMOR; }
		return true;
	}

	return false;
}

int QNN_SubjectPickupCategory(int subject_id)
{
	if (subject_id == QNN_SUBJECT_HEALTH)
		return 1;
	if (subject_id == QNN_SUBJECT_MEGAHEALTH)
		return 2;
	if (subject_id == QNN_SUBJECT_ARMOR_GREEN || subject_id == QNN_SUBJECT_ARMOR_YELLOW || subject_id == QNN_SUBJECT_ARMOR_RED)
		return 3;
	if (subject_id == QNN_SUBJECT_SHOTGUN || subject_id == QNN_SUBJECT_SUPER_SHOTGUN || subject_id == QNN_SUBJECT_NAILGUN || subject_id == QNN_SUBJECT_SUPER_NAILGUN || subject_id == QNN_SUBJECT_GRENADE_LAUNCHER || subject_id == QNN_SUBJECT_ROCKET_LAUNCHER || subject_id == QNN_SUBJECT_THUNDERBOLT)
		return 4;
	if (subject_id == QNN_SUBJECT_SHELLS || subject_id == QNN_SUBJECT_NAILS || subject_id == QNN_SUBJECT_ROCKETS || subject_id == QNN_SUBJECT_CELLS)
		return 5;
	if (subject_id == QNN_SUBJECT_QUAD || subject_id == QNN_SUBJECT_PENT || subject_id == QNN_SUBJECT_RING || subject_id == QNN_SUBJECT_SUIT)
		return 6;
	return 0;
}

#define QNN_VIEW_OFS_Z 22.0f

/* Model's view-cone aperture. Decoupled from the renderer's "fov" cvar
 * (which the engine clamps to [10,170] in SCR_CalcRefdef) so we can open
 * the perception cone past a hemisphere. Registered via
 * QNN_RegisterPerceptionCvars (QNN_IOInit, plus client startup so configs
 * exec'd before the first QNN tick can set it). */
cvar_t qnn_fov = {"qnn_fov", "120"};

/* Idempotent: a cvar set from a config keeps its value across re-calls
 * (QNN_IOInit re-runs per map load). Must run before any config exec
 * that sets qnn_fov — an unregistered cvar is "Unknown command" and the
 * set is silently dropped. */
void QNN_RegisterPerceptionCvars(void)
{
	if (Cvar_FindVar("qnn_fov") == NULL)
		Cvar_RegisterVariable(&qnn_fov);
}

/*
 * Emit FOV cone aperture, in total degrees, read live from "qnn_fov".
 *
 * qnn_fov is the *total* apex angle of the model's view cone, read every
 * call so it can be changed at the console mid-game:
 *   0    SIGHT disabled entirely — QNN_InFov always false. Lets us test
 *        the bot on non-sight modalities only (proximity/sound/memory).
 *   120  the training-time emit geometry (default).
 *   360  full all-around; only the line-of-sight trace gates visibility.
 *
 * Out-of-range values are clamped to [0, 360] in place via Cvar_Set (NQ
 * cvars have no set-time hook), the same idiom SCR_CalcRefdef uses for
 * fov/viewsize, so the console value always reflects the real aperture.
 *
 * Defaults to 120 when the cvar isn't registered (a build path that never
 * calls QNN_IOInit), so a missing registration degrades to the trained
 * geometry rather than silently zeroing sight. The cached pointer's
 * ->value tracks console changes live.
 */
static float QNN_EmitFovDeg(void)
{
	static cvar_t *cv = NULL;

	if (cv == NULL)
		cv = Cvar_FindVar("qnn_fov");
	if (cv == NULL)
		return 120.0f; /* unregistered -> trained default */

	if (cv->value < 0.0f)
		Cvar_Set("qnn_fov", "0");
	else if (cv->value > 360.0f)
		Cvar_Set("qnn_fov", "360");

	return cv->value;
}

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
	float fov_deg;
	trace_t trace;

	fov_deg = QNN_EmitFovDeg();
	if (fov_deg <= 0.0f)
		return false; /* qnn_fov 0 -> no SIGHT, ever (before the close-range path) */

	AngleVectors(view_angles, forward, right, up);
	VectorSubtract(target, player_origin, delta);
	dist = QNN_VecLength(delta);
	if (dist < 1.0f)
		return true;
	dot = DotProduct(forward, delta) / dist;
	cos_half = cosf(fov_deg * 0.5f * (float)(M_PI / 180.0));
	if (dot < cos_half)
		return false;

	if (!cl.worldmodel)
		return true;

	VectorCopy(player_origin, eye);
	eye[2] += QNN_VIEW_OFS_Z;
	memset(&trace, 0, sizeof(trace));
	QNN_TraceLine(eye, target, &trace);
	return trace.fraction >= 1.0f;
}

/* ══════════════════════════════════════════════════════════════════
 * Frag helper (shared)
 *
 * Team detection (QNN_IsSameTeam, QNN_PlayersResetTeamCache) lives in
 * the per-game qnn_players.c — pants-color for NQ (engine team field),
 * userinfo "team" + cl.teamplay for QW.
 * ══════════════════════════════════════════════════════════════════ */

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
		qualifier_id = 0;
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

/* PVS check: returns true if any leaf touched by the entity's bounding
 * box is visible from the viewer's PVS.
 *
 * NQ demos are PVS-culled at record time, so cl_entities only contains
 * entities the player could see — this returns true cheaply for all of
 * them.  QW MVD demos contain ALL entities (server-side recording with
 * no PVS culling), so this filter prevents the model from training on
 * omniscient state.
 *
 * Uses a 32-byte cluster bitmap from the viewer's leaf and tests the
 * entity's center leaf against it. */
qboolean QNN_EntityInPvs(const vec3_t viewer, const vec3_t target)
{
	mleaf_t *vleaf, *tleaf;
	byte *vis;
	int cluster;

	/* qnn_arena8's generated geometry is a fixed 4x2 grid: 64-unit walls,
	 * 512-unit interiors, and therefore a 576-unit cell pitch.  Match identity
	 * is a stronger boundary than renderer PVS state (which can fail open for
	 * zero-portal BSPs), and applies equally to actors, projectiles, and sounds. */
	if (!strncmp(qnn_map_state.requested_map_id, "qnn_arena", 9))
	{
		int viewer_col = (int)floorf((viewer[0] - 64.0f) / 576.0f);
		int viewer_row = (int)floorf((viewer[1] - 64.0f) / 576.0f);
		int target_col = (int)floorf((target[0] - 64.0f) / 576.0f);
		int target_row = (int)floorf((target[1] - 64.0f) / 576.0f);
		return viewer_col == target_col && viewer_row == target_row;
	}

	if (cl.worldmodel == NULL)
		return true; /* fail open */

	vleaf = Mod_PointInLeaf((float *)viewer, cl.worldmodel);
	if (vleaf == NULL || vleaf == cl.worldmodel->leafs)
		return true; /* in solid or root — fail open */

	tleaf = Mod_PointInLeaf((float *)target, cl.worldmodel);
	if (tleaf == NULL || tleaf == cl.worldmodel->leafs)
		return false; /* target in solid */

	if (tleaf == vleaf)
		return true;

	vis = Mod_LeafPVS(vleaf, cl.worldmodel);
	cluster = (int)(tleaf - cl.worldmodel->leafs) - 1;
	if (cluster < 0)
		return false;
	return (vis[cluster >> 3] & (1 << (cluster & 7))) ? true : false;
}

int QNN_EntityClassifyKnown(const qnn_snapshot_t *snapshot,
	qnn_entity_update_t *out_entities, int max_entities,
	qnn_pvs_item_t *out_pvs, int max_pvs, int *out_pvs_count)
{
	int entity_count = 0;
	int pvs_count = 0;
	int entity_num;
	float server_dt = (float)(cl.mtime[0] - cl.mtime[1]);
	/* Apply the BSP PVS explicitly for every transport.  Stock NQ normally
	 * mirrors the server filter by nulling cl_entities[].model, but player
	 * baselines and state-only arena resets can legitimately keep a stale
	 * model pointer.  Geometry is the authoritative visibility boundary. */
	qboolean apply_pvs = true;
	if (server_dt < 0.001f || server_dt > 0.5f)
		server_dt = 1.0f / 20.0f; /* fallback if mtime is stale or bogus */

	for (entity_num = 1; entity_num < cl.num_entities; ++entity_num)
	{
		entity_t *entity;
		const char *model_name;
		int subject_id, qualifier_id;
		float magnitude;
		qboolean is_item, in_fov, is_brush;
		vec3_t anchor_origin;
		float half_extents[3];

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
			qualifier_id = 0;
			magnitude = 0.0f;
		}
		else if (!QNN_ClassifyByModel(model_name, entity->skinnum, &subject_id, &qualifier_id, &magnitude))
			continue;

		/* Runtime players are engine-specific transport.  NQ materializes
		 * them in cl_entities[]; QW carries them in playerstate[].  Let the
		 * engine-local player scanner own all actor updates so QW baseline
		 * shims cannot become live-player tokens by accident. */
		if (subject_id == QNN_SUBJECT_PLAYER)
			continue;

		is_item = (!is_brush && QNN_SubjectIsItem(subject_id));
		QNN_EntityAnchorFromModel(entity_num, entity->origin, anchor_origin, half_extents);
		in_fov = QNN_InFov(snapshot->player_origin, snapshot->player_view_angles, anchor_origin);

		/* Cull entities outside the viewer's PVS. */
		if (apply_pvs && !QNN_EntityInPvs(snapshot->player_origin, anchor_origin))
			continue;

		/* Items and brush movers go into PVS list for timestamp stamping */
		if (is_item || is_brush)
		{
			if (pvs_count < max_pvs)
			{
				out_pvs[pvs_count].entity_num = entity_num;
				out_pvs[pvs_count].subject_id = subject_id;
				VectorCopy(anchor_origin, out_pvs[pvs_count].origin);
				out_pvs[pvs_count].magnitude = magnitude;
				out_pvs[pvs_count].in_fov = in_fov;
				pvs_count++;
			}
			if (is_item)
				continue;
			/* Brush movers also need entity_updates for position/state */
		}

		/* Non-item entities: collect regardless of FOV.
		   FOV is stored for emission modality selection. */
		if (entity_num <= 0)
			continue;

		if (entity_count < max_entities)
		{
			qnn_entity_update_t *eu = &out_entities[entity_count];
			vec3_t delta;

			memset(eu, 0, sizeof(*eu));
			eu->entity_num = entity_num;
			eu->subject_id = subject_id;
			eu->qualifier_id = qualifier_id;
			eu->magnitude = magnitude;
			eu->is_brush = is_brush;
			VectorCopy(anchor_origin, eu->origin);

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

			eu->half_extents[0] = half_extents[0];
			eu->half_extents[1] = half_extents[1];
			eu->half_extents[2] = half_extents[2];

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

	entity_count = QNN_AppendPlayerEntityUpdates(snapshot,
		out_entities, entity_count, max_entities);

	*out_pvs_count = pvs_count;
	return entity_count;
}
